#include "ADG/Builder.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Artifact/FabricArtifactMigration.h"
#include "Fabric/Artifact/FabricSystemContracts.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FifoResourceContract.h"

#include "Common/ArtifactFinalizer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using loom::ArtifactIdentity;
using loom::ArtifactRootReference;
using loom::ArtifactStore;
using loom::CanonicalSemanticBytes;
using loom::SchemaVersion;
using loom::adg::BoundarySpec;
using loom::adg::DesignBuilder;
using loom::adg::FifoSpec;
using loom::adg::PortType;
using loom::adg::SpatialValue;
using loom::fabric::FinalizedFabricRoot;

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

void expectRejected(llvm::StringRef test, llvm::Error error,
                    llvm::StringRef diagnostic) {
  if (!error)
    fail(test, "accepted invalid input");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(diagnostic), message);
}

template <typename T>
void expectRejected(llvm::StringRef test, llvm::Expected<T> value,
                    llvm::StringRef diagnostic) {
  if (value)
    fail(test, "accepted invalid input");
  expectRejected(test, value.takeError(), diagnostic);
}

class TemporaryDirectory {
public:
  explicit TemporaryDirectory(llvm::StringRef test) : test_(test.str()) {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-fabric-queue-discipline-test", path))
      fail(test, error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << test_ << ": unable to remove temporary directory: "
                   << error.message() << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  std::string test_;
  std::string path_;
};

mlir::MLIRContext &context() {
  static mlir::MLIRContext *ctx = [] {
    mlir::DialectRegistry registry;
    registry.insert<::fabric::FabricDialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect>();
    auto *result =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    result->loadAllAvailableDialects();
    return result;
  }();
  return *ctx;
}

mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef test,
                                        llvm::StringRef source) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  if (!module)
    fail(test, "unable to parse Fabric source");
  return module;
}

::fabric::ModuleOp moduleRoot(llvm::StringRef test, mlir::ModuleOp module) {
  ::fabric::ModuleOp selected;
  for (::fabric::ModuleOp candidate : module.getOps<::fabric::ModuleOp>()) {
    if (selected)
      fail(test, "fixture has more than one root");
    selected = candidate;
  }
  if (!selected)
    fail(test, "fixture has no Fabric root");
  return selected;
}

::fabric::SystemOp systemRoot(llvm::StringRef test, mlir::ModuleOp module) {
  ::fabric::SystemOp selected;
  for (::fabric::SystemOp candidate : module.getOps<::fabric::SystemOp>()) {
    if (selected)
      fail(test, "fixture has more than one System root");
    selected = candidate;
  }
  if (!selected)
    fail(test, "fixture has no System root");
  return selected;
}

loom::fabric::FabricModuleTemplateRef
uniqueModuleTemplate(llvm::StringRef test,
                     const loom::fabric::FabricArtifactView &view) {
  std::optional<loom::fabric::FabricModuleTemplateRef> result;
  for (std::uint64_t id = 0;; ++id) {
    auto kind = view.entityKind(id);
    if (!kind)
      break;
    if (*kind != loom::fabric::FabricEntityKind::FabricModuleTemplate)
      continue;
    if (result)
      fail(test, "fixture has more than one canonical module template");
    result = loom::fabric::FabricModuleTemplateRef(id);
  }
  if (!result)
    fail(test, "fixture has no canonical module template");
  return *result;
}

std::string denseI8Assembly(llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  std::string text;
  llvm::raw_string_ostream stream(text);
  mlir::DenseI8ArrayAttr::get(&context(), signedBytes).print(stream);
  return text;
}

::fabric::ResourceContract instructionContextContract(llvm::StringRef test) {
  ::fabric::ResourceContractDeclaration declaration;
  declaration.states = {::fabric::ResourceStateDeclaration{
      ::fabric::StateKey(0),
      {::fabric::CapacityDimensionDeclaration{::fabric::CapacityDimensionKey(0),
                                              ::fabric::CapacityUnits(1),
                                              ::fabric::CapacityUnits(0)}}}};
  declaration.timingContracts = {::fabric::TimingContractDeclaration{
      ::fabric::TimingContractKey(0), {0, 1}}};
  declaration.requesters = {::fabric::RequesterKey(0)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 2;
  declaration.usePatterns = {::fabric::UsePatternDeclaration{
      ::fabric::UsePatternKey(0),
      ::fabric::RequesterKey(0),
      ::fabric::EligibilityKey(0),
      ::fabric::EventKey(0),
      ::fabric::EventKey(1),
      std::nullopt,
      ::fabric::TimingContractKey(0),
      {::fabric::ClaimDeclaration{::fabric::ClaimKey(0), ::fabric::StateKey(0),
                                  ::fabric::CapacityDimensionKey(0),
                                  ::fabric::CapacityUnits(1)}},
      {::fabric::InternalTransactionDeclaration{{::fabric::ClaimKey(0)}}}}};
  return take(test, ::fabric::ResourceContract::create(declaration));
}

std::vector<std::uint8_t> instructionArchitecture(llvm::StringRef test) {
  loom::fabric::RiscVArchitectureDeclaration declaration;
  declaration.xlen = loom::fabric::RiscVXLen::X64;
  declaration.base = loom::fabric::RiscVBase::I;
  declaration.extensions = {loom::fabric::RiscVExtension::M,
                            loom::fabric::RiscVExtension::Zicsr};
  declaration.endianness = loom::fabric::InstructionEndianness::Little;
  declaration.physicalAddressWidthBits = 48;
  declaration.privilegeModes = {loom::fabric::PrivilegeMode::Machine};
  declaration.abiCapabilities = {loom::fabric::RiscVAbi::Lp64};
  declaration.memoryOrdering = loom::fabric::RiscVMemoryOrdering::Rvwmo;
  declaration.syncScopes = {loom::fabric::InstructionSyncScope::Hart};
  declaration.codeModels = {loom::fabric::RiscVCodeModel::MediumAny};
  declaration.relocationModels = {loom::fabric::RelocationModel::Static};
  declaration.runtimeServices = {
      loom::fabric::InstructionRuntimeService::ThreadDispatch,
      loom::fabric::InstructionRuntimeService::SpatialLaunch};
  auto contract =
      take(test, loom::fabric::InstructionCoreArchitecturalContract::create(
                     std::move(declaration)));
  return take(
      test, loom::fabric::encodeInstructionCoreArchitecturalContract(contract));
}

std::vector<std::uint8_t> instructionMicroarchitecture(llvm::StringRef test) {
  loom::fabric::InstructionCoreCommonDeclaration common{
      1,
      {{loom::fabric::InstructionOperationClass::IntegerAlu, 1, 1, 1}},
      instructionContextContract(test)};
  loom::fabric::InOrderMicroarchitectureDeclaration pipeline{1, 1, 1, 1,
                                                             1, 1, 2, 1};
  auto realization = take(
      test,
      loom::fabric::InstructionCoreMicroarchitecturalRealization::createInOrder(
          std::move(common), pipeline));
  return take(test,
              loom::fabric::encodeInstructionCoreMicroarchitecturalRealization(
                  realization));
}

std::string hostCoreSource(llvm::StringRef test) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "fabric.system.host_core architecture = "
         << denseI8Assembly(instructionArchitecture(test))
         << " microarchitecture = "
         << denseI8Assembly(instructionMicroarchitecture(test)) << "\n";
  return text;
}

std::string
accCoreSource(llvm::StringRef test,
              const loom::fabric::FabricImportedModuleTargetRef &target) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "fabric.system.acc_core architecture = "
         << denseI8Assembly(instructionArchitecture(test))
         << " microarchitecture = "
         << denseI8Assembly(instructionMicroarchitecture(test))
         << " spatial_core = "
         << denseI8Assembly(
                loom::fabric::encodeFabricImportedModuleTargetRef(target))
         << "\n";
  return text;
}

std::string
accCoreSystemSource(llvm::StringRef test,
                    const loom::fabric::FabricImportedModuleTargetRef &target) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "module { fabric.system @soc {\n"
         << hostCoreSource(test) << accCoreSource(test, target) << "\n} }\n";
  return text;
}

/// One SpatialCore module whose single tagged FIFO selects the given queue
/// discipline.
FinalizedFabricRoot taggedFifoModule(llvm::StringRef test, ArtifactStore &store,
                                     bool bypassable,
                                     ::fabric::FifoQueueDiscipline discipline,
                                     bool declareDiscipline) {
  DesignBuilder design(store);
  const PortType data = take(test, PortType::bits(32));
  const PortType tag = take(test, PortType::bits(4));
  const PortType tagged = take(test, PortType::taggedBits(32, 4));
  auto spatial = take(test, design.createSpatialCore("fifo", {data, tag},
                                                     {tagged}));
  SpatialValue dataInput = take(test, spatial.input(0));
  SpatialValue tagInput = take(test, spatial.input(1));
  auto boundary = take(
      test, spatial.addBoundary({dataInput, tagInput},
                                BoundarySpec::s2t(data, tag, tagged)));
  FifoSpec spec{tagged, 4, bypassable, std::nullopt};
  if (declareDiscipline)
    spec.queueDiscipline = discipline;
  SpatialValue queued =
      take(test, spatial.addFifo(boundary.front(), spec)).value();
  if (llvm::Error error = spatial.close({queued}))
    fail(test, llvm::toString(std::move(error)));
  auto completed = take(test, std::move(design).finalize());
  require(test, completed.roots().size() == 1,
          "fixture did not publish exactly one Fabric root");
  return completed.roots().front();
}

loom::fabric::FabricFifoOccurrenceRef
uniqueFifo(llvm::StringRef test, const loom::fabric::FabricArtifactView &view) {
  require(test, view.fifoOccurrences().size() == 1,
          "fixture changed its FIFO inventory");
  return view.fifoOccurrences().front();
}

void strictFifoRoundTrip() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  FinalizedFabricRoot finalized =
      taggedFifoModule(test, store, false, ::fabric::FifoQueueDiscipline::StrictFifo,
                       /*declareDiscipline=*/false);
  const loom::fabric::FabricFifoOccurrenceRef fifo =
      uniqueFifo(test, finalized.view());
  require(test,
          finalized.view().fifoQueueDiscipline(fifo) ==
              ::fabric::FifoQueueDiscipline::StrictFifo,
          "absent queue discipline does not project as StrictFifo");

  FinalizedFabricRoot imported = take(
      test, loom::fabric::importEntireFabricRoot(finalized.reference(), store));
  require(test,
          imported.reference().artifact == finalized.reference().artifact &&
              imported.view().fifoQueueDiscipline(fifo) ==
                  ::fabric::FifoQueueDiscipline::StrictFifo,
          "strict 7.1 import changed a StrictFifo occurrence");
}

void perTagVirtualChannelRoundTrip() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  FinalizedFabricRoot finalized = taggedFifoModule(
      test, store, false, ::fabric::FifoQueueDiscipline::PerTagVirtualChannel,
      /*declareDiscipline=*/true);
  const loom::fabric::FabricFifoOccurrenceRef fifo =
      uniqueFifo(test, finalized.view());
  require(test,
          finalized.view().fifoQueueDiscipline(fifo) ==
              ::fabric::FifoQueueDiscipline::PerTagVirtualChannel,
          "declared discipline did not survive finalization");

  FinalizedFabricRoot imported = take(
      test, loom::fabric::importEntireFabricRoot(finalized.reference(), store));
  require(test,
          imported.reference().artifact == finalized.reference().artifact &&
              imported.view().fifoQueueDiscipline(fifo) ==
                  ::fabric::FifoQueueDiscipline::PerTagVirtualChannel,
          "strict 7.1 import changed a PerTagVirtualChannel occurrence");
}

void disciplineSelectionChangesIdentity() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  FinalizedFabricRoot strict =
      taggedFifoModule(test, store, false, ::fabric::FifoQueueDiscipline::StrictFifo,
                       /*declareDiscipline=*/false);
  FinalizedFabricRoot virtualChannel = taggedFifoModule(
      test, store, false, ::fabric::FifoQueueDiscipline::PerTagVirtualChannel,
      /*declareDiscipline=*/true);
  require(test,
          strict.reference().artifact != virtualChannel.reference().artifact,
          "queue discipline selection did not change canonical identity");
}

void coldRebuildKeepsIdentity() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory firstDirectory("cold-first");
  TemporaryDirectory secondDirectory("cold-second");
  ArtifactStore firstStore(firstDirectory.path());
  ArtifactStore secondStore(secondDirectory.path());
  FinalizedFabricRoot first = taggedFifoModule(
      test, firstStore, false, ::fabric::FifoQueueDiscipline::PerTagVirtualChannel,
      /*declareDiscipline=*/true);
  FinalizedFabricRoot second = taggedFifoModule(
      test, secondStore, false,
      ::fabric::FifoQueueDiscipline::PerTagVirtualChannel,
      /*declareDiscipline=*/true);
  require(test, first.reference() == second.reference(),
          "cold rebuild changed the 7.1 identity of equal semantics");
}

void untaggedFifoRejectsVirtualChannel() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType data = take(test, PortType::bits(32));
  auto spatial = take(test, design.createSpatialCore("fifo", {data}, {data}));
  SpatialValue dataInput = take(test, spatial.input(0));
  FifoSpec spec{data, 4, false,
                ::fabric::FifoQueueDiscipline::PerTagVirtualChannel};
  expectRejected(test, spatial.addFifo(dataInput, spec),
                 "rejected the typed FIFO");
}

void virtualChannelRejectsBypassCapability() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType data = take(test, PortType::bits(32));
  const PortType tag = take(test, PortType::bits(4));
  const PortType tagged = take(test, PortType::taggedBits(32, 4));
  auto spatial = take(test, design.createSpatialCore("fifo", {data, tag},
                                                     {tagged}));
  SpatialValue dataInput = take(test, spatial.input(0));
  SpatialValue tagInput = take(test, spatial.input(1));
  auto boundary = take(
      test, spatial.addBoundary({dataInput, tagInput},
                                BoundarySpec::s2t(data, tag, tagged)));
  FifoSpec spec{tagged, 4, true,
                ::fabric::FifoQueueDiscipline::PerTagVirtualChannel};
  expectRejected(test, spatial.addFifo(boundary.front(), spec),
                 "rejected the typed FIFO");
}

ArtifactRootReference publish7_0Twin(llvm::StringRef test, ArtifactStore &store,
                                     const CanonicalSemanticBytes &bytes) {
  ArtifactIdentity identity = loom::finalizeArtifactIdentity(
      loom::fabric::fabricArtifactSchemaV7_0, bytes);
  ArtifactRootReference reference{
      loom::fabric::fabricArtifactSchemaV7_0.identity.str(),
      loom::fabric::fabricArtifactSchemaV7_0.version, identity};
  ArtifactIdentity stored = take(
      test, store.put(loom::fabric::fabricArtifactSchemaV7_0, bytes));
  require(test, stored == identity, "ArtifactStore rewrote a 7.0 identity");
  return reference;
}

void dependencyOrderCodecRanksRowsCanonically() {
  const llvm::StringRef test = __func__;
  const auto reference = [&](std::uint8_t seed) {
    return ArtifactRootReference{
        loom::fabric::fabricArtifactSchemaV7_0.identity.str(),
        loom::fabric::fabricArtifactSchemaV7_0.version,
        take(test, ArtifactIdentity::fromBytes(
                       std::vector<std::uint8_t>(ArtifactIdentity::byteSize,
                                                 seed)))};
  };
  const loom::fabric::FabricDirectDependency low{
      loom::fabric::FabricDependencyRole::ImportedModule, reference(0x10)};
  const loom::fabric::FabricDirectDependency high{
      loom::fabric::FabricDependencyRole::ImportedModule, reference(0x20)};
  auto forward = loom::fabric::canonicalFabricDependencyOrder({low, high});
  auto reversed = loom::fabric::canonicalFabricDependencyOrder({high, low});
  if (!forward)
    fail(test, llvm::toString(forward.takeError()));
  if (!reversed)
    fail(test, llvm::toString(reversed.takeError()));
  require(test, *forward == std::vector<std::uint32_t>({0, 1}) &&
                *reversed == std::vector<std::uint32_t>({1, 0}),
          "dependency order codec did not rank rows by canonical bytes");
}

void migrationRefinalizesModule() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  auto source = parse(test, R"mlir(
    module {
      fabric.module @empty() {
        fabric.yield
      }
    }
  )mlir");
  FinalizedFabricRoot native = take(
      test, loom::fabric::finalizeFabricRoot(moduleRoot(test, *source), store));
  require(test,
          native.reference().schemaVersion == SchemaVersion{7, 1},
          "fixture is not a current loom.fabric artifact");
  CanonicalSemanticBytes bytes =
      take(test, store.get(native.reference()));
  const ArtifactRootReference twin70 = publish7_0Twin(test, store, bytes);
  require(test, twin70.artifact != native.reference().artifact,
          "the 7.0 descriptor shares the 7.1 identity");

  // The ordinary 7.1 importer never silently accepts a 7.0 reference.
  expectRejected(test,
                 loom::fabric::importEntireFabricRoot(twin70, store),
                 "wrong Fabric schema");
  // Migration rejects a reference that is not an exact 7.0 root.
  expectRejected(test,
                 loom::fabric::migrateFabricRootV7_0ToV7_1(
                     native.reference(), store),
                 "exact loom.fabric 7.0 root reference");

  const ArtifactRootReference migrated = take(
      test, loom::fabric::migrateFabricRootV7_0ToV7_1(twin70, store));
  require(test, migrated == native.reference(),
          "migration did not reproduce the native 7.1 identity");
  const ArtifactRootReference repeated = take(
      test, loom::fabric::migrateFabricRootV7_0ToV7_1(twin70, store));
  require(test, repeated == native.reference(),
          "repeated migration changed the 7.1 identity");
}

void migrationRewritesDependencyClosure() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  auto moduleSource = parse(test, R"mlir(
    module { fabric.module @empty() { fabric.yield } }
  )mlir");
  FinalizedFabricRoot nativeModule = take(
      test, loom::fabric::finalizeFabricRoot(moduleRoot(test, *moduleSource),
                                             store));
  const loom::fabric::FabricImportedModuleTargetRef target{
      0, uniqueModuleTemplate(test, nativeModule.view())};
  auto systemSource = parse(test, accCoreSystemSource(test, target));
  FinalizedFabricRoot nativeSystem = take(
      test, loom::fabric::finalizeFabricRoot(systemRoot(test, *systemSource),
                                             {nativeModule.reference()},
                                             store));

  // Fabricate the 7.0 closure: identical canonical payloads under the 7.0
  // descriptor, with the System envelope naming the 7.0 Module reference.
  CanonicalSemanticBytes moduleBytes =
      take(test, store.get(nativeModule.reference()));
  const ArtifactRootReference module70 = publish7_0Twin(test, store, moduleBytes);
  auto decoded = loom::fabric::decodeFabricArtifactEnvelope(
      take(test, store.get(nativeSystem.reference())).bytes());
  require(test, decoded && decoded->dependencies.size() == 1,
          "fixture System lost its ImportedModule dependency");
  auto envelope70 = loom::fabric::encodeFabricArtifactEnvelope(
      decoded->rootKind,
      {loom::fabric::FabricDirectDependency{
          loom::fabric::FabricDependencyRole::ImportedModule, module70}},
      decoded->canonicalMlirBytecode);
  if (!envelope70)
    fail(test, llvm::toString(envelope70.takeError()));
  const ArtifactRootReference system70 = publish7_0Twin(test, store, *envelope70);
  require(test, system70.artifact != nativeSystem.reference().artifact,
          "the 7.0 System shares the 7.1 identity");

  expectRejected(test, loom::fabric::importEntireFabricRoot(system70, store),
                 "wrong Fabric schema");
  const ArtifactRootReference migrated = take(
      test, loom::fabric::migrateFabricRootV7_0ToV7_1(system70, store));
  require(test, migrated == nativeSystem.reference(),
          "recursive migration did not reproduce the native 7.1 System");
}

} // namespace

int main() {
  strictFifoRoundTrip();
  perTagVirtualChannelRoundTrip();
  disciplineSelectionChangesIdentity();
  coldRebuildKeepsIdentity();
  untaggedFifoRejectsVirtualChannel();
  virtualChannelRejectsBypassCapability();
  migrationRefinalizesModule();
  migrationRewritesDependencyClosure();
  dependencyOrderCodecRanksRowsCanonically();
  return EXIT_SUCCESS;
}
