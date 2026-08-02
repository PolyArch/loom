#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricHardwareDomainContracts.h"
#include "Fabric/Artifact/FabricSystemContracts.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/Identity/FabricFuCapabilityTemplate.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "Common/ArtifactFinalizer.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FuCapabilityDomain.h"

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

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

using loom::ArtifactIdentity;
using loom::ArtifactStore;
using loom::CanonicalSemanticBytes;
using loom::fabric::FinalizedFabricRoot;

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

void requireFabricError(llvm::StringRef test, llvm::Error error,
                        loom::fabric::FabricRefErrorKind expected) {
  if (!error)
    fail(test, "accepted an invalid finalized Fabric reference");
  const loom::fabric::FabricRefErrorKind actual =
      loom::fabric::takeFabricRefErrorKind(std::move(error));
  require(test, actual == expected,
          "finalized Fabric reference failure kind changed");
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectRejected(llvm::StringRef test, llvm::Expected<T> value,
                    llvm::StringRef diagnostic) {
  if (value)
    fail(test, "accepted invalid input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(diagnostic), message);
}

class TemporaryDirectory {
public:
  explicit TemporaryDirectory(llvm::StringRef test) : test_(test.str()) {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-fabric-finalizer-test", path))
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
    registry.insert<::fabric::FabricDialect>();
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

::fabric::ModuleOp root(llvm::StringRef test, mlir::ModuleOp module) {
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

void setOperationResourceContracts(llvm::StringRef test, mlir::ModuleOp module,
                                   const ::fabric::ResourceContract &contract) {
  std::vector<std::uint8_t> bytes =
      take(test, ::fabric::encodeResourceContractRecord(contract));
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  module.walk([&](::fabric::OpOp operation) {
    operation->setAttr(
        ::fabric::kResourceContractRecordAttrName,
        mlir::DenseI8ArrayAttr::get(module.getContext(), signedBytes));
  });
}

void setFuCapabilityDomain(
    llvm::StringRef test, ::fabric::FuOp fu,
    std::vector<::fabric::FuCapabilityTemplateSelection> selections) {
  auto domain = take(
      test, ::fabric::FuCapabilityDomainRecord::create(std::move(selections)));
  std::vector<std::uint8_t> bytes =
      take(test, ::fabric::encodeFuCapabilityDomainRecord(domain));
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  fu.setCapabilityTemplatesAttr(::fabric::FuCapabilityDomainAttr::get(
      &context(), mlir::DenseI8ArrayAttr::get(&context(), signedBytes)));
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

std::vector<std::uint8_t> instructionArchitecture(llvm::StringRef test,
                                                  bool x64 = true,
                                                  bool floatingAbi = false) {
  loom::fabric::RiscVArchitectureDeclaration declaration;
  declaration.xlen =
      x64 ? loom::fabric::RiscVXLen::X64 : loom::fabric::RiscVXLen::X32;
  declaration.base = loom::fabric::RiscVBase::I;
  declaration.extensions = {loom::fabric::RiscVExtension::M,
                            loom::fabric::RiscVExtension::Zicsr};
  if (floatingAbi) {
    declaration.extensions.push_back(loom::fabric::RiscVExtension::F);
    declaration.extensions.push_back(loom::fabric::RiscVExtension::D);
  }
  declaration.endianness = loom::fabric::InstructionEndianness::Little;
  declaration.physicalAddressWidthBits = x64 ? 48 : 32;
  declaration.privilegeModes = {loom::fabric::PrivilegeMode::Machine};
  declaration.abiCapabilities = {
      x64 ? (floatingAbi ? loom::fabric::RiscVAbi::Lp64d
                         : loom::fabric::RiscVAbi::Lp64)
          : (floatingAbi ? loom::fabric::RiscVAbi::Ilp32d
                         : loom::fabric::RiscVAbi::Ilp32)};
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

std::string hostCoreSource(llvm::StringRef test, bool x64 = true,
                           bool floatingAbi = false) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "fabric.system.host_core architecture = "
         << denseI8Assembly(instructionArchitecture(test, x64, floatingAbi))
         << " microarchitecture = "
         << denseI8Assembly(instructionMicroarchitecture(test)) << "\n";
  return text;
}

std::string
accCoreSource(llvm::StringRef test,
              const loom::fabric::FabricImportedModuleTargetRef &target,
              bool x64 = true, bool floatingAbi = false) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "fabric.system.acc_core architecture = "
         << denseI8Assembly(instructionArchitecture(test, x64, floatingAbi))
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

std::string attachedAccCoreSystemSource(
    llvm::StringRef test,
    const loom::fabric::FabricImportedModuleTargetRef &target,
    bool attachOutput) {
  constexpr std::uint64_t coreId = 17;
  const loom::fabric::FabricImportedModuleBoundaryEndpointRef moduleInput{
      target.dependencyOrdinal,
      {target.target, loom::fabric::FabricPortDirection::Input, 0}};
  const loom::fabric::FabricImportedModuleBoundaryEndpointRef moduleOutput{
      target.dependencyOrdinal,
      {target.target, loom::fabric::FabricPortDirection::Output, 0}};
  const auto spatialOwner = loom::fabric::FabricTransportEndpointOwnerRef::of(
      loom::fabric::SpatialCoreOccurrenceRef{
          loom::fabric::AccCoreOccurrenceRef(coreId)});
  const auto spatialInput = take(
      test, loom::fabric::FabricSpatialAttachmentEndpointRef::create(
                loom::fabric::FabricTransportEndpointRef{spatialOwner, 0}));
  const auto spatialOutput = take(
      test, loom::fabric::FabricSpatialAttachmentEndpointRef::create(
                loom::fabric::FabricTransportEndpointRef{spatialOwner, 1}));

  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "module { fabric.system @soc {\n"
         << hostCoreSource(test) << "fabric.system.acc_core architecture = "
         << denseI8Assembly(instructionArchitecture(test))
         << " microarchitecture = "
         << denseI8Assembly(instructionMicroarchitecture(test))
         << " spatial_core = "
         << denseI8Assembly(
                loom::fabric::encodeFabricImportedModuleTargetRef(target))
         << " {entity_id = #fabric.entity_id<" << coreId << ">}\n"
         << "fabric.system.spatial_attachment module_endpoint = "
         << denseI8Assembly(
                loom::fabric::encodeFabricImportedModuleBoundaryEndpointRef(
                    moduleInput))
         << " spatial_endpoint = "
         << denseI8Assembly(
                loom::fabric::encodeFabricSpatialAttachmentEndpointRef(
                    spatialInput))
         << "\n";
  if (attachOutput)
    stream << "fabric.system.spatial_attachment module_endpoint = "
           << denseI8Assembly(
                  loom::fabric::encodeFabricImportedModuleBoundaryEndpointRef(
                      moduleOutput))
           << " spatial_endpoint = "
           << denseI8Assembly(
                  loom::fabric::encodeFabricSpatialAttachmentEndpointRef(
                      spatialOutput))
           << "\n";
  stream << "} }\n";
  return text;
}

std::string transportResource(llvm::StringRef test, std::uint64_t id,
                              std::uint32_t width,
                              llvm::ArrayRef<std::uint8_t> crossing = {}) {
  const std::vector<std::uint8_t> contract = take(
      test,
      ::fabric::encodeResourceContractRecord(instructionContextContract(test)));
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "fabric.system.transport_resource ports = (!fabric.bits<" << width
         << ">) -> (!fabric.bits<" << width
         << ">) contract = " << denseI8Assembly(contract);
  if (!crossing.empty())
    stream << " crossing = " << denseI8Assembly(crossing);
  stream << " {entity_id = #fabric.entity_id<" << id << ">}\n";
  return text;
}

std::string clockDomain(llvm::StringRef test, std::uint64_t id,
                        std::uint64_t memberId) {
  auto clock =
      take(test, loom::fabric::ClockDomainContractRecord::create(1'000, 0));
  auto record =
      take(test, loom::fabric::HardwareDomainContractRecord::create(
                     {loom::fabric::FabricInventoryOwnerRef::of(
                         loom::fabric::SystemTransportResourceRef(memberId))},
                     std::move(clock)));
  auto bytes =
      take(test, loom::fabric::encodeHardwareDomainContractRecord(record));
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "fabric.system.hardware_domain contract = "
         << denseI8Assembly(bytes) << " {entity_id = #fabric.entity_id<" << id
         << ">}\n";
  return text;
}

std::string transferPattern(llvm::StringRef test, std::uint64_t resourceId) {
  const loom::fabric::SystemTransportResourceRef resource(resourceId);
  const loom::fabric::FabricTransportEndpointOwnerRef owner =
      loom::fabric::FabricTransportEndpointOwnerRef::of(resource);
  const loom::fabric::FabricUsePatternRef usePattern{
      loom::fabric::FabricUsePatternOwnerRef(
          loom::fabric::FabricInventoryOwnerRef::of(resource)),
      0};
  auto record =
      take(test, loom::fabric::SystemTransferPatternRecord::create(
                     {resource, 0}, {owner, 0}, {{owner, 1}}, usePattern));
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "fabric.system.transfer_pattern contract = "
         << denseI8Assembly(
                loom::fabric::encodeSystemTransferPatternRecord(record))
         << "\n";
  return text;
}

std::string systemConnection(std::uint64_t sourceId,
                             std::uint64_t destinationId) {
  const loom::fabric::FabricTransportEndpointRef source{
      loom::fabric::FabricTransportEndpointOwnerRef::of(
          loom::fabric::SystemTransportResourceRef(sourceId)),
      1};
  const loom::fabric::FabricTransportEndpointRef destination{
      loom::fabric::FabricTransportEndpointOwnerRef::of(
          loom::fabric::SystemTransportResourceRef(destinationId)),
      0};
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "fabric.system.connection source = "
         << denseI8Assembly(loom::fabric::canonicalFabricBytes(source))
         << " destination = "
         << denseI8Assembly(loom::fabric::canonicalFabricBytes(destination))
         << "\n";
  return text;
}

std::string connectedTransportSystemSource(
    llvm::StringRef name, llvm::StringRef test,
    const loom::fabric::FabricImportedModuleTargetRef &target,
    std::uint64_t sourceId, std::uint64_t destinationId, bool reverseOrder) {
  const std::string source = transportResource(test, sourceId, 8);
  const std::string destination = transportResource(test, destinationId, 16);
  const std::string connection = systemConnection(sourceId, destinationId);
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "module { fabric.system @" << name << " {\n"
         << hostCoreSource(test) << accCoreSource(test, target);
  if (reverseOrder)
    stream << destination << connection << source;
  else
    stream << source << destination << connection;
  stream << "} }\n";
  return text;
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

FinalizedFabricRoot publishEmptySpatialCore(llvm::StringRef test,
                                            ArtifactStore &store) {
  auto source = parse(test, R"mlir(
    module { fabric.module @empty() { fabric.yield } }
  )mlir");
  return take(test,
              loom::fabric::finalizeFabricRoot(root(test, *source), store));
}

loom::fabric::FabricFuTemplateRef
uniqueFuTemplate(llvm::StringRef test,
                 const loom::fabric::FabricArtifactView &view) {
  std::optional<loom::fabric::FabricFuTemplateRef> result;
  for (std::uint64_t id = 0;; ++id) {
    std::optional<loom::fabric::FabricEntityKind> kind = view.entityKind(id);
    if (!kind)
      break;
    if (*kind != loom::fabric::FabricEntityKind::FabricFuTemplate)
      continue;
    if (result)
      fail(test, "fixture has more than one canonical FU template");
    result = loom::fabric::FabricFuTemplateRef(id);
  }
  if (!result)
    fail(test, "fixture has no canonical FU template");
  return *result;
}

std::string moduleSource(llvm::StringRef name, bool reverse) {
  const llvm::StringLiteral first = R"mlir(
    %x = fabric.fifo %a [max_depth = 2, bypassable = true]
         : !fabric.bits<32>
    %y = fabric.boundary [s2t] %x, %ta
         : (!fabric.bits<32>, !fabric.bits<4>)
        -> !fabric.bits_tag<32, 4>
  )mlir";
  const llvm::StringLiteral second = R"mlir(
    %u = fabric.fifo %b [max_depth = 3, bypassable = false]
         : !fabric.bits<32>
    %v = fabric.boundary [s2t] %u, %tb
         : (!fabric.bits<32>, !fabric.bits<4>)
        -> !fabric.bits_tag<32, 4>
  )mlir";

  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << "module { fabric.module @" << name
         << "(%a: !fabric.bits<32>, %ta: !fabric.bits<4>, "
            "%b: !fabric.bits<32>, %tb: !fabric.bits<4>) "
            "-> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>) {\n";
  if (reverse)
    stream << second << first;
  else
    stream << first << second;
  stream << "fabric.yield %y, %v : !fabric.bits_tag<32, 4>, "
            "!fabric.bits_tag<32, 4>\n} }\n";
  return stream.str();
}

void canonicalPublicationAndStrictImport() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  mlir::OwningOpRef<mlir::ModuleOp> first =
      parse(test, moduleSource("first", false));
  mlir::OwningOpRef<mlir::ModuleOp> second =
      parse(test, moduleSource("second", true));

  FinalizedFabricRoot firstResult =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *first), store));
  FinalizedFabricRoot secondResult =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *second), store));
  require(test,
          firstResult.reference().artifact == secondResult.reference().artifact,
          "source names or graph-region order changed Fabric identity");
  require(test, firstResult.directDependencies().empty(),
          "Module root retained a direct dependency");
  require(test, firstResult.view().pointConnections().size() == 2,
          "fixed FIFO-to-boundary connections were not imported");

  std::optional<loom::fabric::FabricFifoOccurrenceRef> nonBypassableFifo;
  for (const loom::fabric::FabricPhysicalTraversalRef &traversal :
       firstResult.view().admittedTraversals()) {
    if (llvm::Error error =
            loom::fabric::validateFabricRef(firstResult.view(), traversal))
      fail(test, llvm::toString(std::move(error)));
    if (traversal.kind() !=
        loom::fabric::FabricPhysicalTraversalKind::FifoTraversal)
      continue;
    const auto &fifo =
        std::get<loom::fabric::FabricFifoTraversalPayload>(traversal.payload);
    if (fifo.mode == loom::fabric::FabricFifoTraversalMode::Buffered &&
        !firstResult.view().admitsTraversal(
            loom::fabric::FabricPhysicalTraversalRef::fifoTraversal(
                fifo.owner, loom::fabric::FabricFifoTraversalMode::Bypass)))
      nonBypassableFifo = fifo.owner;
  }
  require(test, nonBypassableFifo.has_value(),
          "non-bypassable FIFO capability was not preserved");

  const loom::fabric::FabricInventoryOwnerRef fifoOwner =
      loom::fabric::FabricInventoryOwnerRef::of(*nonBypassableFifo);
  const ::fabric::ResourceContract *fifoContract =
      firstResult.view().resourceContract(fifoOwner);
  require(test,
          fifoContract && fifoContract->stateCount() == 1 &&
              fifoContract->usePatternCount() == 3,
          "finalized view did not expose the FIFO owner contract");
  requireFabricError(
      test,
      loom::fabric::validateFabricRef(
          firstResult.view(),
          loom::fabric::FabricPhysicalTraversalRef::fifoTraversal(
              *nonBypassableFifo,
              loom::fabric::FabricFifoTraversalMode::Bypass)),
      loom::fabric::FabricRefErrorKind::TraversalNotAdmitted);

  const loom::fabric::FabricPointConnectionPayload connection =
      firstResult.view().pointConnections().front();
  loom::fabric::FabricTransportEndpointRef stale = connection.destination;
  stale.ordinal = firstResult.view().transportEndpointCount(stale.owner);
  requireFabricError(test,
                     loom::fabric::validateFabricRef(firstResult.view(), stale),
                     loom::fabric::FabricRefErrorKind::OrdinalOutOfRange);

  std::vector<std::uint8_t> foreignBytes(ArtifactIdentity::byteSize, 0x5a);
  const ArtifactIdentity foreign =
      take(test, ArtifactIdentity::fromBytes(foreignBytes));
  requireFabricError(test,
                     loom::fabric::checkFabricBinding(
                         firstResult.view(),
                         loom::fabric::FabricImportBinding{
                             foreign, loom::fabric::FabricRootKind::Module}),
                     loom::fabric::FabricRefErrorKind::ForeignArtifact);

  FinalizedFabricRoot imported =
      take(test, loom::fabric::importEntireFabricRoot(firstResult.reference(),
                                                      store));
  require(test,
          imported.reference().artifact == firstResult.reference().artifact,
          "strict import changed Fabric identity");
  require(test,
          imported.canonicalBytes().bytes().equals(
              firstResult.canonicalBytes().bytes()),
          "strict import changed canonical bytes");
}

void visualizationCoordinatesHaveNoFabricSemantics() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  auto firstSource = parse(test, R"mlir(
    module {
      fabric.module @first() attributes {
        coordinates_semantic = false,
        visual_layout = [{node = "resource", x = 0 : i32, y = 0 : i32}]
      } { }
    }
  )mlir");
  auto secondSource = parse(test, R"mlir(
    module {
      fabric.module @second() attributes {
        coordinates_semantic = false,
        visual_layout = [{node = "resource", x = 91 : i32, y = -37 : i32}]
      } { }
    }
  )mlir");
  auto strippedSource = parse(test, R"mlir(
    module {
      fabric.module @stripped() { }
    }
  )mlir");

  FinalizedFabricRoot first = take(
      test, loom::fabric::finalizeFabricRoot(root(test, *firstSource), store));
  FinalizedFabricRoot second = take(
      test, loom::fabric::finalizeFabricRoot(root(test, *secondSource), store));
  FinalizedFabricRoot stripped =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *strippedSource),
                                                  store));
  require(test,
          first.reference().artifact == second.reference().artifact &&
              first.reference().artifact == stripped.reference().artifact,
          "visualization coordinates changed Fabric identity");
  require(test,
          first.canonicalBytes().bytes().equals(
              stripped.canonicalBytes().bytes()) &&
              first.view().rootKind() == stripped.view().rootKind() &&
              first.view().pointConnections() ==
                  stripped.view().pointConnections() &&
              first.view().admittedTraversals() ==
                  stripped.view().admittedTraversals(),
          "removing visualization metadata changed the sealed Fabric view");

  auto semanticSource = parse(test, R"mlir(
    module {
      fabric.module @semantic() attributes {
        coordinates_semantic = true,
        visual_layout = [{node = "resource", x = 0 : i32, y = 0 : i32}]
      } { }
    }
  )mlir");
  expectRejected(
      test,
      loom::fabric::finalizeFabricRoot(root(test, *semanticSource), store),
      "authoring coordinates claim semantic authority");
}

void malformedStoredPayloadIsRejected() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  std::vector<std::uint8_t> bytes = {'n', 'o', 't', '-', 'f',
                                     'a', 'b', 'r', 'i', 'c'};
  CanonicalSemanticBytes malformed(bytes);
  ArtifactIdentity identity =
      take(test, store.put(loom::fabric::fabricArtifactSchema, malformed));
  expectRejected(test,
                 loom::fabric::importEntireFabricRoot(
                     {loom::fabric::fabricArtifactSchema.identity.str(),
                      loom::fabric::fabricArtifactSchema.version, identity},
                     store),
                 "fabric_artifact_invalid");
}

void spatialSwitchConnectivityBecomesTraversals() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  mlir::OwningOpRef<mlir::ModuleOp> source = parse(test, R"mlir(
    module {
      fabric.module @switch_root(
          %a: !fabric.bits<32>, %b: !fabric.bits<32>)
          -> (!fabric.bits<32>, !fabric.bits<32>) {
        %x:2 = fabric.switch [spatial] %a, %b
          [{connectivity_table = ["11", "10"]}]
          : (!fabric.bits<32>, !fabric.bits<32>)
         -> (!fabric.bits<32>, !fabric.bits<32>)
        fabric.yield %x#0, %x#1
          : !fabric.bits<32>, !fabric.bits<32>
      }
    }
  )mlir");

  FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *source), store));
  std::vector<loom::fabric::FabricSwitchTraversalPayload> traversals;
  for (const loom::fabric::FabricPhysicalTraversalRef &traversal :
       finalized.view().admittedTraversals()) {
    if (traversal.kind() !=
        loom::fabric::FabricPhysicalTraversalKind::SwitchTraversal)
      continue;
    traversals.push_back(std::get<loom::fabric::FabricSwitchTraversalPayload>(
        traversal.payload));
  }
  require(test, traversals.size() == 3,
          "switch connectivity did not produce three physical traversals");
  bool input0Output0 = false;
  bool input1Output0 = false;
  bool input1Output1 = false;
  for (const auto &traversal : traversals) {
    input0Output0 |= traversal.input == 0 && traversal.output == 0;
    input1Output0 |= traversal.input == 1 && traversal.output == 0;
    input1Output1 |= traversal.input == 1 && traversal.output == 1;
  }
  require(test, input0Output0 && input1Output0 && input1Output1,
          "switch traversal ordinals do not follow the MSB-left convention");
}

void fuCapabilityTemplatesComeFromThePhysicalGraph() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());

  mlir::OwningOpRef<mlir::ModuleOp> singleSource = parse(test, R"mlir(
    module {
      fabric.module @single(%a: !fabric.bits<32>, %b: !fabric.bits<32>)
          -> !fabric.bits<32> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<32>, %pb = %b : !fabric.bits<32>)
            -> !fabric.bits<32> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<32>,
               %fb = %pb : !fabric.bits<32>)
              -> !fabric.bits<32> {
            %sum = fabric.op [@arith.addi, @arith.subi] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [32 : i32]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %sum : !fabric.bits<32>
          }
        }
        fabric.yield %pe : !fabric.bits<32>
      }
    }
  )mlir");
  expectRejected(
      test, loom::fabric::finalizeFabricRoot(root(test, *singleSource), store),
      "fabric.op is missing its complete resource contract");
  setOperationResourceContracts(
      test, *singleSource,
      ::fabric::oneCycleElasticOperationResourceContract());
  FinalizedFabricRoot single = take(
      test, loom::fabric::finalizeFabricRoot(root(test, *singleSource), store));
  const loom::fabric::FabricFuTemplateRef singleFu =
      uniqueFuTemplate(test, single.view());
  auto singleTemplates = single.view().fuCapabilityTemplates(singleFu);
  require(test, singleTemplates.size() == 1,
          "single operation FU did not produce one capability template");
  require(test,
          singleTemplates.front().activeNodes.size() == 1 &&
              singleTemplates.front().activeEdges.size() == 3,
          "single operation FU capability template is incomplete");
  const loom::fabric::ResolvedFabricOpCapabilityView *singleCapability =
      single.view().resolvedFabricOpCapability(
          singleTemplates.front().activeNodes.front());
  require(test, singleCapability != nullptr,
          "finalized Fabric lost the concrete operation capability");
  require(test,
          singleCapability->implementationFamily ==
              ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
          "concrete operation capability changed implementation family");
  require(test,
          singleCapability->enabledOperationSchemas ==
              std::vector<::dataflow::OperationSchemaId>{
                  ::dataflow::OperationSchemaId::ArithAddI,
                  ::dataflow::OperationSchemaId::ArithSubI},
          "concrete operation capability changed its enabled schema set");
  require(test, singleCapability->physicalPorts.size() == 3,
          "concrete operation capability lost physical ports");
  require(test,
          singleCapability->resourceStateAndTimingContract.usePatternCount() !=
              0,
          "concrete operation capability lost its resource contract");
  require(test, singleCapability->configurationFieldSchema.size() == 1,
          "multi-member operation capability lost its semantic field");

  mlir::OwningOpRef<mlir::ModuleOp> branchSource = parse(test, R"mlir(
    module {
      fabric.module @branch(%a: !fabric.bits<32>, %b: !fabric.bits<32>)
          -> !fabric.bits<32> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<32>, %pb = %b : !fabric.bits<32>)
            -> !fabric.bits<32> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<32>,
               %fb = %pb : !fabric.bits<32>)
              -> !fabric.bits<32> {
            %a0, %a1 = fabric.demux %fa : !fabric.bits<32> -> 2
            %b0, %b1 = fabric.demux %fb : !fabric.bits<32> -> 2
            %sum = fabric.op [@arith.addi] (%a0, %b0)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [32 : i32]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            %product = fabric.op [@arith.muli] (%a1, %b1)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerMultiply>,
               hw_params = {integer_widths = [32 : i32]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            %selected = fabric.mux %sum, %product : !fabric.bits<32>
            fabric.yield %selected : !fabric.bits<32>
          }
        }
        fabric.yield %pe : !fabric.bits<32>
      }
    }
  )mlir");
  setOperationResourceContracts(
      test, *branchSource,
      ::fabric::oneCycleElasticOperationResourceContract());
  expectRejected(
      test, loom::fabric::finalizeFabricRoot(root(test, *branchSource), store),
      "multi-template FU requires an explicit capability domain");
  ::fabric::FuOp branchFuOp;
  branchSource->walk([&](::fabric::FuOp fu) { branchFuOp = fu; });
  require(test, static_cast<bool>(branchFuOp),
          "branch fixture has no fabric.fu");
  setFuCapabilityDomain(
      test, branchFuOp,
      {{{2}, {{0, 0}, {1, 0}, {4, 0}}}, {{3}, {{0, 1}, {1, 1}, {4, 1}}}});
  FinalizedFabricRoot branch = take(
      test, loom::fabric::finalizeFabricRoot(root(test, *branchSource), store));
  const loom::fabric::FabricFuTemplateRef branchFu =
      uniqueFuTemplate(test, branch.view());
  auto branchTemplates = branch.view().fuCapabilityTemplates(branchFu);
  require(test, branchTemplates.size() == 2,
          "branch FU did not produce exactly two coherent templates");
  for (const auto &capability :
       branch.view().resolvedFabricOpCapabilities(branchFu))
    require(test, capability.configurationFieldSchema.empty(),
            "singleton modular operation gained a redundant semantic field");
  for (const loom::fabric::FabricFuCapabilityTemplateRecord &record :
       branchTemplates) {
    unsigned opCount = 0;
    unsigned muxCount = 0;
    unsigned demuxCount = 0;
    for (const loom::fabric::FabricFuTemplateNodeRef &node :
         record.activeNodes) {
      opCount += node.node == loom::fabric::FabricFuNodeKind::Op;
      muxCount += node.node == loom::fabric::FabricFuNodeKind::Mux;
      demuxCount += node.node == loom::fabric::FabricFuNodeKind::Demux;
    }
    require(test,
            opCount == 1 && muxCount == 1 && demuxCount == 2 &&
                record.activeEdges.size() == 6,
            "branch FU capability template contains a mixed route selection");
  }

  FinalizedFabricRoot imported = take(
      test, loom::fabric::importEntireFabricRoot(branch.reference(), store));
  require(test,
          imported.view().fuCapabilityTemplates(branchFu) == branchTemplates,
          "strict import changed the FU capability-template inventory");

  setFuCapabilityDomain(test, branchFuOp, {{{2}, {{0, 0}, {1, 0}, {4, 0}}}});
  FinalizedFabricRoot narrowed = take(
      test, loom::fabric::finalizeFabricRoot(root(test, *branchSource), store));
  require(test, narrowed.reference().artifact != branch.reference().artifact,
          "FU capability-domain change did not change Fabric identity");
}

void systemPublicationUsesExactImportedModule() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  mlir::OwningOpRef<mlir::ModuleOp> moduleSource = parse(test, R"mlir(
    module {
      fabric.module @empty() {
        fabric.yield
      }
    }
  )mlir");
  FinalizedFabricRoot module = take(
      test, loom::fabric::finalizeFabricRoot(root(test, *moduleSource), store));
  const loom::fabric::FabricModuleTemplateRef moduleTemplate =
      uniqueModuleTemplate(test, module.view());

  mlir::OwningOpRef<mlir::ModuleOp> systemSource =
      parse(test, accCoreSystemSource(test, {0, moduleTemplate}));
  FinalizedFabricRoot system = take(
      test, loom::fabric::finalizeFabricRoot(systemRoot(test, *systemSource),
                                             {module.reference()}, store));
  require(test,
          system.view().rootKind() == loom::fabric::FabricRootKind::System,
          "System finalizer returned the wrong root kind");
  require(test,
          system.directDependencies().size() == 1 &&
              system.directDependencies().front().root == module.reference(),
          "System finalizer changed its ImportedModule dependency");
  loom::fabric::FabricSystemRootView view =
      take(test, loom::fabric::requireSystemRoot(system.view()));
  require(test, view.spatialAttachments().empty(),
          "zero-port SpatialCore gained an attachment");
  const auto spatialTarget =
      view.spatialCoreTarget(loom::fabric::AccCoreOccurrenceRef(0));
  require(test,
          spatialTarget && *spatialTarget ==
                               loom::fabric::FabricImportedModuleTargetRef{
                                   0, moduleTemplate},
          "zero-port AccCore lost its exact imported SpatialCore target");

  FinalizedFabricRoot imported = take(
      test, loom::fabric::importEntireFabricRoot(system.reference(), store));
  require(test, imported.reference().artifact == system.reference().artifact,
          "strict System import changed artifact identity");
  loom::fabric::FabricSystemRootView importedView =
      take(test, loom::fabric::requireSystemRoot(imported.view()));
  require(test,
          importedView.spatialCoreTarget(
              loom::fabric::AccCoreOccurrenceRef(0)) == spatialTarget,
          "strict System import changed its AccCore SpatialCore target");
}

void systemRequiresOneCompatibleHostCore() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  auto moduleSource = parse(test, R"mlir(
    module { fabric.module @empty() { fabric.yield } }
  )mlir");
  FinalizedFabricRoot module = take(
      test, loom::fabric::finalizeFabricRoot(root(test, *moduleSource), store));
  const loom::fabric::FabricImportedModuleTargetRef target{
      0, uniqueModuleTemplate(test, module.view())};

  auto finalize = [&](llvm::StringRef source) {
    auto parsed = parse(test, source);
    return loom::fabric::finalizeFabricRoot(systemRoot(test, *parsed),
                                            {module.reference()}, store);
  };

  const std::string missing = "module { fabric.system @missing { " +
                              accCoreSource(test, target) + "} }";
  expectRejected(test, finalize(missing), "exactly one HostCore");

  const std::string duplicate = "module { fabric.system @duplicate { " +
                                hostCoreSource(test) + hostCoreSource(test) +
                                accCoreSource(test, target) + "} }";
  expectRejected(test, finalize(duplicate), "exactly one HostCore");

  auto hostOnlySource = parse(test, "module { fabric.system @host_only { " +
                                        hostCoreSource(test) + "} }");
  expectRejected(test,
                 loom::fabric::finalizeFabricRoot(
                     systemRoot(test, *hostOnlySource), {}, store),
                 "at least one AccCore");

  const std::string incompatible = "module { fabric.system @incompatible { " +
                                   hostCoreSource(test, /*x64=*/false) +
                                   accCoreSource(test, target) + "} }";
  expectRejected(test, finalize(incompatible),
                 "common InstructionCore XLEN and endianness");

  const std::string disjointAbis =
      "module { fabric.system @disjoint_abis { " +
      hostCoreSource(test, /*x64=*/true, /*floatingAbi=*/true) +
      accCoreSource(test, target) + "} }";
  expectRejected(test, finalize(disjointAbis), "no common InstructionCore ABI");
}

void systemPublishesCompleteSpatialAttachments() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  auto moduleSource = parse(test, R"mlir(
    module {
      fabric.module @stream(%input: !fabric.bits<32>)
          -> !fabric.bits<32> {
        %buffered = fabric.fifo %input [max_depth = 2, bypassable = true]
            : !fabric.bits<32>
        fabric.yield %buffered : !fabric.bits<32>
      }
    }
  )mlir");
  FinalizedFabricRoot module = take(
      test, loom::fabric::finalizeFabricRoot(root(test, *moduleSource), store));
  const loom::fabric::FabricModuleTemplateRef moduleTemplate =
      uniqueModuleTemplate(test, module.view());

  auto completeSource =
      parse(test, attachedAccCoreSystemSource(test, {0, moduleTemplate},
                                              /*attachOutput=*/true));
  FinalizedFabricRoot system = take(
      test, loom::fabric::finalizeFabricRoot(systemRoot(test, *completeSource),
                                             {module.reference()}, store));
  loom::fabric::FabricSystemRootView systemView =
      take(test, loom::fabric::requireSystemRoot(system.view()));
  require(test, systemView.spatialAttachments().size() == 2,
          "complete SpatialCore boundary did not retain both attachments");

  bool sawInput = false;
  bool sawOutput = false;
  for (const loom::fabric::FabricSpatialAttachmentRecordView &attachment :
       systemView.spatialAttachments()) {
    require(test,
            attachment.moduleEndpoint.dependencyOrdinal == 0 &&
                attachment.moduleEndpoint.target.module == moduleTemplate,
            "attachment changed its exact ImportedModule target");
    const auto *endpoint = attachment.spatialEndpoint.transport();
    require(test, endpoint != nullptr,
            "token module boundary became a memory attachment");
    const auto direction = system.view().transportEndpointDirection(*endpoint);
    require(test, direction.has_value(),
            "attachment names an unknown SpatialCore endpoint");
    require(test, *direction == attachment.moduleEndpoint.target.direction,
            "attachment changed its boundary direction");
    sawInput |= *direction == loom::fabric::FabricPortDirection::Input;
    sawOutput |= *direction == loom::fabric::FabricPortDirection::Output;
  }
  require(test, sawInput && sawOutput,
          "complete SpatialCore boundary lost an input or output attachment");

  auto incompleteSource =
      parse(test, attachedAccCoreSystemSource(test, {0, moduleTemplate},
                                              /*attachOutput=*/false));
  expectRejected(
      test,
      loom::fabric::finalizeFabricRoot(systemRoot(test, *incompleteSource),
                                       {module.reference()}, store),
      "does not attach every module boundary endpoint exactly once");
}

void systemRejectsUnusedImportedModule() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  auto firstSource = parse(test, R"mlir(
    module { fabric.module @empty() { fabric.yield } }
  )mlir");
  auto secondSource = parse(test, R"mlir(
    module {
      fabric.module @input(%x: !fabric.bits<8>) {
        fabric.yield
      }
    }
  )mlir");
  FinalizedFabricRoot first = take(
      test, loom::fabric::finalizeFabricRoot(root(test, *firstSource), store));
  FinalizedFabricRoot second = take(
      test, loom::fabric::finalizeFabricRoot(root(test, *secondSource), store));
  auto systemSource = parse(
      test,
      accCoreSystemSource(test, {0, loom::fabric::FabricModuleTemplateRef(0)}));
  expectRejected(test,
                 loom::fabric::finalizeFabricRoot(
                     systemRoot(test, *systemSource),
                     {first.reference(), second.reference()}, store),
                 "unused ImportedModule dependency");
}

void systemRejectsWrongImportedModuleTarget() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  auto moduleSource = parse(test, R"mlir(
    module { fabric.module @empty() { fabric.yield } }
  )mlir");
  FinalizedFabricRoot module = take(
      test, loom::fabric::finalizeFabricRoot(root(test, *moduleSource), store));
  auto systemSource =
      parse(test, accCoreSystemSource(
                      test, {0, loom::fabric::FabricModuleTemplateRef(999)}));
  expectRejected(
      test,
      loom::fabric::finalizeFabricRoot(systemRoot(test, *systemSource),
                                       {module.reference()}, store),
      "unknown_entity");
}

void systemRejectsImplicitConnectionFanout() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  FinalizedFabricRoot module = publishEmptySpatialCore(test, store);
  const loom::fabric::FabricImportedModuleTargetRef target{
      0, uniqueModuleTemplate(test, module.view())};
  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << "module { fabric.system @soc {\n"
         << hostCoreSource(test) << accCoreSource(test, target)
         << transportResource(test, 10, 8) << transportResource(test, 20, 16)
         << transportResource(test, 30, 32) << systemConnection(10, 20)
         << systemConnection(10, 30) << "} }\n";
  auto systemSource = parse(test, source);
  expectRejected(
      test,
      loom::fabric::finalizeFabricRoot(systemRoot(test, *systemSource),
                                       {module.reference()}, store),
      "point connection source is connected more than once");
}

void systemPublicationIgnoresAuthoringIdentityAndOrder() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  FinalizedFabricRoot module = publishEmptySpatialCore(test, store);
  const loom::fabric::FabricImportedModuleTargetRef target{
      0, uniqueModuleTemplate(test, module.view())};
  auto firstSource = parse(test, connectedTransportSystemSource(
                                     "first", test, target, 10, 20, false));
  auto secondSource = parse(test, connectedTransportSystemSource(
                                      "second", test, target, 91, 4, true));
  FinalizedFabricRoot first = take(
      test, loom::fabric::finalizeFabricRoot(systemRoot(test, *firstSource),
                                             {module.reference()}, store));
  FinalizedFabricRoot second = take(
      test, loom::fabric::finalizeFabricRoot(systemRoot(test, *secondSource),
                                             {module.reference()}, store));
  require(test, first.reference().artifact == second.reference().artifact,
          "System authoring names, IDs, or order changed artifact identity");
}

void systemRequiresExplicitClockCrossing() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  FinalizedFabricRoot module = publishEmptySpatialCore(test, store);
  const loom::fabric::FabricImportedModuleTargetRef target{
      0, uniqueModuleTemplate(test, module.view())};

  std::string hidden;
  llvm::raw_string_ostream hiddenStream(hidden);
  hiddenStream << "module { fabric.system @hidden {\n"
               << hostCoreSource(test) << accCoreSource(test, target)
               << transportResource(test, 10, 8)
               << transportResource(test, 30, 32) << clockDomain(test, 40, 10)
               << clockDomain(test, 50, 30) << systemConnection(10, 30)
               << "} }\n";
  auto hiddenSource = parse(test, hidden);
  expectRejected(
      test,
      loom::fabric::finalizeFabricRoot(systemRoot(test, *hiddenSource),
                                       {module.reference()}, store),
      "crosses Clock domains without an explicit crossing resource");

  const loom::fabric::SystemTransportResourceRef crossingResource(20);
  auto crossing = take(
      test,
      loom::fabric::ClockCrossingContractRecord::createAsyncFifo(
          {crossingResource, 0},
          loom::fabric::ClockDomainRef(loom::fabric::HardwareDomainRef(40)),
          loom::fabric::ClockDomainRef(loom::fabric::HardwareDomainRef(50)), 4,
          2));
  const std::vector<std::uint8_t> crossingBytes =
      take(test, loom::fabric::encodeClockCrossingContractRecord(crossing));
  std::string explicitCrossing;
  llvm::raw_string_ostream explicitStream(explicitCrossing);
  explicitStream << "module { fabric.system @explicit {\n"
                 << hostCoreSource(test) << accCoreSource(test, target)
                 << transportResource(test, 10, 8)
                 << transportResource(test, 20, 16, crossingBytes)
                 << transportResource(test, 30, 32) << transferPattern(test, 20)
                 << clockDomain(test, 40, 10) << clockDomain(test, 50, 30)
                 << systemConnection(10, 20) << systemConnection(20, 30)
                 << "} }\n";
  auto explicitSource = parse(test, explicitCrossing);
  FinalizedFabricRoot finalized = take(
      test, loom::fabric::finalizeFabricRoot(systemRoot(test, *explicitSource),
                                             {module.reference()}, store));
  auto view = take(test, loom::fabric::requireSystemRoot(finalized.view()));
  require(test, view.transportResources().size() == 3,
          "explicit crossing System lost a transport resource");
  unsigned crossingCount = 0;
  for (loom::fabric::SystemTransportResourceRef resource :
       view.transportResources())
    crossingCount += view.clockCrossing(resource) != nullptr;
  require(test, crossingCount == 1,
          "explicit crossing System changed its crossing contract");
}

} // namespace

int main() {
  canonicalPublicationAndStrictImport();
  visualizationCoordinatesHaveNoFabricSemantics();
  malformedStoredPayloadIsRejected();
  spatialSwitchConnectivityBecomesTraversals();
  fuCapabilityTemplatesComeFromThePhysicalGraph();
  systemPublicationUsesExactImportedModule();
  systemRequiresOneCompatibleHostCore();
  systemPublishesCompleteSpatialAttachments();
  systemRejectsUnusedImportedModule();
  systemRejectsWrongImportedModuleTarget();
  systemRejectsImplicitConnectionFanout();
  systemPublicationIgnoresAuthoringIdentityAndOrder();
  systemRequiresExplicitClockCrossing();
  return EXIT_SUCCESS;
}
