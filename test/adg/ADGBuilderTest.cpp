#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "ADG/Export.h"
#include "ADG/FuLibrary.h"
#include "ADG/MemoryLibrary.h"
#include "Fabric/IR/OperationResourceContract.h"

#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/MemoryOperationPort.h"
#include "Fabric/Identity/FabricFuCapabilityTemplate.h"
#include "Fabric/Identity/FabricRefs.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, llvm::StringRef message) {
  if (!condition)
    fail(test, message.str());
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

void expectError(llvm::StringRef test, llvm::Error error,
                 llvm::StringRef diagnostic) {
  if (!error)
    fail(test, "accepted invalid ADG authoring");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(diagnostic), message);
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef diagnostic) {
  if (value)
    fail(test, "accepted invalid ADG authoring");
  expectError(test, value.takeError(), diagnostic);
}

class TemporaryDirectory {
public:
  explicit TemporaryDirectory(llvm::StringRef test) : test_(test.str()) {
    llvm::SmallString<128> path;
    if (std::error_code error =
            llvm::sys::fs::createUniqueDirectory("loom-adg-builder", path))
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

using loom::adg::BoundarySpec;
using loom::adg::DesignBuilder;
using loom::adg::FifoSpec;
using loom::adg::FuCapabilityTemplateSpec;
using loom::adg::FuConfigurationMode;
using loom::adg::FuRouteSelection;
using loom::adg::FuSpec;
using loom::adg::FuValue;
using loom::adg::LocalMemoryParameters;
using loom::adg::LocalMemoryServiceSpec;
using loom::adg::MemoryConnectivitySpec;
using loom::adg::MemoryEngineSpec;
using loom::adg::MemorySpec;
using loom::adg::OperationCapabilitySpec;
using loom::adg::PeSpec;
using loom::adg::PortType;
using loom::adg::SpatialValue;
using loom::adg::SwitchSpec;
using loom::adg::TemporalPeParameters;

std::uint64_t uniqueEntity(llvm::StringRef test,
                           const loom::fabric::FabricArtifactView &view,
                           loom::fabric::FabricEntityKind expectedKind) {
  std::optional<std::uint64_t> result;
  for (std::uint64_t id = 0;; ++id) {
    std::optional<loom::fabric::FabricEntityKind> kind = view.entityKind(id);
    if (!kind)
      break;
    if (*kind != expectedKind)
      continue;
    if (result)
      fail(test, "finalized root contains duplicate expected entities");
    result = id;
  }
  if (!result)
    fail(test, "finalized root contains no expected entity");
  return *result;
}

std::uint64_t entityCount(const loom::fabric::FabricArtifactView &view,
                          loom::fabric::FabricEntityKind expectedKind) {
  std::uint64_t count = 0;
  for (std::uint64_t id = 0;; ++id) {
    std::optional<loom::fabric::FabricEntityKind> kind = view.entityKind(id);
    if (!kind)
      return count;
    count += *kind == expectedKind;
  }
}

loom::fabric::FabricFuTemplateRef
uniqueFuTemplate(llvm::StringRef test,
                 const loom::fabric::FabricArtifactView &view) {
  return loom::fabric::FabricFuTemplateRef(uniqueEntity(
      test, view, loom::fabric::FabricEntityKind::FabricFuTemplate));
}

OperationCapabilitySpec
integerCapability(::fabric::ImplementationFamilyId family,
                  ::dataflow::OperationSchemaId operation,
                  const PortType &outputType) {
  return OperationCapabilitySpec{
      family,
      ::fabric::ScalarIntegerParams{
          ::fabric::IntegerWidthSet::get({::fabric::IntegerWidth::I32})},
      {operation},
      {outputType},
      ::fabric::oneCycleElasticOperationResourceContract()};
}

::fabric::UnsignedDomain singleton(std::uint64_t value) {
  return take("memory singleton",
              ::fabric::UnsignedDomain::fromCanonical({{value, value}}));
}

::fabric::ResourceContract singleUseResourceContract(llvm::StringRef test) {
  ::fabric::ResourceContractDeclaration declaration;
  declaration.states = {::fabric::ResourceStateDeclaration{
      ::fabric::StateKey(0),
      {{::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1),
        ::fabric::CapacityUnits(0)}}}};
  declaration.requesters = {::fabric::RequesterKey(0)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 2;
  declaration.timingContracts = {{::fabric::TimingContractKey(0), {0, 1}}};
  declaration.usePatterns = {
      {::fabric::UsePatternKey(0),
       ::fabric::RequesterKey(0),
       ::fabric::EligibilityKey(0),
       ::fabric::EventKey(0),
       ::fabric::EventKey(1),
       std::nullopt,
       ::fabric::TimingContractKey(0),
       {{::fabric::ClaimKey(0), ::fabric::StateKey(0),
         ::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1)}},
       {{{::fabric::ClaimKey(0)}}}}};
  return take(test, ::fabric::ResourceContract::create(std::move(declaration)));
}

loom::fabric::InstructionCoreArchitecturalContract
instructionArchitecture(llvm::StringRef test) {
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
  return take(test, loom::fabric::InstructionCoreArchitecturalContract::create(
                        std::move(declaration)));
}

loom::fabric::InstructionCoreMicroarchitecturalRealization
inOrderMicroarchitecture(llvm::StringRef test) {
  loom::fabric::InstructionCoreCommonDeclaration common{
      1,
      {{loom::fabric::InstructionOperationClass::IntegerAlu, 1, 1, 1},
       {loom::fabric::InstructionOperationClass::LoadStore, 1, 2, 1}},
      singleUseResourceContract(test)};
  loom::fabric::InOrderMicroarchitectureDeclaration pipeline{1, 1, 1, 1,
                                                             1, 1, 4, 2};
  return take(
      test,
      loom::fabric::InstructionCoreMicroarchitecturalRealization::createInOrder(
          std::move(common), pipeline));
}

loom::fabric::InstructionCoreMicroarchitecturalRealization
outOfOrderMicroarchitecture(llvm::StringRef test) {
  loom::fabric::InstructionCoreCommonDeclaration common{
      2,
      {{loom::fabric::InstructionOperationClass::IntegerAlu, 2, 1, 1},
       {loom::fabric::InstructionOperationClass::LoadStore, 2, 2, 1}},
      singleUseResourceContract(test)};
  loom::fabric::OutOfOrderMicroarchitectureDeclaration pipeline{
      2, 2, 2, 2, 2, 2, 2, 32, 16, 8, 8, 64, 32, 32};
  return take(test, loom::fabric::InstructionCoreMicroarchitecturalRealization::
                        createOutOfOrder(std::move(common), pipeline));
}

::fabric::MemoryAccessClass systemElementAccess(llvm::StringRef test) {
  auto alignment =
      take(test,
           ::fabric::AlignmentDomain::create(
               take(test, ::fabric::UnsignedDomain::fromCanonical({{0, 63}}))));
  auto read = take(
      test,
      ::fabric::ClosedEnumDomain<::fabric::ReadSubwordSemantics>::fromCanonical(
          {::fabric::ReadSubwordSemantics::Exact}));
  auto write =
      take(test,
           ::fabric::ClosedEnumDomain<::fabric::WriteSubwordSemantics>::
               fromCanonical({::fabric::WriteSubwordSemantics::NotApplicable}));
  return take(
      test, ::fabric::MemoryAccessClass::create(
                ::dataflow::semantics::MemoryAccessForm::Element,
                take(test, ::fabric::UnsignedDomain::fromCanonical({{32, 32}})),
                singleton(1),
                {{::dataflow::semantics::MemoryMaskForm::Absent,
                  ::fabric::InactiveLaneSemantics::NotApplicable}},
                std::move(alignment), std::move(read), std::move(write)));
}

::fabric::MemoryActorContractDomain plainLoadActorDomain(llvm::StringRef test) {
  ::fabric::MemoryActorContractClause plain =
      ::fabric::LoadStorePlainContractClause{{false}};
  return take(test, ::fabric::MemoryActorContractDomain::create(
                        ::dataflow::OperationSchemaId::DataflowLoad, {plain}));
}

::fabric::MemoryServiceContractRecord
systemMemoryContract(llvm::StringRef test, mlir::MLIRContext &context) {
  auto accesses = take(test, ::fabric::ParameterizedMemoryAccessDomain::create(
                                 {systemElementAccess(test)}));
  ::fabric::MemoryServiceContractDeclaration declaration{
      {{0, 4096, ::fabric::MemoryServiceRegionBehavior::Storage, std::nullopt}},
      singleUseResourceContract(test),
      {{plainLoadActorDomain(test),
        std::move(accesses),
        {0},
        128,
        {::fabric::UsePatternKey(0)},
        ::fabric::NoMemoryServiceConsistency{}}}};
  return take(test, ::fabric::MemoryServiceContractRecord::create(
                        &context, ::fabric::MemoryServiceOwnerKind::System,
                        std::move(declaration)));
}

::fabric::MemoryServiceContractRecord
localMemoryContract(llvm::StringRef test, mlir::MLIRContext &context) {
  auto accesses = take(test, ::fabric::ParameterizedMemoryAccessDomain::create(
                                 {systemElementAccess(test)}));
  ::fabric::MemoryServiceContractDeclaration declaration{
      {{0, 4096, ::fabric::MemoryServiceRegionBehavior::Storage, std::nullopt}},
      singleUseResourceContract(test),
      {{plainLoadActorDomain(test),
        std::move(accesses),
        {0},
        128,
        {::fabric::UsePatternKey(0)},
        ::fabric::NoMemoryServiceConsistency{}}}};
  return take(test, ::fabric::MemoryServiceContractRecord::create(
                        &context, ::fabric::MemoryServiceOwnerKind::Local,
                        std::move(declaration)));
}

loom::fabric::CanonicalServiceCapabilitySet
systemMemoryCapabilities(llvm::StringRef test,
                         loom::fabric::ServiceRateContractRecord serviceRate) {
  auto accesses = take(test, ::fabric::ParameterizedMemoryAccessDomain::create(
                                 {systemElementAccess(test)}));
  auto addressDomain =
      take(test, ::fabric::UnsignedDomain::fromCanonical({{0, 4095}}));
  auto domain =
      take(test, loom::fabric::AddressedMemoryCapabilityDomain::create(
                     plainLoadActorDomain(test), std::move(accesses),
                     std::move(addressDomain), 128, std::nullopt));
  auto capability =
      take(test, loom::fabric::CanonicalServiceCapabilityRecord::create(
                     ::dataflow::semantics::ServiceKind::MemoryRead,
                     loom::fabric::CanonicalServiceEndpointRole::Serve,
                     std::move(domain), std::move(serviceRate)));
  return take(test, loom::fabric::CanonicalServiceCapabilitySet::create(
                        {std::move(capability)}));
}

::fabric::MemoryOperationPortDeclaration loadPortDeclaration() {
  ::fabric::ResourceContractDeclaration resource;
  resource.states = {::fabric::ResourceStateDeclaration{
      ::fabric::StateKey(0),
      {{::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1),
        ::fabric::CapacityUnits(0)}}}};
  resource.requesters = {::fabric::RequesterKey(0)};
  resource.eligibilityCount = 1;
  resource.eventCount = 1;
  resource.timingContracts = {{::fabric::TimingContractKey(0), {0}}};
  resource.usePatterns = {{::fabric::UsePatternKey(0),
                           ::fabric::RequesterKey(0),
                           ::fabric::EligibilityKey(0),
                           ::fabric::EventKey(0),
                           ::fabric::EventKey(0),
                           std::nullopt,
                           ::fabric::TimingContractKey(0),
                           {},
                           {{{}}}}};

  auto alignment =
      take("memory alignment",
           ::fabric::AlignmentDomain::create(
               take("memory alignment range",
                    ::fabric::UnsignedDomain::fromCanonical({{0, 63}}))));
  auto read = take(
      "memory read semantics",
      ::fabric::ClosedEnumDomain<::fabric::ReadSubwordSemantics>::fromCanonical(
          {::fabric::ReadSubwordSemantics::Exact}));
  auto write =
      take("memory write semantics",
           ::fabric::ClosedEnumDomain<::fabric::WriteSubwordSemantics>::
               fromCanonical({::fabric::WriteSubwordSemantics::NotApplicable}));
  auto access =
      take("memory access class",
           ::fabric::MemoryAccessClass::create(
               ::dataflow::semantics::MemoryAccessForm::Element, singleton(32),
               singleton(1),
               {{::dataflow::semantics::MemoryMaskForm::Absent,
                 ::fabric::InactiveLaneSemantics::NotApplicable}},
               std::move(alignment), std::move(read), std::move(write)));
  auto accessDomain = take(
      "memory access domain",
      ::fabric::ParameterizedMemoryAccessDomain::create({std::move(access)}));
  ::fabric::MemoryActorContractClause plain =
      ::fabric::LoadStorePlainContractClause{{false}};
  auto actorDomain =
      take("memory actor domain",
           ::fabric::MemoryActorContractDomain::create(
               ::dataflow::OperationSchemaId::DataflowLoad, {plain}));

  return {{0, 1, 2, 3},
          take("memory resource contract",
               ::fabric::ResourceContract::create(std::move(resource))),
          {{::fabric::MemoryPortTransactionProjection::Direct}},
          {{std::move(actorDomain),
            {{::dataflow::semantics::ServiceValueRole::Address, 0},
             {::dataflow::semantics::ServiceValueRole::Data, 2},
             {::dataflow::semantics::ServiceValueRole::Control, 1},
             {::dataflow::semantics::ServiceValueRole::Completion, 3}},
            std::move(accessDomain),
            {::fabric::UsePatternKey(0)}}}};
}

::fabric::MemoryDispatchTarget localMemoryTarget() {
  return ::fabric::MemoryDispatchTarget(
      std::in_place_type<::fabric::LocalMemoryDispatchTarget>);
}

::fabric::MemoryDispatchTarget managerMemoryTarget(std::uint64_t ordinal) {
  return ::fabric::MemoryDispatchTarget(
      std::in_place_type<::fabric::ManagerMemoryDispatchTarget>,
      ::fabric::ManagerMemoryDispatchTarget{ordinal});
}

MemoryConnectivitySpec
operationConnectivity(llvm::StringRef test,
                      ::fabric::MemoryDispatchTarget target) {
  ::fabric::MemoryConnectivityDeclaration declaration;
  declaration.operationPorts = {{{{std::move(target)}}}};
  return take(test, MemoryConnectivitySpec::create(std::move(declaration)));
}

MemoryConnectivitySpec storageConnectivity(llvm::StringRef test) {
  ::fabric::MemoryConnectivityDeclaration declaration;
  declaration.subordinateEndpoints = {
      {1,
       {},
       ::fabric::MemoryProviderAddressTransform::None,
       {localMemoryTarget()}}};
  return take(test, MemoryConnectivitySpec::create(std::move(declaration)));
}

void regularAndIrregularSpatialCoresFinalize() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);

  const PortType bits4 = take(test, PortType::bits(4));
  const PortType bits32 = take(test, PortType::bits(32));
  const PortType bits64 = take(test, PortType::bits(64));
  const PortType tagged32x4 = take(test, PortType::taggedBits(32, 4));

  auto regular = take(
      test, design.createSpatialCore("regular", {bits32, bits4}, {tagged32x4}));
  SpatialValue regularData = take(test, regular.input(0));
  SpatialValue regularTag = take(test, regular.input(1));
  auto regularBoundary = take(
      test, regular.addBoundary({regularData, regularTag},
                                BoundarySpec::s2t(bits32, bits4, tagged32x4)));
  SpatialValue regularQueued =
      take(test, regular.addFifo(regularBoundary.front(),
                                 FifoSpec{tagged32x4, 2, true}));
  if (llvm::Error error = regular.close({regularQueued}))
    fail(test, llvm::toString(std::move(error)));

  auto irregular =
      take(test,
           design.createSpatialCore("irregular", {bits64, bits32, bits4, bits4},
                                    {tagged32x4, tagged32x4}));
  SpatialValue irregularData = take(test, irregular.input(0));
  SpatialValue alternateData = take(test, irregular.input(1));
  SpatialValue irregularTag0 = take(test, irregular.input(2));
  SpatialValue irregularTag1 = take(test, irregular.input(3));
  SpatialValue narrowed =
      take(test, irregular.addFifo(irregularData, FifoSpec{bits32, 3, false}));
  auto switched =
      take(test, irregular.addSwitch({narrowed, alternateData},
                                     SwitchSpec::spatial({bits32, bits32},
                                                         {bits32, bits32},
                                                         {{0, 1}, {1}})));
  auto irregularBoundary0 =
      take(test,
           irregular.addBoundary({switched[0], irregularTag0},
                                 BoundarySpec::s2t(bits32, bits4, tagged32x4)));
  auto irregularBoundary1 =
      take(test,
           irregular.addBoundary({switched[1], irregularTag1},
                                 BoundarySpec::s2t(bits32, bits4, tagged32x4)));
  if (llvm::Error error = irregular.close(
          {irregularBoundary0.front(), irregularBoundary1.front()}))
    fail(test, llvm::toString(std::move(error)));

  loom::adg::FinalizedFabricDesign finalized =
      take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 2,
          "finalized design did not contain both SpatialCore roots");
  for (const loom::fabric::FinalizedFabricRoot &root : finalized.roots()) {
    require(test,
            root.view().rootKind() == loom::fabric::FabricRootKind::Module,
            "SpatialCore finalized with a non-Module root kind");
    require(test, !root.view().admittedTraversals().empty(),
            "SpatialCore lost its physical traversal inventory");

    std::string mlirText;
    llvm::raw_string_ostream stream(mlirText);
    if (llvm::Error error = loom::fabric::writeFabricMlir(root, stream))
      fail(test, llvm::toString(std::move(error)));
    stream.flush();
    require(test, llvm::StringRef(mlirText).contains("fabric.module"),
            "finalized SpatialCore did not export Fabric MLIR");
  }
}

void foreignHandlesAndIncompleteRootsFailClosed() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits32 = take(test, PortType::bits(32));
  expectError(test, PortType::memory({-1}, bits32), "invalid extent");

  auto first =
      take(test, design.createSpatialCore("first", {bits32}, {bits32}));
  auto second =
      take(test, design.createSpatialCore("second", {bits32}, {bits32}));
  SpatialValue foreign = take(test, first.input(0));
  expectError(test, second.addFifo(foreign, FifoSpec{bits32, 1, false}),
              "foreign SpatialValue");

  if (llvm::Error error = first.close({foreign}))
    fail(test, llvm::toString(std::move(error)));
  expectError(test, std::move(design).finalize(),
              "SpatialCore 'second' is not closed");
}

void spatialCoreTemplatesInstantiateAndElaborate() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);

  const PortType bits16 = take(test, PortType::bits(16));
  const PortType bits32 = take(test, PortType::bits(32));

  auto pipeline =
      take(test, design.createSpatialCore("pipeline", {bits16}, {bits16}));
  SpatialValue pipelineInput = take(test, pipeline.input(0));
  SpatialValue pipelineOutput =
      take(test, pipeline.addFifo(pipelineInput, FifoSpec{bits16, 2, true}));
  if (llvm::Error error = pipeline.close({pipelineOutput}))
    fail(test, llvm::toString(std::move(error)));

  auto top = take(test, design.createSpatialCore("top", {bits32}, {bits16}));
  SpatialValue topInput = take(test, top.input(0));
  auto instance = take(test, top.instantiate(pipeline, {topInput}));
  require(test, instance.size() == 1,
          "typed SpatialCore instance returned the wrong result count");
  if (llvm::Error error = top.close(instance))
    fail(test, llvm::toString(std::move(error)));

  loom::adg::FinalizedFabricDesign finalized =
      take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 2,
          "module template design did not publish both requested roots");

  std::string mlirText;
  llvm::raw_string_ostream stream(mlirText);
  if (llvm::Error error =
          loom::fabric::writeFabricMlir(finalized.roots()[1], stream))
    fail(test, llvm::toString(std::move(error)));
  stream.flush();
  require(test, llvm::StringRef(mlirText).contains("fabric.fifo"),
          "module instantiation did not elaborate its physical body");
  require(test, !llvm::StringRef(mlirText).contains("fabric.instantiate"),
          "finalized Fabric retained an authoring-time instantiation");
  require(test, llvm::StringRef(mlirText).contains("!fabric.bits<16>"),
          "module instantiation lost its declared inner endpoint type");
}

void typedPeFuGraphsFinalize() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);

  const PortType bits32 = take(test, PortType::bits(32));
  const PortType tagged32x4 = take(test, PortType::taggedBits(32, 4));

  auto spatial = take(
      test, design.createSpatialCore("spatial-pe", {bits32, bits32}, {bits32}));
  SpatialValue spatialA = take(test, spatial.input(0));
  SpatialValue spatialB = take(test, spatial.input(1));
  auto spatialPe =
      take(test, spatial.addPe({spatialA, spatialB},
                               PeSpec::spatial({bits32, bits32}, {bits32})));
  auto spatialFu =
      take(test, spatialPe.addFu({take(test, spatialPe.input(0)),
                                  take(test, spatialPe.input(1))},
                                 FuSpec{{bits32, bits32}, {bits32}}));
  FuValue fuA = take(test, spatialFu.input(0));
  FuValue fuB = take(test, spatialFu.input(1));
  auto aRoutes = take(test, spatialFu.addDemux(fuA, 2));
  auto bRoutes = take(test, spatialFu.addDemux(fuB, 2));
  auto sum = take(
      test, spatialFu.addOperation(
                {take(test, aRoutes.output(0)), take(test, bRoutes.output(0))},
                integerCapability(
                    ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                    ::dataflow::OperationSchemaId::ArithAddI, bits32)));
  auto product = take(
      test, spatialFu.addOperation(
                {take(test, aRoutes.output(1)), take(test, bRoutes.output(1))},
                integerCapability(
                    ::fabric::ImplementationFamilyId::ScalarIntegerMultiply,
                    ::dataflow::OperationSchemaId::ArithMulI, bits32)));
  auto resultMux =
      take(test, spatialFu.addMux({take(test, sum.output(0)),
                                   take(test, product.output(0))}));
  if (llvm::Error error =
          spatialFu.addCapabilityTemplate(FuCapabilityTemplateSpec{
              {sum}, {{aRoutes, 0}, {bRoutes, 0}, {resultMux, 0}}}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error =
          spatialFu.addCapabilityTemplate(FuCapabilityTemplateSpec{
              {product}, {{aRoutes, 1}, {bRoutes, 1}, {resultMux, 1}}}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = spatialFu.close({take(test, resultMux.output(0))}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = spatialPe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = spatial.close({take(test, spatialPe.output(0))}))
    fail(test, llvm::toString(std::move(error)));

  auto temporal =
      take(test, design.createSpatialCore(
                     "temporal-pe", {tagged32x4, tagged32x4}, {tagged32x4}));
  auto temporalPe = take(
      test, temporal.addPe(
                {take(test, temporal.input(0)), take(test, temporal.input(1))},
                PeSpec::temporal({bits32, bits32}, {tagged32x4},
                                 TemporalPeParameters{
                                     2, FuConfigurationMode::PerFu,
                                     ::fabric::OperandBufferMode::PerInputPort,
                                     2, std::nullopt})));
  auto temporalFu =
      take(test, temporalPe.addFu({take(test, temporalPe.input(0)),
                                   take(test, temporalPe.input(1))},
                                  FuSpec{{bits32, bits32}, {bits32}}));
  auto temporalSum = take(
      test,
      temporalFu.addOperation(
          {take(test, temporalFu.input(0)), take(test, temporalFu.input(1))},
          integerCapability(
              ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
              ::dataflow::OperationSchemaId::ArithAddI, bits32)));
  if (llvm::Error error = temporalFu.addCapabilityTemplate(
          FuCapabilityTemplateSpec{{temporalSum}, {}}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = temporalFu.close({take(test, temporalSum.output(0))}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = temporalPe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = temporal.close({take(test, temporalPe.output(0))}))
    fail(test, llvm::toString(std::move(error)));

  loom::adg::FinalizedFabricDesign finalized =
      take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 2,
          "typed PE/FU authoring did not publish both roots");

  const auto spatialTemplates =
      finalized.roots()[0].view().fuCapabilityTemplates(
          uniqueFuTemplate(test, finalized.roots()[0].view()));
  require(test, spatialTemplates.size() == 2,
          "explicit add-or-multiply routing did not form two FU templates");
  const auto temporalTemplates =
      finalized.roots()[1].view().fuCapabilityTemplates(
          uniqueFuTemplate(test, finalized.roots()[1].view()));
  require(test, temporalTemplates.size() == 1,
          "temporal PE operation did not form one FU template");
  const loom::fabric::FabricPeOccurrenceRef temporalPeRef(
      uniqueEntity(test, finalized.roots()[1].view(),
                   loom::fabric::FabricEntityKind::FabricPeOccurrence));
  const ::fabric::ResourceContract *operandBuffer =
      finalized.roots()[1].view().resourceContract(
          loom::fabric::FabricInventoryOwnerRef::of(temporalPeRef));
  require(test,
          operandBuffer && operandBuffer->stateCount() != 0 &&
              operandBuffer->usePatternCount() != 0,
          "temporal PE lost its operand-buffer resource contract");

  llvm::SmallString<128> outputBase(directory.path());
  llvm::sys::path::append(outputBase, "spatial-pe");
  if (llvm::Error error = loom::adg::exportFabricDesign(
          finalized.roots().front(), store, outputBase))
    fail(test, llvm::toString(std::move(error)));
  llvm::SmallString<128> htmlPath(outputBase);
  htmlPath.append(".html");
  auto exportedHtml = llvm::MemoryBuffer::getFile(htmlPath);
  if (!exportedHtml)
    fail(test, exportedHtml.getError().message());
  require(test,
          exportedHtml.get()->getBuffer().contains("data-view-kind=\"fu\"") &&
              exportedHtml.get()->getBuffer().contains("ScalarIntegerAddSub") &&
              exportedHtml.get()->getBuffer().contains("ScalarIntegerMultiply"),
          "Fabric HTML did not expose the configured FU graph details");
}

void fuCapabilityRowsCorrelateRoutes() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits32 = take(test, PortType::bits(32));

  auto spatial =
      take(test, design.createSpatialCore("correlated-fu", {bits32, bits32},
                                          {bits32, bits32}));
  auto pe = take(
      test, spatial.addPe(
                {take(test, spatial.input(0)), take(test, spatial.input(1))},
                PeSpec::spatial({bits32, bits32}, {bits32, bits32})));
  auto fu =
      take(test, pe.addFu({take(test, pe.input(0)), take(test, pe.input(1))},
                          FuSpec{{bits32, bits32}, {bits32, bits32}}));

  auto inputMux =
      take(test, fu.addMux({take(test, fu.input(0)), take(test, fu.input(1))}));
  auto add =
      take(test,
           fu.addOperation(
               {take(test, inputMux.output(0)), take(test, inputMux.output(0))},
               integerCapability(
                   ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                   ::dataflow::OperationSchemaId::ArithAddI, bits32)));
  auto outputDemux = take(test, fu.addDemux(take(test, add.output(0)), 2));

  if (llvm::Error error = fu.addCapabilityTemplate(
          FuCapabilityTemplateSpec{{add}, {{inputMux, 0}, {outputDemux, 0}}}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = fu.addCapabilityTemplate(
          FuCapabilityTemplateSpec{{add}, {{inputMux, 1}, {outputDemux, 1}}}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = fu.close({take(test, outputDemux.output(0)),
                                    take(test, outputDemux.output(1))}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error =
          spatial.close({take(test, pe.output(0)), take(test, pe.output(1))}))
    fail(test, llvm::toString(std::move(error)));

  auto finalized = take(test, std::move(design).finalize());
  auto templates = finalized.roots().front().view().fuCapabilityTemplates(
      uniqueFuTemplate(test, finalized.roots().front().view()));
  require(test, templates.size() == 2,
          "correlated FU routes admitted a selector Cartesian product");
  for (const auto &record : templates)
    require(test, record.activeNodes.size() == 3,
            "correlated FU template omitted a selected physical node");
}

void typedMemoryFormsFinalize() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);

  const PortType bits0 = take(test, PortType::bits(0));
  const PortType bits32 = take(test, PortType::bits(32));
  const PortType memory32 =
      take(test, PortType::memory({PortType::kDynamicExtent}, bits32));
  auto spatial = take(test, design.createSpatialCore("memory-engine",
                                                     {memory32, bits32, bits0},
                                                     {bits32, bits0}));
  auto outputs =
      take(test,
           spatial.addMemory(
               {take(test, spatial.input(0)), take(test, spatial.input(1)),
                take(test, spatial.input(2))},
               take(test,
                    MemorySpec::create(
                        {memory32, bits32, bits0}, {bits32, bits0}, {0}, {},
                        MemoryEngineSpec::spatial({loadPortDeclaration()}),
                        std::nullopt,
                        operationConnectivity(test, managerMemoryTarget(0))))));
  if (llvm::Error error = spatial.close(outputs))
    fail(test, llvm::toString(std::move(error)));

  const PortType tagged0 = take(test, PortType::taggedBits(0, 4));
  const PortType tagged32 = take(test, PortType::taggedBits(32, 4));
  mlir::MLIRContext localContractContext(
      mlir::MLIRContext::Threading::DISABLED);
  auto temporal = take(test, design.createSpatialCore("temporal-local-memory",
                                                      {tagged32, tagged0},
                                                      {tagged32, tagged0}));
  auto temporalOutputs = take(
      test,
      temporal.addMemory(
          {take(test, temporal.input(0)), take(test, temporal.input(1))},
          take(test, MemorySpec::create(
                         {tagged32, tagged0}, {tagged32, tagged0}, {}, {},
                         MemoryEngineSpec::temporal(4, {loadPortDeclaration()}),
                         take(test, LocalMemoryServiceSpec::create(
                                        4096, localMemoryContract(
                                                  test, localContractContext))),
                         operationConnectivity(test, localMemoryTarget())))));
  if (llvm::Error error = temporal.close(temporalOutputs))
    fail(test, llvm::toString(std::move(error)));

  auto storage =
      take(test, design.createSpatialCore("local-storage", {}, {memory32}));
  auto storageOutputs = take(
      test,
      storage.addMemory(
          {},
          take(test, MemorySpec::create(
                         {}, {memory32}, {}, {0}, std::nullopt,
                         take(test, LocalMemoryServiceSpec::create(
                                        4096, localMemoryContract(
                                                  test, localContractContext))),
                         storageConnectivity(test)))));
  if (llvm::Error error = storage.close(storageOutputs))
    fail(test, llvm::toString(std::move(error)));

  loom::adg::FinalizedFabricDesign finalized =
      take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 3,
          "memory forms did not publish three SpatialCore roots");
  std::size_t localServices = 0;
  std::size_t operationEngines = 0;
  std::size_t storageOnly = 0;
  for (const auto &root : finalized.roots()) {
    const auto &view = root.view();
    const loom::fabric::FabricMemoryOccurrenceRef memory(uniqueEntity(
        test, view, loom::fabric::FabricEntityKind::FabricMemoryOccurrence));
    auto ports = view.memoryOperationPorts(memory);
    if (view.declaresLocalMemoryService(memory))
      ++localServices;
    if (!ports.empty()) {
      ++operationEngines;
      require(test,
              ports.size() == 1 && view.memoryOperationPort(ports.front()) &&
                  view.memoryCapabilityAlternative({ports.front(), 0}),
              "typed memory capability was not preserved by finalization");
      if (view.memorySchedule(memory) == ::fabric::Schedule::Temporal) {
        require(test, view.memoryResidentContextCount(memory) == 4,
                "temporal resident-context inventory was not preserved");
        require(
            test,
            view.inventorySize(
                loom::fabric::FabricInventoryOwnerRef::of(ports.front()),
                loom::fabric::FabricInventoryKind::MemoryOperationContext) == 4,
            "temporal operation-context references were not projected");
      }
    } else {
      ++storageOnly;
    }
  }
  require(test, localServices == 2 && operationEngines == 2 && storageOnly == 1,
          "Fabric lost the orthogonal memory engine and local service forms");
}

void publicMemoryLibraryBuildsHybridLocalMemories() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);

  expectError(test,
              loom::adg::makeHybrid32LocalMemory(
                  {(std::uint64_t(1) << 32) + 1, std::nullopt}),
              "32-bit address capacity");

  auto addMemoryRoot = [&](llvm::StringRef name,
                           LocalMemoryParameters parameters) {
    MemorySpec memory =
        take(test, loom::adg::makeHybrid32LocalMemory(parameters));
    require(test,
            memory.inputTypes().size() == 7 && memory.outputTypes().size() == 3,
            "hybrid local memory changed its maximal typed interface");
    auto spatial =
        take(test, design.createSpatialCore(name, memory.inputTypes(),
                                            memory.outputTypes()));
    std::vector<SpatialValue> inputs;
    inputs.reserve(memory.inputTypes().size());
    for (std::size_t ordinal = 0; ordinal < memory.inputTypes().size();
         ++ordinal)
      inputs.push_back(take(test, spatial.input(ordinal)));
    auto outputs = take(test, spatial.addMemory(inputs, memory));
    if (llvm::Error error = spatial.close(outputs))
      fail(test, llvm::toString(std::move(error)));
  };

  addMemoryRoot("spatial-hybrid-memory", {4096, std::nullopt});
  addMemoryRoot("temporal-hybrid-memory",
                {8192, loom::adg::TemporalMemoryParameters{4, 4}});

  MemorySpec tiered = take(
      test, loom::adg::makeHybrid32LocalMemory({4096, std::nullopt, true}));
  require(test,
          tiered.inputTypes().size() == 8 &&
              tiered.inputTypes().front().kind() == PortType::Kind::Memory,
          "tiered memory did not expose one leading manager capability");
  auto tieredRoot = take(test, design.createSpatialCore(
                                  "tiered-hybrid-memory", tiered.inputTypes(),
                                  tiered.outputTypes()));
  std::vector<SpatialValue> tieredInputs;
  for (std::size_t ordinal = 0; ordinal < tiered.inputTypes().size();
       ++ordinal)
    tieredInputs.push_back(take(test, tieredRoot.input(ordinal)));
  auto tieredOutputs = take(test, tieredRoot.addMemory(tieredInputs, tiered));
  if (llvm::Error error = tieredRoot.close(tieredOutputs))
    fail(test, llvm::toString(std::move(error)));

  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 3,
          "memory library did not publish both schedules and manager form");
  std::size_t spatialCount = 0;
  std::size_t temporalCount = 0;
  std::size_t managerCount = 0;
  for (const auto &root : finalized.roots()) {
    const auto &view = root.view();
    const loom::fabric::FabricMemoryOccurrenceRef memory(uniqueEntity(
        test, view, loom::fabric::FabricEntityKind::FabricMemoryOccurrence));
    require(test,
            view.declaresLocalMemoryService(memory) &&
                view.memoryOperationPorts(memory).size() == 2,
            "memory library lost its local service or load/store ports");
    for (const loom::fabric::FabricMemoryOperationPortRef port :
         view.memoryOperationPorts(memory)) {
      const auto *alternative = view.memoryCapabilityAlternative({port, 0});
      require(test, alternative && alternative->accessDomain,
              "hybrid memory lost its addressed access domain");
      const ::fabric::MemoryAccessClass *element = nullptr;
      for (const ::fabric::MemoryAccessClass &accessClass :
           alternative->accessDomain->accessClasses())
        if (accessClass.accessForm() ==
            ::dataflow::semantics::MemoryAccessForm::Element)
          element = &accessClass;
      require(test, element != nullptr,
              "hybrid memory lost its scalar element access class");
      require(test,
              element->elementWidths().contains(8) &&
                  element->elementWidths().contains(16) &&
                  element->elementWidths().contains(32) &&
                  !element->elementWidths().contains(64),
              "hybrid memory changed its scalar subword width domain");
      const bool reads = alternative->actorContractDomain.actorSchema() ==
                         ::dataflow::OperationSchemaId::DataflowLoad;
      require(
          test,
          reads ? element->readSubwordSemantics().contains(
                      ::fabric::ReadSubwordSemantics::ZeroExtend)
                : element->writeSubwordSemantics().contains(
                      ::fabric::WriteSubwordSemantics::ByteEnable),
          "hybrid memory lost its scalar subword physical guarantee");
    }
    const loom::fabric::FabricMemoryEndpointOwnerRef owner =
        loom::fabric::FabricMemoryEndpointOwnerRef::of(memory);
    const std::uint64_t endpointCount = view.memoryEndpointCount(owner);
    if (endpointCount != 0) {
      require(test, endpointCount == 1,
              "tiered memory exposed more than one manager endpoint");
      const loom::fabric::FabricMemoryEndpointRef endpoint{owner, 0};
      require(test,
              view.memoryEndpointRole(endpoint) ==
                  loom::fabric::FabricMemoryEndpointRole::Manager,
              "tiered memory endpoint is not a manager capability");
      const ::fabric::MemoryConnectivityContractRecord *connectivity =
          view.memoryConnectivity(memory);
      require(test, connectivity != nullptr,
              "tiered memory lost its dispatch eligibility relation");
      for (const auto &port : connectivity->operationPorts())
        require(test,
                port.capabilityTargetDomains.size() == 1 &&
                    port.capabilityTargetDomains.front().size() == 2,
                "tiered memory does not admit both local and manager targets");
      ++managerCount;
    }
    if (view.memorySchedule(memory) == ::fabric::Schedule::Spatial) {
      ++spatialCount;
    } else {
      ++temporalCount;
      require(test, view.memoryResidentContextCount(memory) == 4,
              "temporal memory lost its resident contexts");
    }
  }
  require(test, spatialCount == 2 && temporalCount == 1,
          "memory library changed a requested schedule");
  require(test, managerCount == 1,
          "memory library did not preserve the requested manager form");
}

void builtinPresetsExpandThroughPublicBuilder() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  const auto architecture =
      take(test, loom::adg::getBuiltinInstructionCoreArchitecture());
  const std::array expectedExtensions{
      loom::fabric::RiscVExtension::M,     loom::fabric::RiscVExtension::A,
      loom::fabric::RiscVExtension::F,     loom::fabric::RiscVExtension::D,
      loom::fabric::RiscVExtension::C,     loom::fabric::RiscVExtension::Zicsr,
      loom::fabric::RiscVExtension::Zifencei};
  require(test,
          llvm::equal(architecture.extensions(), expectedExtensions) &&
              llvm::equal(architecture.abiCapabilities(),
                          std::array{loom::fabric::RiscVAbi::Lp64d}),
          "builtin InstructionCore does not cover its exact compiler target");
  struct Expectation {
    loom::adg::BuiltinTargetPreset preset;
    std::uint32_t accCores;
    std::uint32_t spatialPes;
    std::uint32_t temporalPes;
    std::uint32_t spatialMemories;
    std::uint32_t temporalMemories;
  };
  const std::array<Expectation, 3> expectations{{
      {loom::adg::BuiltinTargetPreset::Small, 4, 12, 4, 1, 1},
      {loom::adg::BuiltinTargetPreset::Default, 8, 27, 9, 2, 2},
      {loom::adg::BuiltinTargetPreset::Large, 16, 48, 16, 4, 4},
  }};

  for (const Expectation &expected : expectations) {
    const auto &descriptor =
        loom::adg::getBuiltinTargetDescriptor(expected.preset);
    require(
        test,
        descriptor.scale.accCoreCount == expected.accCores &&
            descriptor.scale.spatialPeCount == expected.spatialPes &&
            descriptor.scale.temporalPeCount == expected.temporalPes &&
            descriptor.scale.spatialMemoryCount == expected.spatialMemories &&
            descriptor.scale.temporalMemoryCount == expected.temporalMemories,
        "builtin descriptor changed its scale contract");

    auto target =
        take(test, loom::adg::buildBuiltinTarget(store, expected.preset));
    require(test, target.roots().size() == 1,
            "builtin expansion did not publish one System root");
    const auto &root = target.roots().front();
    require(
        test,
        root.view().rootKind() == loom::fabric::FabricRootKind::System &&
            root.directDependencies().size() == 1 &&
            entityCount(root.view(),
                        loom::fabric::FabricEntityKind::AccCoreOccurrence) ==
                expected.accCores &&
            entityCount(root.view(),
                        loom::fabric::FabricEntityKind::SystemMemoryService) ==
                1 &&
            entityCount(
                root.view(),
                loom::fabric::FabricEntityKind::SystemServiceEndpoint) == 1,
        "builtin lost its SpatialCore, AccCore, or System memory inventory");

    auto systemView =
        take(test, loom::fabric::requireSystemRoot(root.view()));
    std::size_t memoryAttachments = 0;
    for (const auto &attachment : systemView.spatialAttachments())
      memoryAttachments +=
          attachment.spatialEndpoint.plane() ==
          loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Memory;
    require(test, memoryAttachments == expected.accCores,
            "builtin did not attach one manager capability per AccCore");

    auto module =
        take(test, loom::fabric::importEntireFabricRoot(
                       root.directDependencies().front().root, store));
    const loom::fabric::FabricModuleTemplateRef moduleTemplate(
        uniqueEntity(test, module.view(),
                     loom::fabric::FabricEntityKind::FabricModuleTemplate));
    require(test,
            entityCount(module.view(),
                        loom::fabric::FabricEntityKind::FabricPeOccurrence) ==
                    expected.spatialPes + expected.temporalPes &&
                entityCount(
                    module.view(),
                    loom::fabric::FabricEntityKind::FabricMemoryOccurrence) ==
                    expected.spatialMemories + expected.temporalMemories,
            "builtin SpatialCore lost its PE or memory scale");
    std::size_t wideScalarPorts = 0;
    for (std::uint64_t id = 0;; ++id) {
      const auto kind = module.view().entityKind(id);
      if (!kind)
        break;
      if (*kind !=
          loom::fabric::FabricEntityKind::FabricMemoryOccurrence)
        continue;
      const loom::fabric::FabricMemoryOccurrenceRef memory(id);
      for (const loom::fabric::FabricMemoryOperationPortRef port :
           module.view().memoryOperationPorts(memory)) {
        const auto *alternative =
            module.view().memoryCapabilityAlternative({port, 0});
        require(test, alternative && alternative->accessDomain,
                "builtin memory lost its typed access domain");
        const auto element = llvm::find_if(
            alternative->accessDomain->accessClasses(), [](const auto &access) {
              return access.accessForm() ==
                     ::dataflow::semantics::MemoryAccessForm::Element;
            });
        require(test,
                element != alternative->accessDomain->accessClasses().end() &&
                    element->elementWidths().contains(64),
                "builtin memory does not cover the common 64-bit scalar floor");
        ++wideScalarPorts;
      }
    }
    require(test,
            wideScalarPorts ==
                2 * (expected.spatialMemories + expected.temporalMemories),
            "builtin memory operation-port inventory changed unexpectedly");
    require(
        test,
        module.view().moduleBoundaryEndpointCount(
            moduleTemplate, loom::fabric::FabricPortDirection::Input) ==
            descriptor.scale.gatewayCount + 1,
        "builtin SpatialCore did not expose one shared manager capability");
    const loom::fabric::FabricModuleBoundaryEndpointRef memoryBoundary{
        moduleTemplate, loom::fabric::FabricPortDirection::Input,
        descriptor.scale.gatewayCount};
    require(test,
            module.view().moduleBoundaryEndpointPlane(memoryBoundary) ==
                loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::
                    Memory,
            "builtin manager capability is not on the memory plane");
  }

  const auto preset = loom::adg::BuiltinTargetPreset::Small;
  auto canonical = take(test, loom::adg::buildBuiltinTarget(store, preset));

  DesignBuilder moduleDesign(store);
  auto moduleExpansion =
      take(test, loom::adg::expandBuiltinSpatialCore(moduleDesign, preset));
  if (llvm::Error error =
          moduleExpansion.spatialCore.close(moduleExpansion.outputs))
    fail(test, llvm::toString(std::move(error)));
  auto modules = take(test, std::move(moduleDesign).finalize());
  require(test, modules.roots().size() == 1,
          "public builtin expansion did not publish one SpatialCore");

  DesignBuilder systemDesign(store);
  auto system = take(test, loom::adg::expandBuiltinSystem(
                               systemDesign, preset, modules.roots().front()));
  if (llvm::Error error = system.close())
    fail(test, llvm::toString(std::move(error)));
  auto direct = take(test, std::move(systemDesign).finalize());
  require(test,
          direct.roots().size() == 1 && canonical.roots().size() == 1 &&
              direct.roots().front().reference() ==
                  canonical.roots().front().reference(),
          "public builtin expansion changed the canonical preset identity");

  DesignBuilder customModuleDesign(store);
  auto customExpansion = take(
      test, loom::adg::expandBuiltinSpatialCore(customModuleDesign, preset));
  std::vector<SpatialValue> customOutputs = customExpansion.outputs;
  customOutputs.front() =
      take(test, customExpansion.spatialCore.addFifo(
                     customOutputs.front(),
                     FifoSpec{take(test, PortType::bits(128)), 3, false}));
  if (llvm::Error error = customExpansion.spatialCore.close(customOutputs))
    fail(test, llvm::toString(std::move(error)));
  auto customModules = take(test, std::move(customModuleDesign).finalize());
  require(test,
          customModules.roots().front().reference() !=
              modules.roots().front().reference(),
          "typed builtin extension did not change the custom Fabric identity");
}

void publicFuLibraryBuildsTypedGraphs() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits128 = take(test, PortType::bits(128));

  auto spatial =
      take(test, design.createSpatialCore("fu-library",
                                          {bits128, bits128, bits128, bits128},
                                          {bits128, bits128, bits128}));
  auto pe = take(
      test, spatial.addPe(
                {take(test, spatial.input(0)), take(test, spatial.input(1)),
                 take(test, spatial.input(2)), take(test, spatial.input(3))},
                PeSpec::spatial({bits128, bits128, bits128, bits128},
                                {bits128, bits128, bits128})));
  std::vector<loom::adg::PeValue> inputs;
  for (std::size_t ordinal = 0; ordinal != 4; ++ordinal)
    inputs.push_back(take(test, pe.input(ordinal)));
  if (llvm::Error error = loom::adg::addCoreAluFu(
          pe, llvm::ArrayRef<loom::adg::PeValue>(inputs).take_front(3),
          ::fabric::ResolvedIndexWidthSet::get(
              {::fabric::ResolvedIndexWidth::I64})))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = loom::adg::addMacFu(pe, inputs))
    fail(test, llvm::toString(std::move(error)));
  expectError(test,
              loom::adg::addLoopControlFu(pe, inputs,
                                          ::dataflow::StreamStepKind::Add,
                                          ::dataflow::StreamStepKind::Add),
              "distinct step kinds");
  if (llvm::Error error = loom::adg::addLoopControlFu(
          pe, inputs, ::dataflow::StreamStepKind::Add,
          ::dataflow::StreamStepKind::Sub))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = loom::adg::addVectorComputeFu(pe, inputs))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = loom::adg::addSpecialMathFu(
          pe, llvm::ArrayRef<loom::adg::PeValue>(inputs).take_front(2)))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error =
          spatial.close({take(test, pe.output(0)), take(test, pe.output(1)),
                         take(test, pe.output(2))}))
    fail(test, llvm::toString(std::move(error)));

  auto finalized = take(test, std::move(design).finalize());
  require(test,
          entityCount(finalized.roots().front().view(),
                      loom::fabric::FabricEntityKind::FabricFuOccurrence) == 5,
          "public FU helpers did not create five ordinary FU occurrences");
  bool sawMacDomain = false;
  bool sawLoopControlDomain = false;
  bool sawExactLoopControlContracts = false;
  bool sawStreamSemanticConfiguration = false;
  bool sawVectorSelectSemanticConfiguration = false;
  for (std::uint64_t id = 0;; ++id) {
    auto kind = finalized.roots().front().view().entityKind(id);
    if (!kind)
      break;
    if (*kind != loom::fabric::FabricEntityKind::FabricFuTemplate)
      continue;
    const loom::fabric::FabricFuTemplateRef fu(id);
    auto templates = finalized.roots().front().view().fuCapabilityTemplates(fu);
    if (templates.size() == 8) {
      bool hasRecurrence = false;
      for (const auto &record : templates) {
        unsigned activeOperations = 0;
        for (const auto &node : record.activeNodes)
          activeOperations += node.node == loom::fabric::FabricFuNodeKind::Op;
        hasRecurrence |= activeOperations == 3;
      }
      sawMacDomain |= hasRecurrence;
    }
    if (templates.size() == 7) {
      unsigned fusedTemplates = 0;
      for (const auto &record : templates) {
        unsigned activeOperations = 0;
        for (const auto &node : record.activeNodes)
          activeOperations += node.node == loom::fabric::FabricFuNodeKind::Op;
        fusedTemplates += activeOperations == 2;
      }
      sawLoopControlDomain |= fusedTemplates == 2;
    }
    unsigned exactLoopContracts = 0;
    for (const auto &capability :
         finalized.roots().front().view().resolvedFabricOpCapabilities(fu)) {
      std::uint32_t expectedPatterns = 0;
      switch (capability.implementationFamily) {
      case ::fabric::ImplementationFamilyId::LoopStream:
        sawStreamSemanticConfiguration |=
            capability.configurationFieldSchema.size() == 1;
        [[fallthrough]];
      case ::fabric::ImplementationFamilyId::LoopGate:
        expectedPatterns = 4;
        break;
      case ::fabric::ImplementationFamilyId::LoopCarry:
      case ::fabric::ImplementationFamilyId::LoopInvariant:
        expectedPatterns = 3;
        break;
      default:
        if (capability.implementationFamily ==
            ::fabric::ImplementationFamilyId::FixedVectorValueSelect)
          sawVectorSelectSemanticConfiguration |=
              capability.configurationFieldSchema.size() == 1;
        continue;
      }
      exactLoopContracts +=
          capability.resourceStateAndTimingContract.usePatternCount() ==
          expectedPatterns;
    }
    sawExactLoopControlContracts |= exactLoopContracts == 5;
  }
  require(test, sawMacDomain,
          "MacFu did not expose its complete carry-recurrence domain");
  require(test, sawLoopControlDomain,
          "LoopControlFu did not expose its seven coherent templates");
  require(test, sawExactLoopControlContracts,
          "loop-control resources lost their schema-case use patterns");
  require(test, sawStreamSemanticConfiguration,
          "stream capability lost its typed semantic configuration field");
  require(test, sawVectorSelectSemanticConfiguration,
          "vector select lost its lane-width configuration field");
  std::string text;
  llvm::raw_string_ostream stream(text);
  if (llvm::Error error =
          loom::fabric::writeFabricMlir(finalized.roots().front(), stream))
    fail(test, llvm::toString(std::move(error)));
  stream.flush();
  require(test,
          llvm::StringRef(text).contains("ScalarIntegerAddSub") &&
              llvm::StringRef(text).contains("LoopCarry") &&
              llvm::StringRef(text).contains("LoopStream") &&
              llvm::StringRef(text).contains("LoopInvariant") &&
              llvm::StringRef(text).contains("LoopGate") &&
              llvm::StringRef(text).contains("FixedVectorFloatFma") &&
              llvm::StringRef(text).contains("ScalarMathSqrt"),
          "public FU helpers lost generated implementation-family bindings");
}

void resolvedCapabilityPreservesTypedVectorGeometry() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits128 = take(test, PortType::bits(128));

  auto spatial = take(test, design.createSpatialCore(
                                "f32-vector", {bits128, bits128}, {bits128}));
  auto pe =
      take(test, spatial.addPe({take(test, spatial.input(0)),
                                take(test, spatial.input(1))},
                               PeSpec::spatial({bits128, bits128}, {bits128})));
  auto fu =
      take(test, pe.addFu({take(test, pe.input(0)), take(test, pe.input(1))},
                          FuSpec{{bits128, bits128}, {bits128}}));
  auto operation = take(
      test,
      fu.addOperation(
          {take(test, fu.input(0)), take(test, fu.input(1))},
          OperationCapabilitySpec{
              ::fabric::ImplementationFamilyId::FixedVectorFloatAddSub,
              ::fabric::FixedVectorFloatParams{
                  ::fabric::FloatFormatSet::get({::fabric::FloatFormat::F32}),
                  ::fabric::FloatBehaviorProfile::strictIEEE(), 128},
              {::dataflow::OperationSchemaId::ArithAddF,
               ::dataflow::OperationSchemaId::ArithSubF},
              {bits128},
              ::fabric::oneCycleElasticOperationResourceContract()}));
  if (llvm::Error error =
          fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{operation}, {}}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = fu.close({take(test, operation.output(0))}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = spatial.close({take(test, pe.output(0))}))
    fail(test, llvm::toString(std::move(error)));

  auto finalized = take(test, std::move(design).finalize());
  const loom::fabric::FabricFuTemplateRef fuRef =
      uniqueFuTemplate(test, finalized.roots().front().view());
  auto templates =
      finalized.roots().front().view().fuCapabilityTemplates(fuRef);
  require(test,
          templates.size() == 1 && templates.front().activeNodes.size() == 1,
          "custom vector FU changed its capability template");
  const auto *capability =
      finalized.roots().front().view().resolvedFabricOpCapability(
          templates.front().activeNodes.front());
  require(test, capability != nullptr,
          "custom vector FU lost its concrete capability");
  const auto &typedCapability = std::get<::fabric::FixedVectorFloatParams>(
      capability->parameterizedCapability);
  require(
      test,
      typedCapability.elementFormats.contains(::fabric::FloatFormat::F32) &&
          !typedCapability.elementFormats.contains(::fabric::FloatFormat::F64),
      "custom vector FU changed its typed floating format domain");

  mlir::MLIRContext actorContext(mlir::MLIRContext::Threading::DISABLED);
  auto vectorActor = [&](mlir::Type elementType, std::int64_t lanes) {
    mlir::Type vector = mlir::VectorType::get({lanes}, elementType);
    require(test,
            mlir::cast<mlir::VectorType>(vector).getElementType() ==
                elementType,
            "vector actor changed its element type");
    return ::dataflow::CanonicalActorSchemaProjection{
        ::dataflow::OperationSchemaId::ArithAddF,
        mlir::FunctionType::get(&actorContext, {vector, vector}, {vector}),
        ::dataflow::FloatingPointPayload{}};
  };
  auto f32Actor = vectorActor(mlir::Float32Type::get(&actorContext), 4);
  require(test,
          mlir::cast<mlir::VectorType>(f32Actor.type.getInput(0))
              .getElementType()
              .isF32(),
          "f32 vector actor changed its semantic element type");
  if (llvm::Error error = capability->admit(f32Actor, 32))
    fail(test, llvm::toString(std::move(error)));
  auto f64Actor = vectorActor(mlir::Float64Type::get(&actorContext), 2);
  expectError(test, capability->admit(f64Actor, 32),
              "element type is not admitted");

  loom::frontend::FabricCapabilityIndex index(finalized.roots().front().view());
  require(test, index.admittingOperationResources(f32Actor, 32).size() == 1,
          "Fabric capability index lost the admitted vector resource");
  require(test, index.admittingOperationResources(f64Actor, 32).empty(),
          "Fabric capability index treated equal payload width as semantics");
}

void builtinCoreCapabilitiesCoverTypedDomains() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  auto system = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  auto module =
      take(test, loom::fabric::importEntireFabricRoot(
                     system.roots().front().directDependencies().front().root,
                     store));

  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const auto actor = ::dataflow::CanonicalActorSchemaProjection{
      ::dataflow::OperationSchemaId::ArithIndexCast,
      mlir::FunctionType::get(&context, {mlir::IntegerType::get(&context, 32)},
                              {mlir::IndexType::get(&context)}),
      ::dataflow::NoPayload{}};
  loom::frontend::FabricCapabilityIndex index(module.view());
  require(test, !index.admittingOperationResources(actor, 32).empty(),
          "builtin Fabric rejected its 32-bit resolved index cast");
  require(test, !index.admittingOperationResources(actor, 64).empty(),
          "builtin Fabric rejected its 64-bit resolved index cast");

  mlir::Type f32 = mlir::Float32Type::get(&context);
  const auto floatMultiply = ::dataflow::CanonicalActorSchemaProjection{
      ::dataflow::OperationSchemaId::ArithMulF,
      mlir::FunctionType::get(&context, {f32, f32}, {f32}),
      ::dataflow::FloatingPointPayload{mlir::arith::FastMathFlags::nnan,
                                       std::nullopt}};
  require(test,
          !index.admittingOperationResources(floatMultiply, 32).empty(),
          "strict builtin Fabric did not refine relaxed scalar f32 multiply");

  const auto saturatingAdd = ::dataflow::CanonicalActorSchemaProjection{
      ::dataflow::OperationSchemaId::LLVMSAddSat,
      mlir::FunctionType::get(&context,
                              {mlir::IntegerType::get(&context, 32),
                               mlir::IntegerType::get(&context, 32)},
                              {mlir::IntegerType::get(&context, 32)}),
      ::dataflow::NoPayload{}};
  require(test, !index.admittingOperationResources(saturatingAdd, 32).empty(),
          "builtin Fabric has no scalar saturating arithmetic resource");

  const auto countTrailingZeros =
      ::dataflow::CanonicalActorSchemaProjection{
          ::dataflow::OperationSchemaId::MathCountTrailingZeros,
          mlir::FunctionType::get(&context,
                                  {mlir::IntegerType::get(&context, 32)},
                                  {mlir::IntegerType::get(&context, 32)}),
          ::dataflow::NoPayload{}};
  require(test,
          !index.admittingOperationResources(countTrailingZeros, 32).empty(),
          "builtin Fabric has no scalar zero-count resource");

  mlir::Type vectorI16 =
      mlir::VectorType::get({4}, mlir::IntegerType::get(&context, 16));
  const auto vectorSaturatingAdd =
      ::dataflow::CanonicalActorSchemaProjection{
          ::dataflow::OperationSchemaId::LLVMUAddSat,
          mlir::FunctionType::get(&context, {vectorI16, vectorI16},
                                  {vectorI16}),
          ::dataflow::NoPayload{}};
  require(
      test,
      !index.admittingOperationResources(vectorSaturatingAdd, 32).empty(),
      "builtin Fabric has no fixed-vector saturating arithmetic resource");

  const auto vectorCountLeadingZeros =
      ::dataflow::CanonicalActorSchemaProjection{
          ::dataflow::OperationSchemaId::LLVMCountLeadingZeros,
          mlir::FunctionType::get(&context, {vectorI16}, {vectorI16}),
          ::dataflow::ZeroPoisonPayload{true}};
  require(test,
          !index.admittingOperationResources(vectorCountLeadingZeros, 32)
               .empty(),
          "builtin Fabric has no fixed-vector zero-count resource");
}

void fuBackedgesAreExplicitAndResolved() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  const PortType bits32 = take(test, PortType::bits(32));

  {
    DesignBuilder incomplete(store);
    auto spatial = take(test, incomplete.createSpatialCore(
                                  "unresolved-feedback", {bits32}, {bits32}));
    auto pe = take(test, spatial.addPe({take(test, spatial.input(0))},
                                       PeSpec::spatial({bits32}, {bits32})));
    auto fu = take(
        test, pe.addFu({take(test, pe.input(0))}, FuSpec{{bits32}, {bits32}}));
    auto backedge = take(test, fu.createBackedge(bits32));
    expectError(test, fu.close({backedge.value()}), "unresolved backedge");
  }

  DesignBuilder design(store);
  auto spatial = take(
      test, design.createSpatialCore("resolved-feedback", {bits32}, {bits32}));
  auto pe = take(test, spatial.addPe({take(test, spatial.input(0))},
                                     PeSpec::spatial({bits32}, {bits32})));
  auto fu = take(
      test, pe.addFu({take(test, pe.input(0))}, FuSpec{{bits32}, {bits32}}));
  auto backedge = take(test, fu.createBackedge(bits32));
  auto sum = take(
      test,
      fu.addOperation({take(test, fu.input(0)), backedge.value()},
                      integerCapability(
                          ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                          ::dataflow::OperationSchemaId::ArithAddI, bits32)));
  FuValue sumValue = take(test, sum.output(0));
  if (llvm::Error error = fu.resolveBackedge(std::move(backedge), sumValue))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error =
          fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{sum}, {}}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = fu.close({sumValue}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = spatial.close({take(test, pe.output(0))}))
    fail(test, llvm::toString(std::move(error)));
  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "resolved FU backedge did not finalize");
}

void spatialBackedgesEnableCyclicTopology() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  const PortType bits32 = take(test, PortType::bits(32));

  {
    DesignBuilder incomplete(store);
    auto spatial = take(test, incomplete.createSpatialCore("unresolved-cycle",
                                                           {bits32}, {bits32}));
    auto backedge = take(test, spatial.createBackedge(bits32));
    expectError(test, spatial.close({backedge.value()}), "unresolved backedge");
  }

  DesignBuilder design(store);
  auto spatial =
      take(test, design.createSpatialCore("cyclic-switch", {bits32}, {bits32}));
  auto backedge = take(test, spatial.createBackedge(bits32));
  auto routed = take(
      test,
      spatial.addSwitch({take(test, spatial.input(0)), backedge.value()},
                        SwitchSpec::spatial({bits32, bits32}, {bits32, bits32},
                                            {{0, 1}, {0, 1}})));
  SpatialValue buffered =
      take(test, spatial.addFifo(routed[0], FifoSpec{bits32, 2, true}));
  if (llvm::Error error =
          spatial.resolveBackedge(std::move(backedge), buffered))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = spatial.close({routed[1]}))
    fail(test, llvm::toString(std::move(error)));

  auto finalized = take(test, std::move(design).finalize());
  require(test,
          finalized.roots().size() == 1 &&
              !finalized.roots().front().view().admittedTraversals().empty(),
          "resolved SpatialCore cycle did not finalize as explicit topology");
}

void routedFuLibraryBuildsHeterogeneousBoundaries() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits128 = take(test, PortType::bits(128));

  auto adapter = take(test, design.createSpatialCore(
                                "vector-adapter", {bits128, bits128, bits128},
                                {bits128, bits128, bits128}));
  auto adapterPe =
      take(test, adapter.addPe({take(test, adapter.input(0)),
                                take(test, adapter.input(1)),
                                take(test, adapter.input(2))},
                               PeSpec::spatial({bits128, bits128, bits128},
                                               {bits128, bits128, bits128})));
  std::vector<loom::adg::PeValue> adapterInputs;
  for (std::size_t ordinal = 0; ordinal != 3; ++ordinal)
    adapterInputs.push_back(take(test, adapterPe.input(ordinal)));
  if (llvm::Error error =
          loom::adg::addVectorAdapterFu(adapterPe, adapterInputs))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = adapterPe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = adapter.close({take(test, adapterPe.output(0)),
                                         take(test, adapterPe.output(1)),
                                         take(test, adapterPe.output(2))}))
    fail(test, llvm::toString(std::move(error)));

  auto token = take(test, design.createSpatialCore(
                              "token-control",
                              {bits128, bits128, bits128, bits128, bits128},
                              {bits128, bits128, bits128, bits128}));
  auto tokenPe = take(
      test,
      token.addPe({take(test, token.input(0)), take(test, token.input(1)),
                   take(test, token.input(2)), take(test, token.input(3)),
                   take(test, token.input(4))},
                  PeSpec::spatial({bits128, bits128, bits128, bits128, bits128},
                                  {bits128, bits128, bits128, bits128})));
  std::vector<loom::adg::PeValue> tokenInputs;
  for (std::size_t ordinal = 0; ordinal != 5; ++ordinal)
    tokenInputs.push_back(take(test, tokenPe.input(ordinal)));
  if (llvm::Error error = loom::adg::addTokenControlFu(tokenPe, tokenInputs))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = tokenPe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = token.close(
          {take(test, tokenPe.output(0)), take(test, tokenPe.output(1)),
           take(test, tokenPe.output(2)), take(test, tokenPe.output(3))}))
    fail(test, llvm::toString(std::move(error)));

  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 2,
          "routed FU helpers did not finalize both boundary shapes");
  for (const auto &root : finalized.roots())
    require(test,
            root.view()
                    .fuCapabilityTemplates(uniqueFuTemplate(test, root.view()))
                    .size() == 4,
            "routed FU helper did not derive four complete templates");
  std::string text;
  llvm::raw_string_ostream stream(text);
  for (const auto &root : finalized.roots())
    if (llvm::Error error = loom::fabric::writeFabricMlir(root, stream))
      fail(test, llvm::toString(std::move(error)));
  stream.flush();
  require(test,
          llvm::StringRef(text).contains("FixedVectorParallelize") &&
              llvm::StringRef(text).contains("FixedVectorSerialize") &&
              llvm::StringRef(text).contains("TokenSync") &&
              llvm::StringRef(text).contains("TokenDemux"),
          "routed FU helpers lost heterogeneous operation capabilities");
}

void heterogeneousSystemFinalizes() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  const PortType bits32 = take(test, PortType::bits(32));

  DesignBuilder moduleDesign(store);
  auto firstSpatial =
      take(test, moduleDesign.createSpatialCore("system-spatial-buffered",
                                                {bits32}, {bits32}));
  SpatialValue firstBuffered =
      take(test, firstSpatial.addFifo(take(test, firstSpatial.input(0)),
                                      FifoSpec{bits32, 2, true}));
  if (llvm::Error error = firstSpatial.close({firstBuffered}))
    fail(test, llvm::toString(std::move(error)));
  auto secondSpatial =
      take(test, moduleDesign.createSpatialCore("system-spatial-deep-buffered",
                                                {bits32}, {bits32}));
  SpatialValue secondBuffered =
      take(test, secondSpatial.addFifo(take(test, secondSpatial.input(0)),
                                       FifoSpec{bits32, 3, false}));
  if (llvm::Error error = secondSpatial.close({secondBuffered}))
    fail(test, llvm::toString(std::move(error)));
  loom::adg::FinalizedFabricDesign moduleClosure =
      take(test, std::move(moduleDesign).finalize());
  require(test, moduleClosure.roots().size() == 2,
          "heterogeneous System fixture did not publish both SpatialCores");

  DesignBuilder systemDesign(store);
  auto system = take(test, systemDesign.createSystem("heterogeneous-system"));
  auto firstImported =
      take(test, system.importSpatialCore(moduleClosure.roots()[0]));
  auto secondImported =
      take(test, system.importSpatialCore(moduleClosure.roots()[1]));
  auto architecture = instructionArchitecture(test);
  auto firstCore =
      take(test, system.addAccCore(architecture, inOrderMicroarchitecture(test),
                                   firstImported));
  auto secondCore = take(
      test, system.addAccCore(architecture, outOfOrderMicroarchitecture(test),
                              secondImported));

  auto transport =
      take(test, system.addTransportResource(
                     {{bits32}, {bits32}, singleUseResourceContract(test)}));
  auto pattern = take(test, system.addTransferPattern(transport, 0, {0}, 0));
  if (llvm::Error error =
          system.connect(take(test, firstCore.spatialTransportOutput(0)),
                         take(test, transport.input(0))))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error =
          system.connect(take(test, transport.output(0)),
                         take(test, secondCore.spatialTransportInput(0))))
    fail(test, llvm::toString(std::move(error)));

  auto clock = take(test, system.createHardwareDomain());
  auto serviceRate =
      take(test, system.createServiceRate(
                     clock, 1, 1, 4,
                     loom::fabric::ServiceProgress(
                         std::in_place_type<::fabric::FairEventual>)));
  mlir::MLIRContext contractContext(
      mlir::MLIRContext::Threading::DISABLED);
  auto memoryService = take(test, system.addMemoryService(systemMemoryContract(
                                      test, contractContext)));
  auto memoryEndpoint =
      take(test, system.addServiceEndpoint(
                     memoryService,
                     systemMemoryCapabilities(test, std::move(serviceRate))));
  auto clockContract =
      take(test, loom::fabric::ClockDomainContractRecord::create(1'000, 0));
  if (llvm::Error error = clock.close(
          {firstCore.instructionCoreDomainMember(),
           firstCore.spatialCoreDomainMember(),
           secondCore.instructionCoreDomainMember(),
           secondCore.spatialCoreDomainMember(), transport.domainMember(),
           pattern.domainMember(), memoryService.domainMember(),
           memoryEndpoint.domainMember()},
          std::move(clockContract)))
    fail(test, llvm::toString(std::move(error)));

  if (llvm::Error error = system.close())
    fail(test, llvm::toString(std::move(error)));
  loom::adg::FinalizedFabricDesign finalized =
      take(test, std::move(systemDesign).finalize());
  require(test, finalized.roots().size() == 1,
          "System design did not publish one root");
  const auto &root = finalized.roots().front();
  require(test, root.view().rootKind() == loom::fabric::FabricRootKind::System,
          "System Builder published the wrong root kind");
  require(test, root.directDependencies().size() == 2,
          "heterogeneous System did not retain both SpatialCore types");
  for (const auto &module : moduleClosure.roots()) {
    bool found = false;
    for (const auto &dependency : root.directDependencies())
      found |= dependency.root == module.reference();
    require(test, found,
            "heterogeneous System changed an exact SpatialCore dependency");
  }
  require(test,
          entityCount(root.view(),
                      loom::fabric::FabricEntityKind::AccCoreOccurrence) == 2,
          "heterogeneous System lost an AccCore occurrence");
  require(test,
          entityCount(root.view(),
                      loom::fabric::FabricEntityKind::SystemMemoryService) == 1,
          "heterogeneous System lost its memory service");
  require(test,
          entityCount(root.view(),
                      loom::fabric::FabricEntityKind::SystemServiceEndpoint) ==
              1,
          "heterogeneous System lost its service endpoint");
  auto systemView = take(test, loom::fabric::requireSystemRoot(root.view()));
  require(test, systemView.spatialAttachments().size() == 4,
          "System Builder did not attach every SpatialCore boundary");
  require(
      test,
      systemView.transportResources().size() == 1 &&
          systemView.transferPatterns(systemView.transportResources().front())
                  .size() == 1,
      "System Builder lost its explicit transport resource or pattern");
  require(test, root.view().pointConnections().size() == 2,
          "System Builder lost its arbitrary directed transport path");

  std::string mlirText;
  llvm::raw_string_ostream stream(mlirText);
  if (llvm::Error error = loom::fabric::writeFabricMlir(root, stream))
    fail(test, llvm::toString(std::move(error)));
  stream.flush();
  require(test, llvm::StringRef(mlirText).contains("fabric.system"),
          "finalized System did not export Fabric MLIR");

  llvm::SmallString<128> outputBase(directory.path());
  llvm::sys::path::append(outputBase, "heterogeneous-system");
  if (llvm::Error error =
          loom::adg::exportFabricDesign(root, store, outputBase))
    fail(test, llvm::toString(std::move(error)));

  llvm::SmallString<128> mlirPath(outputBase);
  mlirPath.append(".mlir");
  llvm::SmallString<128> htmlPath(outputBase);
  htmlPath.append(".html");
  auto exportedMlir = llvm::MemoryBuffer::getFile(mlirPath);
  if (!exportedMlir)
    fail(test, exportedMlir.getError().message());
  auto exportedHtml = llvm::MemoryBuffer::getFile(htmlPath);
  if (!exportedHtml)
    fail(test, exportedHtml.getError().message());
  require(test, exportedMlir.get()->getBuffer().contains("fabric.system"),
          "paired export did not write the canonical Fabric MLIR projection");
  const llvm::StringRef html = exportedHtml.get()->getBuffer();
  const std::size_t firstSpatialView =
      html.find("data-view-kind=\"spatial-core\"");
  const std::size_t secondSpatialView =
      html.find("data-view-kind=\"spatial-core\"", firstSpatialView + 1);
  require(
      test,
      html.contains("data-layout-engine=\"loom-layered-v1\"") &&
          html.contains("data-view-kind=\"system-overview\"") &&
          html.contains("data-view-kind=\"system-noc\"") &&
          html.contains("data-view-kind=\"system\"") &&
          html.contains("data-view-kind=\"spatial-core\"") &&
          html.contains("data-entity-kind=\"fabric.acc_core_occurrence\"") &&
          html.contains("data-entity-kind=\"fabric.fifo_occurrence\"") &&
          html.contains("data-x=\"") && html.contains("data-y=\""),
      "Fabric HTML did not contain the precomputed two-level topology");
  require(test,
          firstSpatialView != llvm::StringRef::npos &&
              secondSpatialView != llvm::StringRef::npos,
          "Fabric HTML did not preserve both heterogeneous SpatialCore views");
  const std::size_t overviewBegin =
      html.find("data-view-kind=\"system-overview\"");
  const std::size_t overviewEnd = html.find("</svg>", overviewBegin);
  require(test,
          overviewBegin != llvm::StringRef::npos &&
              overviewEnd != llvm::StringRef::npos,
          "Fabric HTML has no bounded System overview");
  const llvm::StringRef overview =
      html.slice(overviewBegin, overviewEnd + llvm::StringRef("</svg>").size());
  require(
      test,
      overview.contains("data-entity-kind=\"visual.noc_summary\"") &&
          overview.contains("data-entity-kind=\"fabric.acc_core_occurrence\""),
      "System overview lost its AccCore or NoC architecture summary");
  require(test,
          !overview.contains(
              "data-entity-kind=\"fabric.system_transport_resource\""),
          "System overview exposed individual NoC transport resources");
  const std::size_t nocBegin = html.find(" data-view-kind=\"system-noc\"");
  const std::size_t nocEnd = html.find("</svg>", nocBegin);
  require(test,
          nocBegin != llvm::StringRef::npos && nocEnd != llvm::StringRef::npos,
          "Fabric HTML has no bounded NoC topology view");
  const llvm::StringRef noc =
      html.slice(nocBegin, nocEnd + llvm::StringRef("</svg>").size());
  require(test,
          noc.contains("data-entity-kind=\"fabric.acc_core_occurrence\"") &&
              noc.contains("data-entity-kind=\"fabric.system_memory_service\""),
          "NoC topology lost an architecture participant");
  require(test,
          !noc.contains("data-entity-kind=\"fabric.module_dependency\"") &&
              !noc.contains(
                  "data-entity-kind=\"fabric.system_service_endpoint\"") &&
              !noc.contains("data-entity-kind=\"fabric.hardware_domain\""),
          "NoC topology exposed a detail-only node");
  require(test,
          !html.contains("forceSimulation") && !html.contains("dagre.layout") &&
              !html.contains("elk.layout"),
          "Fabric HTML contains a browser-side graph layout engine");
}

} // namespace

int main() {
  regularAndIrregularSpatialCoresFinalize();
  foreignHandlesAndIncompleteRootsFailClosed();
  spatialCoreTemplatesInstantiateAndElaborate();
  typedPeFuGraphsFinalize();
  fuCapabilityRowsCorrelateRoutes();
  typedMemoryFormsFinalize();
  publicMemoryLibraryBuildsHybridLocalMemories();
  builtinPresetsExpandThroughPublicBuilder();
  publicFuLibraryBuildsTypedGraphs();
  resolvedCapabilityPreservesTypedVectorGeometry();
  builtinCoreCapabilitiesCoverTypedDomains();
  fuBackedgesAreExplicitAndResolved();
  spatialBackedgesEnableCyclicTopology();
  routedFuLibraryBuildsHeterogeneousBoundaries();
  heterogeneousSystemFinalizes();
  return EXIT_SUCCESS;
}
