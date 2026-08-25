#include "ADGBuilderTestSupport.h"

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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::adg::test {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, llvm::StringRef message) {
  if (!condition)
    fail(test, message.str());
}

void expectError(llvm::StringRef test, llvm::Error error,
                 llvm::StringRef diagnostic) {
  if (!error)
    fail(test, "accepted invalid ADG authoring");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(diagnostic), message);
}

void expectFabricRefError(llvm::StringRef test, llvm::Error error,
                          loom::fabric::FabricRefErrorKind expected) {
  if (!error)
    fail(test, "accepted invalid Fabric reference");
  require(test,
          loom::fabric::takeFabricRefErrorKind(std::move(error)) == expected,
          "Fabric reference failure kind changed");
}

TemporaryDirectory::TemporaryDirectory(llvm::StringRef test)
    : test_(test.str()) {
  llvm::SmallString<128> path;
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory("loom-adg-builder", path))
    fail(test, error.message());
  path_ = path.str().str();
}

TemporaryDirectory::~TemporaryDirectory() {
  if (std::error_code error = llvm::sys::fs::remove_directories(path_))
    llvm::errs() << test_ << ": unable to remove temporary directory: "
                 << error.message() << '\n';
}

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
  const auto width = llvm::find_if(
      ::fabric::integerWidthDomain, [&](::fabric::IntegerWidth candidate) {
        return ::fabric::getBitWidth(candidate) == outputType.width();
      });
  if (width == ::fabric::integerWidthDomain.end())
    fail("integerCapability", "test port has no scalar integer width");
  return OperationCapabilitySpec{
      family,
      ::fabric::ScalarIntegerParams{::fabric::IntegerWidthSet::get({*width})},
      {operation},
      {outputType},
      ::fabric::oneCycleElasticOperationResourceContract()};
}

::fabric::UnsignedDomain singleton(std::uint64_t value) {
  return take("memory singleton",
              ::fabric::UnsignedDomain::fromCanonical({{value, value}}));
}

::fabric::MemoryAddressDomain rootRelativeAddress(std::uint64_t indexBits) {
  return take(
      "root-relative address domain",
      ::fabric::MemoryAddressDomain::rootRelative(singleton(indexBits)));
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
                std::move(alignment), std::move(read), std::move(write),
                rootRelativeAddress(64)));
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
  auto access = take("memory access class",
                     ::fabric::MemoryAccessClass::create(
                         ::dataflow::semantics::MemoryAccessForm::Element,
                         singleton(32), singleton(1),
                         {{::dataflow::semantics::MemoryMaskForm::Absent,
                           ::fabric::InactiveLaneSemantics::NotApplicable}},
                         std::move(alignment), std::move(read),
                         std::move(write), rootRelativeAddress(32)));
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

MemoryConnectivitySpec
operationConnectivityWithInternalEdge(llvm::StringRef test,
                                      ::fabric::MemoryDispatchTarget target) {
  ::fabric::MemoryConnectivityDeclaration declaration;
  declaration.operationPorts = {{{{std::move(target)}}}};
  declaration.internalConnections = {{2, 0}};
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
                                 FifoSpec{tagged32x4, 2, true}))
          .value();
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
      take(test, irregular.addFifo(irregularData, FifoSpec{bits32, 3, false}))
          .value();
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
      take(test, pipeline.addFifo(pipelineInput, FifoSpec{bits16, 2, true}))
          .value();
  if (llvm::Error error = pipeline.close({pipelineOutput}))
    fail(test, llvm::toString(std::move(error)));
  const auto childClocks = take(
      test, pipeline.domainSlots(loom::fabric::FabricClockResetKind::Clock));
  const auto childResets = take(
      test, pipeline.domainSlots(loom::fabric::FabricClockResetKind::Reset));

  auto top = take(test, design.createSpatialCore("top", {bits32}, {bits16}));
  SpatialValue topInput = take(test, top.input(0));
  const auto parentClock = take(
      test, top.declareDomainSlot(loom::fabric::FabricClockResetKind::Clock));
  const auto parentReset = take(
      test, top.declareDomainSlot(loom::fabric::FabricClockResetKind::Reset));
  auto instance =
      take(test, top.instantiate(pipeline, {topInput},
                                 {{childClocks.front(), parentClock},
                                  {childResets.front(), parentReset}}));
  require(test, instance.size() == 1,
          "typed SpatialCore instance returned the wrong result count");
  for (const auto &member : {take(test, top.inputDomainMember(0)),
                             take(test, top.outputDomainMember(0))}) {
    if (llvm::Error error = top.assignDomainSlot(member, parentClock))
      fail(test, llvm::toString(std::move(error)));
    if (llvm::Error error = top.assignDomainSlot(member, parentReset))
      fail(test, llvm::toString(std::move(error)));
  }
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
  const auto productTemplate = take(
      test, spatialFu.addCapabilityTemplateWithHandle(FuCapabilityTemplateSpec{
                {product}, {{aRoutes, 1}, {bRoutes, 1}, {resultMux, 1}}}));
  const auto sumTemplate = take(
      test, spatialFu.addCapabilityTemplateWithHandle(FuCapabilityTemplateSpec{
                {sum}, {{aRoutes, 0}, {bRoutes, 0}, {resultMux, 0}}}));
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
  const auto finalizedSum = take(test, finalized.resolve(sumTemplate));
  const auto finalizedProduct = take(test, finalized.resolve(productTemplate));
  require(test,
          finalizedSum.artifact == finalized.roots()[0].reference().artifact &&
              finalizedProduct.artifact ==
                  finalized.roots()[0].reference().artifact &&
              finalizedSum.entity != finalizedProduct.entity,
          "FU capability handles lost their exact finalized targets");
  const auto hasFamily = [&](const auto &reference,
                             ::fabric::ImplementationFamilyId family) {
    const auto records =
        finalized.roots()[0].view().fuCapabilityTemplates(reference.entity.fu);
    if (reference.entity.ordinal >= records.size())
      return false;
    for (const auto &node : records[reference.entity.ordinal].activeNodes) {
      if (node.node != loom::fabric::FabricFuNodeKind::Op)
        continue;
      const auto *capability =
          finalized.roots()[0].view().resolvedFabricOpCapability(node);
      if (capability && capability->implementationFamily == family)
        return true;
    }
    return false;
  };
  require(
      test,
      hasFamily(finalizedSum,
                ::fabric::ImplementationFamilyId::ScalarIntegerAddSub) &&
          hasFamily(finalizedProduct,
                    ::fabric::ImplementationFamilyId::ScalarIntegerMultiply),
      "FU capability handle correspondence changed row semantics");
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

void temporalResourceGrantFinalizes() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType tagged32 = take(test, PortType::taggedBits(32, 4));

  auto spatial =
      take(test, design.createSpatialCore("temporal-round-robin",
                                          {tagged32, tagged32}, {tagged32}));
  auto routed = take(
      test,
      spatial.addSwitch(
          {take(test, spatial.input(0)), take(test, spatial.input(1))},
          SwitchSpec::temporal({tagged32, tagged32}, {tagged32}, {{0, 1}}, 4,
                               ::fabric::TemporalSwitchRoundRobin{{0, 1}, 0})));
  if (llvm::Error error = spatial.close(routed.values()))
    fail(test, llvm::toString(std::move(error)));

  auto finalized = take(test, std::move(design).finalize());
  const auto &view = finalized.roots().front().view();
  const loom::fabric::FabricSwitchOccurrenceRef sw(uniqueEntity(
      test, view, loom::fabric::FabricEntityKind::FabricSwitchOccurrence));
  const ::fabric::ResourceContract *contract =
      view.resourceContract(loom::fabric::FabricInventoryOwnerRef::of(sw));
  require(test, contract != nullptr && contract->requesterCount() == 2,
          "temporal switch lost its competing requesters");
  const std::optional<::fabric::GrantPolicyView> policy =
      contract->grantPolicy();
  const auto *roundRobin =
      policy ? std::get_if<::fabric::RoundRobinView>(&*policy) : nullptr;
  require(test,
          roundRobin && roundRobin->requesterCycle().size() == 2 &&
              roundRobin->resetCursor().ordinal() == 0,
          "temporal switch lost its deterministic round-robin policy");
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
  if (llvm::Error error = spatial.close(outputs.values()))
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
  if (llvm::Error error = temporal.close(temporalOutputs.values()))
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
  if (llvm::Error error = storage.close(storageOutputs.values()))
    fail(test, llvm::toString(std::move(error)));

  auto duplicates = take(
      test,
      design.createSpatialCore("duplicate-memory-engines",
                               {memory32, bits32, bits0, memory32, bits32,
                                bits0, memory32, bits32, bits0},
                               {bits32, bits0, bits32, bits0, bits32, bits0}));
  std::vector<SpatialValue> duplicateOutputs;
  for (unsigned base : {0U, 3U, 6U}) {
    auto memoryOutputs = take(
        test,
        duplicates.addMemory(
            {take(test, duplicates.input(base)),
             take(test, duplicates.input(base + 1)),
             take(test, duplicates.input(base + 2))},
            take(test, MemorySpec::create(
                           {memory32, bits32, bits0}, {bits32, bits0}, {0}, {},
                           MemoryEngineSpec::spatial({loadPortDeclaration()}),
                           std::nullopt,
                           base == 6 ? operationConnectivityWithInternalEdge(
                                           test, managerMemoryTarget(0))
                                     : operationConnectivity(
                                           test, managerMemoryTarget(0))))));
    duplicateOutputs.insert(duplicateOutputs.end(),
                            memoryOutputs.values().begin(),
                            memoryOutputs.values().end());
  }
  if (llvm::Error error = duplicates.close(duplicateOutputs))
    fail(test, llvm::toString(std::move(error)));

  loom::adg::FinalizedFabricDesign finalized =
      take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 4,
          "memory forms did not publish four SpatialCore roots");
  std::size_t localServices = 0;
  std::size_t operationEngines = 0;
  std::size_t storageOnly = 0;
  for (const auto &root : finalized.roots()) {
    const auto &view = root.view();
    std::vector<loom::fabric::FabricMemoryOccurrenceRef> memories;
    for (std::uint64_t id = 0;; ++id) {
      auto kind = view.entityKind(id);
      if (!kind)
        break;
      if (*kind == loom::fabric::FabricEntityKind::FabricMemoryOccurrence)
        memories.emplace_back(id);
    }
    std::vector<loom::fabric::FabricMemoryEngineTemplateRef> engineTemplates;
    for (const loom::fabric::FabricMemoryOccurrenceRef memory : memories) {
      auto ports = view.memoryOperationPorts(memory);
      if (view.declaresLocalMemoryService(memory)) {
        ++localServices;
        const auto *service = view.localMemoryService(memory);
        require(test,
                service && service->regions().size() == 1 &&
                    service->regions().front().addressBaseBytes == 0 &&
                    service->regions().front().sizeBytes == 4096 &&
                    service->capabilities().size() == 1 &&
                    service->resourceContract().usePatternCount() == 1,
                "local memory service contract was not preserved");
      }
      if (!ports.empty()) {
        ++operationEngines;
        const auto engine = view.memoryEngineTemplateOf(memory);
        require(test, engine && view.memoryEngineTemplate(*engine),
                "Operation Engine has no canonical template projection");
        engineTemplates.push_back(*engine);
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
                  loom::fabric::FabricInventoryKind::MemoryOperationContext) ==
                  4,
              "temporal operation-context references were not projected");
        }
      } else {
        ++storageOnly;
        require(test, !view.memoryEngineTemplateOf(memory),
                "storage-only memory acquired an Operation Engine template");
      }
    }
    std::vector<std::uint64_t> uniqueTemplateIds;
    for (loom::fabric::FabricMemoryEngineTemplateRef engine : engineTemplates)
      if (!llvm::is_contained(uniqueTemplateIds, engine.id()))
        uniqueTemplateIds.push_back(engine.id());
    if (memories.size() == 3) {
      require(test,
              engineTemplates.size() == 3 && uniqueTemplateIds.size() == 2,
              "memory template dedup ignored an exact semantic delta");
      std::optional<loom::fabric::FabricMemoryEngineTemplateRef> edgeTemplate;
      std::optional<loom::fabric::FabricMemoryEngineTemplateRef> plainTemplate;
      for (std::uint64_t id : uniqueTemplateIds) {
        const loom::fabric::FabricMemoryEngineTemplateRef engine(id);
        const auto *record = view.memoryEngineTemplate(engine);
        if (record && record->internalConnections.size() == 1)
          edgeTemplate = engine;
        else if (record && record->internalConnections.empty())
          plainTemplate = engine;
      }
      require(test, edgeTemplate && plainTemplate,
              "memory template semantic delta was not preserved");
      const auto *edgeEngine = view.memoryEngineTemplate(*edgeTemplate);
      require(test, edgeEngine && edgeEngine->internalConnections.size() == 1,
              "memory template lost its internal connection relation");
      const loom::fabric::FabricMemoryEngineTemplateInternalConnectionRef edge{
          *edgeTemplate, {*edgeTemplate, 2}, {*edgeTemplate, 0}};
      if (llvm::Error error = loom::fabric::validateFabricRef(view, edge))
        fail(test, llvm::toString(std::move(error)));
      expectFabricRefError(
          test,
          loom::fabric::validateFabricRef(
              view,
              loom::fabric::FabricMemoryEngineTemplateOperationPortRef{
                  *edgeTemplate, 1}),
          loom::fabric::FabricRefErrorKind::OrdinalOutOfRange);
      expectFabricRefError(
          test,
          loom::fabric::validateFabricRef(
              view,
              loom::fabric::FabricMemoryEngineTemplateInternalConnectionRef{
                  *edgeTemplate, {*edgeTemplate, 3}, {*edgeTemplate, 0}}),
          loom::fabric::FabricRefErrorKind::TraversalNotAdmitted);
      expectFabricRefError(
          test,
          loom::fabric::validateFabricRef(
              view,
              loom::fabric::FabricMemoryEngineTemplateInternalConnectionRef{
                  *edgeTemplate, {*plainTemplate, 2}, {*edgeTemplate, 0}}),
          loom::fabric::FabricRefErrorKind::WrongOwner);
    }
    require(
        test,
        entityCount(
            view, loom::fabric::FabricEntityKind::FabricMemoryEngineTemplate) ==
            uniqueTemplateIds.size(),
        "Memory Operation Engine template inventory is not canonical");
  }
  require(test, localServices == 2 && operationEngines == 5 && storageOnly == 1,
          "Fabric lost the orthogonal memory engine and local service forms");
}

void publicMemoryLibraryBuildsHybridLocalMemories() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const loom::adg::MemoryInterfaceParameters interface{
      loom::adg::MemoryAccessDomainParameters{128, std::nullopt, 4,
                                              singleton(64)},
      64, 128};

  expectError(
      test,
      loom::adg::makeHybrid32LocalMemory(
          {(std::uint64_t(1) << 32) + 1, interface, std::nullopt, false}),
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
    if (llvm::Error error = spatial.close(outputs.values()))
      fail(test, llvm::toString(std::move(error)));
  };

  addMemoryRoot("spatial-hybrid-memory",
                {4096, interface, std::nullopt, false});
  addMemoryRoot(
      "temporal-hybrid-memory",
      {8192, interface, loom::adg::TemporalMemoryParameters{4, 4}, false});

  MemorySpec tiered = take(test, loom::adg::makeHybrid32LocalMemory(
                                     {4096, interface, std::nullopt, true}));
  require(test,
          tiered.inputTypes().size() == 8 &&
              tiered.inputTypes().front().kind() == PortType::Kind::Memory,
          "tiered memory did not expose one leading manager capability");
  auto tieredRoot = take(test, design.createSpatialCore("tiered-hybrid-memory",
                                                        tiered.inputTypes(),
                                                        tiered.outputTypes()));
  std::vector<SpatialValue> tieredInputs;
  for (std::size_t ordinal = 0; ordinal < tiered.inputTypes().size(); ++ordinal)
    tieredInputs.push_back(take(test, tieredRoot.input(ordinal)));
  auto tieredOutputs = take(test, tieredRoot.addMemory(tieredInputs, tiered));
  if (llvm::Error error = tieredRoot.close(tieredOutputs.values()))
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
      require(test,
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

void publicMemoryLibraryBuildsPortVariants() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const loom::adg::MemoryInterfaceParameters interface{
      loom::adg::MemoryAccessDomainParameters{128, std::nullopt, 4,
                                              singleton(64)},
      64, 128};
  const std::array<loom::adg::LocalMemoryPortVariant, 4> variants = {
      loom::adg::LocalMemoryPortVariant::ElementOnly,
      loom::adg::LocalMemoryPortVariant::VectorOnly,
      loom::adg::LocalMemoryPortVariant::SeparateElementVector,
      loom::adg::LocalMemoryPortVariant::SharedElementVector};
  for (auto [ordinal, variant] : llvm::enumerate(variants)) {
    auto variantMemory = loom::adg::makeVariant32LocalMemory(
        {4096, interface, std::nullopt, false}, variant);
    if (!variantMemory)
      fail(test, "variant " + std::to_string(ordinal) + ": " +
                     llvm::toString(variantMemory.takeError()));
    MemorySpec memory = std::move(*variantMemory);
    auto spatial =
        take(test, design.createSpatialCore(
                       "memory-port-variant-" + std::to_string(ordinal),
                       memory.inputTypes(), memory.outputTypes()));
    std::vector<SpatialValue> inputs;
    for (std::size_t input = 0; input < memory.inputTypes().size(); ++input)
      inputs.push_back(take(test, spatial.input(input)));
    auto outputs = take(test, spatial.addMemory(inputs, memory));
    if (llvm::Error error = spatial.close(outputs.values()))
      fail(test, llvm::toString(std::move(error)));
  }

  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == variants.size(),
          "memory port variants did not finalize independently");
  std::uint64_t elementOnly = 0;
  std::uint64_t vectorOnly = 0;
  std::uint64_t separate = 0;
  std::uint64_t shared = 0;
  for (const auto &root : finalized.roots()) {
    const auto &view = root.view();
    const loom::fabric::FabricMemoryOccurrenceRef memory(uniqueEntity(
        test, view, loom::fabric::FabricEntityKind::FabricMemoryOccurrence));
    const auto ports = view.memoryOperationPorts(memory);
    std::uint64_t elementPorts = 0;
    std::uint64_t vectorPorts = 0;
    std::uint64_t sharedPorts = 0;
    for (const auto port : ports) {
      const auto *alternative = view.memoryCapabilityAlternative({port, 0});
      require(test, alternative && alternative->accessDomain,
              "memory port variant has no access domain");
      bool hasElement = false;
      bool hasVector = false;
      for (const auto &access : alternative->accessDomain->accessClasses()) {
        hasElement |= access.accessForm() ==
                      ::dataflow::semantics::MemoryAccessForm::Element;
        hasVector |= access.accessForm() !=
                     ::dataflow::semantics::MemoryAccessForm::Element;
      }
      elementPorts += hasElement && !hasVector;
      vectorPorts += hasVector && !hasElement;
      sharedPorts += hasElement && hasVector;
    }
    if (ports.size() == 2 && elementPorts == 2)
      ++elementOnly;
    else if (ports.size() == 2 && vectorPorts == 2)
      ++vectorOnly;
    else if (ports.size() == 4 && elementPorts == 2 && vectorPorts == 2)
      ++separate;
    else if (ports.size() == 2 && sharedPorts == 2)
      ++shared;
    else
      fail(test, "memory port variant has an unexpected physical contract");
  }
  require(test,
          elementOnly == 1 && vectorOnly == 1 && separate == 1 && shared == 1,
          "memory port variant inventory is not complete");
}

void publicMemoryRecipeKeepsIndependentEndpointWidths() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);

  auto indexWidths =
      take(test, ::fabric::UnsignedDomain::fromCanonical({{48, 48}, {64, 64}}));
  const loom::adg::MemoryInterfaceParameters interface{
      loom::adg::MemoryAccessDomainParameters{192, 256, 8,
                                              std::move(indexWidths)},
      64, 96};
  LocalMemoryParameters parameters;
  parameters.capacityBytes = 4096;
  parameters.interface = interface;
  MemorySpec memory =
      take(test, loom::adg::makeGeneral64LocalMemory(parameters));
  const std::vector<std::uint32_t> expectedInputWidths = {64,  256, 8, 0, 64,
                                                          256, 192, 8, 0};
  require(test,
          memory.inputTypes().size() == expectedInputWidths.size() &&
              memory.outputTypes().size() == 3,
          "general memory did not expose its independent typed endpoints");
  for (auto [type, width] :
       llvm::zip_equal(memory.inputTypes(), expectedInputWidths))
    require(test, type.kind() == PortType::Kind::Bits && type.width() == width,
            "general memory changed an input endpoint width");
  require(test,
          memory.outputTypes()[0].width() == 192 &&
              memory.outputTypes()[1].width() == 0 &&
              memory.outputTypes()[2].width() == 0,
          "general memory changed an output endpoint width");

  auto spatial =
      take(test,
           design.createSpatialCore("independent-memory-widths",
                                    memory.inputTypes(), memory.outputTypes()));
  std::vector<SpatialValue> inputs;
  for (std::size_t ordinal = 0; ordinal != memory.inputTypes().size();
       ++ordinal)
    inputs.push_back(take(test, spatial.input(ordinal)));
  auto outputs = take(test, spatial.addMemory(inputs, memory));
  if (llvm::Error error = spatial.close(outputs.values()))
    fail(test, llvm::toString(std::move(error)));

  auto finalized = take(test, std::move(design).finalize());
  const auto &view = finalized.roots().front().view();
  const loom::fabric::FabricMemoryOccurrenceRef occurrence(uniqueEntity(
      test, view, loom::fabric::FabricEntityKind::FabricMemoryOccurrence));
  const auto definition = view.memoryEngineTemplateOf(occurrence);
  require(test, definition.has_value(),
          "general memory lost its operation-engine definition");
  const auto *engine = view.memoryEngineTemplate(*definition);
  require(test, engine && engine->operationPorts.size() == 2,
          "general memory lost its load/store operation ports");
  const std::array<::fabric::MemoryInternalConnectionDeclaration, 5>
      expectedConnections{{
          {9, 6},
          {10, 3},
          {10, 8},
          {11, 3},
          {11, 8},
      }};
  require(test,
          engine &&
              llvm::equal(engine->internalConnections, expectedConnections,
                          [](const auto &actual, const auto &expected) {
                            return actual.sourceEndpointOrdinal ==
                                       expected.sourceEndpointOrdinal &&
                                   actual.sinkEndpointOrdinal ==
                                       expected.sinkEndpointOrdinal;
                          }),
          "general memory changed its internal forwarding relation");
  for (const auto &port : engine->operationPorts) {
    bool sawDirect = false;
    bool sawIndexed = false;
    bool sawIndex48Lanes5 = false;
    bool sawIndex64Lanes4 = false;
    bool sawPointer32Lanes8 = false;
    bool sawPointer64Lanes4 = false;
    for (const auto &alternative : port.capabilityAlternatives()) {
      require(test, alternative.accessDomain.has_value(),
              "general memory capability lost its access domain");
      bool hasIndexed = false;
      for (const auto &access : alternative.accessDomain->accessClasses()) {
        hasIndexed |= access.accessForm() ==
                      ::dataflow::semantics::MemoryAccessForm::Indexed;
        if (access.accessForm() !=
                ::dataflow::semantics::MemoryAccessForm::Indexed ||
            !access.elementWidths().contains(8))
          continue;
        if (const auto *widths = access.rootRelativeIndexWidths()) {
          sawIndex48Lanes5 |= widths->contains(48) && !widths->contains(64) &&
                              access.flattenedLaneCounts().contains(5) &&
                              !access.flattenedLaneCounts().contains(6);
          sawIndex64Lanes4 |= widths->contains(64) && !widths->contains(48) &&
                              access.flattenedLaneCounts().contains(4) &&
                              !access.flattenedLaneCounts().contains(5);
          continue;
        }
        if (const auto *formats = access.addressPointerFormats()) {
          const ::fabric::PointerFormat p32{
              0, 32, 32, ::loom::PointerLayoutKind::StableIntegral};
          const ::fabric::PointerFormat p64{
              0, 64, 64, ::loom::PointerLayoutKind::StableIntegral};
          sawPointer32Lanes8 |= formats->contains(p32) &&
                                !formats->contains(p64) &&
                                access.flattenedLaneCounts().contains(8) &&
                                !access.flattenedLaneCounts().contains(9);
          sawPointer64Lanes4 |= formats->contains(p64) &&
                                !formats->contains(p32) &&
                                access.flattenedLaneCounts().contains(4) &&
                                !access.flattenedLaneCounts().contains(5);
        }
      }
      require(test, alternative.admissibleUsePatterns.size() == 1,
              "general memory capability has ambiguous use patterns");
      const std::uint32_t pattern =
          alternative.admissibleUsePatterns.front().ordinal();
      sawDirect |= !hasIndexed && pattern == 0;
      sawIndexed |= hasIndexed && pattern == 1;
    }
    require(test, sawDirect && sawIndexed,
            "general memory did not separate direct and indexed access");
    require(test, sawIndex48Lanes5 && sawIndex64Lanes4,
            "general memory lost index-width/lane-count correlation");
    require(test, sawPointer32Lanes8 && sawPointer64Lanes4,
            "general memory lost pointer-format/lane-count correlation");
    require(test,
            port.resourceContract()
                        .usePattern(::fabric::UsePatternKey(0))
                        .internalTransactionCount == 1 &&
                port.resourceContract()
                        .usePattern(::fabric::UsePatternKey(1))
                        .internalTransactionCount == 8,
            "general memory did not derive its exact transaction capacities");
  }
  const auto *service = view.localMemoryService(occurrence);
  require(test, service && !service->capabilities().empty(),
          "general memory lost its local service contract");
  for (const auto &capability : service->capabilities())
    require(test, capability.serviceBeatWidthBits == 96,
            "general memory changed its independent service beat width");
}

void nonModuleTemplatesMatchDirectAuthoring() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  const PortType bits0 = take(test, PortType::bits(0));
  const PortType bits16 = take(test, PortType::bits(16));
  const PortType bits32 = take(test, PortType::bits(32));
  const PortType tagged0 = take(test, PortType::taggedBits(0, 4));
  const PortType tagged32 = take(test, PortType::taggedBits(32, 4));
  mlir::MLIRContext contractContext(mlir::MLIRContext::Threading::DISABLED);

  {
    DesignBuilder invalidDesign(store);
    auto invalidRoot =
        take(test, invalidDesign.createSpatialCore("invalid-template", {}, {}));
    auto mismatched = invalidRoot.createPeTemplate(
        "pe", {bits16, bits32}, PeSpec::spatial({bits16, bits32}, {bits32}));
    if (mismatched)
      fail(test, "spatial PE template accepted nonuniform port widths");
    expectError(test, mismatched.takeError(), "uniform width");
  }

  const auto makeMemory = [&]() {
    return take(
        test,
        MemorySpec::create(
            {tagged32, tagged0}, {tagged32, tagged0}, {}, {},
            MemoryEngineSpec::temporal(4, {loadPortDeclaration()}),
            take(test, LocalMemoryServiceSpec::create(
                           4096, localMemoryContract(test, contractContext))),
            operationConnectivity(test, localMemoryTarget())));
  };

  const auto build = [&](bool useTemplates) -> loom::ArtifactRootReference {
    DesignBuilder design(store);
    auto root =
        take(test, design.createSpatialCore(
                       "non-module-template-equivalence",
                       {bits32, bits32, bits32, bits32, tagged32, tagged0},
                       {bits32, bits32, tagged32, tagged0}));
    const auto clock =
        take(test,
             root.declareDomainSlot(loom::fabric::FabricClockResetKind::Clock));
    const auto reset =
        take(test,
             root.declareDomainSlot(loom::fabric::FabricClockResetKind::Reset));

    const auto assign = [&](const ModuleDomainMemberHandle &member) {
      if (llvm::Error error = root.assignDomainSlot(member, clock))
        fail(test, llvm::toString(std::move(error)));
      if (llvm::Error error = root.assignDomainSlot(member, reset))
        fail(test, llvm::toString(std::move(error)));
    };
    for (std::size_t ordinal = 0; ordinal != 6; ++ordinal)
      assign(take(test, root.inputDomainMember(ordinal)));
    for (std::size_t ordinal = 0; ordinal != 4; ++ordinal)
      assign(take(test, root.outputDomainMember(ordinal)));

    std::vector<SpatialValue> outputs;
    if (!useTemplates) {
      auto pe = take(
          test,
          root.addPe({take(test, root.input(0)), take(test, root.input(1))},
                     PeSpec::spatial({bits32, bits32}, {bits32})));
      auto fu = take(
          test, pe.addFu({take(test, pe.input(0)), take(test, pe.input(1))},
                         FuSpec{{bits32, bits32}, {bits32}}));
      auto sum = take(
          test, fu.addOperation(
                    {take(test, fu.input(0)), take(test, fu.input(1))},
                    integerCapability(
                        ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                        ::dataflow::OperationSchemaId::ArithAddI, bits32)));
      if (llvm::Error error =
              fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{sum}, {}}))
        fail(test, llvm::toString(std::move(error)));
      if (llvm::Error error = fu.close({take(test, sum.output(0))}))
        fail(test, llvm::toString(std::move(error)));
      if (llvm::Error error = pe.close())
        fail(test, llvm::toString(std::move(error)));
      assign(pe.domainMember());
      assign(take(test, pe.instructionContextMember(0)));
      assign(fu.domainMember());
      assign(sum.domainMember());

      auto sw = take(
          test, root.addSwitch(
                    {take(test, root.input(2)), take(test, root.input(3))},
                    SwitchSpec::spatial({bits32, bits32}, {bits32}, {{0, 1}})));
      assign(sw.domainMember());

      auto memory = take(test, root.addMemory({take(test, root.input(4)),
                                               take(test, root.input(5))},
                                              makeMemory()));
      assign(memory.domainMember());
      assign(take(test, memory.operationPortMember(0)));
      require(test, memory.localServiceMember().has_value(),
              "direct memory lost its local service handle");
      assign(*memory.localServiceMember());

      outputs = {take(test, pe.output(0)), sw[0], memory.values()[0],
                 memory.values()[1]};
    } else {
      auto pe = take(test, root.createPeTemplate(
                               "pe", {bits32, bits32},
                               PeSpec::spatial({bits32, bits32}, {bits32})));
      auto fu = take(
          test, pe.createFuTemplate("fu", FuSpec{{bits32, bits32}, {bits32}}));
      auto sum = take(
          test, fu.addOperation(
                    {take(test, fu.input(0)), take(test, fu.input(1))},
                    integerCapability(
                        ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                        ::dataflow::OperationSchemaId::ArithAddI, bits32)));
      const auto sumOwner = take(test, sum.templateOwner());
      auto ambiguousCapability = fu.addCapabilityTemplateWithHandle(
          FuCapabilityTemplateSpec{{sum}, {}});
      if (ambiguousCapability)
        fail(test, "named FU returned an occurrence-local capability handle");
      expectError(test, ambiguousCapability.takeError(),
                  "do not have a unique finalized occurrence handle");
      if (llvm::Error error =
              fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{sum}, {}}))
        fail(test, llvm::toString(std::move(error)));
      expectError(test, fu.close({take(test, sum.output(0))}),
                  "closed with closeTemplate");
      const auto fuTemplate =
          take(test, fu.closeTemplate({take(test, sum.output(0))}));
      auto fuInstance =
          take(test, pe.instantiate(fuTemplate, {take(test, pe.input(0)),
                                                 take(test, pe.input(1))}));
      expectError(test, pe.close(), "closed with closeTemplate");
      const auto peTemplate = take(test, pe.closeTemplate());
      auto peInstance =
          take(test, root.instantiate(peTemplate, {take(test, root.input(0)),
                                                   take(test, root.input(1))}));
      assign(take(test, root.moduleMember(peInstance.occurrenceOwner())));
      assign(take(
          test, root.moduleMember(take(
                    test, peInstance.project(take(
                              test, peTemplate.instructionContextOwner(0)))))));
      assign(take(
          test, root.moduleMember(take(
                    test, peInstance.project(fuInstance.occurrenceOwner())))));
      assign(take(test, root.moduleMember(take(
                            test, peInstance.project(take(
                                      test, fuInstance.project(sumOwner)))))));

      const auto switchTemplate =
          take(test, root.createSwitchTemplate(
                         "switch", SwitchSpec::spatial({bits32, bits32},
                                                       {bits32}, {{0, 1}})));
      auto sw = take(
          test, root.instantiate(switchTemplate, {take(test, root.input(2)),
                                                  take(test, root.input(3))}));
      assign(take(test, root.moduleMember(sw.occurrenceOwner())));

      const auto memoryTemplate =
          take(test, root.createMemoryTemplate("memory", makeMemory()));
      auto memory = take(
          test, root.instantiate(memoryTemplate, {take(test, root.input(4)),
                                                  take(test, root.input(5))}));
      assign(take(test, root.moduleMember(memory.occurrenceOwner())));
      assign(take(
          test, root.moduleMember(take(
                    test, memory.project(take(
                              test, memoryTemplate.operationPortOwner(0)))))));
      const auto localService = memoryTemplate.localServiceOwner();
      require(test, localService.has_value(),
              "memory template lost its local service handle");
      assign(take(
          test, root.moduleMember(take(test, memory.project(*localService)))));

      outputs = {peInstance[0], sw[0], memory[0], memory[1]};
    }

    if (llvm::Error error = root.close(outputs))
      fail(test, llvm::toString(std::move(error)));
    auto finalized = take(test, std::move(design).finalize());
    require(test, finalized.roots().size() == 1,
            "template equivalence design published an unexpected root count");
    return finalized.roots().front().reference();
  };

  const loom::ArtifactRootReference direct = build(false);
  const loom::ArtifactRootReference instantiated = build(true);
  const loom::ArtifactRootReference independent =
      buildIndependentNonModuleTemplateOracle(test, store);
  require(test, direct == instantiated && direct == independent,
          "non-Module authoring paths changed canonical Fabric identity");
}

void nestedNonModuleTemplatesSurviveModuleComposition() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits32 = take(test, PortType::bits(32));

  auto child = take(
      test, design.createSpatialCore("template-child", {bits32}, {bits32}));
  const auto childClock = take(
      test, child.declareDomainSlot(loom::fabric::FabricClockResetKind::Clock));
  const auto childReset = take(
      test, child.declareDomainSlot(loom::fabric::FabricClockResetKind::Reset));
  const auto assignChild = [&](const ModuleDomainMemberHandle &member) {
    if (llvm::Error error = child.assignDomainSlot(member, childClock))
      fail(test, llvm::toString(std::move(error)));
    if (llvm::Error error = child.assignDomainSlot(member, childReset))
      fail(test, llvm::toString(std::move(error)));
  };
  assignChild(take(test, child.inputDomainMember(0)));
  assignChild(take(test, child.outputDomainMember(0)));

  auto pe =
      take(test, child.createPeTemplate("pe", {bits32},
                                        PeSpec::spatial({bits32}, {bits32})));
  auto fu = take(test, pe.createFuTemplate("fu", FuSpec{{bits32}, {bits32}}));
  auto add = take(
      test,
      fu.addOperation({take(test, fu.input(0)), take(test, fu.input(0))},
                      integerCapability(
                          ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                          ::dataflow::OperationSchemaId::ArithAddI, bits32)));
  const auto nodeOwner = take(test, add.templateOwner());
  if (llvm::Error error =
          fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{add}, {}}))
    fail(test, llvm::toString(std::move(error)));
  const auto fuTemplate =
      take(test, fu.closeTemplate({take(test, add.output(0))}));
  auto fuInstance =
      take(test, pe.instantiate(fuTemplate, {take(test, pe.input(0))}));
  const auto peTemplate = take(test, pe.closeTemplate());
  auto peInstance =
      take(test, child.instantiate(peTemplate, {take(test, child.input(0))}));
  assignChild(take(test, child.moduleMember(peInstance.occurrenceOwner())));
  assignChild(take(
      test, child.moduleMember(take(
                test, peInstance.project(take(
                          test, peTemplate.instructionContextOwner(0)))))));
  assignChild(
      take(test, child.moduleMember(take(
                     test, peInstance.project(fuInstance.occurrenceOwner())))));
  assignChild(
      take(test, child.moduleMember(
                     take(test, peInstance.project(take(
                                    test, fuInstance.project(nodeOwner)))))));
  if (llvm::Error error = child.close(peInstance.values()))
    fail(test, llvm::toString(std::move(error)));

  const auto childClocks =
      take(test, child.domainSlots(loom::fabric::FabricClockResetKind::Clock));
  const auto childResets =
      take(test, child.domainSlots(loom::fabric::FabricClockResetKind::Reset));
  auto parent = take(
      test, design.createSpatialCore("template-parent", {bits32}, {bits32}));
  const auto parentClock =
      take(test,
           parent.declareDomainSlot(loom::fabric::FabricClockResetKind::Clock));
  const auto parentReset =
      take(test,
           parent.declareDomainSlot(loom::fabric::FabricClockResetKind::Reset));
  auto childInstance =
      take(test, parent.instantiate(child, {take(test, parent.input(0))},
                                    {{childClocks.front(), parentClock},
                                     {childResets.front(), parentReset}}));
  for (const auto &member : {take(test, parent.inputDomainMember(0)),
                             take(test, parent.outputDomainMember(0))}) {
    if (llvm::Error error = parent.assignDomainSlot(member, parentClock))
      fail(test, llvm::toString(std::move(error)));
    if (llvm::Error error = parent.assignDomainSlot(member, parentReset))
      fail(test, llvm::toString(std::move(error)));
  }
  if (llvm::Error error = parent.close(childInstance))
    fail(test, llvm::toString(std::move(error)));

  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 2,
          "nested non-Module template composition lost a Module root");
  require(test,
          finalized.roots()[0].reference() == finalized.roots()[1].reference(),
          "Module composition changed the nested template identity");
  for (const auto &root : finalized.roots())
    require(
        test,
        entityCount(root.view(),
                    loom::fabric::FabricEntityKind::FabricPeOccurrence) == 1 &&
            entityCount(root.view(),
                        loom::fabric::FabricEntityKind::FabricFuOccurrence) ==
                1,
        "Module composition changed the nested physical inventory");
}

void runBuilderTests() {
  regularAndIrregularSpatialCoresFinalize();
  foreignHandlesAndIncompleteRootsFailClosed();
  spatialCoreTemplatesInstantiateAndElaborate();
  typedPeFuGraphsFinalize();
  temporalResourceGrantFinalizes();
  fuCapabilityRowsCorrelateRoutes();
  typedMemoryFormsFinalize();
  publicMemoryLibraryBuildsHybridLocalMemories();
  publicMemoryLibraryBuildsPortVariants();
  publicMemoryRecipeKeepsIndependentEndpointWidths();
  nonModuleTemplatesMatchDirectAuthoring();
  nestedNonModuleTemplatesSurviveModuleComposition();
  runBuilderTemplateTests();
}

} // namespace loom::adg::test
