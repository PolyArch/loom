#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/IndexWidth.h"
#include "Config/ResolvedConfig.h"
#include "ConfiguredHardwareProjectionInternal.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Deployment/HardwareConfigurationImage.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/Identity/FabricPeConfiguration.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingClosureProjection.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "Mapping/Tech/TechMappingGenerator.h"
#include "PnR/MappingObjective.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialPnrGenerator.h"
#include "PnR/System/SystemPnrGenerator.h"
#include "PnR/System/SystemPnrSearchDomain.h"
#include "Simulator/CGRAAdmission.h"
#include "Simulator/CGRASimulator.h"
#include "Simulator/SimulationArtifacts.h"

#include "ConfigurationABITestSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "configured hardware projection test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectError(llvm::Expected<T> value, llvm::StringRef expected) {
  if (value)
    fail(("accepted invalid input; expected '" + expected + "'").str());
  const std::string diagnostic = llvm::toString(value.takeError());
  if (!llvm::StringRef(diagnostic).contains(expected))
    fail("unexpected rejection: " + diagnostic);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-configured-hardware-projection", path))
      fail("cannot create ArtifactStore directory: " + error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << "cannot remove test directory: " << error.message()
                   << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  std::string path_;
};

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry.insert<::dataflow::DataflowDialect, ::mapping::MappingDialect,
                  ::fabric::FabricDialect, mlir::arith::ArithDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::vector::VectorDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

std::string byteList(llvm::ArrayRef<std::uint8_t> bytes) {
  std::string result = "[";
  for (auto [ordinal, byte] : llvm::enumerate(bytes)) {
    if (ordinal)
      result += ", ";
    result += std::to_string(static_cast<std::int8_t>(byte));
  }
  return result + "]";
}

std::string identityAttr(const loom::ArtifactIdentity &identity) {
  return "#mapping.artifact_identity<" + byteList(identity.bytes()) + ">";
}

mlir::OwningOpRef<mlir::ModuleOp>
parseSpatial(mlir::MLIRContext &context,
             const loom::CanonicalSemanticBytes &bytes) {
  std::string text = "module {\n";
  text.append(reinterpret_cast<const char *>(bytes.bytes().data()),
              bytes.bytes().size());
  text += "}\n";
  return mlir::parseSourceString<mlir::ModuleOp>(text, &context);
}

::mapping::PhysicalRefinementAssignmentAttr nonemptyRefinementPlaceholder(
    mlir::MLIRContext &context,
    const loom::fabric::FabricFuOccurrenceNodeRef &owner) {
  const loom::fabric::FabricPhysicalRefinementDomainRef domain{
      loom::fabric::FabricRefinementOwnerRef(
          loom::fabric::FabricInventoryOwnerRef::of(owner)),
      0};
  const auto domainBytes = loom::fabric::canonicalFabricBytes(domain);
  std::vector<std::int8_t> signedDomainBytes;
  signedDomainBytes.reserve(domainBytes.size());
  for (std::uint8_t byte : domainBytes)
    signedDomainBytes.push_back(static_cast<std::int8_t>(byte));
  auto domainAttr = ::mapping::FabricPhysicalRefinementDomainRefAttr::get(
      &context, mlir::DenseI8ArrayAttr::get(&context, signedDomainBytes));
  const std::array<std::int8_t, 1> valueBytes = {0};
  auto value = ::mapping::OwnerTypedValueAttr::get(
      &context, mlir::DenseI8ArrayAttr::get(&context, valueBytes));
  return ::mapping::PhysicalRefinementAssignmentAttr::get(&context, domainAttr,
                                                          value);
}

void requireRefinementRejected(
    ::mapping::SpatialOp root,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const loom::mapping::TechMappingView &tech,
    const loom::fabric::FabricArtifactView &fabric,
    llvm::StringRef ownerDescription) {
  llvm::Error error =
      loom::mapping::verifySpatialMappingBase(root, dataflow, tech, fabric);
  if (!error) {
    std::string message = "strict SpatialMapping import accepted a nonempty ";
    message += ownerDescription.str();
    message += " refinement without an owner codec";
    fail(message);
  }
  const std::string diagnostic = llvm::toString(std::move(error));
  if (llvm::StringRef(diagnostic)
          .find(
              "nonempty physical refinement requires its owner value codec") ==
      llvm::StringRef::npos)
    fail("nonempty refinement was rejected outside the owner-codec gate");
}

loom::ResolvedObjectiveCatalogs spatialObjectiveCatalogs() {
  loom::ResolvedObjectiveCatalogs catalogs;
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  catalogs.dimensions = {
      {loom::ResolvedMappingViolationObjectiveSource{
           loom::ResolvedPnrViolationKind::UnroutedObligation},
       loom::ResolvedObjectiveDirection::Minimize,
       loom::resolvedObjectiveInteger(0), loom::resolvedObjectiveInteger(1), 0,
       maximum},
      {loom::ResolvedMappingViolationObjectiveSource{
           loom::ResolvedPnrViolationKind::CapacityOveruse},
       loom::ResolvedObjectiveDirection::Minimize,
       loom::resolvedObjectiveInteger(0), loom::resolvedObjectiveInteger(1), 0,
       maximum},
      {loom::ResolvedMappingMeasureObjectiveSource{static_cast<std::uint32_t>(
           loom::pnr::MappingMeasureKind::TotalSelectedTraversalClaim)},
       loom::ResolvedObjectiveDirection::Minimize,
       loom::resolvedObjectiveInteger(0), loom::resolvedObjectiveInteger(1), 0,
       maximum},
  };
  catalogs.weightedLevels = {{{{0, 1}, {1, 1}, {2, 1}}}};
  catalogs.totalOrderings = {{{0}}};
  return catalogs;
}

dataflow::CanonicalDataflowArtifact buildDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @shuffle(
      %start: none, %lhs: vector<2xi8>, %rhs: vector<1xi8>)
      -> vector<3xi8>
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %result = vector.shuffle %lhs, %rhs [1, 2, 0]
        : vector<2xi8>, vector<1xi8>
    %retired:2 = dataflow.sync %start, %result
        : (none, vector<3xi8>) -> (none, vector<3xi8>)
    dataflow.graph.return values(%retired#1 : vector<3xi8>) streams() memories()
        complete(%retired#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %lhs: vector<2xi8>, %rhs: vector<1xi8>) ctrl (%ctrl: none) {
    %result, %done = dataflow.graph.launch @shuffle deps(%ctrl)
        values(%lhs, %rhs) stream_inputs() memories() stream_outputs()
        : (none, vector<2xi8>, vector<1xi8>) -> (vector<3xi8>, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host(%lhs: vector<2xi8>, %rhs: vector<1xi8>) {
    %thread = dataflow.thread.launch @worker(%lhs, %rhs)
        : (vector<2xi8>, vector<1xi8>) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalActorSchemaProjection
alternateShuffleProjection(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func private @alternate(
      %lhs: vector<2xi8>, %rhs: vector<1xi8>) -> vector<3xi8> {
    %result = vector.shuffle %lhs, %rhs [0, 2, 1]
        : vector<2xi8>, vector<1xi8>
    return %result : vector<3xi8>
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse alternate vector-shuffle fixture");
  mlir::func::FuncOp function = *module->getOps<mlir::func::FuncOp>().begin();
  auto shuffles = function.getOps<mlir::vector::ShuffleOp>();
  if (shuffles.empty())
    fail("alternate vector-shuffle fixture has no shuffle actor");
  mlir::vector::ShuffleOp shuffle = *shuffles.begin();
  return take(dataflow::projectRegisteredActorSchemaProjection(shuffle));
}

loom::fabric::FinalizedFabricRoot
buildFabric(const loom::ArtifactStore &store) {
  using namespace loom::adg;

  const PortType bits128 = take(PortType::bits(128));
  const std::vector<PortType> inputs(3, bits128);
  const std::vector<PortType> outputs(2, bits128);
  DesignBuilder builder(store);
  auto spatial = take(
      builder.createSpatialCore("configured-vector-shuffle", inputs, outputs));
  std::vector<SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal < inputs.size(); ++ordinal)
    spatialInputs.push_back(take(spatial.input(ordinal)));
  auto pe =
      take(spatial.addPe(spatialInputs, PeSpec::spatial(inputs, outputs)));
  std::vector<PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal < inputs.size(); ++ordinal)
    peInputs.push_back(take(pe.input(ordinal)));
  auto fu = take(pe.addFu(peInputs, FuSpec{inputs, outputs}));
  std::vector<FuValue> fuInputs;
  for (std::size_t ordinal = 0; ordinal < inputs.size(); ++ordinal)
    fuInputs.push_back(take(fu.input(ordinal)));
  const ::fabric::IntegerWidthSet integerWidths =
      ::fabric::IntegerWidthSet::get({::fabric::IntegerWidth::I8});
  const ::fabric::FloatFormatSet noFloatFormats;
  auto shuffle = take(fu.addOperation(
      {fuInputs[0], fuInputs[1]},
      OperationCapabilitySpec{
          ::fabric::ImplementationFamilyId::FixedVectorShuffle,
          ::fabric::FixedVectorShuffleParams{integerWidths, noFloatFormats, 128,
                                             128, 128, 32, 16},
          {::dataflow::OperationSchemaId::VectorShuffle},
          {bits128},
          ::fabric::oneCycleElasticOperationResourceContract()}));
  auto shuffled = take(shuffle.output(0));
  const std::vector<PortType> tokenTypes(2, bits128);
  auto sync = take(fu.addOperation(
      {fuInputs[2], shuffled},
      OperationCapabilitySpec{
          ::fabric::ImplementationFamilyId::TokenSync,
          ::fabric::RoutedTokenParams{128, 4},
          {::dataflow::OperationSchemaId::DataflowSync},
          tokenTypes,
          ::fabric::oneCycleElasticOperationResourceContract()}));
  requireSuccess(
      fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{shuffle, sync}, {}}));
  requireSuccess(fu.close({take(sync.output(0)), take(sync.output(1))}));
  requireSuccess(pe.close());
  std::vector<SpatialValue> spatialOutputs;
  for (std::size_t ordinal = 0; ordinal < outputs.size(); ++ordinal)
    spatialOutputs.push_back(take(pe.output(ordinal)));
  requireSuccess(spatial.close(spatialOutputs));
  auto finalized = take(std::move(builder).finalize());
  if (finalized.roots().size() != 1)
    fail("Fabric fixture did not publish exactly one root");
  return std::move(finalized.roots().front());
}

::fabric::ResourceContract
sharedTransportResourceContract(std::uint32_t residentRouteCapacity) {
  ::fabric::ResourceContractDeclaration declaration;
  declaration.states = {{::fabric::StateKey(0),
                         {{::fabric::CapacityDimensionKey(0),
                           ::fabric::CapacityUnits(residentRouteCapacity),
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
  return take(::fabric::ResourceContract::create(std::move(declaration)));
}

loom::fabric::InstructionCoreMicroarchitecturalRealization
inOrderMicroarchitecture() {
  loom::fabric::InstructionCoreCommonDeclaration common{
      1,
      {{loom::fabric::InstructionOperationClass::IntegerAlu, 1, 1, 1}},
      ::fabric::oneCycleElasticOperationResourceContract()};
  loom::fabric::InOrderMicroarchitectureDeclaration pipeline{1, 1, 1, 1,
                                                             1, 1, 2, 1};
  return take(
      loom::fabric::InstructionCoreMicroarchitecturalRealization::createInOrder(
          std::move(common), pipeline));
}

loom::fabric::FinalizedFabricRoot
buildSystem(const loom::fabric::FinalizedFabricRoot &module,
            llvm::ArrayRef<mlir::Type> messagePayloads,
            const loom::ArtifactStore &store) {
  using namespace loom::adg;

  DesignBuilder design(store);
  auto system = take(design.createSystem("configured-vector-system"));
  const auto imported = take(system.importSpatialCore(module));
  const auto architecture =
      take(loom::adg::getBuiltinInstructionCoreArchitecture());
  const auto microarchitecture = inOrderMicroarchitecture();
  const auto host = take(system.addHostCore(architecture, microarchitecture));
  const auto core =
      take(system.addAccCore(architecture, microarchitecture, imported));

  auto clock = take(system.createHardwareDomain());
  const auto rate = take(system.createServiceRate(
      clock, 1, 1, 1,
      loom::fabric::ServiceProgress(
          std::in_place_type<::fabric::FairEventual>)));
  const auto messageDomain = take(
      loom::fabric::MessageTransferCapabilityDomain::create(messagePayloads));
  const auto initiateCapability =
      take(loom::fabric::CanonicalServiceCapabilityRecord::create(
          dataflow::semantics::ServiceKind::MessageTransfer,
          loom::fabric::CanonicalServiceEndpointRole::Initiate, messageDomain,
          rate));
  const auto serveCapability =
      take(loom::fabric::CanonicalServiceCapabilityRecord::create(
          dataflow::semantics::ServiceKind::MessageTransfer,
          loom::fabric::CanonicalServiceEndpointRole::Serve, messageDomain,
          rate));
  const auto initiateSet =
      take(loom::fabric::CanonicalServiceCapabilitySet::create(
          {initiateCapability}));
  const auto serveSet = take(
      loom::fabric::CanonicalServiceCapabilitySet::create({serveCapability}));
  const auto carrier = take(PortType::bits(128));
  const auto hostSource =
      take(system.addServiceEndpoint(host, initiateSet, carrier));
  const auto hostSink =
      take(system.addServiceEndpoint(host, serveSet, carrier));
  const auto coreSource =
      take(system.addServiceEndpoint(core, initiateSet, carrier));
  const auto coreSink =
      take(system.addServiceEndpoint(core, serveSet, carrier));
  const std::array sources{hostSource, coreSource};
  const std::array sinks{hostSink, coreSink};

  const auto transportContract = sharedTransportResourceContract(16);
  const std::array<std::vector<std::uint32_t>, 3> patterns = {
      std::vector<std::uint32_t>{0}, std::vector<std::uint32_t>{1},
      std::vector<std::uint32_t>{0, 1}};
  std::vector<SystemTransportResource> routers;
  routers.reserve(2);
  std::vector<HardwareDomainMember> clockMembers = {
      host.domainMember(),
      core.instructionCoreDomainMember(),
      core.spatialCoreDomainMember(),
      hostSource.domainMember(),
      hostSink.domainMember(),
      coreSource.domainMember(),
      coreSink.domainMember()};
  for (std::size_t ordinal = 0; ordinal != sources.size(); ++ordinal) {
    routers.push_back(take(system.addTransportResource(
        {{carrier, carrier}, {carrier, carrier}, transportContract})));
    clockMembers.push_back(routers.back().domainMember());
    for (std::size_t input = 0; input != 2; ++input)
      for (const auto &outputs : patterns) {
        const auto pattern =
            take(system.addTransferPattern(routers.back(), input, outputs, 0));
        clockMembers.push_back(pattern.domainMember());
      }
    requireSuccess(system.connect(take(sources[ordinal].transport()),
                                  take(routers[ordinal].input(0))));
    requireSuccess(system.connect(take(routers[ordinal].output(0)),
                                  take(sinks[ordinal].transport())));
  }
  for (std::size_t ordinal = 0; ordinal != routers.size(); ++ordinal)
    requireSuccess(
        system.connect(take(routers[ordinal].output(1)),
                       take(routers[(ordinal + 1) % routers.size()].input(1))));

  requireSuccess(clock.close(
      clockMembers,
      take(loom::fabric::ClockDomainContractRecord::create(1'000, 0))));
  requireSuccess(system.close());
  auto finalized = take(std::move(design).finalize());
  if (finalized.roots().size() != 1)
    fail("System fixture did not publish exactly one root");
  return take(loom::fabric::importEntireFabricRoot(
      finalized.roots().front().reference(), store));
}

loom::mapping::FinalizedSpatialMappingConstraintSet
buildConstraints(mlir::MLIRContext &context,
                 const dataflow::CanonicalDataflowProgramView &dataflow,
                 const loom::mapping::TechMappingView &tech,
                 const loom::fabric::FabricArtifactView &fabric,
                 const loom::ArtifactStore &store) {
  const std::string text = "module {\n  mapping.constraints.spatial dataflow(" +
                           identityAttr(dataflow.identity()) +
                           ") tech_mapping(" + identityAttr(tech.identity()) +
                           ") fabric(" + identityAttr(fabric.identity()) +
                           ") {\n  }\n}\n";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  if (!module)
    fail("cannot parse empty Spatial constraint set");
  auto roots = module->getOps<::mapping::ConstraintsSpatialOp>();
  return take(loom::mapping::finalizeSpatialMappingConstraintSet(
      *roots.begin(), dataflow, tech, fabric, store));
}

void exactVectorMappingDerivesConfigurationAndExecutes() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();

  auto dataflowArtifact = buildDataflow(context);
  const auto dataflowReference =
      take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  const auto fabric = buildFabric(store);

  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.objectiveCatalogs = spatialObjectiveCatalogs();
  resolved.dse.spatialPnr.temporaryViolations.admitted = {
      loom::ResolvedPnrViolationKind::UnroutedObligation,
      loom::ResolvedPnrViolationKind::CapacityOveruse};
  resolved.dse.spatialPnr.objectiveSelection = {0, 0};
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  const auto techConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  const std::array<dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};
  auto generatedTech = loom::mapping::generateTechMappings(
      {dataflow, covers, fabric.view(), techConfig, store});
  const auto *techCandidates =
      std::get_if<loom::mapping::GeneratedTechMappings>(&generatedTech);
  if (!techCandidates || techCandidates->candidates.size() != 1)
    fail("TechMapping fixture did not produce one candidate");
  const auto tech = take(loom::mapping::importTechMapping(
      techCandidates->candidates.front(), store));
  const auto constraints =
      buildConstraints(context, dataflow, tech.view(), fabric.view(), store);

  auto &search = resolved.dse.spatialPnr.search;
  search.initializer.seedAttemptCount = 1;
  search.actionProposal = {0, 1, 0};
  search.annealing.calibrationProposalCount = 1;
  search.annealing.fallbackTemperature = 1;
  search.annealing.minimumTemperature = 1;
  search.annealing.coolingRatio = {1, 2};
  search.annealing.proposalsPerLevelBase = 1;
  search.annealing.proposalsPerMovableDecision = 0;
  search.exactRepair = {loom::ResolvedPnrExactRepairKind::Disabled, 0, 0};
  const auto pnrConfig =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(resolved));
  const auto physicalTiming =
      take(loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
          fabric.view()));
  auto generatedSpatial = loom::pnr::generateSpatialMappings(
      {dataflow, tech.view(), fabric.view(), physicalTiming, pnrConfig,
       constraints.view(), store});
  const auto *spatialCandidates =
      std::get_if<loom::pnr::GeneratedSpatialMappings>(&generatedSpatial);
  if (!spatialCandidates || spatialCandidates->candidates.size() != 1) {
    const std::string diagnostic = std::visit(
        [](const auto &outcome) {
          using Outcome = std::decay_t<decltype(outcome)>;
          if constexpr (std::is_same_v<Outcome,
                                       loom::pnr::GeneratedSpatialMappings>)
            return std::string("generated an unexpected candidate count");
          else if constexpr (std::is_same_v<
                                 Outcome,
                                 loom::pnr::InterruptedSpatialPnrGeneration>)
            return (llvm::Twine("interrupted at ") +
                    loom::pnr::spatialPnrInterruptionStageSpelling(
                        outcome.snapshot.stage))
                .str();
          else
            return outcome.diagnostic;
        },
        generatedSpatial);
    fail("SpatialMapping fixture did not produce one candidate: " + diagnostic);
  }
  const auto spatial = take(loom::mapping::importSpatialMapping(
      spatialCandidates->candidates.front(), store));

  const auto fields = spatial.view().configuredHardware().fields();
  const auto projectedComputeBindings = spatial.view().computeBindings();
  if (projectedComputeBindings.size() != 1)
    fail("configured projection fixture has the wrong binding count");
  const auto selectedPe =
      fabric.view().parentPeOf(projectedComputeBindings.front().occurrence);
  if (!selectedPe)
    fail("configured projection fixture FU has no parent PE");
  const auto peSchema =
      take(fabric.view().spatialPeConfigurationSchema(*selectedPe));
  std::size_t selectedPeFieldCount = 0;
  for (const auto &field : peSchema.fields())
    if (!field.port ||
        field.port->fu == projectedComputeBindings.front().occurrence)
      ++selectedPeFieldCount;
  if (fields.size() != selectedPeFieldCount + 3)
    fail("complete Mapping did not derive the PE, FU topology, and operation "
         "fields");
  std::size_t peFieldCount = 0;
  std::size_t topologyFieldCount = 0;
  std::size_t operationFieldCount = 0;
  std::size_t peRouteCount = 0;
  for (const auto &field : fields) {
    const auto &owner = field.slot.field.owner.catalog();
    if (owner.kind() == loom::fabric::FabricInventoryOwnerKind::PeOccurrence) {
      ++peFieldCount;
      const auto descriptor =
          llvm::find_if(peSchema.fields(), [&](const auto &candidate) {
            return candidate.reference == field.slot.field;
          });
      if (descriptor == peSchema.fields().end())
        fail("configured PE field is absent from its Fabric schema");
      const auto value =
          take(peSchema.decode(field.slot.field, field.value.bytes()));
      if (descriptor->kind ==
          loom::fabric::FabricPeConfigurationFieldKind::Activation) {
        const auto *active = std::get_if<loom::fabric::FabricPeActive>(&value);
        if (!active ||
            active->fu != projectedComputeBindings.front().occurrence)
          fail("configured PE activation selected the wrong FU");
      } else if (std::holds_alternative<loom::fabric::FabricPeRoute>(value)) {
        ++peRouteCount;
      } else if (!std::holds_alternative<loom::fabric::FabricPeDisconnected>(
                     value)) {
        fail("configured PE selector has an unexpected semantic value");
      }
    } else if (owner.kind() ==
               loom::fabric::FabricInventoryOwnerKind::FuOccurrence) {
      ++topologyFieldCount;
    } else if (owner.kind() ==
               loom::fabric::FabricInventoryOwnerKind::FuOccurrenceNode) {
      ++operationFieldCount;
    } else {
      fail("configured compute field has an unexpected owner");
    }
    auto relation =
        take(fabric.view().semanticFieldRelation(field.slot.field, context));
    requireSuccess(relation.validateSemanticValue(field.value.bytes()));
  }
  if (peFieldCount != selectedPeFieldCount || peRouteCount == 0 ||
      topologyFieldCount != 1 || operationFieldCount != 2)
    fail("configured compute fields have the wrong semantic owners");

  const std::array<mlir::Type, 4> messagePayloads = {
      mlir::NoneType::get(&context),
      mlir::VectorType::get({2}, mlir::IntegerType::get(&context, 8)),
      mlir::VectorType::get({1}, mlir::IntegerType::get(&context, 8)),
      mlir::VectorType::get({3}, mlir::IntegerType::get(&context, 8))};
  const auto system = buildSystem(fabric, messagePayloads, store);
  const auto occurrence = loom::fabric::SpatialCoreOccurrenceRef{
      system.view().accCoreOccurrences().front()};
  std::vector<loom::hardware::test::ConfigurationFieldEncodingOverride>
      directOverrides;
  for (const auto &selected : fields) {
    auto relation =
        take(fabric.view().semanticFieldRelation(selected.slot.field, context));
    if (relation.kind() !=
        loom::fabric::FabricSemanticFieldRelationKind::Direct)
      continue;
    if (!relation.directEncodedBitCount())
      fail("direct Fabric relation has no encoded width");
    const auto physicalSlot =
        take(loom::fabric::FabricPhysicalConfigurationSlotRef::create(
            loom::fabric::SpatialCoreInternalConfigurationSlotRef{
                occurrence, selected.slot}));
    directOverrides.push_back(
        {loom::fabric::configurationField(physicalSlot),
         loom::hardware::DirectBitsEncoding{*relation.directEncodedBitCount()},
         std::vector<std::uint8_t>(selected.value.bytes().begin(),
                                   selected.value.bytes().end())});
  }
  auto abiDraft = take(loom::hardware::test::makeCompleteConfigurationABIDraft(
      system, directOverrides));
  const auto abi = take(
      loom::hardware::finalizeConfigurationABI(std::move(abiDraft), store));
  if (abi.abi().programmingUnits().size() != 1)
    fail("configuration image fixture did not derive one programming unit");
  const auto image = take(loom::deployment::finalizeHardwareConfigurationImage(
      {abi.reference(),
       abi.abi().programmingUnits().front().id,
       {loom::deployment::ConfigurationImageSourceKind::SpatialMapping,
        spatial.reference()}},
      store));
  const auto reimported =
      take(loom::deployment::importHardwareConfigurationImage(image.reference(),
                                                              store));
  if (reimported.reference() != image.reference() ||
      reimported.image().payloadBitCount() !=
          abi.abi().programmingUnits().front().payloadBitCount ||
      !reimported.image().payload().equals(image.image().payload()))
    fail("hardware configuration image did not round-trip exactly");

  resolved.dse.systemPnr.temporaryViolations.admitted = {
      loom::ResolvedPnrViolationKind::UnroutedObligation,
      loom::ResolvedPnrViolationKind::CapacityOveruse};
  resolved.dse.systemPnr.objectiveSelection = {0, 0};
  auto &systemSearch = resolved.dse.systemPnr.search;
  systemSearch.initializer.seedAttemptCount = 1;
  systemSearch.routing.negotiationIterationLimit = 8;
  systemSearch.actionProposal = {0, 1, 0};
  systemSearch.annealing.calibrationProposalCount = 1;
  systemSearch.annealing.fallbackTemperature = 1;
  systemSearch.annealing.minimumTemperature = 1;
  systemSearch.annealing.coolingRatio = {1, 2};
  systemSearch.annealing.proposalsPerLevelBase = 1;
  systemSearch.annealing.proposalsPerMovableDecision = 0;
  systemSearch.exactRepair = {loom::ResolvedPnrExactRepairKind::Disabled, 0, 0};
  const auto systemView = take(loom::fabric::requireSystemRoot(system.view()));
  std::vector<dataflow::RootThreadLaunchRef> rootThreads;
  for (const auto &root : dataflow.rootThreadLaunches())
    rootThreads.push_back(root.ref);
  const auto systemConstraints =
      take(loom::mapping::finalizeEmptySystemMappingConstraintSet(
          dataflow, systemView, rootThreads, store));
  const auto partition =
      take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
          dataflow, systemConstraints.view().rootThreadLaunches()));
  const auto systemConfig =
      take(loom::pnr::projectResolvedSystemPnrConfigView(resolved));
  const auto searchDomain = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflow, systemView, systemConfig, systemConstraints, partition,
      loom::pnr::SystemHierarchicalGraphSearchInput{{spatial.reference()}},
      store));
  const auto systemPhysicalTiming = take(
      loom::fabric::projectNormalizedSystemPhysicalTimingProfiles(systemView));
  auto generatedSystem = loom::pnr::generateSystemMappings(
      {dataflow, systemView, systemPhysicalTiming, searchDomain, systemConfig,
       systemConstraints, store});
  const auto *systemCandidates =
      std::get_if<loom::pnr::GeneratedSystemMappings>(&generatedSystem);
  if (!systemCandidates || systemCandidates->candidates.size() != 1) {
    const std::string diagnostic = std::visit(
        [](const auto &outcome) {
          using Outcome = std::decay_t<decltype(outcome)>;
          if constexpr (std::is_same_v<Outcome,
                                       loom::pnr::GeneratedSystemMappings>)
            return std::string("generated an unexpected candidate count");
          else if constexpr (std::is_same_v<
                                 Outcome,
                                 loom::pnr::InterruptedSystemPnrGeneration>)
            return (llvm::Twine("interrupted at ") +
                    loom::pnr::systemPnrInterruptionStageSpelling(
                        outcome.snapshot.stage))
                .str();
          else
            return outcome.diagnostic;
        },
        generatedSystem);
    fail("SystemMapping fixture did not produce one candidate: " + diagnostic);
  }
  const auto systemMapping = take(loom::mapping::importSystemMapping(
      systemCandidates->candidates.front(), store));
  const auto closure = take(loom::mapping::projectSystemMappingClosure(
      dataflow, systemView, systemMapping.view(), store));
  const auto replayedClosure = take(loom::mapping::projectSystemMappingClosure(
      dataflow, systemView, systemMapping.view(), store));
  if (closure.capacityCells.empty() || closure.resourceActivations.empty() ||
      closure.capacityCells.size() != replayedClosure.capacityCells.size() ||
      closure.resourceActivations.size() !=
          replayedClosure.resourceActivations.size())
    fail("SystemMapping closure projection is incomplete or unstable");
  bool foundDirectOwner = false;
  bool foundSpatialOwner = false;
  for (const auto &[ordinal, cell] : llvm::enumerate(closure.capacityCells)) {
    if (cell.baselineOccupancy > cell.capacity)
      fail("closure projection published an over-capacity baseline");
    foundDirectOwner |=
        cell.physicalOwner.kind() ==
        loom::fabric::FabricPhysicalOccurrenceOwnerKind::DirectSystemOwner;
    foundSpatialOwner |=
        cell.physicalOwner.kind() ==
        loom::fabric::FabricPhysicalOccurrenceOwnerKind::SpatialCoreInternal;
    const auto &replayed = replayedClosure.capacityCells[ordinal];
    if (cell.physicalOwner != replayed.physicalOwner ||
        cell.state != replayed.state || cell.dimension != replayed.dimension ||
        cell.capacity != replayed.capacity ||
        cell.baselineOccupancy != replayed.baselineOccupancy)
      fail("closure projection changed canonical capacity-cell order");
  }
  if (!foundDirectOwner || !foundSpatialOwner)
    fail("closure projection lost a physical owner namespace");
  for (const auto &activation : closure.resourceActivations) {
    if (activation.relationDomain.empty() ||
        activation.triggerAlternatives.empty())
      fail("closure projection published an incomplete activation member");
    for (const auto &claim : activation.capacityClaims)
      if (claim.capacityCellOrdinal >= closure.capacityCells.size() ||
          claim.amount == 0)
        fail("closure projection published a foreign capacity claim");
  }
  const auto systemImage =
      take(loom::deployment::finalizeHardwareConfigurationImage(
          {abi.reference(),
           abi.abi().programmingUnits().front().id,
           {loom::deployment::ConfigurationImageSourceKind::SystemMapping,
            systemMapping.reference()}},
          store));
  if (systemImage.reference() == image.reference() ||
      !systemImage.image().payload().equals(image.image().payload()))
    fail("SystemMapping source did not preserve the exact physical payload");

  expectError(
      loom::deployment::finalizeHardwareConfigurationImage(
          {abi.reference(),
           abi.abi().programmingUnits().front().id,
           {loom::deployment::ConfigurationImageSourceKind::SystemMapping,
            spatial.reference()}},
          store),
      "SystemMapping");

  const auto decoded = take(abi.abi().decode(image.image().programmingUnitId(),
                                             image.image().payload()));
  const auto physical = take(loom::mapping::qualifyConfiguredHardwareProjection(
      spatial, abi.abi().fabricSystem(), occurrence));
  expectError(loom::mapping::qualifyConfiguredHardwareProjection(
                  spatial, abi.abi().fabricSystem(),
                  loom::fabric::SpatialCoreOccurrenceRef{
                      loom::fabric::AccCoreOccurrenceRef(999)}),
              "no imported Module");
  for (const auto &selected : physical.fields()) {
    const auto found = llvm::find_if(decoded, [&](const auto &value) {
      return value.slot == selected.slot;
    });
    if (found == decoded.end() || !llvm::ArrayRef<std::uint8_t>(found->value)
                                       .equals(selected.value.bytes()))
      fail("configuration image changed a Mapping-owned semantic field");
  }

  std::vector<std::uint8_t> corrupted(image.canonicalBytes().bytes().begin(),
                                      image.canonicalBytes().bytes().end());
  if (corrupted.empty())
    fail("configuration image canonical bytes are empty");
  corrupted.back() ^= 1U;
  const auto corruptedIdentity =
      take(store.put(loom::deployment::hardwareConfigurationImageSchema,
                     loom::CanonicalSemanticBytes(std::move(corrupted))));
  expectError(
      loom::deployment::importHardwareConfigurationImage(
          {loom::deployment::hardwareConfigurationImageSchema.identity.str(),
           loom::deployment::hardwareConfigurationImageSchema.version,
           corruptedIdentity},
          store),
      "payload");

  std::vector<std::uint8_t> noncanonical(image.canonicalBytes().bytes().begin(),
                                         image.canonicalBytes().bytes().end());
  if (noncanonical.size() < 4)
    fail("configuration image canonical frame is truncated");
  std::uint32_t headerSize =
      (static_cast<std::uint32_t>(noncanonical[0]) << 24) |
      (static_cast<std::uint32_t>(noncanonical[1]) << 16) |
      (static_cast<std::uint32_t>(noncanonical[2]) << 8) |
      static_cast<std::uint32_t>(noncanonical[3]);
  if (headerSize == std::numeric_limits<std::uint32_t>::max())
    fail("configuration image canonical header is too large");
  ++headerSize;
  noncanonical[0] = static_cast<std::uint8_t>(headerSize >> 24);
  noncanonical[1] = static_cast<std::uint8_t>(headerSize >> 16);
  noncanonical[2] = static_cast<std::uint8_t>(headerSize >> 8);
  noncanonical[3] = static_cast<std::uint8_t>(headerSize);
  noncanonical.insert(noncanonical.begin() + 4, ' ');
  const auto noncanonicalIdentity =
      take(store.put(loom::deployment::hardwareConfigurationImageSchema,
                     loom::CanonicalSemanticBytes(std::move(noncanonical))));
  expectError(
      loom::deployment::importHardwareConfigurationImage(
          {loom::deployment::hardwareConfigurationImageSchema.identity.str(),
           loom::deployment::hardwareConfigurationImageSchema.version,
           noncanonicalIdentity},
          store),
      "not canonical");

  const auto realizations = tech.view().computeRealizations();
  const loom::mapping::TechComputeActorView *actorBinding = nullptr;
  mlir::Operation *shuffleActor = nullptr;
  std::uint64_t shuffleRealization = 0;
  for (const auto &realization : realizations) {
    for (const auto &candidate : realization.actors) {
      const auto actor = take(dataflow.resolve(candidate.actor));
      auto schema = dataflow::projectRegisteredActorSchemaProjection(actor.op);
      if (!schema)
        fail(llvm::toString(schema.takeError()));
      if (schema->schema != ::dataflow::OperationSchemaId::VectorShuffle)
        continue;
      if (actorBinding)
        fail("configured projection fixture has multiple shuffle actors");
      actorBinding = &candidate;
      shuffleActor = actor.op;
      shuffleRealization = realization.entityId;
    }
  }
  if (!actorBinding)
    fail("configured projection fixture has no shuffle actor");
  const loom::mapping::SpatialComputeBindingView *binding = nullptr;
  for (const auto &candidate : projectedComputeBindings) {
    if (candidate.realization == shuffleRealization) {
      if (binding)
        fail("configured projection fixture has duplicate spatial bindings");
      binding = &candidate;
    }
  }
  if (!binding)
    fail("configured shuffle realization has no spatial binding");
  auto occurrenceOperation = loom::fabric::deriveFabricFuOccurrenceNode(
      fabric.view(), actorBinding->fabricOperation, binding->occurrence);
  if (!occurrenceOperation)
    fail(llvm::toString(occurrenceOperation.takeError()));
  const auto *capability =
      fabric.view().resolvedFabricOpCapability(*occurrenceOperation);
  if (!capability || capability->configurationFieldSchema.size() != 1)
    fail("configured operation has no exact Fabric field schema");
  auto actorProjection =
      take(dataflow::projectRegisteredActorSchemaProjection(shuffleActor));
  const auto indexBitWidth = take(loom::getIndexBitWidth(shuffleActor));
  const auto expectedValue = take(capability->encodeSemanticConfiguration(
      capability->configurationFieldSchema.front(), actorProjection,
      indexBitWidth, actorBinding->operandPorts, actorBinding->resultPorts,
      nullptr));
  const loom::fabric::FabricSemanticConfigFieldRef expectedField{
      loom::fabric::FabricConfigurationOwnerRef(
          loom::fabric::FabricInventoryOwnerRef::of(*occurrenceOperation)),
      capability->configurationFieldSchema.front().ordinal};
  const loom::mapping::ConfiguredHardwareFieldValueView *projectedField =
      nullptr;
  for (const auto &field : fields)
    if (std::holds_alternative<
            loom::fabric::FabricStaticConfigurationResidency>(
            field.slot.residency) &&
        field.slot.field == expectedField) {
      if (projectedField)
        fail("configured projection duplicated the shuffle field");
      projectedField = &field;
    }
  if (!projectedField ||
      !projectedField->value.bytes().equals(expectedValue.bytes()))
    fail("configured projection changed the exact physical slot or value");

  const auto refinement =
      nonemptyRefinementPlaceholder(context, *occurrenceOperation);
  const auto refinementArray = mlir::ArrayAttr::get(&context, {refinement});
  auto computeRefinement = parseSpatial(context, spatial.canonicalBytes());
  if (!computeRefinement)
    fail("cannot parse compute-refinement rejection fixture");
  auto computeRoot = *computeRefinement->getOps<::mapping::SpatialOp>().begin();
  auto computeBindings =
      computeRoot.getBody().front().getOps<::mapping::ComputeBindingOp>();
  if (computeBindings.empty())
    fail("configured projection fixture has no compute binding operation");
  (*computeBindings.begin()).setRefinementsAttr(refinementArray);
  requireRefinementRejected(computeRoot, dataflow, tech.view(), fabric.view(),
                            "compute");

  auto routeRefinement = parseSpatial(context, spatial.canonicalBytes());
  if (!routeRefinement)
    fail("cannot parse route-refinement rejection fixture");
  auto routeRoot = *routeRefinement->getOps<::mapping::SpatialOp>().begin();
  auto routeTrees =
      routeRoot.getBody().front().getOps<::mapping::RouteTreeOp>();
  if (routeTrees.empty())
    fail("configured projection fixture has no route tree operation");
  auto routeNodes =
      (*routeTrees.begin()).getBody().front().getOps<::mapping::RouteNodeOp>();
  if (routeNodes.empty())
    fail("configured projection fixture has no route node operation");
  (*routeNodes.begin()).setRefinementsAttr(refinementArray);
  requireRefinementRejected(routeRoot, dataflow, tech.view(), fabric.view(),
                            "route");

  const auto alternateProjection = alternateShuffleProjection(context);
  const auto alternateValue = take(capability->encodeSemanticConfiguration(
      capability->configurationFieldSchema.front(), alternateProjection,
      indexBitWidth, actorBinding->operandPorts, actorBinding->resultPorts,
      nullptr));
  if (alternateValue.bytes().equals(expectedValue.bytes()))
    fail("different shuffle masks collapsed to one physical configuration");

  std::vector<loom::mapping::ConfiguredHardwareFieldValueView> equalRows = {
      *projectedField, *projectedField};
  const auto deduplicated =
      take(loom::mapping::detail::canonicalizeConfiguredHardwareProjection(
          std::move(equalRows)));
  if (deduplicated.fields().size() != 1)
    fail("equal requirements for one configuration slot did not deduplicate");

  std::vector<loom::mapping::ConfiguredHardwareFieldValueView> conflictingRows =
      {*projectedField, {projectedField->slot, alternateValue}};
  auto conflict =
      loom::mapping::detail::canonicalizeConfiguredHardwareProjection(
          std::move(conflictingRows));
  if (conflict)
    fail("conflicting requirements for one configuration slot were accepted");
  const std::string conflictDiagnostic = llvm::toString(conflict.takeError());
  if (llvm::StringRef(conflictDiagnostic).find("conflicting semantic values") ==
      llvm::StringRef::npos)
    fail("configuration conflict returned the wrong diagnostic");

  auto secondContext = *projectedField;
  if (binding->context.ordinal ==
      std::numeric_limits<loom::fabric::FabricOrdinal>::max())
    fail("configured projection fixture context cannot be incremented");
  auto distinctContext = binding->context;
  ++distinctContext.ordinal;
  secondContext.slot.residency = distinctContext;
  std::vector<loom::mapping::ConfiguredHardwareFieldValueView> contextRows = {
      *projectedField, std::move(secondContext)};
  const auto separated =
      take(loom::mapping::detail::canonicalizeConfiguredHardwareProjection(
          std::move(contextRows)));
  if (separated.fields().size() != 2)
    fail("independently configurable contexts collapsed into one slot");

  const auto prepared = take(loom::sim::prepareCgraExecution(
      dataflowReference, fabric.reference(),
      spatialCandidates->candidates.front(), store));
  const auto summary = prepared.summary();
  if (summary.semanticConfigurationFieldCount != fields.size() ||
      summary.computeActorCount != 2 ||
      summary.computeTransitionPhysicalUseCount == 0 ||
      summary.routeTreeCount == 0 || summary.routeSinkCount == 0)
    fail("CGRA cold admission discarded Mapping-owned configuration");

  const dataflow::RootedGraphLaunchRef launch{
      dataflow.rootThreadLaunches().front().ref,
      dataflow.staticGraphLaunches().front().ref};
  loom::sim::SpatialSimulationWorkload workloadDraft{launch};
  workloadDraft.valueInputPlan = {loom::sim::RuntimeValueInput{},
                                  loom::sim::RuntimeValueInput{}};
  workloadDraft.observableContract.valueResults = {0};
  const auto workload =
      take(loom::sim::finalizeSimulationWorkload(workloadDraft, dataflow));
  loom::sim::SpatialSimulationRuntimeInputDraft runtimeDraft{
      workload.identity()};
  runtimeDraft.runtimeValues = {
      {0,
       {1,
        {loom::sim::SemanticLane::defined(llvm::APInt(8, 1)),
         loom::sim::SemanticLane::defined(llvm::APInt(8, 2))}}},
      {1, {1, {loom::sim::SemanticLane::defined(llvm::APInt(8, 3))}}}};
  const auto runtime = take(loom::sim::finalizeSimulationRuntimeInput(
      runtimeDraft, workload, dataflow));
  const auto outcome = take(loom::sim::simulateCgraWorkload(
      prepared, workload, runtime, /*maxEventFrames=*/128));
  if (outcome.state != loom::sim::SpatialExecutionSessionState::Retired ||
      !outcome.retired || outcome.counters.actorCommitCount == 0 ||
      outcome.counters.physicalRequestCount == 0 ||
      outcome.counters.physicalRetirementCount == 0)
    fail("mapped vector shuffle did not retire through physical execution");
  if (outcome.retired->observations.valueResults.size() != 1)
    fail("mapped vector shuffle produced the wrong result count");
  const auto *published = std::get_if<loom::sim::PublishedValueResult>(
      &outcome.retired->observations.valueResults.front());
  if (!published || published->value.tokenCount != 1 ||
      published->value.lanes.size() != 3 ||
      !(published->value.lanes[0] ==
        loom::sim::SemanticLane::defined(llvm::APInt(8, 2))) ||
      !(published->value.lanes[1] ==
        loom::sim::SemanticLane::defined(llvm::APInt(8, 3))) ||
      !(published->value.lanes[2] ==
        loom::sim::SemanticLane::defined(llvm::APInt(8, 1))))
    fail("mapped vector shuffle changed lane semantics");
}

} // namespace

int main() {
  exactVectorMappingDerivesConfigurationAndExecutes();
  llvm::outs() << "configured hardware projection tests passed\n";
  return 0;
}
