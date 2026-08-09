#include "ADG/Builder.h"
#include "Common/ArtifactStore.h"
#include "Common/IndexWidth.h"
#include "Config/ResolvedConfig.h"
#include "ConfiguredHardwareProjectionInternal.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "Mapping/Tech/TechMappingGenerator.h"
#include "PnR/MappingObjective.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialPnrGenerator.h"
#include "Simulator/CGRAAdmission.h"
#include "Simulator/CGRASimulator.h"
#include "Simulator/SimulationArtifacts.h"

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
  resolved.dse.spatialPnr.objectiveSelection = {0, 0, {}};
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
  auto generatedSpatial = loom::pnr::generateSpatialMappings(
      {dataflow, tech.view(), fabric.view(), pnrConfig, constraints.view(),
       store});
  const auto *spatialCandidates =
      std::get_if<loom::pnr::GeneratedSpatialMappings>(&generatedSpatial);
  if (!spatialCandidates || spatialCandidates->candidates.size() != 1) {
    const std::string diagnostic = std::visit(
        [](const auto &outcome) {
          using Outcome = std::decay_t<decltype(outcome)>;
          if constexpr (std::is_same_v<Outcome,
                                       loom::pnr::GeneratedSpatialMappings>)
            return std::string("generated an unexpected candidate count");
          else
            return outcome.diagnostic;
        },
        generatedSpatial);
    fail("SpatialMapping fixture did not produce one candidate: " + diagnostic);
  }
  const auto spatial = take(loom::mapping::importSpatialMapping(
      spatialCandidates->candidates.front(), store));

  const auto fields = spatial.view().configuredHardware().fields();
  if (fields.size() != 3)
    fail("complete Mapping did not derive the FU topology and operation "
         "fields");
  std::size_t topologyFieldCount = 0;
  std::size_t operationFieldCount = 0;
  for (const auto &field : fields) {
    const auto &owner = field.slot.field.owner.catalog();
    if (owner.kind() == loom::fabric::FabricInventoryOwnerKind::FuOccurrence) {
      ++topologyFieldCount;
    } else if (owner.kind() ==
               loom::fabric::FabricInventoryOwnerKind::FuOccurrenceNode) {
      ++operationFieldCount;
    } else {
      fail("configured compute field has a non-FU occurrence owner");
    }
    auto relation =
        take(fabric.view().semanticFieldRelation(field.slot.field, context));
    requireSuccess(relation.validateSemanticValue(field.value.bytes()));
  }
  if (topologyFieldCount != 1 || operationFieldCount != 2)
    fail("configured compute fields have the wrong semantic owners");
  const auto bindings = spatial.view().computeBindings();
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
  for (const auto &candidate : bindings) {
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
