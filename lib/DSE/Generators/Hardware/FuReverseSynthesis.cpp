#include "DSE/FuReverseSynthesis.h"

#include "ADG/Builder.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Mapping/Tech/TechMappingGenerator.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::dse {
namespace {

struct AdmittedGraph final {
  ::dataflow::GraphRef graph;
  ::dataflow::ActorRef arithmeticActor;
  ::dataflow::CanonicalActorSchemaProjection arithmetic;
  ::dataflow::ActorRef syncActor;
  ::dataflow::CanonicalActorSchemaProjection sync;
};

llvm::Error failure(FuReverseSynthesisFailure kind,
                    const llvm::Twine &message) {
  return llvm::make_error<FuReverseSynthesisError>(kind, message.str());
}

llvm::Expected<::dataflow::CanonicalActorSchemaProjection>
projectActor(::mlir::Operation *operation,
             FuReverseSynthesisFailure failureKind) {
  auto projection =
      ::dataflow::projectRegisteredActorSchemaProjection(operation);
  if (!projection)
    return failure(failureKind, llvm::toString(projection.takeError()));
  return std::move(*projection);
}

bool hasExactSegments(llvm::ArrayRef<std::int32_t> actual,
                      std::array<std::int32_t, 3> expected) {
  return actual == llvm::ArrayRef<std::int32_t>(expected);
}

llvm::Expected<AdmittedGraph>
admitGraph(const ::dataflow::CanonicalDataflowProgramView &dataflow,
           ::dataflow::GraphRef graph) {
  auto resolved = dataflow.resolve(graph);
  if (!resolved)
    return failure(FuReverseSynthesisFailure::InvalidGraphReference,
                   llvm::toString(resolved.takeError()));
  auto graphOp = llvm::dyn_cast<::dataflow::GraphOp>(resolved->op);
  if (!graphOp || graphOp.isExternal())
    return failure(FuReverseSynthesisFailure::UnsupportedGraphInterface,
                   "reverse FU synthesis requires a defined canonical graph");

  ::mlir::FunctionType type = graphOp.getFunctionType();
  ::mlir::Type i32 = ::mlir::IntegerType::get(graphOp.getContext(), 32);
  if (type.getNumInputs() != 2 || type.getNumResults() != 1 ||
      !llvm::all_of(type.getInputs(),
                    [&](::mlir::Type input) { return input == i32; }) ||
      type.getResult(0) != i32 ||
      !hasExactSegments(graphOp.getInputSegmentSizes(), {2, 0, 0}) ||
      !hasExactSegments(graphOp.getResultSegmentSizes(), {1, 0, 0}))
    return failure(
        FuReverseSynthesisFailure::UnsupportedGraphInterface,
        "reverse FU synthesis requires exactly two i32 value inputs and one "
        "i32 value result");

  ::mlir::Block &body = graphOp.getBody().front();
  if (body.getNumArguments() != 3 ||
      !llvm::isa<::mlir::NoneType>(body.getArgument(0).getType()))
    return failure(FuReverseSynthesisFailure::UnsupportedGraphInterface,
                   "canonical graph boundary does not expose the required "
                   "start and value inputs");
  auto returnOp =
      llvm::dyn_cast<::dataflow::GraphReturnOp>(body.getTerminator());
  if (!returnOp || returnOp.getValues().size() != 1 ||
      !returnOp.getStreams().empty() || !returnOp.getMemories().empty() ||
      returnOp.getComplete().size() != 1)
    return failure(FuReverseSynthesisFailure::UnsupportedGraphInterface,
                   "canonical graph return must expose one value and one "
                   "completion frontier");

  std::optional<::dataflow::CanonicalActorView> arithmeticActor;
  std::optional<::dataflow::CanonicalActorView> syncActor;
  std::size_t actorCount = 0;
  for (const ::dataflow::CanonicalActorView &actor : dataflow.actors()) {
    if (actor.graph != graph)
      continue;
    ++actorCount;
    const std::optional<::dataflow::OperationSchemaId> schema =
        ::dataflow::operationSchemaOf(actor.op);
    if (schema == ::dataflow::OperationSchemaId::ArithAddI ||
        schema == ::dataflow::OperationSchemaId::ArithSubI) {
      if (arithmeticActor)
        return failure(FuReverseSynthesisFailure::UnsupportedActorInventory,
                       "canonical graph contains more than one arithmetic "
                       "actor");
      arithmeticActor = actor;
      continue;
    }
    if (schema == ::dataflow::OperationSchemaId::DataflowSync) {
      if (syncActor)
        return failure(FuReverseSynthesisFailure::UnsupportedActorInventory,
                       "canonical graph contains more than one token-sync "
                       "actor");
      syncActor = actor;
      continue;
    }
    return failure(FuReverseSynthesisFailure::UnsupportedActorSchema,
                   "canonical graph contains an actor outside the scalar "
                   "integer add/sub and token-sync domain");
  }
  if (actorCount != 2 || !arithmeticActor || !syncActor)
    return failure(FuReverseSynthesisFailure::UnsupportedActorInventory,
                   "canonical graph must contain one arithmetic actor and "
                   "one token-sync actor");

  auto arithmetic =
      projectActor(arithmeticActor->op,
                   FuReverseSynthesisFailure::UnsupportedActorProjection);
  if (!arithmetic)
    return arithmetic.takeError();
  auto sync = projectActor(
      syncActor->op, FuReverseSynthesisFailure::UnsupportedActorProjection);
  if (!sync)
    return sync.takeError();

  const auto *overflow =
      std::get_if<::dataflow::IntegerOverflowPayload>(&arithmetic->payload);
  if (!overflow ||
      overflow->flags != ::mlir::arith::IntegerOverflowFlags::none ||
      arithmetic->type.getNumInputs() != 2 ||
      arithmetic->type.getNumResults() != 1 ||
      !llvm::all_of(arithmetic->type.getInputs(),
                    [&](::mlir::Type input) { return input == i32; }) ||
      arithmetic->type.getResult(0) != i32)
    return failure(FuReverseSynthesisFailure::UnsupportedActorProjection,
                   "arithmetic actor must be overflow-free (i32, i32) -> i32");
  if (!std::holds_alternative<::dataflow::NoPayload>(sync->payload) ||
      sync->type.getNumInputs() != 2 || sync->type.getNumResults() != 2 ||
      !llvm::isa<::mlir::NoneType>(sync->type.getInput(0)) ||
      sync->type.getInput(1) != i32 ||
      !llvm::isa<::mlir::NoneType>(sync->type.getResult(0)) ||
      sync->type.getResult(1) != i32)
    return failure(FuReverseSynthesisFailure::UnsupportedActorProjection,
                   "token-sync actor must have exact (none, i32) -> (none, "
                   "i32) semantics");

  ::mlir::Operation *arithmeticOp = arithmeticActor->op;
  ::mlir::Operation *syncOp = syncActor->op;
  if (arithmeticOp->getOperand(0) != body.getArgument(1) ||
      arithmeticOp->getOperand(1) != body.getArgument(2) ||
      syncOp->getOperand(0) != body.getArgument(0) ||
      syncOp->getOperand(1) != arithmeticOp->getResult(0) ||
      returnOp.getValues().front() != syncOp->getResult(1) ||
      returnOp.getComplete().front() != syncOp->getResult(0))
    return failure(FuReverseSynthesisFailure::UnsupportedGraphTopology,
                   "canonical graph does not have the exact add/sub-to-sync "
                   "edge and boundary correspondence");

  return AdmittedGraph{graph, arithmeticActor->ref, std::move(*arithmetic),
                       syncActor->ref, std::move(*sync)};
}

llvm::Expected<::fabric::CanonicalImplementationCapability> deriveCapability(
    llvm::ArrayRef<::dataflow::CanonicalActorSchemaProjection> actors,
    ::fabric::ImplementationFamilyId expectedFamily) {
  auto capability =
      ::fabric::deriveCanonicalImplementationCapability(expectedFamily, actors);
  if (!capability)
    return failure(FuReverseSynthesisFailure::CapabilityDerivationRejected,
                   llvm::toString(capability.takeError()));
  if (capability->family != expectedFamily)
    return failure(FuReverseSynthesisFailure::CapabilityDerivationRejected,
                   "canonical capability derivation selected an unexpected "
                   "implementation family");
  return std::move(*capability);
}

llvm::Error verifyIdentityCorrespondence(
    const ::fabric::CanonicalImplementationCapability &capability,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    llvm::ArrayRef<std::uint32_t> inputWidths,
    llvm::ArrayRef<std::uint32_t> resultWidths) {
  std::vector<std::uint64_t> operands(actor.type.getNumInputs());
  std::vector<std::uint64_t> results(actor.type.getNumResults());
  for (std::size_t ordinal = 0; ordinal < operands.size(); ++ordinal)
    operands[ordinal] = ordinal;
  for (std::size_t ordinal = 0; ordinal < results.size(); ++ordinal)
    results[ordinal] = ordinal;
  if (llvm::Error error =
          ::fabric::verifyImplementationFamilyPortCorrespondence(
              capability.family, capability.parameters, actor, operands,
              results, inputWidths, resultWidths))
    return failure(FuReverseSynthesisFailure::CapabilityDerivationRejected,
                   llvm::toString(std::move(error)));
  return llvm::Error::success();
}

struct PreparedSynthesisDomain final {
  std::vector<::dataflow::GraphRef> graphs;
  std::vector<AdmittedGraph> admitted;
  ::fabric::CanonicalImplementationCapability arithmeticCapability;
  ::fabric::CanonicalImplementationCapability syncCapability;
};

llvm::Expected<PreparedSynthesisDomain>
prepareSynthesisDomain(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                       llvm::ArrayRef<::dataflow::GraphRef> graphs) {
  if (graphs.empty())
    return failure(FuReverseSynthesisFailure::EmptyGraphSet,
                   "reverse FU synthesis requires a non-empty graph set");
  for (::dataflow::GraphRef graph : graphs)
    if (graph.artifact != dataflow.identity())
      return failure(FuReverseSynthesisFailure::InvalidGraphReference,
                     "reverse FU synthesis received a foreign graph ref");

  std::vector<::dataflow::GraphRef> orderedGraphs(graphs.begin(), graphs.end());
  llvm::sort(orderedGraphs, [](const auto &left, const auto &right) {
    return left.entity.value() < right.entity.value();
  });
  if (std::adjacent_find(orderedGraphs.begin(), orderedGraphs.end()) !=
      orderedGraphs.end())
    return failure(FuReverseSynthesisFailure::DuplicateGraph,
                   "reverse FU synthesis graph set contains a duplicate");

  std::vector<AdmittedGraph> admitted;
  admitted.reserve(orderedGraphs.size());
  for (::dataflow::GraphRef graph : orderedGraphs) {
    auto value = admitGraph(dataflow, graph);
    if (!value)
      return value.takeError();
    admitted.push_back(std::move(*value));
  }

  std::vector<::dataflow::CanonicalActorSchemaProjection> arithmeticActors;
  std::vector<::dataflow::CanonicalActorSchemaProjection> syncActors;
  arithmeticActors.reserve(admitted.size());
  syncActors.reserve(admitted.size());
  for (const AdmittedGraph &graph : admitted) {
    arithmeticActors.push_back(graph.arithmetic);
    syncActors.push_back(graph.sync);
  }
  auto arithmeticCapability = deriveCapability(
      arithmeticActors, ::fabric::ImplementationFamilyId::ScalarIntegerAddSub);
  if (!arithmeticCapability)
    return arithmeticCapability.takeError();
  auto syncCapability =
      deriveCapability(syncActors, ::fabric::ImplementationFamilyId::TokenSync);
  if (!syncCapability)
    return syncCapability.takeError();

  const std::array<std::uint32_t, 2> arithmeticInputs = {32, 32};
  const std::array<std::uint32_t, 1> arithmeticResults = {32};
  const std::array<std::uint32_t, 2> syncInputs = {0, 32};
  const std::array<std::uint32_t, 2> syncResults = {0, 32};
  for (const AdmittedGraph &graph : admitted) {
    if (llvm::Error error = verifyIdentityCorrespondence(
            *arithmeticCapability, graph.arithmetic, arithmeticInputs,
            arithmeticResults))
      return std::move(error);
    if (llvm::Error error = verifyIdentityCorrespondence(
            *syncCapability, graph.sync, syncInputs, syncResults))
      return std::move(error);
  }
  return PreparedSynthesisDomain{std::move(orderedGraphs), std::move(admitted),
                                 std::move(*arithmeticCapability),
                                 std::move(*syncCapability)};
}

llvm::Expected<::loom::adg::DesignBuilder> buildFabricDesign(
    const ::fabric::CanonicalImplementationCapability &arithmeticCapability,
    const ::fabric::CanonicalImplementationCapability &syncCapability,
    const ArtifactStore &store) {
  using namespace ::loom::adg;
  constexpr std::uint32_t residentContexts = 1;
  auto bits0 = PortType::bits(0);
  auto bits32 = PortType::bits(32);
  auto tagged32 = PortType::taggedBits(32, 1);
  if (!bits0 || !bits32 || !tagged32)
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   "Fabric rejected the bounded synthesis port types");

  const std::vector<PortType> outerInputs(3, *tagged32);
  const std::vector<PortType> outerOutputs(2, *tagged32);
  const std::vector<PortType> peInputTypes(3, *bits32);
  const std::vector<PortType> fuOutputTypes(2, *bits32);
  DesignBuilder design(store);
  auto spatial = design.createSpatialCore("scalar-add-sub-synthesis",
                                          outerInputs, outerOutputs);
  if (!spatial)
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   llvm::toString(spatial.takeError()));
  std::vector<SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal < outerInputs.size(); ++ordinal) {
    auto input = spatial->input(ordinal);
    if (!input)
      return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                     llvm::toString(input.takeError()));
    spatialInputs.push_back(*input);
  }
  auto pe = spatial->addPe(
      spatialInputs,
      PeSpec::temporal(
          peInputTypes, outerOutputs,
          TemporalPeParameters{residentContexts,
                               FuConfigurationMode::PerInstruction,
                               ::fabric::OperandBufferMode::PerInstruction,
                               residentContexts, std::nullopt}));
  if (!pe)
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   llvm::toString(pe.takeError()));
  std::vector<PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal < outerInputs.size(); ++ordinal) {
    auto input = pe->input(ordinal);
    if (!input)
      return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                     llvm::toString(input.takeError()));
    peInputs.push_back(*input);
  }

  const std::vector<PortType> fuInputs = {*bits0, *bits32, *bits32};
  auto fu = pe->addFu(peInputs, FuSpec{fuInputs, fuOutputTypes});
  if (!fu)
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   llvm::toString(fu.takeError()));
  std::vector<FuValue> inputs;
  for (std::size_t ordinal = 0; ordinal < fuInputs.size(); ++ordinal) {
    auto input = fu->input(ordinal);
    if (!input)
      return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                     llvm::toString(input.takeError()));
    inputs.push_back(*input);
  }

  auto arithmetic = fu->addOperation(
      {inputs[1], inputs[2]},
      OperationCapabilitySpec{
          arithmeticCapability.family,
          arithmeticCapability.parameters,
          arithmeticCapability.enabledSchemas,
          {*bits32},
          ::fabric::oneCycleElasticOperationResourceContract()});
  if (!arithmetic)
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   llvm::toString(arithmetic.takeError()));
  auto arithmeticResult = arithmetic->output(0);
  if (!arithmeticResult)
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   llvm::toString(arithmeticResult.takeError()));
  auto sync = fu->addOperation(
      {inputs[0], *arithmeticResult},
      OperationCapabilitySpec{
          syncCapability.family,
          syncCapability.parameters,
          syncCapability.enabledSchemas,
          {*bits0, *bits32},
          ::fabric::oneCycleElasticOperationResourceContract()});
  if (!sync)
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   llvm::toString(sync.takeError()));
  if (llvm::Error error = fu->addCapabilityTemplate(
          FuCapabilityTemplateSpec{{*arithmetic, *sync}, {}}))
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   llvm::toString(std::move(error)));
  auto completion = sync->output(0);
  auto value = sync->output(1);
  if (!completion || !value)
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   "synthesized token-sync outputs do not resolve");
  if (llvm::Error error = fu->close({*completion, *value}))
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   llvm::toString(std::move(error)));
  if (llvm::Error error = pe->close())
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   llvm::toString(std::move(error)));
  std::vector<SpatialValue> outputs;
  for (std::size_t ordinal = 0; ordinal < outerOutputs.size(); ++ordinal) {
    auto output = pe->output(ordinal);
    if (!output)
      return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                     llvm::toString(output.takeError()));
    outputs.push_back(*output);
  }
  if (llvm::Error error = spatial->close(outputs))
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   llvm::toString(std::move(error)));
  return std::move(design);
}

llvm::Expected<::loom::fabric::FinalizedFabricRoot> materializeFabric(
    const ::fabric::CanonicalImplementationCapability &arithmeticCapability,
    const ::fabric::CanonicalImplementationCapability &syncCapability,
    const ArtifactStore &store) {
  auto design = buildFabricDesign(arithmeticCapability, syncCapability, store);
  if (!design)
    return design.takeError();
  auto finalized = std::move(*design).finalize();
  if (!finalized)
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   llvm::toString(finalized.takeError()));
  if (finalized->roots().size() != 1)
    return failure(FuReverseSynthesisFailure::FabricFinalizationFailed,
                   "bounded synthesis did not publish exactly one Fabric "
                   "root");
  return finalized->roots().front();
}

struct MappingMaterializationAttempt final {
  std::optional<::loom::mapping::FinalizedTechMapping> mapping;
  std::optional<FuReverseSynthesisFailure> termination;
  std::string diagnostic;
};

llvm::Expected<MappingMaterializationAttempt>
materializeMapping(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                   ::dataflow::GraphRef graph,
                   const ::loom::fabric::FinalizedFabricRoot &fabric,
                   const ::loom::mapping::ResolvedTechMappingConfigView &config,
                   const ArtifactStore &store,
                   ExecutionControlView executionControl) {
  const std::array graphs = {graph};
  const auto outcome = ::loom::mapping::generateTechMappings(
      {dataflow, graphs, fabric.view(), config, store, executionControl});
  if (const auto *generated =
          std::get_if<::loom::mapping::GeneratedTechMappings>(&outcome)) {
    if (generated->candidates.empty())
      return MappingMaterializationAttempt{
          std::nullopt, FuReverseSynthesisFailure::MappingInfeasible,
          "TechMapping produced no coverage candidate"};
    auto imported = ::loom::mapping::importTechMapping(
        generated->candidates.front(), store);
    if (!imported)
      return failure(FuReverseSynthesisFailure::MappingInternal,
                     llvm::toString(imported.takeError()));
    return MappingMaterializationAttempt{
        std::move(*imported), std::nullopt, {}};
  }
  if (std::holds_alternative<::loom::mapping::ProvenInfeasibleTechMapping>(
          outcome))
    return MappingMaterializationAttempt{
        std::nullopt, FuReverseSynthesisFailure::MappingInfeasible,
        "TechMapping proved the synthesized FU infeasible"};
  if (const auto *interrupted =
          std::get_if<::loom::mapping::InterruptedTechMappingGeneration>(
              &outcome)) {
    std::optional<::loom::mapping::FinalizedTechMapping> retained;
    if (!interrupted->candidates.empty()) {
      auto imported = ::loom::mapping::importTechMapping(
          interrupted->candidates.front(), store);
      if (!imported)
        return failure(FuReverseSynthesisFailure::MappingInternal,
                       llvm::toString(imported.takeError()));
      retained = std::move(*imported);
    }
    return MappingMaterializationAttempt{
        std::move(retained), FuReverseSynthesisFailure::CancelledOrTimeout,
        "TechMapping was interrupted before coverage closure"};
  }
  if (std::holds_alternative<::loom::mapping::IncompleteTechMappingGeneration>(
          outcome))
    return MappingMaterializationAttempt{
        std::nullopt, FuReverseSynthesisFailure::MappingIncomplete,
        "TechMapping did not establish complete coverage"};
  if (const auto *invalid =
          std::get_if<::loom::mapping::InvalidTechMappingGeneration>(&outcome))
    return failure(FuReverseSynthesisFailure::MappingInvalid,
                   invalid->diagnostic);
  const auto &internal =
      std::get<::loom::mapping::InternalTechMappingGeneration>(outcome);
  return failure(FuReverseSynthesisFailure::MappingInternal,
                 internal.diagnostic);
}

bool sameActorBinding(const ::loom::mapping::TechComputeActorView &left,
                      const ::loom::mapping::TechComputeActorView &right) {
  return left.actor == right.actor &&
         left.fabricOperation == right.fabricOperation &&
         left.operandPorts == right.operandPorts &&
         left.resultPorts == right.resultPorts;
}

bool sameBoundaryBinding(
    const ::loom::mapping::TechComputeBoundaryView &left,
    const ::loom::mapping::TechComputeBoundaryView &right) {
  return left.actor == right.actor && left.direction == right.direction &&
         left.portOrdinal == right.portOrdinal &&
         left.fabricPort == right.fabricPort;
}

bool sameCoverageWitness(const FuSynthesisCoverageWitness &left,
                         const FuSynthesisCoverageWitness &right) {
  return left.graph == right.graph && left.fabric == right.fabric &&
         left.capabilityTemplate == right.capabilityTemplate &&
         left.actors.size() == right.actors.size() &&
         left.boundaries.size() == right.boundaries.size() &&
         llvm::equal(left.actors, right.actors, sameActorBinding) &&
         llvm::equal(left.boundaries, right.boundaries, sameBoundaryBinding);
}

llvm::Expected<std::vector<FuSynthesisCoverageWitness>>
buildCoverage(llvm::ArrayRef<AdmittedGraph> graphs,
              const ::dataflow::CanonicalDataflowProgramView &dataflow,
              const ::loom::fabric::FinalizedFabricRoot &fabric) {
  const auto templates = fabric.view().fuTemplates();
  if (templates.size() != 1 ||
      fabric.view().fuCapabilityTemplates(templates.front()).size() != 1)
    return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                   "synthesized Fabric does not contain its unique FU "
                   "capability template");
  const ::loom::fabric::FabricFuCapabilityTemplateRef capability{
      templates.front(), 0};
  const auto operations =
      fabric.view().resolvedFabricOpCapabilities(templates.front());
  const auto arithmetic = llvm::find_if(operations, [](const auto &op) {
    return op.implementationFamily ==
           ::fabric::ImplementationFamilyId::ScalarIntegerAddSub;
  });
  const auto sync = llvm::find_if(operations, [](const auto &op) {
    return op.implementationFamily ==
           ::fabric::ImplementationFamilyId::TokenSync;
  });
  if (arithmetic == operations.end() || sync == operations.end())
    return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                   "synthesized FU does not expose both operation owners");

  std::vector<FuSynthesisCoverageWitness> witnesses;
  witnesses.reserve(graphs.size());
  for (const AdmittedGraph &graph : graphs) {
    std::vector<::loom::mapping::TechComputeActorView> actors = {
        {graph.arithmeticActor, arithmetic->occurrence, {0, 1}, {0}},
        {graph.syncActor, sync->occurrence, {0, 1}, {0, 1}}};
    std::vector<::loom::mapping::TechComputeBoundaryView> boundaries = {
        {graph.arithmeticActor,
         ::loom::fabric::FabricPortDirection::Input,
         0,
         {templates.front(), ::loom::fabric::FabricPortDirection::Input, 1}},
        {graph.arithmeticActor,
         ::loom::fabric::FabricPortDirection::Input,
         1,
         {templates.front(), ::loom::fabric::FabricPortDirection::Input, 2}},
        {graph.syncActor,
         ::loom::fabric::FabricPortDirection::Input,
         0,
         {templates.front(), ::loom::fabric::FabricPortDirection::Input, 0}},
        {graph.syncActor,
         ::loom::fabric::FabricPortDirection::Output,
         0,
         {templates.front(), ::loom::fabric::FabricPortDirection::Output, 0}},
        {graph.syncActor,
         ::loom::fabric::FabricPortDirection::Output,
         1,
         {templates.front(), ::loom::fabric::FabricPortDirection::Output, 1}}};
    const ::loom::mapping::TechComputeRealizationView prospective{
        0, capability, actors, boundaries};
    if (llvm::Error error =
            ::loom::mapping::verifyTechComputeRealizationClosure(
                prospective, dataflow, fabric.view()))
      return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                     llvm::toString(std::move(error)));
    witnesses.push_back({graph.graph, fabric.view().identity(), capability,
                         std::move(actors), std::move(boundaries)});
  }
  return witnesses;
}

llvm::Error verifyMaterializedCoverage(
    const ::loom::mapping::FinalizedTechMapping &mapping,
    const FuSynthesisCoverageWitness &witness,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FinalizedFabricRoot &fabric) {
  if (mapping.view().dataflowIdentity() != witness.graph.artifact ||
      mapping.view().fabricIdentity() != witness.fabric ||
      mapping.view().covers().size() != 1 ||
      mapping.view().covers().front() != witness.graph ||
      mapping.view().computeRealizations().size() != 1 ||
      !mapping.view().memoryRealizations().empty())
    return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                   "materialized TechMapping does not have exact graph and "
                   "Fabric ownership");
  const ::loom::mapping::TechComputeRealizationView prospective{
      0, witness.capabilityTemplate, witness.actors, witness.boundaries};
  if (llvm::Error error = ::loom::mapping::verifyTechComputeRealizationClosure(
          prospective, dataflow, fabric.view()))
    return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                   llvm::toString(std::move(error)));
  auto expected = ::loom::mapping::canonicalTechMatchRowKey(
      prospective, dataflow.identity());
  if (!expected)
    return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                   llvm::toString(expected.takeError()));
  auto actual = ::loom::mapping::canonicalTechMatchRowKey(
      mapping.view().computeRealizations().front(), dataflow.identity());
  if (!actual)
    return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                   llvm::toString(actual.takeError()));
  if (*expected != *actual)
    return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                   "materialized TechMapping differs from its synthesis "
                   "coverage witness");
  return llvm::Error::success();
}

} // namespace

char FuReverseSynthesisError::ID = 0;

void FuReverseSynthesisError::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code FuReverseSynthesisError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<const FuSynthesisCoverageWitness *>
resolveFuSynthesisCoverage(const ScalarIntegerAddSubFuSynthesisResult &result,
                           const FuSynthesisCoverageWitness &witness) {
  if (result.coverage().empty() ||
      witness.graph.artifact != result.coverage().front().graph.artifact ||
      witness.fabric != result.fabric().view().identity())
    return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                   "coverage witness belongs to another synthesis result");
  const auto published = llvm::find_if(
      result.coverage(), [&](const FuSynthesisCoverageWitness &candidate) {
        return sameCoverageWitness(candidate, witness);
      });
  if (published == result.coverage().end())
    return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                   "coverage witness is absent from the synthesis result");
  return &*published;
}

llvm::Expected<ScalarIntegerAddSubFuSynthesisAttempt>
attemptScalarIntegerAddSubFuSynthesis(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::GraphRef> graphs,
    const ::loom::mapping::ResolvedTechMappingConfigView &mappingConfig,
    const ArtifactStore &store, ExecutionControlView executionControl) {
  auto prepared = prepareSynthesisDomain(dataflow, graphs);
  if (!prepared)
    return prepared.takeError();
  if (executionControl.stopRequested())
    return failure(FuReverseSynthesisFailure::CancelledOrTimeout,
                   "reverse FU synthesis was interrupted before Fabric "
                   "materialization");

  auto fabric = materializeFabric(prepared->arithmeticCapability,
                                  prepared->syncCapability, store);
  if (!fabric)
    return fabric.takeError();
  auto coverage = buildCoverage(prepared->admitted, dataflow, *fabric);
  if (!coverage)
    return coverage.takeError();

  std::vector<::loom::mapping::FinalizedTechMapping> mappings;
  mappings.reserve(prepared->graphs.size());
  std::optional<FuReverseSynthesisFailure> termination;
  std::string terminationMessage;
  std::uint64_t consumedGraphBindings = 0;
  for (auto [graph, witness] : llvm::zip(prepared->graphs, *coverage)) {
    if (executionControl.stopRequested()) {
      termination = FuReverseSynthesisFailure::CancelledOrTimeout;
      terminationMessage =
          "reverse FU synthesis was interrupted between graph bindings";
      break;
    }
    ++consumedGraphBindings;
    auto mapping = materializeMapping(dataflow, graph, *fabric, mappingConfig,
                                      store, executionControl);
    if (!mapping)
      return mapping.takeError();
    if (mapping->mapping) {
      if (llvm::Error error = verifyMaterializedCoverage(
              *mapping->mapping, witness, dataflow, *fabric))
        return std::move(error);
      mappings.push_back(std::move(*mapping->mapping));
    }
    if (mapping->termination) {
      termination = mapping->termination;
      terminationMessage = std::move(mapping->diagnostic);
      break;
    }
  }
  return ScalarIntegerAddSubFuSynthesisAttempt{
      std::move(*fabric), std::move(mappings),           std::move(*coverage),
      termination,        std::move(terminationMessage), consumedGraphBindings};
}

llvm::Expected<ScalarIntegerAddSubFuSynthesisResult>
synthesizeScalarIntegerAddSubFu(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::GraphRef> graphs,
    const ::loom::mapping::ResolvedTechMappingConfigView &mappingConfig,
    const ArtifactStore &store, ExecutionControlView executionControl) {
  auto attempt = attemptScalarIntegerAddSubFuSynthesis(
      dataflow, graphs, mappingConfig, store, executionControl);
  if (!attempt)
    return attempt.takeError();
  if (attempt->termination_)
    return failure(*attempt->termination_, attempt->terminationMessage_);
  return ScalarIntegerAddSubFuSynthesisResult{std::move(attempt->fabric_),
                                              std::move(attempt->mappings_),
                                              std::move(attempt->coverage_)};
}

llvm::Error verifyScalarIntegerAddSubFuFabricLineage(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FinalizedFabricRoot &fabric,
    const ArtifactStore &store) {
  std::vector<::dataflow::GraphRef> graphs;
  graphs.reserve(dataflow.graphs().size());
  for (const ::dataflow::CanonicalGraphView &graph : dataflow.graphs())
    graphs.push_back(graph.ref);
  auto prepared = prepareSynthesisDomain(dataflow, graphs);
  if (!prepared)
    return prepared.takeError();
  auto design = buildFabricDesign(prepared->arithmeticCapability,
                                  prepared->syncCapability, store);
  if (!design)
    return design.takeError();
  auto expected = std::move(*design).deriveRootIdentities();
  if (!expected)
    return expected.takeError();
  if (expected->size() != 1 || expected->front() != fabric.reference().artifact)
    return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                   "Fabric lineage output is not the exact synthesis of its "
                   "complete Dataflow graph domain");
  return llvm::Error::success();
}

llvm::Error verifyScalarIntegerAddSubFuMappingLineage(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FinalizedFabricRoot &fabric,
    const ::loom::mapping::FinalizedTechMapping &mapping,
    const ArtifactStore &store) {
  if (llvm::Error error =
          verifyScalarIntegerAddSubFuFabricLineage(dataflow, fabric, store))
    return error;
  if (mapping.view().covers().size() != 1)
    return failure(FuReverseSynthesisFailure::CoverageNotEstablished,
                   "TechMapping lineage must cover exactly one graph");
  const ::dataflow::GraphRef graph = mapping.view().covers().front();
  auto admitted = admitGraph(dataflow, graph);
  if (!admitted)
    return admitted.takeError();
  const std::array admittedGraphs = {*admitted};
  auto coverage = buildCoverage(admittedGraphs, dataflow, fabric);
  if (!coverage)
    return coverage.takeError();
  return verifyMaterializedCoverage(mapping, coverage->front(), dataflow,
                                    fabric);
}

} // namespace loom::dse
