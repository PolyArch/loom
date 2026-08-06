#include "DSE/SpatialMappingFeedbackCandidateGenerator.h"

#include "DSE/DataflowRewriteCandidateGenerator.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/Transforms/DataflowRewrite.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/Request.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "PnR/MappingObjective.h"
#include "PnR/SpatialPnrProblem.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom::dse {
namespace {

enum InputSlot : std::uint32_t {
  DataflowInput,
  SpatialMappingInput,
  ConstraintInput,
  EvidenceInput,
  WorkloadInput,
  RuntimeInput,
  InputSlotCount,
};

const std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots = {{
        {CandidateGeneratorInputSlotRef(DataflowInput), "canonical_dataflow",
         PlanValueRole::CandidateSet, &::dataflow::canonicalDataflowSchema,
         PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorInputSlotRef(SpatialMappingInput), "spatial_mapping",
         PlanValueRole::CandidateSet, &::loom::mapping::mappingArtifactSchema,
         PlanValueCardinality::FiniteSet},
        {CandidateGeneratorInputSlotRef(ConstraintInput), "spatial_constraints",
         PlanValueRole::CandidateSet,
         &::loom::mapping::mappingConstraintSetSchema,
         PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorInputSlotRef(EvidenceInput), "mapping_evidence",
         PlanValueRole::EvidenceSet,
         &::loom::evaluation::EvaluationEvidence::artifactSchema,
         PlanValueCardinality::FiniteSet},
        {CandidateGeneratorInputSlotRef(WorkloadInput), "workload",
         PlanValueRole::CandidateSet, &::loom::sim::simulationWorkloadSchema,
         PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorInputSlotRef(RuntimeInput), "runtime_input",
         PlanValueRole::CandidateSet,
         &::loom::sim::simulationRuntimeInputSchema,
         PlanValueCardinality::ExactlyOne},
    }};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {
    {{CandidateGeneratorOutputSlotRef(0), "canonical_dataflow",
      PlanValueRole::CandidateSet, &::dataflow::canonicalDataflowSchema,
      PlanValueCardinality::FiniteSet}}};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 1> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "mapping_feedback_rewrite"},
}};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "spatial_mapping_feedback_generator_invalid: " + message);
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = ::loom::pnr::adoptResolvedSpatialPnrConfigView(
      ::loom::pnr::resolvedSpatialPnrConfigSchemaDescriptorBytes(), bytes,
      digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

const CandidateGeneratorDescriptor descriptor{
    spatialMappingFeedbackCandidateGeneratorKind,
    "mapping.spatial_feedback",
    "loom.mapping.spatial_feedback.generator.v1",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{
        ::loom::pnr::resolvedSpatialPnrConfigSchemaDescriptorBytes(),
        validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    &dataflowRewriteCandidateLineagePayloadContract(),
};

const ArtifactRootReference &
singleInput(llvm::ArrayRef<CandidateGeneratorInputBinding> bindings,
            InputSlot slot) {
  return bindings[slot].artifacts.front();
}

ArtifactRootReference rootReference(const ArtifactSchemaDescriptor &schema,
                                    const ArtifactIdentity &identity) {
  return {schema.identity.str(), schema.version, identity};
}

bool isExactSingleton(llvm::ArrayRef<ArtifactRootReference> references,
                      const ArtifactRootReference &reference) {
  return references.size() == 1 && references.front() == reference;
}

llvm::Error
validateCgraRequest(const ::loom::evaluation::EvaluationRequest &request,
                    const ArtifactRootReference &dataflow,
                    const ArtifactRootReference &fabric,
                    const ArtifactRootReference &spatialMapping,
                    const ArtifactRootReference &workload,
                    const ArtifactRootReference &runtimeInput) {
  if (request.modelBinding().descriptorRef() !=
      ::loom::evaluation::models::cgraSimulationModelDescriptorRef())
    return invalid("Evidence uses a non-CGRA model");
  if (!isExactSingleton(
          request.subjectBindings().subjects(
              ::loom::evaluation::models::cgraSimulationProgramRole()),
          dataflow) ||
      !isExactSingleton(
          request.subjectBindings().subjects(
              ::loom::evaluation::models::cgraSimulationHardwareRole()),
          fabric) ||
      !isExactSingleton(
          request.subjectBindings().subjects(
              ::loom::evaluation::models::cgraSimulationSpatialMappingRole()),
          spatialMapping))
    return invalid("Evidence subjects differ from the exact Mapping closure");
  if (!request.workload() || *request.workload() != workload ||
      !request.runtimeInput() || *request.runtimeInput() != runtimeInput)
    return invalid("Evidence uses a foreign workload or runtime input");
  return llvm::Error::success();
}

struct MatchedEvidence final {
  bool completed = false;
};

llvm::Expected<MatchedEvidence>
matchEvidence(const ArtifactRootReference &dataflow,
              const ArtifactRootReference &fabric,
              const ArtifactRootReference &spatialMapping,
              const ArtifactRootReference &workload,
              const ArtifactRootReference &runtimeInput,
              llvm::ArrayRef<ArtifactRootReference> evidenceReferences,
              std::vector<bool> &consumed, const ArtifactStore &store) {
  auto resolved = ::loom::evaluation::models::resolveCgraSimulationCase(
      spatialMapping, workload, runtimeInput, store);
  if (!resolved)
    return resolved.takeError();
  if (resolved->canonicalDataflow != dataflow || resolved->fabric != fabric)
    return invalid("CGRA case resolution differs from Mapping owners");

  for (std::size_t ordinal = 0; ordinal != evidenceReferences.size();
       ++ordinal) {
    if (consumed[ordinal])
      continue;
    auto evidence = ::loom::evaluation::importEvaluationEvidence(
        evidenceReferences[ordinal], resolved->resolution, store);
    if (!evidence) {
      llvm::consumeError(evidence.takeError());
      continue;
    }
    auto request = ::loom::evaluation::importEvaluationRequest(
        evidence->requestRef(), resolved->resolution, store);
    if (!request)
      return request.takeError();
    if (llvm::Error error = validateCgraRequest(
            *request, dataflow, fabric, spatialMapping, workload, runtimeInput))
      return std::move(error);
    consumed[ordinal] = true;
    return MatchedEvidence{evidence->outcomeKind() ==
                           ::loom::evaluation::EvidenceOutcomeKind::Completed};
  }
  return invalid("no exact EvaluationEvidence exists for a SpatialMapping");
}

struct RankedLogicalNet final {
  ::loom::pnr::SpatialMappingTraversalClaimContribution contribution;
  std::vector<std::uint8_t> canonicalKey;
};

llvm::Expected<std::vector<RankedLogicalNet>> rankLogicalNets(
    const ArtifactIdentity &dataflowIdentity,
    const ::loom::pnr::SpatialMappingTraversalClaimProjection &projection) {
  std::vector<RankedLogicalNet> ranked;
  ranked.reserve(projection.logicalNets.size());
  for (const auto &contribution : projection.logicalNets) {
    auto key = ::dataflow::encodeDataflowReference(dataflowIdentity,
                                                   contribution.logicalNet);
    if (!key)
      return key.takeError();
    ranked.push_back({contribution, std::move(*key)});
  }
  llvm::sort(ranked,
             [](const RankedLogicalNet &lhs, const RankedLogicalNet &rhs) {
               if (lhs.contribution.value != rhs.contribution.value)
                 return lhs.contribution.value > rhs.contribution.value;
               return lhs.canonicalKey < rhs.canonicalKey;
             });
  return ranked;
}

struct FeedbackCandidate final {
  ArtifactRootReference reference;
  CandidateGeneratorLineageEdge lineage;
};

struct RankedActor final {
  ::dataflow::ActorRef actor;
  std::vector<std::uint8_t> canonicalKey;
};

llvm::Expected<std::vector<RankedActor>>
feedbackActors(const ::dataflow::CanonicalDataflowProgramView &view,
               const ::dataflow::CanonicalGraphProducerEndpointRef &producer) {
  std::vector<RankedActor> actors;
  auto append = [&](::dataflow::ActorRef actor) -> llvm::Error {
    auto key = ::dataflow::encodeDataflowReference(view.identity(), actor);
    if (!key)
      return key.takeError();
    actors.push_back({actor, std::move(*key)});
    return llvm::Error::success();
  };

  if (const auto *result =
          std::get_if<::dataflow::ActorTokenResultRef>(&producer))
    if (llvm::Error error = append(result->actor))
      return std::move(error);

  auto consumers = view.graphConsumers(producer);
  if (!consumers)
    return consumers.takeError();
  for (const auto &consumer : *consumers) {
    const auto *operand =
        std::get_if<::dataflow::ActorTokenOperandRef>(&consumer);
    if (operand)
      if (llvm::Error error = append(operand->actor))
        return std::move(error);
  }

  llvm::sort(actors, [](const RankedActor &lhs, const RankedActor &rhs) {
    return lhs.canonicalKey < rhs.canonicalKey;
  });
  actors.erase(std::unique(actors.begin(), actors.end(),
                           [](const RankedActor &lhs, const RankedActor &rhs) {
                             return lhs.actor == rhs.actor;
                           }),
               actors.end());
  return actors;
}

llvm::Expected<std::optional<FeedbackCandidate>> materializeFeedback(
    const ArtifactRootReference &parentReference,
    const ::dataflow::CanonicalDataflowArtifact &parent,
    const ::dataflow::CanonicalDataflowProgramView &parentView,
    const ::loom::pnr::SpatialMappingTraversalClaimProjection &projection,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ArtifactStore &store) {
  auto ranked = rankLogicalNets(parent.identity(), projection);
  if (!ranked)
    return ranked.takeError();
  ::loom::frontend::FabricCapabilityIndex capabilities(fabric);

  for (const RankedLogicalNet &logicalNet : *ranked) {
    if (logicalNet.contribution.value == 0)
      break;
    auto actors =
        feedbackActors(parentView, logicalNet.contribution.logicalNet);
    if (!actors)
      return actors.takeError();
    for (const RankedActor &actor : *actors) {
      auto decisions =
          ::dataflow::enumerateElementwiseVectorDecompositionDecisions(
              parent, actor.actor);
      if (!decisions)
        return decisions.takeError();
      for (const ::dataflow::DataflowRewriteDecision &decision : *decisions) {
        auto child = ::dataflow::materializeDataflowRewrite(parent, decision);
        if (!child)
          return child.takeError();
        if (!*child)
          return invalid("typed feedback rewrite produced an identity");
        auto miss = capabilities.firstInadmissibleActor(**child);
        if (!miss)
          return miss.takeError();
        if (*miss)
          continue;
        auto published = ::dataflow::publishCanonicalDataflow(**child, store);
        if (!published)
          return published.takeError();
        auto payload = ::dataflow::encodeDataflowRewriteDecision(decision);
        if (!payload)
          return payload.takeError();
        return std::optional<FeedbackCandidate>{FeedbackCandidate{
            *published,
            CandidateGeneratorLineageEdge{
                CandidateGeneratorLineageEdgeKind::CandidateDecision,
                CandidateGeneratorOutputSlotRef(0),
                *published,
                {parentReference},
                std::move(*payload)}}};
      }
    }
  }
  return std::optional<FeedbackCandidate>{};
}

llvm::Expected<CandidateGeneratorInvocationOutcome>
invokeProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
               const ResolvedCandidateGeneratorBinding &binding,
               const ArtifactStore &store) {
  auto config = ::loom::pnr::adoptResolvedSpatialPnrConfigView(
      ::loom::pnr::resolvedSpatialPnrConfigSchemaDescriptorBytes(),
      binding.canonicalConfigBytes(), binding.configDigest());
  if (!config)
    return config.takeError();

  auto constraints = ::loom::mapping::importSpatialMappingConstraintSet(
      singleInput(inputs, ConstraintInput), store);
  if (!constraints)
    return constraints.takeError();
  const ArtifactRootReference dataflowReference =
      rootReference(::dataflow::canonicalDataflowSchema,
                    constraints->view().dataflowIdentity());
  const ArtifactRootReference techReference =
      rootReference(::loom::mapping::mappingArtifactSchema,
                    constraints->view().techMappingIdentity());
  const ArtifactRootReference fabricReference =
      rootReference(::loom::fabric::fabricArtifactSchema,
                    constraints->view().fabricIdentity());
  if (singleInput(inputs, DataflowInput) != dataflowReference)
    return invalid("constraint Dataflow owner is outside the bound input set");

  auto dataflow = ::dataflow::importCanonicalDataflow(dataflowReference, store);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();
  auto tech = ::loom::mapping::importTechMapping(techReference, store);
  if (!tech)
    return tech.takeError();
  auto fabric = ::loom::fabric::importEntireFabricRoot(fabricReference, store);
  if (!fabric)
    return fabric.takeError();
  if (tech->view().dataflowIdentity() != dataflowView->identity() ||
      tech->view().fabricIdentity() != fabric->view().identity())
    return invalid("constraint owner tuple is not a closed D/T/F relation");

  auto frozen = ::loom::pnr::freezeSpatialPnrProblem(
      *dataflowView, tech->view(), fabric->view(), *config,
      constraints->view());
  if (!frozen)
    return frozen.takeError();

  const ArtifactRootReference &workload = singleInput(inputs, WorkloadInput);
  const ArtifactRootReference &runtimeInput = singleInput(inputs, RuntimeInput);
  std::vector<bool> consumedEvidence(inputs[EvidenceInput].artifacts.size(),
                                     false);
  std::vector<ArtifactRootReference> outputs;
  std::vector<CandidateGeneratorLineageEdge> lineage;
  bool proofNotEstablished = false;

  for (const ArtifactRootReference &mappingReference :
       inputs[SpatialMappingInput].artifacts) {
    auto mapping =
        ::loom::mapping::importSpatialMapping(mappingReference, store);
    if (!mapping)
      return mapping.takeError();
    if (mapping->view().dataflowIdentity() != dataflowView->identity() ||
        mapping->view().techMappingIdentity() != tech->view().identity() ||
        mapping->view().fabricIdentity() != fabric->view().identity())
      return invalid("SpatialMapping differs from the exact constraint owners");
    if (llvm::Error error = ::loom::mapping::admitSpatialMappingConstraints(
            *dataflowView, tech->view(), fabric->view(), constraints->view(),
            mapping->view()))
      return std::move(error);

    auto evidence = matchEvidence(
        dataflowReference, fabricReference, mappingReference, workload,
        runtimeInput, inputs[EvidenceInput].artifacts, consumedEvidence, store);
    if (!evidence)
      return evidence.takeError();
    if (!evidence->completed) {
      proofNotEstablished = true;
      continue;
    }

    auto projection = ::loom::pnr::projectSpatialMappingTraversalClaims(
        **frozen, mapping->view());
    if (!projection)
      return projection.takeError();
    auto candidate =
        materializeFeedback(dataflowReference, *dataflow, *dataflowView,
                            *projection, fabric->view(), store);
    if (!candidate)
      return candidate.takeError();
    if (!*candidate)
      continue;
    outputs.push_back((*candidate)->reference);
    lineage.push_back(std::move((*candidate)->lineage));
  }
  if (llvm::is_contained(consumedEvidence, false))
    return invalid("EvaluationEvidence set contains an unmatched record");

  CandidateGeneratorOutputBinding output{CandidateGeneratorOutputSlotRef(0),
                                         std::move(outputs)};
  if (proofNotEstablished)
    return CandidateGeneratorInvocationOutcome{
        IncompleteCandidateGeneratorInvocation{
            CandidateGeneratorIncompleteReason::ProofNotEstablished,
            {std::move(output)},
            std::move(lineage)}};
  return CandidateGeneratorInvocationOutcome{
      CompletedCandidateGeneratorInvocation{{std::move(output)},
                                            std::move(lineage)}};
}

const CandidateGeneratorProvider provider{descriptor.reference(),
                                          invokeProvider};

void canonicalizeReferences(std::vector<ArtifactRootReference> &references) {
  llvm::sort(references, artifactRootReferenceLess);
  references.erase(std::unique(references.begin(), references.end()),
                   references.end());
}

} // namespace

const CandidateGeneratorDescriptor &
spatialMappingFeedbackCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerSpatialMappingFeedbackCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindSpatialMappingFeedbackCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> canonicalDataflowPrograms,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    const ArtifactRootReference &constraints,
    llvm::ArrayRef<ArtifactRootReference> evaluationEvidence,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput) {
  if (llvm::Error error = registerSpatialMappingFeedbackCandidateGenerator())
    return std::move(error);
  std::vector<ArtifactRootReference> dataflows(
      canonicalDataflowPrograms.begin(), canonicalDataflowPrograms.end());
  std::vector<ArtifactRootReference> mappings(spatialMappings.begin(),
                                              spatialMappings.end());
  std::vector<ArtifactRootReference> evidence(evaluationEvidence.begin(),
                                              evaluationEvidence.end());
  canonicalizeReferences(dataflows);
  canonicalizeReferences(mappings);
  canonicalizeReferences(evidence);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(DataflowInput), std::move(dataflows)},
      {CandidateGeneratorInputSlotRef(SpatialMappingInput),
       std::move(mappings)},
      {CandidateGeneratorInputSlotRef(ConstraintInput), {constraints}},
      {CandidateGeneratorInputSlotRef(EvidenceInput), std::move(evidence)},
      {CandidateGeneratorInputSlotRef(WorkloadInput), {workload}},
      {CandidateGeneratorInputSlotRef(RuntimeInput), {runtimeInput}},
  };
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveSpatialMappingFeedbackCandidateGeneratorBinding(
    const ::loom::pnr::ResolvedPnrConfigView &config) {
  if (llvm::Error error = registerSpatialMappingFeedbackCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

} // namespace loom::dse
