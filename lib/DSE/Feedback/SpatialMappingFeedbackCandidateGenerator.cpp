#include "DSE/SpatialMappingFeedbackCandidateGenerator.h"

#include "DSE/DataflowRewriteCandidateGenerator.h"
#include "DSE/StructuredOwnershipInvocationInternal.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/MappingDebugLog.h"
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
#include <iterator>
#include <limits>
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
    "loom.mapping.spatial_feedback.generator.v2",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{
        ::loom::pnr::resolvedSpatialPnrConfigSchemaDescriptorBytes(),
        validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    &dataflowRewriteCandidateLineagePayloadContract(),
    ProviderForm::InProcess,
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
  std::size_t mappingOrdinal = 0;
  bool completed = false;
};

struct PreparedSpatialMapping final {
  ArtifactRootReference reference;
  ::loom::mapping::FinalizedSpatialMapping mapping;
  ::loom::evaluation::models::ResolvedCgraSimulationCase cgraCase;
};

llvm::Expected<::loom::evaluation::CaseArtifactResolution>
mergeCaseResolutions(llvm::ArrayRef<PreparedSpatialMapping> mappings) {
  std::vector<::loom::evaluation::CaseArtifactResolution::Entry> entries;
  for (const PreparedSpatialMapping &mapping : mappings)
    entries.insert(entries.end(), mapping.cgraCase.resolution.entries().begin(),
                   mapping.cgraCase.resolution.entries().end());
  llvm::sort(entries, [](const auto &lhs, const auto &rhs) {
    return artifactRootReferenceLess(lhs.artifact, rhs.artifact);
  });

  std::vector<::loom::evaluation::CaseArtifactResolution::Entry> merged;
  merged.reserve(entries.size());
  for (auto &entry : entries) {
    if (!merged.empty() && merged.back().artifact == entry.artifact) {
      if (merged.back().dependencyClosure != entry.dependencyClosure)
        return invalid("CGRA case resolutions disagree on an artifact closure");
      continue;
    }
    merged.push_back(std::move(entry));
  }
  return ::loom::evaluation::CaseArtifactResolution::get(std::move(merged));
}

llvm::Expected<MatchedEvidence>
classifyEvidence(const ArtifactRootReference &reference,
                 llvm::ArrayRef<PreparedSpatialMapping> mappings,
                 const ::loom::evaluation::CaseArtifactResolution &resolution,
                 const ArtifactRootReference &workload,
                 const ArtifactRootReference &runtimeInput,
                 const ArtifactStore &store) {
  auto requestReference =
      ::loom::evaluation::importEvaluationEvidenceRequestReference(reference,
                                                                   store);
  if (!requestReference)
    return requestReference.takeError();

  auto request = ::loom::evaluation::importEvaluationRequest(*requestReference,
                                                             resolution, store);
  if (!request)
    return request.takeError();
  if (request->modelBinding().descriptorRef() !=
      ::loom::evaluation::models::cgraSimulationModelDescriptorRef())
    return invalid("Evidence uses a non-CGRA model");
  const auto subjects = request->subjectBindings().subjects(
      ::loom::evaluation::models::cgraSimulationSpatialMappingRole());
  if (subjects.size() != 1)
    return invalid("Evidence subjects differ from the exact Mapping closure");
  auto found = llvm::lower_bound(mappings, subjects.front(),
                                 [](const PreparedSpatialMapping &mapping,
                                    const ArtifactRootReference &sought) {
                                   return artifactRootReferenceLess(
                                       mapping.reference, sought);
                                 });
  if (found == mappings.end() || found->reference != subjects.front())
    return invalid("Evidence subjects differ from the exact Mapping closure");
  const std::size_t matchedMapping =
      static_cast<std::size_t>(std::distance(mappings.begin(), found));

  const PreparedSpatialMapping &mapping = mappings[matchedMapping];
  auto evidence = ::loom::evaluation::importEvaluationEvidence(
      reference, mapping.cgraCase.resolution, store);
  if (!evidence)
    return evidence.takeError();
  if (llvm::Error error = validateCgraRequest(
          *request, mapping.cgraCase.canonicalDataflow, mapping.cgraCase.fabric,
          mapping.reference, workload, runtimeInput))
    return std::move(error);
  return MatchedEvidence{
      matchedMapping, evidence->outcomeKind() ==
                          ::loom::evaluation::EvidenceOutcomeKind::Completed};
}

llvm::Expected<MatchedEvidence>
matchEvidence(std::size_t mappingOrdinal,
              llvm::ArrayRef<MatchedEvidence> evidence,
              std::vector<bool> &consumed) {
  for (std::size_t ordinal = 0; ordinal != evidence.size(); ++ordinal) {
    if (consumed[ordinal] || evidence[ordinal].mappingOrdinal != mappingOrdinal)
      continue;
    consumed[ordinal] = true;
    return evidence[ordinal];
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
    const ArtifactStore &store, std::uint64_t &decisionAttempts) {
  auto ranked = rankLogicalNets(parent.identity(), projection);
  if (!ranked)
    return ranked.takeError();
  ::loom::frontend::FabricCapabilityIndex capabilities(fabric);

  for (auto indexedNet : llvm::enumerate(*ranked)) {
    const std::size_t netRank = indexedNet.index();
    const RankedLogicalNet &logicalNet = indexedNet.value();
    if (logicalNet.contribution.value == 0)
      break;
    auto actors =
        feedbackActors(parentView, logicalNet.contribution.logicalNet);
    if (!actors)
      return actors.takeError();
    ::loom::mapping_debug::emit(::loom::mapping_debug::Level::Detail,
                                ::loom::mapping_debug::Stage::SpatialPnr,
                                ::loom::mapping_debug::Event::Candidate,
                                [&](llvm::json::Object &fields) {
                                  fields["operation"] = "mapping_feedback_net";
                                  fields["rank"] = netRank;
                                  fields["selected_traversal_claim"] =
                                      logicalNet.contribution.value;
                                  fields["actor_count"] = actors->size();
                                });
    for (const RankedActor &actor : *actors) {
      auto decisions =
          ::dataflow::enumerateElementwiseVectorDecompositionDecisions(
              parent, actor.actor);
      if (!decisions)
        return decisions.takeError();
      auto actorView = parentView.resolve(actor.actor);
      if (!actorView)
        return actorView.takeError();
      ::loom::mapping_debug::emit(
          ::loom::mapping_debug::Level::Detail,
          ::loom::mapping_debug::Stage::SpatialPnr,
          ::loom::mapping_debug::Event::Candidate,
          [&](llvm::json::Object &fields) {
            fields["operation"] = "mapping_feedback_actor";
            fields["rank"] = netRank;
            fields["actor_operation"] = actorView->op->getName().getStringRef();
            fields["decision_count"] = decisions->size();
          });
      for (const ::dataflow::DataflowRewriteDecision &decision : *decisions) {
        if (decisionAttempts == std::numeric_limits<std::uint64_t>::max())
          return invalid("feedback decision accounting overflows u64");
        ++decisionAttempts;
        auto child = ::dataflow::materializeDataflowRewrite(parent, decision);
        if (!child)
          return child.takeError();
        if (!*child)
          return invalid("typed feedback rewrite produced an identity");
        auto miss = capabilities.firstInadmissibleActor(**child);
        if (!miss)
          return miss.takeError();
        const bool inadmissible = miss->has_value();
        ::loom::mapping_debug::emit(
            ::loom::mapping_debug::Level::Detail,
            ::loom::mapping_debug::Stage::SpatialPnr,
            ::loom::mapping_debug::Event::Candidate,
            [&](llvm::json::Object &fields) {
              fields["operation"] = "mapping_feedback_decision";
              fields["rank"] = netRank;
              fields["fabric_admissible"] = !inadmissible;
              if (inadmissible)
                fields["inadmissible_schema"] =
                    ::dataflow::operationSchemaSpelling((*miss)->schema);
              if (const auto *chunk =
                      std::get_if<::dataflow::ElementwiseVectorChunkRewrite>(
                          &decision)) {
                fields["rewrite_mode"] = "leading_chunk";
                fields["leading_blocks_per_chunk"] =
                    chunk->leadingBlocksPerChunk;
              } else {
                fields["rewrite_mode"] = "scalarize";
              }
            });
        if (inadmissible)
          continue;
        auto published = ::dataflow::publishCanonicalDataflow(**child, store);
        if (!published)
          return published.takeError();
        if (StructuredOwnershipInvocation *invocation =
                detail::StructuredOwnershipInvocationAccess::current())
          if (llvm::Error error = detail::StructuredOwnershipInvocationAccess::
                  recordDataflowRewriteCandidate(*invocation, parentReference,
                                                 *published, decision, store))
            return std::move(error);
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

llvm::Expected<CandidateGeneratorProviderResult>
invokeProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
               const ResolvedCandidateGeneratorBinding &binding,
               const ArtifactStore &store, const BlobStore &blobs) {
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
  std::vector<PreparedSpatialMapping> mappings;
  mappings.reserve(inputs[SpatialMappingInput].artifacts.size());
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
    auto cgraCase = ::loom::evaluation::models::resolveCgraSimulationCase(
        mappingReference, workload, runtimeInput, store);
    if (!cgraCase)
      return cgraCase.takeError();
    if (cgraCase->canonicalDataflow != dataflowReference ||
        cgraCase->fabric != fabricReference)
      return invalid("CGRA case resolution differs from Mapping owners");
    mappings.push_back(
        {mappingReference, std::move(*mapping), std::move(*cgraCase)});
  }

  std::vector<MatchedEvidence> evidence;
  evidence.reserve(inputs[EvidenceInput].artifacts.size());
  auto mergedResolution = mergeCaseResolutions(mappings);
  if (!mergedResolution)
    return mergedResolution.takeError();
  for (const ArtifactRootReference &reference :
       inputs[EvidenceInput].artifacts) {
    auto classified = classifyEvidence(reference, mappings, *mergedResolution,
                                       workload, runtimeInput, store);
    if (!classified)
      return classified.takeError();
    evidence.push_back(std::move(*classified));
  }
  std::vector<bool> consumedEvidence(evidence.size(), false);
  std::vector<ArtifactRootReference> outputs;
  std::vector<CandidateGeneratorLineageEdge> lineage;
  bool proofNotEstablished = false;
  std::uint64_t decisionAttempts = 0;

  for (std::size_t mappingOrdinal = 0; mappingOrdinal != mappings.size();
       ++mappingOrdinal) {
    PreparedSpatialMapping &mapping = mappings[mappingOrdinal];
    auto matched = matchEvidence(mappingOrdinal, evidence, consumedEvidence);
    if (!matched)
      return matched.takeError();
    if (!matched->completed) {
      proofNotEstablished = true;
      continue;
    }

    auto projection = ::loom::pnr::projectSpatialMappingTraversalClaims(
        **frozen, mapping.mapping.view());
    if (!projection)
      return projection.takeError();
    auto candidate = materializeFeedback(
        dataflowReference, *dataflow, *dataflowView, *projection,
        fabric->view(), store, decisionAttempts);
    if (!candidate)
      return candidate.takeError();
    if (!*candidate)
      continue;
    outputs.push_back((*candidate)->reference);
    lineage.push_back(std::move((*candidate)->lineage));
  }
  if (llvm::is_contained(consumedEvidence, false))
    return invalid("EvaluationEvidence set contains an unmatched record");

  ::loom::mapping_debug::emit(
      ::loom::mapping_debug::Level::Summary,
      ::loom::mapping_debug::Stage::SpatialPnr,
      ::loom::mapping_debug::Event::Statistics,
      [&](llvm::json::Object &fields) {
        fields["operation"] = "mapping_feedback_summary";
        fields["mapping_count"] = mappings.size();
        fields["decision_attempts"] = decisionAttempts;
        fields["candidate_publications"] = outputs.size();
        fields["proof_not_established"] = proofNotEstablished;
      });

  CandidateGeneratorOutputBinding output{CandidateGeneratorOutputSlotRef(0),
                                         std::move(outputs)};
  if (proofNotEstablished)
    return CandidateGeneratorProviderResult{
        IncompleteCandidateGeneratorResult{
            CandidateGeneratorIncompleteReason::ProofNotEstablished,
            {std::move(output)},
            std::move(lineage)},
        {{CandidateGeneratorWorkUnitRef(0), decisionAttempts,
          decisionAttempts}}};
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{{std::move(output)},
                                        std::move(lineage)},
      {{CandidateGeneratorWorkUnitRef(0), decisionAttempts, decisionAttempts}}};
}

const CandidateGeneratorProvider provider{
    descriptor.reference(),
    CandidateGeneratorInProcessProvider{invokeProvider}};

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
