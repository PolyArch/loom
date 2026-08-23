#include "DSE/RootCompleteTechMappingCandidateGenerator.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/MappingDebugLog.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/Tech/TechMappingGenerator.h"
#include "Mapping/Tech/TechMappingHardwareDemand.h"
#include "PnR/SpatialRootSupplyAdmission.h"

#include "llvm/ADT/STLExtras.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom::dse {
namespace {

enum InputSlot : std::uint32_t {
  DataflowCandidatesInput,
  FabricInput,
  InputSlotCount,
};

enum ApplicationInputSlot : std::uint32_t {
  ApplicationDataflowInput,
  ApplicationSystemConstraintsInput,
  ApplicationFabricInput,
  ApplicationInputSlotCount,
};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots = {{
        {CandidateGeneratorInputSlotRef(DataflowCandidatesInput),
         "canonical_dataflow", PlanValueRole::CandidateSet,
         &::dataflow::canonicalDataflowSchema, PlanValueCardinality::FiniteSet},
        {CandidateGeneratorInputSlotRef(FabricInput), "fabric",
         PlanValueRole::CandidateSet, &::loom::fabric::fabricArtifactSchema,
         PlanValueCardinality::ExactlyOne},
    }};

constexpr std::array<CandidateGeneratorInputSlotDescriptor,
                     ApplicationInputSlotCount>
    applicationInputSlots = {{
        {CandidateGeneratorInputSlotRef(ApplicationDataflowInput), "dataflow",
         PlanValueRole::CandidateSet, &::dataflow::canonicalDataflowSchema,
         PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorInputSlotRef(ApplicationSystemConstraintsInput),
         "system_constraints", PlanValueRole::CandidateSet,
         &::loom::mapping::mappingConstraintSetSchema,
         PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorInputSlotRef(ApplicationFabricInput), "fabric",
         PlanValueRole::CandidateSet, &::loom::fabric::fabricArtifactSchema,
         PlanValueCardinality::ExactlyOne},
    }};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {
    {{CandidateGeneratorOutputSlotRef(0), "tech_mapping",
      PlanValueRole::CandidateSet, &::loom::mapping::mappingArtifactSchema,
      PlanValueCardinality::FiniteSet}}};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 4> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "match_row_attempt"},
    {CandidateGeneratorWorkUnitRef(1), "partial_cover_expansion"},
    {CandidateGeneratorWorkUnitRef(2), "candidate_evaluation"},
    {CandidateGeneratorWorkUnitRef(3), "publication_slot"},
}};

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = ::loom::mapping::adoptResolvedTechMappingConfigView(
      ::loom::mapping::resolvedTechMappingConfigSchemaDescriptorBytes(), bytes,
      digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

llvm::Error validateComputeContextFeedback(
    llvm::ArrayRef<std::uint8_t> bytes,
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ArtifactStore &store) {
  const ArtifactRootReference *fabricReference = nullptr;
  for (const CandidateGeneratorInputBinding &binding : inputs) {
    for (const ArtifactRootReference &artifact : binding.artifacts) {
      if (artifact.schemaIdentity !=
              ::loom::fabric::fabricArtifactSchema.identity ||
          artifact.schemaVersion !=
              ::loom::fabric::fabricArtifactSchema.version)
        continue;
      if (fabricReference)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "tech_mapping_generator_feedback_invalid: input closure has "
            "multiple Fabric roots");
      fabricReference = &artifact;
    }
  }
  if (!fabricReference)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "tech_mapping_generator_feedback_invalid: input closure has no "
        "Fabric root");
  auto fabric = ::loom::fabric::importEntireFabricRoot(*fabricReference, store);
  if (!fabric)
    return fabric.takeError();
  if (fabric->view().rootKind() != ::loom::fabric::FabricRootKind::Module)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "tech_mapping_generator_feedback_invalid: target is not a Module");
  auto adopted = ::loom::mapping::adoptTechMappingComputeContextHallFeedback(
      bytes, fabric->view());
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

const CandidateGeneratorOwnerFeedbackPayloadContract feedbackContract{
    ::loom::mapping::techMappingComputeContextHallFeedbackSchemaBytes(),
    validateComputeContextFeedback};

llvm::Expected<CandidateGeneratorProviderResult> invokeRootCompleteProvider(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store, const BlobStore &blobs,
    const CandidateGeneratorInvocationView &invocation);

llvm::Expected<CandidateGeneratorProviderResult> invokeApplicationGraphProvider(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store, const BlobStore &blobs,
    const CandidateGeneratorInvocationView &invocation);

const CandidateGeneratorDescriptor descriptor{
    rootCompleteTechMappingCandidateGeneratorKind,
    "mapping.root_complete_tech_mapping",
    "loom.mapping.root_complete_tech_mapping.generator.v7",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{
        ::loom::mapping::resolvedTechMappingConfigSchemaDescriptorBytes(),
        validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    nullptr,
    ProviderForm::InProcess,
    &feedbackContract,
};

const CandidateGeneratorDescriptor applicationGraphDescriptor{
    applicationGraphTechMappingCandidateGeneratorKind,
    "mapping.application_graph_tech_mapping",
    "loom.mapping.application_graph_tech_mapping.generator.v10",
    applicationInputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{
        ::loom::mapping::resolvedTechMappingConfigSchemaDescriptorBytes(),
        validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    nullptr,
    ProviderForm::InProcess,
    &feedbackContract,
};

llvm::Error accumulate(std::uint64_t source, std::uint64_t &target,
                       llvm::StringRef subject) {
  if (source > std::numeric_limits<std::uint64_t>::max() - target)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "root_complete_tech_mapping_generator_invalid: " + subject +
            " accounting overflows u64");
  target += source;
  return llvm::Error::success();
}

llvm::Error
accumulate(const ::loom::mapping::TechMappingGenerationAccounting &source,
           ::loom::mapping::TechMappingGenerationAccounting &target) {
  if (llvm::Error error =
          accumulate(source.matchRowAttempts, target.matchRowAttempts,
                     "match-row attempt"))
    return error;
  if (llvm::Error error =
          accumulate(source.matchRowFirstVisits, target.matchRowFirstVisits,
                     "match-row first visit"))
    return error;
  if (llvm::Error error = accumulate(source.matchRowCursorResumptions,
                                     target.matchRowCursorResumptions,
                                     "match-row cursor resumption"))
    return error;
  if (llvm::Error error =
          accumulate(source.matchRowReplayVisits, target.matchRowReplayVisits,
                     "match-row replay visit"))
    return error;
  if (llvm::Error error = accumulate(source.memoryRowFrontierLimits,
                                     target.memoryRowFrontierLimits,
                                     "memory-row frontier limit"))
    return error;
  if (llvm::Error error =
          accumulate(source.partialCoverExpansions,
                     target.partialCoverExpansions, "partial-cover expansion"))
    return error;
  if (llvm::Error error = accumulate(source.constructiveCoverSearchInvocations,
                                     target.constructiveCoverSearchInvocations,
                                     "constructive-cover search invocation"))
    return error;
  if (llvm::Error error = accumulate(source.constructiveCoverCompletedChecks,
                                     target.constructiveCoverCompletedChecks,
                                     "constructive-cover completed check"))
    return error;
  if (llvm::Error error = accumulate(source.constructiveCoverPublications,
                                     target.constructiveCoverPublications,
                                     "constructive-cover publication"))
    return error;
  if (llvm::Error error = accumulate(source.computeContextProjectionWork,
                                     target.computeContextProjectionWork,
                                     "compute-context projection work"))
    return error;
  if (llvm::Error error = accumulate(source.computeContextMatchingChecks,
                                     target.computeContextMatchingChecks,
                                     "compute-context matching check"))
    return error;
  if (llvm::Error error = accumulate(source.computeContextRejectedChecks,
                                     target.computeContextRejectedChecks,
                                     "compute-context rejected check"))
    return error;
  if (llvm::Error error = accumulate(source.computeContextMatchingWork,
                                     target.computeContextMatchingWork,
                                     "compute-context matching work"))
    return error;
  if (llvm::Error error = accumulate(source.memorySupplyProjectionWork,
                                     target.memorySupplyProjectionWork,
                                     "memory-supply projection work"))
    return error;
  if (llvm::Error error =
          accumulate(source.memorySupplyChecks, target.memorySupplyChecks,
                     "memory-supply check"))
    return error;
  if (llvm::Error error = accumulate(source.memorySupplyPartialChecks,
                                     target.memorySupplyPartialChecks,
                                     "memory-supply partial check"))
    return error;
  if (llvm::Error error =
          accumulate(source.memorySupplyFullChecks,
                     target.memorySupplyFullChecks, "memory-supply full check"))
    return error;
  if (llvm::Error error = accumulate(source.memorySupplyRejectedChecks,
                                     target.memorySupplyRejectedChecks,
                                     "memory-supply rejected check"))
    return error;
  if (llvm::Error error = accumulate(source.memorySupplyEmptyDomainRejections,
                                     target.memorySupplyEmptyDomainRejections,
                                     "memory-supply empty-domain rejection"))
    return error;
  if (llvm::Error error =
          accumulate(source.memorySupplyExclusiveResourceRejections,
                     target.memorySupplyExclusiveResourceRejections,
                     "memory-supply exclusive-resource rejection"))
    return error;
  if (llvm::Error error = accumulate(source.memorySupplySpatialPortRejections,
                                     target.memorySupplySpatialPortRejections,
                                     "memory-supply Spatial-port rejection"))
    return error;
  if (llvm::Error error =
          accumulate(source.memorySupplyTemporalIngressRejections,
                     target.memorySupplyTemporalIngressRejections,
                     "memory-supply Temporal-ingress rejection"))
    return error;
  if (llvm::Error error =
          accumulate(source.memorySupplyInternalConnectionRejections,
                     target.memorySupplyInternalConnectionRejections,
                     "memory-supply internal-connection rejection"))
    return error;
  if (llvm::Error error =
          accumulate(source.memorySupplyResidentCapacityRejections,
                     target.memorySupplyResidentCapacityRejections,
                     "memory-supply resident-capacity rejection"))
    return error;
  if (llvm::Error error =
          accumulate(source.memorySupplyJointAssignmentRejections,
                     target.memorySupplyJointAssignmentRejections,
                     "memory-supply joint-assignment rejection"))
    return error;
  if (llvm::Error error = accumulate(source.memorySupplySearchWork,
                                     target.memorySupplySearchWork,
                                     "memory-supply search work"))
    return error;
  if (llvm::Error error =
          accumulate(source.candidateEvaluations, target.candidateEvaluations,
                     "candidate evaluation"))
    return error;
  return accumulate(source.publicationSlots, target.publicationSlots,
                    "publication slot");
}

std::vector<CandidateGeneratorWorkUnitSummary> workSummary(
    const ::loom::mapping::TechMappingGenerationAccounting &accounting) {
  return {
      {CandidateGeneratorWorkUnitRef(0), accounting.matchRowAttempts,
       accounting.matchRowAttempts},
      {CandidateGeneratorWorkUnitRef(1), accounting.partialCoverExpansions,
       accounting.partialCoverExpansions},
      {CandidateGeneratorWorkUnitRef(2), accounting.candidateEvaluations,
       accounting.candidateEvaluations},
      {CandidateGeneratorWorkUnitRef(3), accounting.publicationSlots,
       accounting.publicationSlots},
  };
}

std::optional<std::vector<std::uint8_t>> encodeFeedback(
    const std::optional<::loom::mapping::TechMappingComputeContextHallDeficit>
        &feedback) {
  if (!feedback)
    return std::nullopt;
  return ::loom::mapping::encodeTechMappingComputeContextHallFeedback(
      *feedback);
}

void emitFeedback(
    const std::optional<::loom::mapping::TechMappingComputeContextHallDeficit>
        &feedback) {
  if (!feedback)
    return;
  ::loom::mapping_debug::emit(
      ::loom::mapping_debug::Level::Summary,
      ::loom::mapping_debug::Stage::TechMapping,
      ::loom::mapping_debug::Event::MappingFailure,
      [&](llvm::json::Object &fields) {
        fields["failure_scope"] = "tech_cover_compute_context_hall_demand";
        fields["closure_status"] = "proven_infeasible";
        fields["proof_scope"] = "observed_cover_relation";
        fields["cover_compute_demand_count"] = feedback->coverDemandCount();
        fields["cover_compute_maximum_matching"] =
            feedback->coverMaximumMatching();
        fields["hall_demand_count"] = feedback->hallDemandCount();
        fields["hall_context_value_count"] = feedback->hallContextValueCount();
        fields["hall_deficit"] = feedback->deficit();
        llvm::json::Array groups;
        for (const auto &group : feedback->groups()) {
          llvm::json::Object value;
          value["demand_count"] = group.demandCount;
          value["compatible_context_count"] = group.compatibleContexts.size();
          groups.push_back(std::move(value));
        }
        fields["capability_groups"] = std::move(groups);
      });
}

llvm::Expected<std::optional<CandidateGeneratorIncompleteReason>>
consumeTechMappingOutcome(
    ::loom::mapping::TechMappingGenerationOutcome outcome,
    ::loom::mapping::TechMappingGenerationAccounting &accounting,
    std::vector<ArtifactRootReference> &outputs,
    std::vector<CandidateGeneratorLineageEdge> &lineage,
    std::optional<::loom::mapping::TechMappingComputeContextHallDeficit>
        &feedback) {
  const auto &currentAccounting = std::visit(
      [](const auto &result)
          -> const ::loom::mapping::TechMappingGenerationAccounting & {
        return result.accounting;
      },
      outcome);
  if (llvm::Error error = accumulate(currentAccounting, accounting))
    return std::move(error);
  const auto &currentFeedback = std::visit(
      [](const auto &result)
          -> const ::loom::mapping::TechMappingGenerationFeedback & {
        return result.feedback;
      },
      outcome);
  if (currentFeedback.computeContextHall)
    ::loom::mapping::retainTechMappingComputeContextHallFeedback(
        feedback, *currentFeedback.computeContextHall);
  if (auto *generated =
          std::get_if<::loom::mapping::GeneratedTechMappings>(&outcome)) {
    for (ArtifactRootReference &candidate : generated->candidates) {
      lineage.push_back(CandidateGeneratorLineageEdge{
          CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
          CandidateGeneratorOutputSlotRef(0),
          candidate,
          {},
          {}});
      outputs.push_back(std::move(candidate));
    }
    if (generated->termination ==
        ::loom::mapping::TechMappingGenerationTermination::SemanticLimitReached)
      return std::optional<CandidateGeneratorIncompleteReason>{
          CandidateGeneratorIncompleteReason::SemanticLimitReached};
    return std::nullopt;
  }
  if (std::holds_alternative<::loom::mapping::ProvenInfeasibleTechMapping>(
          outcome))
    return std::nullopt;
  if (std::holds_alternative<::loom::mapping::IncompleteTechMappingGeneration>(
          outcome))
    return std::optional<CandidateGeneratorIncompleteReason>{
        CandidateGeneratorIncompleteReason::ProofNotEstablished};
  if (auto *interrupted =
          std::get_if<::loom::mapping::InterruptedTechMappingGeneration>(
              &outcome)) {
    for (ArtifactRootReference &candidate : interrupted->candidates) {
      lineage.push_back(CandidateGeneratorLineageEdge{
          CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
          CandidateGeneratorOutputSlotRef(0),
          candidate,
          {},
          {}});
      outputs.push_back(std::move(candidate));
    }
    return std::optional<CandidateGeneratorIncompleteReason>{
        CandidateGeneratorIncompleteReason::CancelledOrTimeout};
  }
  if (const auto *invalid =
          std::get_if<::loom::mapping::InvalidTechMappingGeneration>(&outcome))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "tech_mapping_generator_adapter_invalid: " +
                                       invalid->diagnostic);
  const auto &internal =
      std::get<::loom::mapping::InternalTechMappingGeneration>(outcome);
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "tech_mapping_generator_adapter_execution_failed: " +
          internal.diagnostic);
}

llvm::Expected<CandidateGeneratorProviderResult>
invokeProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
               const ResolvedCandidateGeneratorBinding &binding,
               const ArtifactStore &store,
               const ExecutionControlView &executionControl,
               std::optional<std::uint64_t> maximumOutputs) {
  auto config = ::loom::mapping::adoptResolvedTechMappingConfigView(
      ::loom::mapping::resolvedTechMappingConfigSchemaDescriptorBytes(),
      binding.canonicalConfigBytes(), binding.configDigest());
  if (!config)
    return config.takeError();

  auto fabric = ::loom::fabric::importEntireFabricRoot(
      inputBindings[FabricInput].artifacts.front(), store);
  if (!fabric)
    return fabric.takeError();

  std::vector<ArtifactRootReference> outputs;
  std::vector<CandidateGeneratorLineageEdge> lineage;
  ::loom::mapping::TechMappingGenerationAccounting accounting;
  std::optional<::loom::mapping::TechMappingComputeContextHallDeficit> feedback;
  std::optional<CandidateGeneratorIncompleteReason> incompleteReason;
  const auto rememberIncomplete =
      [&](CandidateGeneratorIncompleteReason reason) {
        if (!incompleteReason ||
            reason == CandidateGeneratorIncompleteReason::CancelledOrTimeout ||
            (*incompleteReason !=
                 CandidateGeneratorIncompleteReason::CancelledOrTimeout &&
             reason == CandidateGeneratorIncompleteReason::ProofNotEstablished))
          incompleteReason = reason;
      };
  for (const auto indexedDataflow :
       llvm::enumerate(inputBindings[DataflowCandidatesInput].artifacts)) {
    if (maximumOutputs && outputs.size() >= *maximumOutputs) {
      rememberIncomplete(
          CandidateGeneratorIncompleteReason::SemanticLimitReached);
      break;
    }
    const ArtifactRootReference &dataflowReference = indexedDataflow.value();
    auto artifact =
        ::dataflow::importCanonicalDataflow(dataflowReference, store);
    if (!artifact)
      return artifact.takeError();
    auto dataflow = artifact->view();
    if (!dataflow)
      return dataflow.takeError();
    if (dataflow->graphs().empty())
      continue;

    std::vector<::dataflow::GraphRef> completeCover;
    completeCover.reserve(dataflow->graphs().size());
    for (const ::dataflow::CanonicalGraphView &graph : dataflow->graphs())
      completeCover.push_back(graph.ref);
    std::optional<::loom::mapping::ResolvedTechMappingConfigView>
        boundedConfig;
    const ::loom::mapping::ResolvedTechMappingConfigView *generationConfig =
        &*config;
    if (maximumOutputs) {
      auto derived =
          ::loom::mapping::deriveTechMappingConfigWithPublicationLimit(
              *config, *maximumOutputs - outputs.size());
      if (!derived)
        return derived.takeError();
      boundedConfig = std::move(*derived);
      generationConfig = &*boundedConfig;
    }
    auto incomplete =
        consumeTechMappingOutcome(::loom::mapping::generateTechMappings(
                                      {*dataflow, completeCover, fabric->view(),
                                       *generationConfig, store,
                                       executionControl}),
                                  accounting, outputs, lineage, feedback);
    if (!incomplete)
      return incomplete.takeError();
    if (*incomplete)
      rememberIncomplete(**incomplete);
    if (incompleteReason ==
        CandidateGeneratorIncompleteReason::CancelledOrTimeout)
      break;
    if (maximumOutputs && outputs.size() >= *maximumOutputs &&
        indexedDataflow.index() + 1 !=
            inputBindings[DataflowCandidatesInput].artifacts.size()) {
      rememberIncomplete(
          CandidateGeneratorIncompleteReason::SemanticLimitReached);
      break;
    }
  }

  emitFeedback(feedback);
  if (incompleteReason)
    return CandidateGeneratorProviderResult{
        IncompleteCandidateGeneratorResult{
            *incompleteReason,
            {{CandidateGeneratorOutputSlotRef(0), std::move(outputs)}},
            std::move(lineage)},
        workSummary(accounting), encodeFeedback(feedback)};
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{CandidateGeneratorOutputSlotRef(0), std::move(outputs)}},
          std::move(lineage)},
      workSummary(accounting), encodeFeedback(feedback)};
}

llvm::Expected<CandidateGeneratorProviderResult> invokeRootCompleteProvider(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store, const BlobStore &,
    const CandidateGeneratorInvocationView &invocation) {
  return invokeProvider(inputBindings, binding, store,
                        invocation.executionControl(),
                        invocation.maximumOutputArtifacts(
                            CandidateGeneratorOutputSlotRef(0)));
}

llvm::Expected<CandidateGeneratorProviderResult> invokeApplicationGraphProvider(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store, const BlobStore &,
    const CandidateGeneratorInvocationView &invocation) {
  auto config = ::loom::mapping::adoptResolvedTechMappingConfigView(
      ::loom::mapping::resolvedTechMappingConfigSchemaDescriptorBytes(),
      binding.canonicalConfigBytes(), binding.configDigest());
  if (!config)
    return config.takeError();
  auto dataflowArtifact = ::dataflow::importCanonicalDataflow(
      inputBindings[ApplicationDataflowInput].artifacts.front(), store);
  if (!dataflowArtifact)
    return dataflowArtifact.takeError();
  auto dataflow = dataflowArtifact->view();
  if (!dataflow)
    return dataflow.takeError();
  auto constraints = ::loom::mapping::importSystemMappingConstraintSet(
      inputBindings[ApplicationSystemConstraintsInput].artifacts.front(),
      store);
  if (!constraints)
    return constraints.takeError();
  if (constraints->view().dataflowIdentity() != dataflow->identity())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "application_graph_tech_mapping_generator_invalid: constraints "
        "bind a foreign Dataflow");
  auto fabric = ::loom::fabric::importEntireFabricRoot(
      inputBindings[ApplicationFabricInput].artifacts.front(), store);
  if (!fabric)
    return fabric.takeError();
  ArtifactRootReference systemReference{
      ::loom::fabric::fabricArtifactSchema.identity.str(),
      ::loom::fabric::fabricArtifactSchema.version,
      constraints->view().fabricIdentity()};
  auto systemArtifact =
      ::loom::fabric::importEntireFabricRoot(systemReference, store);
  if (!systemArtifact)
    return systemArtifact.takeError();
  auto system = ::loom::fabric::requireSystemRoot(systemArtifact->view());
  if (!system)
    return system.takeError();
  const bool attached = llvm::any_of(
      system->artifact().accCoreOccurrences(), [&](const auto core) {
        auto target = system->spatialCoreTarget(core);
        return target &&
               target->dependencyOrdinal <
                   system->artifact().importedModules().size() &&
               system->artifact()
                       .importedModules()[target->dependencyOrdinal]
                       .identity() == fabric->view().identity();
      });
  if (!attached)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "application_graph_tech_mapping_generator_invalid: Fabric Module "
        "is not attached to the constrained System");

  std::vector<::dataflow::GraphRef> graphs;
  llvm::Error graphError = llvm::Error::success();
  for (const auto &root : constraints->view().rootThreadLaunches()) {
    dataflow->forEachRootedGraphLaunch(
        [&](::dataflow::RootedGraphLaunchRef launch) {
          if (graphError || launch.rootThreadLaunch != root)
            return;
          auto graph = dataflow->resolve(launch);
          if (graph)
            graphs.push_back(*graph);
          else
            graphError = graph.takeError();
        });
    if (graphError)
      return std::move(graphError);
  }
  llvm::sort(graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.entity.value() < rhs.entity.value();
  });
  graphs.erase(std::unique(graphs.begin(), graphs.end()), graphs.end());

  std::vector<ArtifactRootReference> outputs;
  std::vector<CandidateGeneratorLineageEdge> lineage;
  ::loom::mapping::TechMappingGenerationAccounting accounting;
  std::uint64_t rootSupplyAdmissions = 0;
  std::uint64_t rootSupplyRejections = 0;
  std::uint64_t admittedGraphCount = 0;
  std::uint64_t unadmittedGraphCount = 0;
  std::uint64_t rootSupplyDeterministicWork = 0;
  std::uint64_t rootSupplyConstructionNanoseconds = 0;
  std::optional<::loom::mapping::TechMappingComputeContextHallDeficit> feedback;
  std::optional<CandidateGeneratorIncompleteReason> incompleteReason;
  const std::optional<std::uint64_t> maximumOutputs =
      invocation.maximumOutputArtifacts(CandidateGeneratorOutputSlotRef(0));
  bool outputDemandReached = false;
  const auto rememberIncomplete =
      [&](CandidateGeneratorIncompleteReason reason) {
        if (!incompleteReason ||
            reason == CandidateGeneratorIncompleteReason::CancelledOrTimeout ||
            (*incompleteReason !=
                 CandidateGeneratorIncompleteReason::CancelledOrTimeout &&
             reason == CandidateGeneratorIncompleteReason::ProofNotEstablished))
          incompleteReason = reason;
      };
  for (const auto indexedGraph : llvm::enumerate(graphs)) {
    const ::dataflow::GraphRef &graph = indexedGraph.value();
    if (maximumOutputs && outputs.size() >= *maximumOutputs) {
      outputDemandReached = true;
      rememberIncomplete(
          CandidateGeneratorIncompleteReason::SemanticLimitReached);
      break;
    }
    const std::uint64_t remainingGraphs = graphs.size() - indexedGraph.index();
    const std::uint64_t remainingOutputs =
        maximumOutputs ? *maximumOutputs - outputs.size()
                       : config->candidatePublicationLimit();
    const std::uint64_t fairGraphLimit =
        maximumOutputs ? remainingOutputs / remainingGraphs
                       : config->candidatePublicationLimit();
    if (fairGraphLimit == 0) {
      outputDemandReached = true;
      rememberIncomplete(
          CandidateGeneratorIncompleteReason::SemanticLimitReached);
      break;
    }
    const std::uint64_t graphPublicationLimit =
        std::min(config->candidatePublicationLimit(), fairGraphLimit);
    const std::array cover = {graph};
    std::uint64_t admittedForGraph = 0;
    std::uint64_t candidateOrdinalForGraph = 0;
    auto enumeration = ::loom::mapping::enumerateTechMappingCandidates(
        {*dataflow, cover, fabric->view(), *config, store,
         invocation.executionControl()},
        [&](const ArtifactRootReference &candidate)
            -> llvm::Expected<
                ::loom::mapping::TechMappingCandidateEnumerationControl> {
          const std::uint64_t candidateOrdinal = candidateOrdinalForGraph++;
          const auto constructionBegin = std::chrono::steady_clock::now();
          auto tech = ::loom::mapping::importTechMapping(candidate, store);
          if (!tech)
            return tech.takeError();
          auto admission = ::loom::pnr::analyzeSpatialRootSupply(
              tech->view(), *dataflow, fabric->view());
          if (!admission)
            return admission.takeError();
          const auto elapsed =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - constructionBegin)
                  .count();
          if (elapsed > 0)
            if (llvm::Error error =
                    accumulate(static_cast<std::uint64_t>(elapsed),
                               rootSupplyConstructionNanoseconds,
                               "root-supply construction nanosecond"))
              return std::move(error);
          if (llvm::Error error = accumulate(admission->deterministicWork,
                                             rootSupplyDeterministicWork,
                                             "root-supply deterministic work"))
            return std::move(error);
          std::uint64_t residualSinkCount = 0;
          for (const auto &net : tech->view().residualLogicalNets())
            if (llvm::Error error =
                    accumulate(net.sinks.size(), residualSinkCount,
                               "residual logical-net sink"))
              return std::move(error);
          ::loom::mapping_debug::emit(
              ::loom::mapping_debug::Level::Decision,
              ::loom::mapping_debug::Stage::TechMapping,
              ::loom::mapping_debug::Event::Candidate,
              [&](llvm::json::Object &fields) {
                fields["candidate_ordinal"] = candidateOrdinal;
                fields["tech_mapping"] =
                    formatArtifactIdentityHex(candidate.artifact);
                fields["compute_realization_count"] =
                    tech->view().computeRealizations().size();
                fields["memory_realization_count"] =
                    tech->view().memoryRealizations().size();
                fields["residual_logical_net_count"] =
                    tech->view().residualLogicalNets().size();
                fields["residual_sink_count"] = residualSinkCount;
                fields["compute_demand_count"] = admission->computeDemandCount;
                fields["compute_context_value_count"] =
                    admission->computeContextValueCount;
                fields["compute_context_edge_count"] =
                    admission->computeContextEdgeCount;
                fields["compute_context_maximum_matching"] =
                    admission->computeContextMaximumMatching;
                fields["compute_hall_demand_count"] =
                    admission->computeHallDemandCount;
                fields["compute_hall_context_value_count"] =
                    admission->computeHallContextValueCount;
                llvm::json::Array hallRealizations;
                for (const std::uint64_t realization :
                     admission->computeHallRealizations)
                  hallRealizations.push_back(realization);
                fields["compute_hall_realizations"] =
                    std::move(hallRealizations);
                fields["memory_demand_count"] = admission->memoryDemandCount;
                fields["memory_occurrence_value_count"] =
                    admission->memoryOccurrenceValueCount;
                fields["memory_occurrence_choice_count"] =
                    admission->memoryOccurrenceChoiceCount;
                fields["memory_exclusive_relation_count"] =
                    admission->memoryExclusiveRelationCount;
                fields["memory_assignment_attempts"] =
                    admission->memoryAssignmentAttempts;
                fields["memory_supply_failure"] = ::loom::mapping::
                    spatialMemoryOccurrenceSupplyFailureKindSpelling(
                        admission->memoryFailure);
                fields["disposition"] =
                    admission->disposition ==
                            ::loom::pnr::SpatialRootSupplyDisposition::
                                ProvenInfeasible
                        ? "proven_infeasible"
                        : "admissible";
              });
          if (admission->disposition ==
              ::loom::pnr::SpatialRootSupplyDisposition::ProvenInfeasible) {
            ++rootSupplyRejections;
            ::loom::mapping_debug::emit(
                ::loom::mapping_debug::Level::Decision,
                ::loom::mapping_debug::Stage::TechMapping,
                ::loom::mapping_debug::Event::MappingFailure,
                [&](llvm::json::Object &fields) {
                  fields["failure_scope"] = "root_supply_admission";
                  fields["closure_status"] = "proven_infeasible";
                  fields["tech_mapping"] =
                      formatArtifactIdentityHex(candidate.artifact);
                  fields["compute_demand_count"] =
                      admission->computeDemandCount;
                  fields["compute_context_value_count"] =
                      admission->computeContextValueCount;
                  fields["compute_context_maximum_matching"] =
                      admission->computeContextMaximumMatching;
                  fields["compute_hall_demand_count"] =
                      admission->computeHallDemandCount;
                  fields["compute_hall_context_value_count"] =
                      admission->computeHallContextValueCount;
                  fields["memory_demand_count"] = admission->memoryDemandCount;
                  fields["memory_occurrence_value_count"] =
                      admission->memoryOccurrenceValueCount;
                  fields["memory_occurrence_choice_count"] =
                      admission->memoryOccurrenceChoiceCount;
                  fields["memory_exclusive_relation_count"] =
                      admission->memoryExclusiveRelationCount;
                  fields["memory_assignment_attempts"] =
                      admission->memoryAssignmentAttempts;
                  fields["memory_supply_failure"] = ::loom::mapping::
                      spatialMemoryOccurrenceSupplyFailureKindSpelling(
                          admission->memoryFailure);
                  fields["diagnostic"] = admission->diagnostic;
                });
            return ::loom::mapping::TechMappingCandidateEnumerationControl::
                Continue;
          }

          ++rootSupplyAdmissions;
          if (!llvm::is_contained(outputs, candidate)) {
            outputs.push_back(candidate);
            lineage.push_back(CandidateGeneratorLineageEdge{
                CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
                CandidateGeneratorOutputSlotRef(0),
                candidate,
                {},
                {}});
            ++admittedForGraph;
          }
          if (maximumOutputs && outputs.size() >= *maximumOutputs) {
            outputDemandReached = true;
            return ::loom::mapping::TechMappingCandidateEnumerationControl::
                Stop;
          }
          return admittedForGraph >= graphPublicationLimit
                     ? ::loom::mapping::TechMappingCandidateEnumerationControl::
                           Stop
                     : ::loom::mapping::TechMappingCandidateEnumerationControl::
                           Continue;
        });
    if (!enumeration)
      return enumeration.takeError();
    if (admittedForGraph != 0) {
      ++admittedGraphCount;
    } else {
      ++unadmittedGraphCount;
      ::loom::mapping_debug::emit(
          ::loom::mapping_debug::Level::Summary,
          ::loom::mapping_debug::Stage::TechMapping,
          ::loom::mapping_debug::Event::MappingFailure,
          [&](llvm::json::Object &fields) {
            fields["failure_scope"] = "graph_root_supply_frontier";
            fields["closure_status"] = enumeration->interruption
                                           ? "cancelled_or_timeout"
                                           : "proof_not_established";
            fields["graph"] = graph.entity.value();
            fields["candidate_evaluations"] =
                enumeration->accounting.candidateEvaluations;
            fields["root_supply_rejections"] = rootSupplyRejections;
          });
    }
    if (enumeration->feedback.computeContextHall)
      ::loom::mapping::retainTechMappingComputeContextHallFeedback(
          feedback, *enumeration->feedback.computeContextHall);
    if (llvm::Error error = accumulate(enumeration->accounting, accounting))
      return std::move(error);
    if (llvm::Error error =
            accumulate(admittedForGraph, accounting.publicationSlots,
                       "admitted publication slot"))
      return std::move(error);
    if (enumeration->interruption)
      rememberIncomplete(
          CandidateGeneratorIncompleteReason::CancelledOrTimeout);
    else if (outputDemandReached)
      rememberIncomplete(
          CandidateGeneratorIncompleteReason::SemanticLimitReached);
    else if (enumeration->termination ==
             ::loom::mapping::TechMappingGenerationTermination::
                 SemanticLimitReached)
      rememberIncomplete(
          admittedForGraph >= graphPublicationLimit
              ? CandidateGeneratorIncompleteReason::SemanticLimitReached
              : CandidateGeneratorIncompleteReason::ProofNotEstablished);
    if (incompleteReason ==
        CandidateGeneratorIncompleteReason::CancelledOrTimeout)
      break;
    if (outputDemandReached)
      break;
  }
  if (admittedGraphCount != graphs.size()) {
    outputs.clear();
    lineage.clear();
    rememberIncomplete(
        incompleteReason.value_or(
            CandidateGeneratorIncompleteReason::ProofNotEstablished));
  }
  ::loom::mapping_debug::emit(
      ::loom::mapping_debug::Level::Summary,
      ::loom::mapping_debug::Stage::TechMapping,
      ::loom::mapping_debug::Event::Statistics,
      [&](llvm::json::Object &fields) {
        fields["statistics_kind"] = "application_tech_root_supply_frontier";
        fields["required_graph_count"] = graphs.size();
        fields["admitted_graph_count"] = admittedGraphCount;
        fields["unadmitted_graph_count"] = unadmittedGraphCount;
        fields["root_supply_admissions"] = rootSupplyAdmissions;
        fields["root_supply_rejections"] = rootSupplyRejections;
        fields["root_supply_construction_time_ns"] =
            rootSupplyConstructionNanoseconds;
        fields["root_supply_deterministic_work"] = rootSupplyDeterministicWork;
        fields["memory_row_frontier_limits"] =
            accounting.memoryRowFrontierLimits;
        fields["constructive_cover_search_invocations"] =
            accounting.constructiveCoverSearchInvocations;
        fields["constructive_cover_completed_checks"] =
            accounting.constructiveCoverCompletedChecks;
        fields["constructive_cover_publications"] =
            accounting.constructiveCoverPublications;
        fields["compute_context_projection_work"] =
            accounting.computeContextProjectionWork;
        fields["compute_context_matching_checks"] =
            accounting.computeContextMatchingChecks;
        fields["compute_context_rejected_checks"] =
            accounting.computeContextRejectedChecks;
        fields["compute_context_matching_work"] =
            accounting.computeContextMatchingWork;
        fields["memory_supply_projection_work"] =
            accounting.memorySupplyProjectionWork;
        fields["memory_supply_checks"] = accounting.memorySupplyChecks;
        fields["memory_supply_partial_checks"] =
            accounting.memorySupplyPartialChecks;
        fields["memory_supply_full_checks"] = accounting.memorySupplyFullChecks;
        fields["memory_supply_rejected_checks"] =
            accounting.memorySupplyRejectedChecks;
        fields["memory_supply_empty_domain_rejections"] =
            accounting.memorySupplyEmptyDomainRejections;
        fields["memory_supply_exclusive_resource_rejections"] =
            accounting.memorySupplyExclusiveResourceRejections;
        fields["memory_supply_spatial_port_rejections"] =
            accounting.memorySupplySpatialPortRejections;
        fields["memory_supply_temporal_ingress_rejections"] =
            accounting.memorySupplyTemporalIngressRejections;
        fields["memory_supply_internal_connection_rejections"] =
            accounting.memorySupplyInternalConnectionRejections;
        fields["memory_supply_resident_capacity_rejections"] =
            accounting.memorySupplyResidentCapacityRejections;
        fields["memory_supply_joint_assignment_rejections"] =
            accounting.memorySupplyJointAssignmentRejections;
        fields["memory_supply_search_work"] = accounting.memorySupplySearchWork;
        fields["partial_cover_expansions"] = accounting.partialCoverExpansions;
        fields["candidate_evaluations"] = accounting.candidateEvaluations;
        fields["candidate_publications"] = outputs.size();
      });
  emitFeedback(feedback);
  if (incompleteReason)
    return CandidateGeneratorProviderResult{
        IncompleteCandidateGeneratorResult{
            *incompleteReason,
            {{CandidateGeneratorOutputSlotRef(0), std::move(outputs)}},
            std::move(lineage)},
        workSummary(accounting), encodeFeedback(feedback)};
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{CandidateGeneratorOutputSlotRef(0), std::move(outputs)}},
          std::move(lineage)},
      workSummary(accounting), encodeFeedback(feedback)};
}

const CandidateGeneratorProvider provider{
    descriptor.reference(),
    CandidateGeneratorInProcessProvider{invokeRootCompleteProvider}};

const CandidateGeneratorProvider applicationGraphProvider{
    applicationGraphDescriptor.reference(),
    CandidateGeneratorInProcessProvider{invokeApplicationGraphProvider}};

} // namespace

const CandidateGeneratorDescriptor &
rootCompleteTechMappingCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerRootCompleteTechMappingCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindRootCompleteTechMappingCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> dataflowCandidates,
    const ArtifactRootReference &fabric) {
  if (llvm::Error error = registerRootCompleteTechMappingCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(DataflowCandidatesInput),
       dataflowCandidates.vec()},
      {CandidateGeneratorInputSlotRef(FabricInput), {fabric}},
  };
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveRootCompleteTechMappingCandidateGeneratorBinding(
    const ::loom::mapping::ResolvedTechMappingConfigView &config) {
  if (llvm::Error error = registerRootCompleteTechMappingCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

const CandidateGeneratorDescriptor &
applicationGraphTechMappingCandidateGeneratorDescriptor() {
  return applicationGraphDescriptor;
}

llvm::Error registerApplicationGraphTechMappingCandidateGenerator() {
  if (llvm::Error error =
          registerCandidateGeneratorDescriptor(applicationGraphDescriptor))
    return error;
  return registerCandidateGeneratorProvider(applicationGraphProvider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindApplicationGraphTechMappingCandidateGeneratorInputs(
    const ArtifactRootReference &dataflow,
    const ArtifactRootReference &systemConstraints,
    const ArtifactRootReference &fabric) {
  if (llvm::Error error =
          registerApplicationGraphTechMappingCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(ApplicationDataflowInput), {dataflow}},
      {CandidateGeneratorInputSlotRef(ApplicationSystemConstraintsInput),
       {systemConstraints}},
      {CandidateGeneratorInputSlotRef(ApplicationFabricInput), {fabric}},
  };
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          applicationGraphDescriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveApplicationGraphTechMappingCandidateGeneratorBinding(
    const ::loom::mapping::ResolvedTechMappingConfigView &config) {
  if (llvm::Error error =
          registerApplicationGraphTechMappingCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      applicationGraphDescriptor.reference(), config.canonicalViewBytes(),
      config.digest());
}

} // namespace loom::dse
