#include "DSE/RootCompleteTechMappingCandidateGenerator.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/Tech/TechMappingGenerator.h"

#include "llvm/ADT/STLExtras.h"

#include <array>
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
    "loom.mapping.root_complete_tech_mapping.generator.v4",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{
        ::loom::mapping::resolvedTechMappingConfigSchemaDescriptorBytes(),
        validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    nullptr,
    ProviderForm::InProcess,
};

const CandidateGeneratorDescriptor applicationGraphDescriptor{
    applicationGraphTechMappingCandidateGeneratorKind,
    "mapping.application_graph_tech_mapping",
    "loom.mapping.application_graph_tech_mapping.generator.v5",
    applicationInputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{
        ::loom::mapping::resolvedTechMappingConfigSchemaDescriptorBytes(),
        validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    nullptr,
    ProviderForm::InProcess,
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
  if (llvm::Error error =
          accumulate(source.partialCoverExpansions,
                     target.partialCoverExpansions, "partial-cover expansion"))
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

llvm::Expected<std::optional<CandidateGeneratorIncompleteReason>>
consumeTechMappingOutcome(
    ::loom::mapping::TechMappingGenerationOutcome outcome,
    ::loom::mapping::TechMappingGenerationAccounting &accounting,
    std::vector<ArtifactRootReference> &outputs,
    std::vector<CandidateGeneratorLineageEdge> &lineage) {
  const auto &currentAccounting = std::visit(
      [](const auto &result)
          -> const ::loom::mapping::TechMappingGenerationAccounting & {
        return result.accounting;
      },
      outcome);
  if (llvm::Error error = accumulate(currentAccounting, accounting))
    return std::move(error);
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
               const ExecutionControlView &executionControl) {
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
  for (const ArtifactRootReference &dataflowReference :
       inputBindings[DataflowCandidatesInput].artifacts) {
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
    auto incomplete =
        consumeTechMappingOutcome(::loom::mapping::generateTechMappings(
                                      {*dataflow, completeCover, fabric->view(),
                                       *config, store, executionControl}),
                                  accounting, outputs, lineage);
    if (!incomplete)
      return incomplete.takeError();
    if (*incomplete)
      rememberIncomplete(**incomplete);
    if (incompleteReason ==
        CandidateGeneratorIncompleteReason::CancelledOrTimeout)
      break;
  }

  if (incompleteReason)
    return CandidateGeneratorProviderResult{
        IncompleteCandidateGeneratorResult{
            *incompleteReason,
            {{CandidateGeneratorOutputSlotRef(0), std::move(outputs)}},
            std::move(lineage)},
        workSummary(accounting)};
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{CandidateGeneratorOutputSlotRef(0), std::move(outputs)}},
          std::move(lineage)},
      workSummary(accounting)};
}

llvm::Expected<CandidateGeneratorProviderResult> invokeRootCompleteProvider(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store, const BlobStore &,
    const CandidateGeneratorInvocationView &invocation) {
  return invokeProvider(inputBindings, binding, store,
                        invocation.executionControl());
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
  for (const ::dataflow::GraphRef &graph : graphs) {
    const std::array cover = {graph};
    auto incomplete = consumeTechMappingOutcome(
        ::loom::mapping::generateTechMappings({*dataflow, cover, fabric->view(),
                                               *config, store,
                                               invocation.executionControl()}),
        accounting, outputs, lineage);
    if (!incomplete)
      return incomplete.takeError();
    if (*incomplete)
      rememberIncomplete(**incomplete);
    if (incompleteReason ==
        CandidateGeneratorIncompleteReason::CancelledOrTimeout)
      break;
  }
  if (incompleteReason)
    return CandidateGeneratorProviderResult{
        IncompleteCandidateGeneratorResult{
            *incompleteReason,
            {{CandidateGeneratorOutputSlotRef(0), std::move(outputs)}},
            std::move(lineage)},
        workSummary(accounting)};
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{CandidateGeneratorOutputSlotRef(0), std::move(outputs)}},
          std::move(lineage)},
      workSummary(accounting)};
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
