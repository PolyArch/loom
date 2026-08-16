#include "DSE/RootCompleteSpatialPnrCandidateGenerator.h"

#include "DSE/MappingCandidateGenerator.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactText.h"
#include "Common/MappingDebugLog.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "PnR/FabricTopologyQualityDiagnostic.h"
#include "PnR/SpatialPnrGenerator.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace loom::dse {
namespace {

enum InputSlot : std::uint32_t {
  TechMappingCandidatesInput,
  FabricInput,
  PhysicalTimingProfileInput,
  InputSlotCount,
};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots = {{
        {CandidateGeneratorInputSlotRef(TechMappingCandidatesInput),
         "tech_mapping", PlanValueRole::CandidateSet,
         &::loom::mapping::mappingArtifactSchema,
         PlanValueCardinality::FiniteSet},
        {CandidateGeneratorInputSlotRef(FabricInput), "fabric",
         PlanValueRole::CandidateSet, &::loom::fabric::fabricArtifactSchema,
         PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorInputSlotRef(PhysicalTimingProfileInput),
         "physical_timing_profile", PlanValueRole::CandidateSet,
         &::loom::fabric::fabricPhysicalTimingProfileArtifactSchema,
         PlanValueCardinality::ExactlyOne},
    }};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {
    {{CandidateGeneratorOutputSlotRef(0), "spatial_mapping",
      PlanValueRole::CandidateSet, &::loom::mapping::mappingArtifactSchema,
      PlanValueCardinality::FiniteSet}}};

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = ::loom::pnr::adoptResolvedSpatialPnrConfigView(
      ::loom::pnr::resolvedSpatialPnrConfigSchemaDescriptorBytes(), bytes,
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

const CandidateGeneratorDescriptor descriptor{
    rootCompleteSpatialPnrCandidateGeneratorKind,
    "mapping.root_complete_spatial_pnr",
    "loom.mapping.root_complete_spatial_pnr.generator.v14",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{
        ::loom::pnr::resolvedSpatialPnrConfigSchemaDescriptorBytes(),
        validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    pnrCandidateGeneratorWorkUnits,
    nullptr,
    ProviderForm::InProcess,
};

struct CachedDataflow final {
  ::dataflow::CanonicalDataflowArtifact artifact;
  ::dataflow::CanonicalDataflowProgramView view;
};

struct DataflowImportStatistics final {
  std::uint64_t requests = 0;
  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  std::uint64_t constructionNanoseconds = 0;
  std::uint64_t retainedBytes = 0;
  std::uint64_t deterministicWork = 0;
};

void saturatingAdd(std::uint64_t &value, std::uint64_t amount) {
  value = amount > std::numeric_limits<std::uint64_t>::max() - value
              ? std::numeric_limits<std::uint64_t>::max()
              : value + amount;
}

template <typename T>
void accountArray(llvm::ArrayRef<T> values, std::uint64_t &bytes,
                  std::uint64_t &work) {
  const std::uint64_t count = values.size();
  const std::uint64_t elementBytes = sizeof(T);
  saturatingAdd(bytes,
                count > std::numeric_limits<std::uint64_t>::max() / elementBytes
                    ? std::numeric_limits<std::uint64_t>::max()
                    : count * elementBytes);
  saturatingAdd(work, count);
}

void accountRetainedDataflow(const CachedDataflow &cached,
                             DataflowImportStatistics &statistics) {
  std::uint64_t bytes = sizeof(CachedDataflow);
  std::uint64_t work = cached.artifact.canonicalBytes().bytes().size();
  saturatingAdd(bytes, cached.artifact.canonicalBytes().bytes().size());
  accountArray(cached.view.graphs(), bytes, work);
  accountArray(cached.view.actors(), bytes, work);
  accountArray(cached.view.rootThreadLaunches(), bytes, work);
  accountArray(cached.view.staticGraphLaunches(), bytes, work);
  accountArray(cached.view.logicalMemoryRoots(), bytes, work);
  saturatingAdd(statistics.retainedBytes, bytes);
  saturatingAdd(statistics.deterministicWork, work);
}

void emitDataflowImportStatistics(const DataflowImportStatistics &statistics) {
  ::loom::mapping_debug::emit(
      ::loom::mapping_debug::Level::Summary,
      ::loom::mapping_debug::Stage::SpatialPnr,
      ::loom::mapping_debug::Event::DerivedContext,
      [&](llvm::json::Object &fields) {
        fields["context_kind"] = "root_complete_spatial_dataflow_import";
        fields["cache_requests"] = statistics.requests;
        fields["cache_hits"] = statistics.hits;
        fields["cache_misses"] = statistics.misses;
        fields["construction_count"] = statistics.misses;
        fields["construction_time_ns"] = statistics.constructionNanoseconds;
        fields["retained_bytes"] = statistics.retainedBytes;
        fields["deterministic_work"] = statistics.deterministicWork;
      });
}

void canonicalizeReferences(std::vector<ArtifactRootReference> &references) {
  llvm::sort(references, artifactRootReferenceLess);
  references.erase(std::unique(references.begin(), references.end()),
                   references.end());
}

std::vector<CandidateGeneratorLineageEdge>
mechanicalLineage(llvm::ArrayRef<ArtifactRootReference> outputs) {
  std::vector<CandidateGeneratorLineageEdge> lineage;
  lineage.reserve(outputs.size());
  for (const ArtifactRootReference &output : outputs)
    lineage.push_back(CandidateGeneratorLineageEdge{
        CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
        CandidateGeneratorOutputSlotRef(0),
        output,
        {},
        {}});
  return lineage;
}

CompletedCandidateGeneratorResult
completed(std::vector<ArtifactRootReference> outputs) {
  canonicalizeReferences(outputs);
  auto lineage = mechanicalLineage(outputs);
  std::vector<CandidateGeneratorOutputBinding> bindings = {
      {CandidateGeneratorOutputSlotRef(0), std::move(outputs)}};
  return {std::move(bindings), std::move(lineage)};
}

IncompleteCandidateGeneratorResult
incomplete(CandidateGeneratorIncompleteReason reason,
           std::vector<ArtifactRootReference> outputs) {
  canonicalizeReferences(outputs);
  auto lineage = mechanicalLineage(outputs);
  std::vector<CandidateGeneratorOutputBinding> bindings = {
      {CandidateGeneratorOutputSlotRef(0), std::move(outputs)}};
  return {reason, std::move(bindings), std::move(lineage)};
}

llvm::Error
accumulateWorkSummary(llvm::ArrayRef<CandidateGeneratorWorkUnitSummary> source,
                      std::vector<CandidateGeneratorWorkUnitSummary> &target) {
  if (source.size() != target.size())
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "root-complete Spatial PnR work summaries have different widths");
  for (std::size_t ordinal = 0; ordinal != source.size(); ++ordinal) {
    if (!(source[ordinal].unit == target[ordinal].unit))
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "root-complete Spatial PnR work summaries have different units");
    if (source[ordinal].planned > std::numeric_limits<std::uint64_t>::max() -
                                      target[ordinal].planned ||
        source[ordinal].consumed > std::numeric_limits<std::uint64_t>::max() -
                                       target[ordinal].consumed)
      return llvm::createStringError(
          std::make_error_code(std::errc::value_too_large),
          "root-complete Spatial PnR work summary overflows u64");
    target[ordinal].planned += source[ordinal].planned;
    target[ordinal].consumed += source[ordinal].consumed;
  }
  return llvm::Error::success();
}

llvm::Expected<CandidateGeneratorProviderResult> invokeRootCompleteProvider(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store, const BlobStore &blobs,
    const CandidateGeneratorInvocationView &invocation) {
  auto config = ::loom::pnr::adoptResolvedSpatialPnrConfigView(
      ::loom::pnr::resolvedSpatialPnrConfigSchemaDescriptorBytes(),
      binding.canonicalConfigBytes(), binding.configDigest());
  if (!config)
    return config.takeError();

  auto fabric = ::loom::fabric::importEntireFabricRoot(
      inputBindings[FabricInput].artifacts.front(), store);
  if (!fabric)
    return fabric.takeError();
  auto physicalTiming = ::loom::fabric::importFabricPhysicalTimingProfile(
      inputBindings[PhysicalTimingProfileInput].artifacts.front(),
      fabric->view(), store);
  if (!physicalTiming)
    return physicalTiming.takeError();
  auto derivedContexts = ::loom::pnr::buildFabricDerivedContextBundle(
      fabric->view(), *physicalTiming);
  if (!derivedContexts)
    return derivedContexts.takeError();
  ::loom::pnr::emitFabricDerivedContextStatistics(
      *derivedContexts, ::loom::mapping_debug::Stage::SpatialPnr, 0, 1, 0, 1);
  auto topology =
      ::loom::pnr::analyzeFabricTopologyQualityForDiagnostics(fabric->view());
  if (!topology)
    return topology.takeError();

  std::map<ArtifactRootReference, std::unique_ptr<CachedDataflow>,
           decltype(&artifactRootReferenceLess)>
      dataflowCache(&artifactRootReferenceLess);
  DataflowImportStatistics dataflowImportStatistics;
  auto emitDataflowImportStatisticsOnExit = llvm::scope_exit(
      [&] { emitDataflowImportStatistics(dataflowImportStatistics); });
  std::vector<ArtifactRootReference> outputs;
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
  std::vector<CandidateGeneratorWorkUnitSummary> workSummary =
      spatialPnrCandidateGeneratorWorkSummary({});
  const std::optional<std::uint64_t> maximumOutputs =
      invocation.maximumOutputArtifacts(CandidateGeneratorOutputSlotRef(0));
  for (const auto indexedTech :
       llvm::enumerate(inputBindings[TechMappingCandidatesInput].artifacts)) {
    const std::size_t techOrdinal = indexedTech.index();
    const ArtifactRootReference &techReference = indexedTech.value();
    auto tech = ::loom::mapping::importTechMapping(techReference, store);
    if (!tech)
      return tech.takeError();
    if (tech->view().fabricIdentity() != fabric->view().identity())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "root_complete_spatial_pnr_generator_invalid: TechMapping binds "
          "a foreign Fabric");

    ArtifactRootReference dataflowReference{
        ::dataflow::canonicalDataflowSchema.identity.str(),
        ::dataflow::canonicalDataflowSchema.version,
        tech->view().dataflowIdentity()};
    ++dataflowImportStatistics.requests;
    auto cached = dataflowCache.find(dataflowReference);
    if (cached == dataflowCache.end()) {
      const auto constructionBegin = std::chrono::steady_clock::now();
      auto artifact =
          ::dataflow::importCanonicalDataflow(dataflowReference, store);
      if (!artifact)
        return artifact.takeError();
      auto view = artifact->view();
      if (!view)
        return view.takeError();
      cached = dataflowCache
                   .emplace(dataflowReference,
                            std::make_unique<CachedDataflow>(CachedDataflow{
                                std::move(*artifact), std::move(*view)}))
                   .first;
      ++dataflowImportStatistics.misses;
      saturatingAdd(
          dataflowImportStatistics.constructionNanoseconds,
          static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - constructionBegin)
                  .count()));
      accountRetainedDataflow(*cached->second, dataflowImportStatistics);
    } else {
      ++dataflowImportStatistics.hits;
    }
    if (cached->second->artifact.identity() != dataflowReference.artifact ||
        cached->second->view.identity() != dataflowReference.artifact)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "root_complete_spatial_pnr_generator_invalid: cached Dataflow "
          "identity does not match the exact TechMapping lineage");
    const ::dataflow::CanonicalDataflowProgramView &dataflow =
        cached->second->view;

    auto constraints =
        ::loom::mapping::finalizeEmptySpatialMappingConstraintSet(
            dataflow, tech->view(), fabric->view(), store);
    if (!constraints)
      return constraints.takeError();

    ::loom::pnr::SpatialPnrGenerationOutcome outcome =
        ::loom::pnr::generateSpatialMappings(
            {dataflow, tech->view(), fabric->view(), *physicalTiming, *config,
             constraints->view(), store, defaultCandidateWorkerCount(),
             invocation.executionControl(), &*derivedContexts,
             *topology ? &**topology : nullptr});
    const auto invocationWorkSummary = std::visit(
        [](const auto &value) {
          return spatialPnrCandidateGeneratorWorkSummary(value.accounting);
        },
        outcome);
    if (llvm::Error error =
            accumulateWorkSummary(invocationWorkSummary, workSummary))
      return std::move(error);
    if (auto *generated =
            std::get_if<::loom::pnr::GeneratedSpatialMappings>(&outcome)) {
      if (auto reason = pnrGenerationIncompleteReason(generated->termination))
        rememberIncomplete(*reason);
      outputs.insert(outputs.end(),
                     std::make_move_iterator(generated->candidates.begin()),
                     std::make_move_iterator(generated->candidates.end()));
      canonicalizeReferences(outputs);
      if (maximumOutputs && outputs.size() >= *maximumOutputs) {
        const bool droppedOutputs = outputs.size() > *maximumOutputs;
        if (droppedOutputs)
          outputs.erase(outputs.begin() +
                            static_cast<std::size_t>(*maximumOutputs),
                        outputs.end());
        const bool skippedTechMappings =
            techOrdinal + 1 !=
            inputBindings[TechMappingCandidatesInput].artifacts.size();
        if (droppedOutputs || skippedTechMappings)
          rememberIncomplete(
              CandidateGeneratorIncompleteReason::SemanticLimitReached);
        if (droppedOutputs || skippedTechMappings)
          break;
      }
      continue;
    }
    if (const auto *infeasible =
            std::get_if<::loom::pnr::ProvenInfeasibleSpatialMapping>(
                &outcome)) {
      ::loom::mapping_debug::emit(
          ::loom::mapping_debug::Level::Summary,
          ::loom::mapping_debug::Stage::SpatialPnr,
          ::loom::mapping_debug::Event::MappingFailure,
          [&](llvm::json::Object &fields) {
            fields["failure_scope"] = "invocation";
            fields["closure_status"] = "proven_infeasible";
            fields["tech_mapping_ordinal"] =
                static_cast<std::uint64_t>(techOrdinal);
            fields["tech_mapping"] =
                formatArtifactIdentityHex(techReference.artifact);
            fields["diagnostic"] = infeasible->diagnostic;
          });
      continue;
    }
    if (const auto *partial =
            std::get_if<::loom::pnr::IncompleteSpatialPnrGeneration>(
                &outcome)) {
      const CandidateGeneratorIncompleteReason reason =
          partial->reason == ::loom::pnr::IncompleteSpatialPnrGenerationReason::
                                 SemanticLimitReached
              ? CandidateGeneratorIncompleteReason::SemanticLimitReached
              : CandidateGeneratorIncompleteReason::ProofNotEstablished;
      ::loom::mapping_debug::emit(
          ::loom::mapping_debug::Level::Summary,
          ::loom::mapping_debug::Stage::SpatialPnr,
          ::loom::mapping_debug::Event::MappingFailure,
          [&](llvm::json::Object &fields) {
            fields["failure_scope"] = "invocation";
            fields["closure_status"] =
                partial->reason ==
                        ::loom::pnr::IncompleteSpatialPnrGenerationReason::
                            SemanticLimitReached
                    ? "semantic_limit_reached"
                    : "proof_not_established";
            fields["tech_mapping_ordinal"] =
                static_cast<std::uint64_t>(techOrdinal);
            fields["tech_mapping"] =
                formatArtifactIdentityHex(techReference.artifact);
            fields["diagnostic"] = partial->diagnostic;
            fields["seed_attempts"] = partial->accounting.seedAttemptSlots;
            fields["prepared_seeds"] = partial->accounting.preparedSeeds;
            fields["exact_repair_invocations"] =
                partial->accounting.exactRepairInvocations;
            fields["exact_repair_region_decisions"] =
                partial->accounting.exactRepairRegionDecisions;
            fields["exact_repair_solver_calls"] =
                partial->accounting.exactRepairSolverCalls;
            fields["final_closure_attempts"] =
                partial->accounting.finalClosureAttempts;
          });
      rememberIncomplete(reason);
      continue;
    }
    if (auto *interrupted =
            std::get_if<::loom::pnr::InterruptedSpatialPnrGeneration>(
                &outcome)) {
      outputs.insert(outputs.end(),
                     std::make_move_iterator(interrupted->candidates.begin()),
                     std::make_move_iterator(interrupted->candidates.end()));
      rememberIncomplete(
          CandidateGeneratorIncompleteReason::CancelledOrTimeout);
      break;
    }
    if (std::holds_alternative<::loom::pnr::UnsupportedSpatialPnrGeneration>(
            outcome))
      return CandidateGeneratorProviderResult{
          incomplete(CandidateGeneratorIncompleteReason::Unsupported,
                     std::move(outputs)),
          std::move(workSummary)};
    if (const auto *invalid =
            std::get_if<::loom::pnr::InvalidSpatialPnrGeneration>(&outcome))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "root_complete_spatial_pnr_generator_invalid: " +
              invalid->diagnostic);
    const auto &internal =
        std::get<::loom::pnr::InternalSpatialPnrGeneration>(outcome);
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "root_complete_spatial_pnr_generator_execution_failed: " +
            internal.diagnostic);
  }

  if (incompleteReason)
    return CandidateGeneratorProviderResult{
        incomplete(*incompleteReason, std::move(outputs)),
        std::move(workSummary)};
  return CandidateGeneratorProviderResult{completed(std::move(outputs)),
                                          std::move(workSummary)};
}

const CandidateGeneratorProvider provider{
    descriptor.reference(),
    CandidateGeneratorInProcessProvider{invokeRootCompleteProvider}};

} // namespace

const CandidateGeneratorDescriptor &
rootCompleteSpatialPnrCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerRootCompleteSpatialPnrCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindRootCompleteSpatialPnrCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> techMappingCandidates,
    const ArtifactRootReference &fabric,
    const ArtifactRootReference &physicalTimingProfile) {
  if (llvm::Error error = registerRootCompleteSpatialPnrCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(TechMappingCandidatesInput),
       techMappingCandidates.vec()},
      {CandidateGeneratorInputSlotRef(FabricInput), {fabric}},
      {CandidateGeneratorInputSlotRef(PhysicalTimingProfileInput),
       {physicalTimingProfile}},
  };
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveRootCompleteSpatialPnrCandidateGeneratorBinding(
    const ::loom::pnr::ResolvedPnrConfigView &config) {
  if (llvm::Error error = registerRootCompleteSpatialPnrCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

} // namespace loom::dse
