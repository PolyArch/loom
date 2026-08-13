#include "DSE/RootCompleteSpatialPnrCandidateGenerator.h"

#include "DSE/MappingCandidateGenerator.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactText.h"
#include "Common/MappingDebugLog.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "PnR/SpatialPnrGenerator.h"

#include "llvm/ADT/STLExtras.h"

#include <array>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace loom::dse {
namespace {

enum InputSlot : std::uint32_t {
  TechMappingCandidatesInput,
  FabricInput,
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
    const ArtifactStore &store, const BlobStore &blobs);

const CandidateGeneratorDescriptor descriptor{
    rootCompleteSpatialPnrCandidateGeneratorKind,
    "mapping.root_complete_spatial_pnr",
    "loom.mapping.root_complete_spatial_pnr.generator.v6",
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
    const ArtifactStore &store, const BlobStore &blobs) {
  auto config = ::loom::pnr::adoptResolvedSpatialPnrConfigView(
      ::loom::pnr::resolvedSpatialPnrConfigSchemaDescriptorBytes(),
      binding.canonicalConfigBytes(), binding.configDigest());
  if (!config)
    return config.takeError();

  auto fabric = ::loom::fabric::importEntireFabricRoot(
      inputBindings[FabricInput].artifacts.front(), store);
  if (!fabric)
    return fabric.takeError();

  std::map<ArtifactIdentity::Storage, std::unique_ptr<CachedDataflow>>
      dataflowCache;
  std::vector<ArtifactRootReference> outputs;
  std::vector<CandidateGeneratorWorkUnitSummary> workSummary =
      spatialPnrCandidateGeneratorWorkSummary({});
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

    const ArtifactIdentity::Storage dataflowKey =
        tech->view().dataflowIdentity().bytes();
    auto cached = dataflowCache.find(dataflowKey);
    if (cached == dataflowCache.end()) {
      ArtifactRootReference dataflowReference{
          ::dataflow::canonicalDataflowSchema.identity.str(),
          ::dataflow::canonicalDataflowSchema.version,
          tech->view().dataflowIdentity()};
      auto artifact =
          ::dataflow::importCanonicalDataflow(dataflowReference, store);
      if (!artifact)
        return artifact.takeError();
      auto view = artifact->view();
      if (!view)
        return view.takeError();
      cached = dataflowCache
                   .emplace(dataflowKey,
                            std::make_unique<CachedDataflow>(CachedDataflow{
                                std::move(*artifact), std::move(*view)}))
                   .first;
    }
    const ::dataflow::CanonicalDataflowProgramView &dataflow =
        cached->second->view;

    auto constraints =
        ::loom::mapping::finalizeEmptySpatialMappingConstraintSet(
            dataflow, tech->view(), fabric->view(), store);
    if (!constraints)
      return constraints.takeError();

    ::loom::pnr::SpatialPnrGenerationOutcome outcome =
        ::loom::pnr::generateSpatialMappings(
            {dataflow, tech->view(), fabric->view(), *config,
             constraints->view(), store, defaultCandidateWorkerCount()});
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
      outputs.insert(outputs.end(),
                     std::make_move_iterator(generated->candidates.begin()),
                     std::make_move_iterator(generated->candidates.end()));
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
      return CandidateGeneratorProviderResult{
          incomplete(reason, std::move(outputs)), std::move(workSummary)};
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
    const ArtifactRootReference &fabric) {
  if (llvm::Error error = registerRootCompleteSpatialPnrCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(TechMappingCandidatesInput),
       techMappingCandidates.vec()},
      {CandidateGeneratorInputSlotRef(FabricInput), {fabric}},
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
