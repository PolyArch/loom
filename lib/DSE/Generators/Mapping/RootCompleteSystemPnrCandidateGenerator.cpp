#include "DSE/RootCompleteSystemPnrCandidateGenerator.h"

#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "PnR/System/SystemPnrSearchDomain.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace loom::dse {
namespace {

enum InputSlot : std::uint32_t {
  DataflowInput,
  SpatialMappingCandidatesInput,
  FabricInput,
  InputSlotCount,
};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots = {{
        {CandidateGeneratorInputSlotRef(DataflowInput), "dataflow",
         PlanValueRole::CandidateSet, &::dataflow::canonicalDataflowSchema,
         PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorInputSlotRef(SpatialMappingCandidatesInput),
         "spatial_mapping", PlanValueRole::CandidateSet,
         &::loom::mapping::mappingArtifactSchema,
         PlanValueCardinality::FiniteSet},
        {CandidateGeneratorInputSlotRef(FabricInput), "fabric",
         PlanValueRole::CandidateSet, &::loom::fabric::fabricArtifactSchema,
         PlanValueCardinality::ExactlyOne},
    }};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {
    {{CandidateGeneratorOutputSlotRef(0), "system_mapping",
      PlanValueRole::CandidateSet, &::loom::mapping::mappingArtifactSchema,
      PlanValueCardinality::FiniteSet}}};

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = ::loom::pnr::adoptResolvedSystemPnrConfigView(
      ::loom::pnr::resolvedSystemPnrConfigSchemaDescriptorBytes(), bytes,
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
    rootCompleteSystemPnrCandidateGeneratorKind,
    "mapping.root_complete_system_pnr",
    "loom.mapping.root_complete_system_pnr.generator.v1",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{
        ::loom::pnr::resolvedSystemPnrConfigSchemaDescriptorBytes(),
        validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    rootCompleteSystemPnrCandidateGeneratorWorkUnits,
    nullptr,
    ProviderForm::InProcess,
};

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
  llvm::sort(outputs, artifactRootReferenceLess);
  outputs.erase(std::unique(outputs.begin(), outputs.end()), outputs.end());
  auto lineage = mechanicalLineage(outputs);
  return {{{CandidateGeneratorOutputSlotRef(0), std::move(outputs)}},
          std::move(lineage)};
}

IncompleteCandidateGeneratorResult
incomplete(CandidateGeneratorIncompleteReason reason) {
  return {reason, {{CandidateGeneratorOutputSlotRef(0), {}}}, {}};
}

llvm::Error validateSpatialMappingOwners(
    llvm::ArrayRef<ArtifactRootReference> spatialMappingReferences,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &system,
    const ArtifactStore &store) {
  for (const ArtifactRootReference &reference : spatialMappingReferences) {
    auto spatial = ::loom::mapping::importSpatialMapping(reference, store);
    if (!spatial)
      return spatial.takeError();
    if (spatial->view().dataflowIdentity() != dataflow.identity())
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "root-complete System SpatialMapping has a foreign Dataflow owner");
    const bool attached = llvm::any_of(
        system.artifact().accCoreOccurrences(), [&](const auto core) {
          const auto target = system.spatialCoreTarget(core);
          return target &&
                 target->dependencyOrdinal <
                     system.artifact().importedModules().size() &&
                 system.artifact()
                         .importedModules()[target->dependencyOrdinal]
                         .identity() == spatial->view().fabricIdentity();
        });
    if (!attached)
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "root-complete System SpatialMapping Fabric is not attached to "
          "the System");
  }
  return llvm::Error::success();
}

llvm::Expected<CandidateGeneratorProviderResult> invokeRootCompleteProvider(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store, const BlobStore &blobs) {
  (void)blobs;
  auto config = ::loom::pnr::adoptResolvedSystemPnrConfigView(
      ::loom::pnr::resolvedSystemPnrConfigSchemaDescriptorBytes(),
      binding.canonicalConfigBytes(), binding.configDigest());
  if (!config)
    return config.takeError();

  auto dataflowArtifact = ::dataflow::importCanonicalDataflow(
      inputBindings[DataflowInput].artifacts.front(), store);
  if (!dataflowArtifact)
    return dataflowArtifact.takeError();
  auto dataflow = dataflowArtifact->view();
  if (!dataflow)
    return dataflow.takeError();
  auto fabricArtifact = ::loom::fabric::importEntireFabricRoot(
      inputBindings[FabricInput].artifacts.front(), store);
  if (!fabricArtifact)
    return fabricArtifact.takeError();
  auto system = ::loom::fabric::requireSystemRoot(fabricArtifact->view());
  if (!system)
    return system.takeError();

  if (llvm::Error error = validateSpatialMappingOwners(
          inputBindings[SpatialMappingCandidatesInput].artifacts, *dataflow,
          *system, store))
    return std::move(error);

  std::vector<::dataflow::RootThreadLaunchRef> roots;
  roots.reserve(dataflow->rootThreadLaunches().size());
  for (const auto &root : dataflow->rootThreadLaunches())
    roots.push_back(root.ref);
  if (roots.empty())
    return CandidateGeneratorProviderResult{
        completed({}), rootCompleteSystemPnrCandidateGeneratorWorkSummary({})};

  auto constraints = ::loom::mapping::finalizeEmptySystemMappingConstraintSet(
      *dataflow, *system, roots, store);
  if (!constraints)
    return constraints.takeError();
  auto partition = ::loom::pnr::projectWholeDomainPresburgerPartitionPlan(
      *dataflow, constraints->view().rootThreadLaunches());
  if (!partition)
    return partition.takeError();
  ::loom::pnr::SystemHierarchicalGraphSearchInput graphSearch{
      inputBindings[SpatialMappingCandidatesInput].artifacts};
  auto searchDomain = ::loom::pnr::projectSystemPnrSearchDomain(
      *dataflow, *system, *config, *constraints, *partition, graphSearch,
      store);
  if (!searchDomain) {
    bool unsupported = false;
    llvm::Error remaining = llvm::handleErrors(
        searchDomain.takeError(),
        [&](const ::loom::pnr::UnsupportedSystemPnrSearchDomain &) {
          unsupported = true;
        });
    if (remaining)
      return std::move(remaining);
    if (unsupported)
      return CandidateGeneratorProviderResult{
          incomplete(CandidateGeneratorIncompleteReason::Unsupported),
          rootCompleteSystemPnrCandidateGeneratorWorkSummary({})};
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "root-complete System H projection lost its failure cause");
  }

  ::loom::pnr::SystemPnrGenerationOutcome outcome =
      ::loom::pnr::generateSystemMappings(
          {*dataflow, *system, *searchDomain, *config, *constraints, store});
  if (auto *generated =
          std::get_if<::loom::pnr::GeneratedSystemMappings>(&outcome))
    return CandidateGeneratorProviderResult{
        completed(std::move(generated->candidates)),
        rootCompleteSystemPnrCandidateGeneratorWorkSummary(
            generated->accounting)};
  if (const auto *infeasible =
          std::get_if<::loom::pnr::ProvenInfeasibleSystemMapping>(&outcome))
    return CandidateGeneratorProviderResult{
        completed({}), rootCompleteSystemPnrCandidateGeneratorWorkSummary(
                           infeasible->accounting)};
  if (const auto *partial =
          std::get_if<::loom::pnr::IncompleteSystemPnrGeneration>(&outcome)) {
    const CandidateGeneratorIncompleteReason reason =
        partial->reason == ::loom::pnr::IncompleteSystemPnrGenerationReason::
                               SemanticLimitReached
            ? CandidateGeneratorIncompleteReason::SemanticLimitReached
            : CandidateGeneratorIncompleteReason::ProofNotEstablished;
    return CandidateGeneratorProviderResult{
        incomplete(reason), rootCompleteSystemPnrCandidateGeneratorWorkSummary(
                                partial->accounting)};
  }
  if (const auto *invalid =
          std::get_if<::loom::pnr::InvalidSystemPnrGeneration>(&outcome))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "root_complete_system_pnr_generator_invalid: " + invalid->diagnostic);
  const auto &internal =
      std::get<::loom::pnr::InternalSystemPnrGeneration>(outcome);
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "root_complete_system_pnr_generator_execution_failed: " +
          internal.diagnostic);
}

const CandidateGeneratorProvider provider{
    descriptor.reference(),
    CandidateGeneratorInProcessProvider{invokeRootCompleteProvider}};

} // namespace

const CandidateGeneratorDescriptor &
rootCompleteSystemPnrCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerRootCompleteSystemPnrCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindRootCompleteSystemPnrCandidateGeneratorInputs(
    const ArtifactRootReference &dataflow,
    llvm::ArrayRef<ArtifactRootReference> spatialMappingCandidates,
    const ArtifactRootReference &fabric) {
  if (llvm::Error error = registerRootCompleteSystemPnrCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(DataflowInput), {dataflow}},
      {CandidateGeneratorInputSlotRef(SpatialMappingCandidatesInput),
       spatialMappingCandidates.vec()},
      {CandidateGeneratorInputSlotRef(FabricInput), {fabric}},
  };
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveRootCompleteSystemPnrCandidateGeneratorBinding(
    const ::loom::pnr::ResolvedPnrConfigView &config) {
  if (llvm::Error error = registerRootCompleteSystemPnrCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

std::vector<CandidateGeneratorWorkUnitSummary>
rootCompleteSystemPnrCandidateGeneratorWorkSummary(
    const ::loom::pnr::SystemPnrGenerationAccounting &accounting) {
  const std::array<std::uint64_t, 2> consumed = {
      accounting.initializerAssignmentAttempts,
      accounting.endpointExpansionSlots};
  std::vector<CandidateGeneratorWorkUnitSummary> result;
  result.reserve(consumed.size());
  for (std::size_t ordinal = 0; ordinal != consumed.size(); ++ordinal)
    result.push_back({CandidateGeneratorWorkUnitRef(ordinal), consumed[ordinal],
                      consumed[ordinal]});
  return result;
}

} // namespace loom::dse
