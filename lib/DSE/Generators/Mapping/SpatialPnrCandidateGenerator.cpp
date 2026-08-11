#include "DSE/MappingCandidateGenerator.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "PnR/PnrConfig.h"

#include "llvm/ADT/ArrayRef.h"

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

enum InputSlot : std::uint32_t {
  DataflowInput,
  TechMappingInput,
  FabricInput,
  ConstraintInput,
  InputSlotCount,
};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots = {
        {{CandidateGeneratorInputSlotRef(DataflowInput), "dataflow",
          PlanValueRole::CandidateSet, &::dataflow::canonicalDataflowSchema,
          PlanValueCardinality::ExactlyOne},
         {CandidateGeneratorInputSlotRef(TechMappingInput), "tech_mapping",
          PlanValueRole::CandidateSet, &::loom::mapping::mappingArtifactSchema,
          PlanValueCardinality::ExactlyOne},
         {CandidateGeneratorInputSlotRef(FabricInput), "fabric",
          PlanValueRole::CandidateSet, &::loom::fabric::fabricArtifactSchema,
          PlanValueCardinality::ExactlyOne},
         {CandidateGeneratorInputSlotRef(ConstraintInput),
          "spatial_constraints", PlanValueRole::CandidateSet,
          &::loom::mapping::mappingConstraintSetSchema,
          PlanValueCardinality::ExactlyOne}}};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {
    {{CandidateGeneratorOutputSlotRef(0), "spatial_mapping",
      PlanValueRole::CandidateSet, &::loom::mapping::mappingArtifactSchema,
      PlanValueCardinality::FiniteSet}}};

llvm::Error validateSpatialConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                  const ComponentViewDigest &digest) {
  auto adopted = ::loom::pnr::adoptResolvedSpatialPnrConfigView(
      ::loom::pnr::resolvedSpatialPnrConfigSchemaDescriptorBytes(), bytes,
      digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

const CandidateGeneratorDescriptor descriptor{
    spatialPnrCandidateGeneratorKind,
    "mapping.spatial_pnr",
    "loom.mapping.spatial_pnr.generator.v1",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{
        ::loom::pnr::resolvedSpatialPnrConfigSchemaDescriptorBytes(),
        validateSpatialConfig},
    CandidateGeneratorDeterminism::Deterministic,
    pnrCandidateGeneratorWorkUnits,
    nullptr,
    ProviderForm::InProcess,
};

llvm::Expected<CandidateGeneratorProviderResult>
invokeSpatialProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
                      const ResolvedCandidateGeneratorBinding &binding,
                      const ArtifactStore &store, const BlobStore &blobs) {
  ::loom::pnr::SpatialPnrGenerationOutcome outcome =
      invokeSpatialPnrCandidateGenerator(inputs, binding, store,
                                         defaultCandidateWorkerCount());
  if (auto *generated =
          std::get_if<::loom::pnr::GeneratedSpatialMappings>(&outcome)) {
    std::vector<CandidateGeneratorLineageEdge> lineageEdges;
    lineageEdges.reserve(generated->candidates.size());
    for (const ArtifactRootReference &candidate : generated->candidates)
      lineageEdges.push_back(CandidateGeneratorLineageEdge{
          CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
          CandidateGeneratorOutputSlotRef(0),
          candidate,
          {},
          {}});
    return CandidateGeneratorProviderResult{
        CompletedCandidateGeneratorResult{{{CandidateGeneratorOutputSlotRef(0),
                                            std::move(generated->candidates)}},
                                          std::move(lineageEdges)},
        spatialPnrCandidateGeneratorWorkSummary(generated->accounting)};
  }
  if (const auto *infeasible =
          std::get_if<::loom::pnr::ProvenInfeasibleSpatialMapping>(&outcome))
    return CandidateGeneratorProviderResult{
        CompletedCandidateGeneratorResult{
            {{CandidateGeneratorOutputSlotRef(0), {}}}, {}},
        spatialPnrCandidateGeneratorWorkSummary(infeasible->accounting)};
  if (const auto *incomplete =
          std::get_if<::loom::pnr::IncompleteSpatialPnrGeneration>(&outcome)) {
    const CandidateGeneratorIncompleteReason reason =
        incomplete->reason ==
                ::loom::pnr::IncompleteSpatialPnrGenerationReason::
                    SemanticLimitReached
            ? CandidateGeneratorIncompleteReason::SemanticLimitReached
            : CandidateGeneratorIncompleteReason::ProofNotEstablished;
    return CandidateGeneratorProviderResult{
        IncompleteCandidateGeneratorResult{
            reason, {{CandidateGeneratorOutputSlotRef(0), {}}}, {}},
        spatialPnrCandidateGeneratorWorkSummary(incomplete->accounting)};
  }
  if (const auto *unsupported =
          std::get_if<::loom::pnr::UnsupportedSpatialPnrGeneration>(&outcome))
    return CandidateGeneratorProviderResult{
        IncompleteCandidateGeneratorResult{
            CandidateGeneratorIncompleteReason::Unsupported,
            {{CandidateGeneratorOutputSlotRef(0), {}}},
            {}},
        spatialPnrCandidateGeneratorWorkSummary(unsupported->accounting)};
  if (const auto *invalid =
          std::get_if<::loom::pnr::InvalidSpatialPnrGeneration>(&outcome))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "spatial_pnr_generator_invalid: " +
                                       invalid->diagnostic);
  const auto &internal =
      std::get<::loom::pnr::InternalSpatialPnrGeneration>(outcome);
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "spatial_pnr_generator_execution_failed: " +
                                     internal.diagnostic);
}

const CandidateGeneratorProvider provider{
    descriptor.reference(),
    CandidateGeneratorInProcessProvider{invokeSpatialProvider}};

::loom::pnr::SpatialPnrGenerationOutcome invalidOutcome(std::string message) {
  return ::loom::pnr::InvalidSpatialPnrGeneration{
      ::loom::pnr::InvalidSpatialPnrGenerationReason::FrozenInput,
      {},
      std::move(message)};
}

const ArtifactRootReference &
singleInput(llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
            InputSlot slot) {
  return inputBindings[slot].artifacts.front();
}

} // namespace

const CandidateGeneratorDescriptor &spatialPnrCandidateGeneratorDescriptor() {
  return descriptor;
}

std::vector<CandidateGeneratorWorkUnitSummary>
spatialPnrCandidateGeneratorWorkSummary(
    const ::loom::pnr::SpatialPnrGenerationAccounting &accounting) {
  const std::array<std::uint64_t, pnrCandidateGeneratorWorkUnits.size()>
      consumed = {
          accounting.seedAttemptSlots,
          accounting.initializerAssignmentAttempts,
          accounting.endpointExpansionSlots,
          accounting.negotiationIterationSlots,
          accounting.calibrationProposalSlots,
          accounting.annealingBaseProposalSlots,
          accounting.annealingMovableProposalSlots,
          accounting.focusedClosureProposalSlots,
          accounting.exactRepairRegionDecisions,
          accounting.exactRepairSolverCalls,
      };
  std::vector<CandidateGeneratorWorkUnitSummary> summary;
  summary.reserve(consumed.size());
  for (std::size_t ordinal = 0; ordinal != consumed.size(); ++ordinal)
    summary.push_back({CandidateGeneratorWorkUnitRef(ordinal),
                       consumed[ordinal], consumed[ordinal]});
  return summary;
}

llvm::Error registerSpatialPnrCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindSpatialPnrCandidateGeneratorInputs(
    const ArtifactRootReference &dataflow,
    const ArtifactRootReference &techMapping,
    const ArtifactRootReference &fabric,
    const ArtifactRootReference &constraints) {
  if (llvm::Error error = registerSpatialPnrCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(DataflowInput), {dataflow}},
      {CandidateGeneratorInputSlotRef(TechMappingInput), {techMapping}},
      {CandidateGeneratorInputSlotRef(FabricInput), {fabric}},
      {CandidateGeneratorInputSlotRef(ConstraintInput), {constraints}},
  };
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveSpatialPnrCandidateGeneratorBinding(
    const ::loom::pnr::ResolvedPnrConfigView &config) {
  if (llvm::Error error = registerSpatialPnrCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

::loom::pnr::SpatialPnrGenerationOutcome invokeSpatialPnrCandidateGenerator(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store, std::uint32_t candidateWorkerCount) {
  if (binding.descriptorRef() != descriptor.reference())
    return invalidOutcome("binding does not select the Spatial PnR generator");
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), inputBindings))
    return invalidOutcome(llvm::toString(std::move(error)));

  auto config = ::loom::pnr::adoptResolvedSpatialPnrConfigView(
      ::loom::pnr::resolvedSpatialPnrConfigSchemaDescriptorBytes(),
      binding.canonicalConfigBytes(), binding.configDigest());
  if (!config)
    return invalidOutcome(llvm::toString(config.takeError()));

  auto dataflowArtifact = ::dataflow::importCanonicalDataflow(
      singleInput(inputBindings, DataflowInput), store);
  if (!dataflowArtifact)
    return invalidOutcome(llvm::toString(dataflowArtifact.takeError()));
  auto dataflow = dataflowArtifact->view();
  if (!dataflow)
    return invalidOutcome(llvm::toString(dataflow.takeError()));

  auto fabric = ::loom::fabric::importEntireFabricRoot(
      singleInput(inputBindings, FabricInput), store);
  if (!fabric)
    return invalidOutcome(llvm::toString(fabric.takeError()));
  auto tech = ::loom::mapping::importTechMapping(
      singleInput(inputBindings, TechMappingInput), store);
  if (!tech)
    return invalidOutcome(llvm::toString(tech.takeError()));
  auto constraints = ::loom::mapping::importSpatialMappingConstraintSet(
      singleInput(inputBindings, ConstraintInput), store);
  if (!constraints)
    return invalidOutcome(llvm::toString(constraints.takeError()));

  if (tech->view().dataflowIdentity() != dataflow->identity() ||
      tech->view().fabricIdentity() != fabric->view().identity() ||
      constraints->view().dataflowIdentity() != dataflow->identity() ||
      constraints->view().techMappingIdentity() != tech->view().identity() ||
      constraints->view().fabricIdentity() != fabric->view().identity())
    return invalidOutcome("D/T/F/K binding has inconsistent artifact owners");

  return ::loom::pnr::generateSpatialMappings({*dataflow, tech->view(),
                                               fabric->view(), *config,
                                               constraints->view(), store,
                                               candidateWorkerCount});
}

} // namespace loom::dse
