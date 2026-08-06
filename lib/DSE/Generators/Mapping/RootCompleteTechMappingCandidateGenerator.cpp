#include "DSE/RootCompleteTechMappingCandidateGenerator.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Tech/TechMappingGenerator.h"

#include <array>
#include <cstdint>
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

constexpr std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots = {{
        {CandidateGeneratorInputSlotRef(DataflowCandidatesInput),
         "canonical_dataflow", PlanValueRole::CandidateSet,
         &::dataflow::canonicalDataflowSchema, PlanValueCardinality::FiniteSet},
        {CandidateGeneratorInputSlotRef(FabricInput), "fabric",
         PlanValueRole::CandidateSet, &::loom::fabric::fabricArtifactSchema,
         PlanValueCardinality::ExactlyOne},
    }};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {
    {{CandidateGeneratorOutputSlotRef(0), "tech_mapping",
      PlanValueRole::CandidateSet, &::loom::mapping::mappingArtifactSchema,
      PlanValueCardinality::FiniteSet}}};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 3> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "match_row_attempt"},
    {CandidateGeneratorWorkUnitRef(1), "partial_cover_expansion"},
    {CandidateGeneratorWorkUnitRef(2), "publication_slot"},
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

llvm::Expected<CandidateGeneratorInvocationOutcome> invokeRootCompleteProvider(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store);

const CandidateGeneratorDescriptor descriptor{
    rootCompleteTechMappingCandidateGeneratorKind,
    "mapping.root_complete_tech_mapping",
    "loom.mapping.root_complete_tech_mapping.generator.v1",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{
        ::loom::mapping::resolvedTechMappingConfigSchemaDescriptorBytes(),
        validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
};

llvm::Expected<CandidateGeneratorInvocationOutcome> invokeRootCompleteProvider(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store) {
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

    std::vector<::dataflow::GraphRef> covers;
    covers.reserve(dataflow->graphs().size());
    for (const ::dataflow::CanonicalGraphView &graph : dataflow->graphs())
      covers.push_back(graph.ref);

    ::loom::mapping::TechMappingGenerationOutcome outcome =
        ::loom::mapping::generateTechMappings(
            {*dataflow, covers, fabric->view(), *config, store});
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
      continue;
    }
    if (std::holds_alternative<::loom::mapping::ProvenInfeasibleTechMapping>(
            outcome))
      continue;
    if (std::holds_alternative<
            ::loom::mapping::IncompleteTechMappingGeneration>(outcome))
      return CandidateGeneratorInvocationOutcome{
          IncompleteCandidateGeneratorInvocation{
              CandidateGeneratorIncompleteReason::ProofNotEstablished,
              {{CandidateGeneratorOutputSlotRef(0), std::move(outputs)}},
              std::move(lineage)}};
    if (const auto *invalid =
            std::get_if<::loom::mapping::InvalidTechMappingGeneration>(
                &outcome))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "root_complete_tech_mapping_generator_invalid: " +
              invalid->diagnostic);
    const auto &internal =
        std::get<::loom::mapping::InternalTechMappingGeneration>(outcome);
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "root_complete_tech_mapping_generator_execution_failed: " +
            internal.diagnostic);
  }

  return CandidateGeneratorInvocationOutcome{
      CompletedCandidateGeneratorInvocation{
          {{CandidateGeneratorOutputSlotRef(0), std::move(outputs)}},
          std::move(lineage)}};
}

const CandidateGeneratorProvider provider{descriptor.reference(),
                                          invokeRootCompleteProvider};

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

} // namespace loom::dse
