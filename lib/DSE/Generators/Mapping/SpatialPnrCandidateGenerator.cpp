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

constexpr std::array<const ArtifactSchemaDescriptor *, 1> dataflowSchemas = {
    &::dataflow::canonicalDataflowSchema};
constexpr std::array<const ArtifactSchemaDescriptor *, 1> mappingSchemas = {
    &::loom::mapping::mappingArtifactSchema};
constexpr std::array<const ArtifactSchemaDescriptor *, 1> fabricSchemas = {
    &::loom::fabric::fabricArtifactSchema};
constexpr std::array<const ArtifactSchemaDescriptor *, 1> constraintSchemas = {
    &::loom::mapping::mappingConstraintSetSchema};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots = {
        {{CandidateGeneratorInputSlotRef(DataflowInput), "dataflow",
          dataflowSchemas, ArtifactCollectionBounds{1, 1}},
         {CandidateGeneratorInputSlotRef(TechMappingInput), "tech_mapping",
          mappingSchemas, ArtifactCollectionBounds{1, 1}},
         {CandidateGeneratorInputSlotRef(FabricInput), "fabric", fabricSchemas,
          ArtifactCollectionBounds{1, 1}},
         {CandidateGeneratorInputSlotRef(ConstraintInput),
          "spatial_constraints", constraintSchemas,
          ArtifactCollectionBounds{1, 1}}}};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {
    {{CandidateGeneratorOutputSlotRef(0), "spatial_mapping",
      &::loom::mapping::mappingArtifactSchema,
      ArtifactCollectionBounds{0, ArtifactCollectionBounds::unbounded}}}};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 10> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "seed_attempt"},
    {CandidateGeneratorWorkUnitRef(1), "assignment_attempt_per_seed"},
    {CandidateGeneratorWorkUnitRef(2), "endpoint_expansion"},
    {CandidateGeneratorWorkUnitRef(3), "negotiation_iteration"},
    {CandidateGeneratorWorkUnitRef(4), "calibration_proposal"},
    {CandidateGeneratorWorkUnitRef(5), "proposal_per_level_base"},
    {CandidateGeneratorWorkUnitRef(6), "proposal_per_movable_decision"},
    {CandidateGeneratorWorkUnitRef(7), "focused_closure_proposal"},
    {CandidateGeneratorWorkUnitRef(8), "exact_repair_region_decision"},
    {CandidateGeneratorWorkUnitRef(9), "exact_repair_solver_call"},
}};

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
    CandidateGeneratorConfigViewContract{
        ::loom::pnr::resolvedSpatialPnrConfigSchemaDescriptorBytes(),
        validateSpatialConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    {},
};

::loom::pnr::SpatialPnrGenerationOutcome invalidOutcome(std::string message) {
  return ::loom::pnr::InvalidSpatialPnrGeneration{
      ::loom::pnr::InvalidSpatialPnrGenerationReason::FrozenInput,
      {},
      std::move(message)};
}

const ArtifactRootReference &
singleInput(const ResolvedCandidateGeneratorBinding &binding, InputSlot slot) {
  return binding.inputBindings()[slot].artifacts.front();
}

} // namespace

const CandidateGeneratorDescriptor &spatialPnrCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerSpatialPnrCandidateGenerator() {
  return registerCandidateGeneratorDescriptor(descriptor);
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveSpatialPnrCandidateGeneratorBinding(
    const ArtifactRootReference &dataflow,
    const ArtifactRootReference &techMapping,
    const ArtifactRootReference &fabric,
    const ArtifactRootReference &constraints,
    const ::loom::pnr::ResolvedPnrConfigView &config) {
  if (llvm::Error error = registerSpatialPnrCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(DataflowInput), {dataflow}},
      {CandidateGeneratorInputSlotRef(TechMappingInput), {techMapping}},
      {CandidateGeneratorInputSlotRef(FabricInput), {fabric}},
      {CandidateGeneratorInputSlotRef(ConstraintInput), {constraints}},
  };
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), std::move(bindings), config.canonicalViewBytes(),
      config.digest());
}

::loom::pnr::SpatialPnrGenerationOutcome invokeSpatialPnrCandidateGenerator(
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store) {
  if (binding.descriptorRef() != descriptor.reference())
    return invalidOutcome("binding does not select the Spatial PnR generator");

  auto config = ::loom::pnr::adoptResolvedSpatialPnrConfigView(
      ::loom::pnr::resolvedSpatialPnrConfigSchemaDescriptorBytes(),
      binding.canonicalConfigBytes(), binding.configDigest());
  if (!config)
    return invalidOutcome(llvm::toString(config.takeError()));

  auto dataflowArtifact = ::dataflow::importCanonicalDataflow(
      singleInput(binding, DataflowInput), store);
  if (!dataflowArtifact)
    return invalidOutcome(llvm::toString(dataflowArtifact.takeError()));
  auto dataflow = dataflowArtifact->view();
  if (!dataflow)
    return invalidOutcome(llvm::toString(dataflow.takeError()));

  auto fabric = ::loom::fabric::importEntireFabricRoot(
      singleInput(binding, FabricInput), store);
  if (!fabric)
    return invalidOutcome(llvm::toString(fabric.takeError()));
  auto tech = ::loom::mapping::importTechMapping(
      singleInput(binding, TechMappingInput), store);
  if (!tech)
    return invalidOutcome(llvm::toString(tech.takeError()));
  auto constraints = ::loom::mapping::importSpatialMappingConstraintSet(
      singleInput(binding, ConstraintInput), store);
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
                                               constraints->view(), store});
}

} // namespace loom::dse
