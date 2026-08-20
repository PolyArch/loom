#include "DSE/RootCompleteSystemPnrCandidateGenerator.h"

#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingHardwareDemand.h"
#include "PnR/System/SystemPnrDerivedContext.h"
#include "PnR/System/SystemPnrSearchDomain.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstddef>
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
  PhysicalTimingProfilesInput,
  InputSlotCount,
};

enum ApplicationInputSlot : std::uint32_t {
  ApplicationDataflowInput,
  ApplicationSpatialMappingCandidatesInput,
  ApplicationFabricInput,
  ApplicationPhysicalTimingProfilesInput,
  ApplicationSystemConstraintsInput,
  ApplicationInputSlotCount,
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
        {CandidateGeneratorInputSlotRef(PhysicalTimingProfilesInput),
         "physical_timing_profile", PlanValueRole::CandidateSet,
         &::loom::fabric::fabricPhysicalTimingProfileArtifactSchema,
         PlanValueCardinality::FiniteSet},
    }};

constexpr std::array<CandidateGeneratorInputSlotDescriptor,
                     ApplicationInputSlotCount>
    applicationInputSlots = {{
        {CandidateGeneratorInputSlotRef(ApplicationDataflowInput), "dataflow",
         PlanValueRole::CandidateSet, &::dataflow::canonicalDataflowSchema,
         PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorInputSlotRef(
             ApplicationSpatialMappingCandidatesInput),
         "spatial_mapping", PlanValueRole::CandidateSet,
         &::loom::mapping::mappingArtifactSchema,
         PlanValueCardinality::FiniteSet},
        {CandidateGeneratorInputSlotRef(ApplicationFabricInput), "fabric",
         PlanValueRole::CandidateSet, &::loom::fabric::fabricArtifactSchema,
         PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorInputSlotRef(ApplicationPhysicalTimingProfilesInput),
         "physical_timing_profile", PlanValueRole::CandidateSet,
         &::loom::fabric::fabricPhysicalTimingProfileArtifactSchema,
         PlanValueCardinality::FiniteSet},
        {CandidateGeneratorInputSlotRef(ApplicationSystemConstraintsInput),
         "system_constraints", PlanValueRole::CandidateSet,
         &::loom::mapping::mappingConstraintSetSchema,
         PlanValueCardinality::ExactlyOne},
    }};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {
    {{CandidateGeneratorOutputSlotRef(0), "system_mapping",
      PlanValueRole::CandidateSet, &::loom::mapping::mappingArtifactSchema,
      PlanValueCardinality::FiniteSet}}};

std::size_t defaultSystemPartitionCount(
    const ::loom::fabric::FabricSystemRootView &system) {
  constexpr std::size_t binaryCyclicPartitionCount = 2;
  return std::min(system.artifact().accCoreOccurrences().size(),
                  binaryCyclicPartitionCount);
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = ::loom::pnr::adoptResolvedSystemPnrConfigView(
      ::loom::pnr::resolvedSystemPnrConfigSchemaDescriptorBytes(), bytes,
      digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

llvm::Error validateCapacityPressureFeedback(
    llvm::ArrayRef<std::uint8_t> bytes,
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ArtifactStore &store) {
  std::optional<ArtifactRootReference> system;
  std::vector<ArtifactRootReference> spatialMappings;
  for (const CandidateGeneratorInputBinding &binding : inputs)
    for (const ArtifactRootReference &input : binding.artifacts) {
      if (input.schemaIdentity ==
              ::loom::fabric::fabricArtifactSchema.identity &&
          input.schemaVersion == ::loom::fabric::fabricArtifactSchema.version) {
        auto imported = ::loom::fabric::importEntireFabricRoot(input, store);
        if (!imported)
          return imported.takeError();
        if (imported->view().rootKind() !=
            ::loom::fabric::FabricRootKind::System)
          continue;
        if (system)
          return llvm::createStringError(
              llvm::inconvertibleErrorCode(),
              "system_pnr_generator_feedback_invalid: input closure has "
              "multiple System roots");
        system = input;
        continue;
      }
      if (input.schemaIdentity ==
              ::loom::mapping::mappingArtifactSchema.identity &&
          input.schemaVersion == ::loom::mapping::mappingArtifactSchema.version)
        spatialMappings.push_back(input);
    }
  if (!system || spatialMappings.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "system_pnr_generator_feedback_invalid: input closure lacks its exact "
        "System or SpatialMapping frontier");
  auto adopted = ::loom::mapping::adoptSystemAccCoreCapacityPressure(
      bytes, *system, spatialMappings, store);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

const CandidateGeneratorOwnerFeedbackPayloadContract feedbackContract{
    ::loom::mapping::systemAccCoreCapacityPressureSchemaBytes(),
    validateCapacityPressureFeedback};

llvm::Expected<CandidateGeneratorProviderResult> invokeRootCompleteProvider(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store, const BlobStore &blobs,
    const CandidateGeneratorInvocationView &invocation);

llvm::Expected<CandidateGeneratorProviderResult> invokeApplicationProvider(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store, const BlobStore &blobs,
    const CandidateGeneratorInvocationView &invocation);

const CandidateGeneratorDescriptor descriptor{
    rootCompleteSystemPnrCandidateGeneratorKind,
    "mapping.root_complete_system_pnr",
    "loom.mapping.root_complete_system_pnr.generator.v9",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{
        ::loom::pnr::resolvedSystemPnrConfigSchemaDescriptorBytes(),
        validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    pnrCandidateGeneratorWorkUnits,
    nullptr,
    ProviderForm::InProcess,
    &feedbackContract,
};

const CandidateGeneratorDescriptor applicationDescriptor{
    applicationSystemPnrCandidateGeneratorKind,
    "mapping.application_system_pnr",
    "loom.mapping.application_system_pnr.generator.v8",
    applicationInputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{
        ::loom::pnr::resolvedSystemPnrConfigSchemaDescriptorBytes(),
        validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    pnrCandidateGeneratorWorkUnits,
    nullptr,
    ProviderForm::InProcess,
    &feedbackContract,
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
incomplete(CandidateGeneratorIncompleteReason reason,
           std::vector<ArtifactRootReference> outputs = {}) {
  llvm::sort(outputs, artifactRootReferenceLess);
  outputs.erase(std::unique(outputs.begin(), outputs.end()), outputs.end());
  auto lineage = mechanicalLineage(outputs);
  return {reason,
          {{CandidateGeneratorOutputSlotRef(0), std::move(outputs)}},
          std::move(lineage)};
}

llvm::Expected<std::optional<std::vector<std::uint8_t>>>
encodeCapacityPressureFeedback(
    const ::loom::pnr::IncompleteSystemPnrGeneration &incomplete,
    const ArtifactRootReference &systemReference,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    const ::loom::fabric::FabricSystemRootView &system) {
  if (!incomplete.importedCapacityPressure)
    return std::optional<std::vector<std::uint8_t>>{};
  const auto &pressure = *incomplete.importedCapacityPressure;
  if (pressure.witness.namespaceOrdinal == 0 ||
      pressure.witness.namespaceOrdinal >
          system.artifact().accCoreOccurrences().size())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "system_pnr_generator_feedback_invalid: capacity witness has no exact "
        "AccCore occurrence");
  const auto witnessCore =
      system.artifact().accCoreOccurrences()[static_cast<std::size_t>(
          pressure.witness.namespaceOrdinal - 1)];
  const auto target = system.spatialCoreTarget(witnessCore);
  if (!target ||
      target->dependencyOrdinal >= system.artifact().importedModules().size())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "system_pnr_generator_feedback_invalid: witness AccCore has no Module");
  const auto &targetModule =
      system.artifact().importedModules()[target->dependencyOrdinal];
  std::uint64_t compatibleAccCoreCount = 0;
  for (const auto core : system.artifact().accCoreOccurrences()) {
    const auto coreTarget = system.spatialCoreTarget(core);
    if (!coreTarget || coreTarget->dependencyOrdinal >=
                           system.artifact().importedModules().size())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "system_pnr_generator_feedback_invalid: AccCore has no Module");
    if (system.artifact()
            .importedModules()[coreTarget->dependencyOrdinal]
            .identity() == targetModule.identity())
      ++compatibleAccCoreCount;
  }
  auto feedback = ::loom::mapping::SystemAccCoreCapacityPressure::get(
      systemReference,
      ArtifactRootReference{::loom::fabric::fabricArtifactSchema.identity.str(),
                            ::loom::fabric::fabricArtifactSchema.version,
                            targetModule.identity()},
      spatialMappings.vec(), compatibleAccCoreCount,
      pressure.assignmentAttempts, pressure.witness.usage,
      pressure.witness.capacity);
  if (!feedback)
    return feedback.takeError();
  return std::optional<std::vector<std::uint8_t>>(
      ::loom::mapping::encodeSystemAccCoreCapacityPressure(*feedback));
}

llvm::Expected<std::vector<::loom::fabric::FabricPhysicalTimingProfileView>>
importPhysicalTimingProfiles(llvm::ArrayRef<ArtifactRootReference> references,
                             const ::loom::fabric::FabricSystemRootView &system,
                             const ArtifactStore &store) {
  std::vector<ArtifactRootReference> canonical(references.begin(),
                                               references.end());
  llvm::sort(canonical, artifactRootReferenceLess);
  if (std::adjacent_find(canonical.begin(), canonical.end()) != canonical.end())
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "System physical timing profile set contains a duplicate root");

  std::vector<::loom::fabric::FabricPhysicalTimingProfileView> profiles;
  profiles.reserve(canonical.size());
  for (const ArtifactRootReference &reference : canonical) {
    auto owner = ::loom::fabric::resolveFabricPhysicalTimingProfileOwner(
        reference, store);
    if (!owner)
      return owner.takeError();
    const ::loom::fabric::FabricArtifactView *module = nullptr;
    for (const auto core : system.artifact().accCoreOccurrences()) {
      const auto target = system.spatialCoreTarget(core);
      if (!target || target->dependencyOrdinal >=
                         system.artifact().importedModules().size())
        return llvm::createStringError(
            std::make_error_code(std::errc::invalid_argument),
            "System AccCore physical timing target does not resolve");
      const auto &candidate =
          system.artifact().importedModules()[target->dependencyOrdinal];
      if (candidate.identity() == *owner)
        module = &candidate;
    }
    if (!module)
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "System physical timing profile has no attached Module owner");
    auto profile = ::loom::fabric::importFabricPhysicalTimingProfile(
        reference, *module, store);
    if (!profile)
      return profile.takeError();
    profiles.push_back(std::move(*profile));
  }
  llvm::sort(profiles, [](const auto &lhs, const auto &rhs) {
    return lhs.fabricIdentity().bytes() < rhs.fabricIdentity().bytes();
  });
  return profiles;
}

llvm::Expected<CandidateGeneratorProviderResult> invokeRootCompleteProvider(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store, const BlobStore &blobs,
    const CandidateGeneratorInvocationView &invocation) {
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
  ::loom::pnr::DerivedContextCacheAccess staticAccess;
  auto staticContext =
      ::loom::pnr::buildSystemStaticContext(*system, &staticAccess);
  if (!staticContext)
    return staticContext.takeError();
  ::loom::pnr::emitSystemStaticContextStatistics(
      *staticContext, ::loom::mapping_debug::Stage::SystemPnr,
      staticAccess.hits, staticAccess.misses);
  auto physicalTimingProfiles = importPhysicalTimingProfiles(
      inputBindings[PhysicalTimingProfilesInput].artifacts, *system, store);
  if (!physicalTimingProfiles)
    return physicalTimingProfiles.takeError();

  std::vector<::dataflow::RootThreadLaunchRef> roots;
  roots.reserve(dataflow->rootThreadLaunches().size());
  for (const auto &root : dataflow->rootThreadLaunches())
    roots.push_back(root.ref);
  if (roots.empty()) {
    if (llvm::Error error = ::loom::pnr::validateSystemSpatialMappingSet(
            *dataflow, *system,
            inputBindings[SpatialMappingCandidatesInput].artifacts, store))
      return std::move(error);
    return CandidateGeneratorProviderResult{
        completed({}), rootCompleteSystemPnrCandidateGeneratorWorkSummary({})};
  }

  auto constraints = ::loom::mapping::finalizeEmptySystemMappingConstraintSet(
      *dataflow, *system, roots, store);
  if (!constraints)
    return constraints.takeError();
  ::loom::pnr::DerivedContextCacheAccess activeAccess;
  auto activeContext = ::loom::pnr::buildSystemActiveContext(
      *staticContext, *dataflow, *system, *physicalTimingProfiles, *constraints,
      inputBindings[SpatialMappingCandidatesInput].artifacts, store,
      &activeAccess);
  if (!activeContext)
    return activeContext.takeError();
  ::loom::pnr::emitSystemActiveContextStatistics(
      *activeContext, ::loom::mapping_debug::Stage::SystemPnr,
      activeAccess.hits, activeAccess.misses);
  auto partition = ::loom::pnr::projectCyclicPresburgerPartitionPlan(
      *dataflow, constraints->view().rootThreadLaunches(),
      defaultSystemPartitionCount(*system));
  if (!partition)
    return partition.takeError();
  ::loom::pnr::SystemHierarchicalGraphSearchInput graphSearch{
      inputBindings[SpatialMappingCandidatesInput].artifacts};
  auto searchDomain = ::loom::pnr::projectSystemPnrSearchDomain(
      *dataflow, *system, *config, *constraints, *partition, graphSearch, store,
      &*activeContext);
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
          {*dataflow, *system, *physicalTimingProfiles, *searchDomain, *config,
           *constraints, store, invocation.executionControl(), &*staticContext,
           &*activeContext});
  if (auto *generated =
          std::get_if<::loom::pnr::GeneratedSystemMappings>(&outcome)) {
    auto reason = pnrGenerationIncompleteReason(generated->termination);
    return CandidateGeneratorProviderResult{
        reason ? CandidateGeneratorProviderOutcome(
                     incomplete(*reason, std::move(generated->candidates)))
               : CandidateGeneratorProviderOutcome(
                     completed(std::move(generated->candidates))),
        rootCompleteSystemPnrCandidateGeneratorWorkSummary(
            generated->accounting)};
  }
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
    auto feedback = encodeCapacityPressureFeedback(
        *partial, inputBindings[FabricInput].artifacts.front(),
        inputBindings[SpatialMappingCandidatesInput].artifacts, *system);
    if (!feedback)
      return feedback.takeError();
    return CandidateGeneratorProviderResult{
        incomplete(reason),
        rootCompleteSystemPnrCandidateGeneratorWorkSummary(partial->accounting),
        std::move(*feedback)};
  }
  if (auto *interrupted =
          std::get_if<::loom::pnr::InterruptedSystemPnrGeneration>(&outcome))
    return CandidateGeneratorProviderResult{
        incomplete(CandidateGeneratorIncompleteReason::CancelledOrTimeout,
                   std::move(interrupted->candidates)),
        rootCompleteSystemPnrCandidateGeneratorWorkSummary(
            interrupted->accounting)};
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

llvm::Expected<CandidateGeneratorProviderResult> invokeApplicationProvider(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store, const BlobStore &,
    const CandidateGeneratorInvocationView &invocation) {
  auto config = ::loom::pnr::adoptResolvedSystemPnrConfigView(
      ::loom::pnr::resolvedSystemPnrConfigSchemaDescriptorBytes(),
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
  auto fabricArtifact = ::loom::fabric::importEntireFabricRoot(
      inputBindings[ApplicationFabricInput].artifacts.front(), store);
  if (!fabricArtifact)
    return fabricArtifact.takeError();
  auto system = ::loom::fabric::requireSystemRoot(fabricArtifact->view());
  if (!system)
    return system.takeError();
  ::loom::pnr::DerivedContextCacheAccess staticAccess;
  auto staticContext =
      ::loom::pnr::buildSystemStaticContext(*system, &staticAccess);
  if (!staticContext)
    return staticContext.takeError();
  ::loom::pnr::emitSystemStaticContextStatistics(
      *staticContext, ::loom::mapping_debug::Stage::SystemPnr,
      staticAccess.hits, staticAccess.misses);
  auto physicalTimingProfiles = importPhysicalTimingProfiles(
      inputBindings[ApplicationPhysicalTimingProfilesInput].artifacts, *system,
      store);
  if (!physicalTimingProfiles)
    return physicalTimingProfiles.takeError();
  auto constraints = ::loom::mapping::importSystemMappingConstraintSet(
      inputBindings[ApplicationSystemConstraintsInput].artifacts.front(),
      store);
  if (!constraints)
    return constraints.takeError();
  if (constraints->view().dataflowIdentity() != dataflow->identity() ||
      constraints->view().fabricIdentity() != system->artifact().identity())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "application_system_pnr_generator_invalid: constraints bind foreign "
        "Dataflow or Fabric owners");
  ::loom::pnr::DerivedContextCacheAccess activeAccess;
  auto activeContext = ::loom::pnr::buildSystemActiveContext(
      *staticContext, *dataflow, *system, *physicalTimingProfiles, *constraints,
      inputBindings[ApplicationSpatialMappingCandidatesInput].artifacts, store,
      &activeAccess);
  if (!activeContext)
    return activeContext.takeError();
  ::loom::pnr::emitSystemActiveContextStatistics(
      *activeContext, ::loom::mapping_debug::Stage::SystemPnr,
      activeAccess.hits, activeAccess.misses);

  auto partition = ::loom::pnr::projectCyclicPresburgerPartitionPlan(
      *dataflow, constraints->view().rootThreadLaunches(),
      defaultSystemPartitionCount(*system));
  if (!partition)
    return partition.takeError();
  ::loom::pnr::SystemHierarchicalGraphSearchInput graphSearch{
      inputBindings[ApplicationSpatialMappingCandidatesInput].artifacts};
  auto searchDomain = ::loom::pnr::projectSystemPnrSearchDomain(
      *dataflow, *system, *config, *constraints, *partition, graphSearch, store,
      &*activeContext);
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
        llvm::inconvertibleErrorCode(),
        "application_system_pnr_generator_invalid: search-domain projection "
        "lost its failure cause");
  }

  ::loom::pnr::SystemPnrGenerationOutcome outcome =
      ::loom::pnr::generateSystemMappings(
          {*dataflow, *system, *physicalTimingProfiles, *searchDomain, *config,
           *constraints, store, invocation.executionControl(), &*staticContext,
           &*activeContext});
  if (auto *generated =
          std::get_if<::loom::pnr::GeneratedSystemMappings>(&outcome)) {
    auto reason = pnrGenerationIncompleteReason(generated->termination);
    return CandidateGeneratorProviderResult{
        reason ? CandidateGeneratorProviderOutcome(
                     incomplete(*reason, std::move(generated->candidates)))
               : CandidateGeneratorProviderOutcome(
                     completed(std::move(generated->candidates))),
        rootCompleteSystemPnrCandidateGeneratorWorkSummary(
            generated->accounting)};
  }
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
    auto feedback = encodeCapacityPressureFeedback(
        *partial, inputBindings[ApplicationFabricInput].artifacts.front(),
        inputBindings[ApplicationSpatialMappingCandidatesInput].artifacts,
        *system);
    if (!feedback)
      return feedback.takeError();
    return CandidateGeneratorProviderResult{
        incomplete(reason),
        rootCompleteSystemPnrCandidateGeneratorWorkSummary(partial->accounting),
        std::move(*feedback)};
  }
  if (auto *interrupted =
          std::get_if<::loom::pnr::InterruptedSystemPnrGeneration>(&outcome))
    return CandidateGeneratorProviderResult{
        incomplete(CandidateGeneratorIncompleteReason::CancelledOrTimeout,
                   std::move(interrupted->candidates)),
        rootCompleteSystemPnrCandidateGeneratorWorkSummary(
            interrupted->accounting)};
  if (const auto *invalid =
          std::get_if<::loom::pnr::InvalidSystemPnrGeneration>(&outcome))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "application_system_pnr_generator_invalid: " + invalid->diagnostic);
  const auto &internal =
      std::get<::loom::pnr::InternalSystemPnrGeneration>(outcome);
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "application_system_pnr_generator_execution_failed: " +
          internal.diagnostic);
}

const CandidateGeneratorProvider provider{
    descriptor.reference(),
    CandidateGeneratorInProcessProvider{invokeRootCompleteProvider}};

const CandidateGeneratorProvider applicationProvider{
    applicationDescriptor.reference(),
    CandidateGeneratorInProcessProvider{invokeApplicationProvider}};

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
    const ArtifactRootReference &fabric,
    llvm::ArrayRef<ArtifactRootReference> physicalTimingProfiles) {
  if (llvm::Error error = registerRootCompleteSystemPnrCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(DataflowInput), {dataflow}},
      {CandidateGeneratorInputSlotRef(SpatialMappingCandidatesInput),
       spatialMappingCandidates.vec()},
      {CandidateGeneratorInputSlotRef(FabricInput), {fabric}},
      {CandidateGeneratorInputSlotRef(PhysicalTimingProfilesInput),
       physicalTimingProfiles.vec()},
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

const CandidateGeneratorDescriptor &
applicationSystemPnrCandidateGeneratorDescriptor() {
  return applicationDescriptor;
}

llvm::Error registerApplicationSystemPnrCandidateGenerator() {
  if (llvm::Error error =
          registerCandidateGeneratorDescriptor(applicationDescriptor))
    return error;
  return registerCandidateGeneratorProvider(applicationProvider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindApplicationSystemPnrCandidateGeneratorInputs(
    const ArtifactRootReference &dataflow,
    llvm::ArrayRef<ArtifactRootReference> spatialMappingCandidates,
    const ArtifactRootReference &fabric,
    llvm::ArrayRef<ArtifactRootReference> physicalTimingProfiles,
    const ArtifactRootReference &systemConstraints) {
  if (llvm::Error error = registerApplicationSystemPnrCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(ApplicationDataflowInput), {dataflow}},
      {CandidateGeneratorInputSlotRef(ApplicationSpatialMappingCandidatesInput),
       spatialMappingCandidates.vec()},
      {CandidateGeneratorInputSlotRef(ApplicationFabricInput), {fabric}},
      {CandidateGeneratorInputSlotRef(ApplicationPhysicalTimingProfilesInput),
       physicalTimingProfiles.vec()},
      {CandidateGeneratorInputSlotRef(ApplicationSystemConstraintsInput),
       {systemConstraints}},
  };
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          applicationDescriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveApplicationSystemPnrCandidateGeneratorBinding(
    const ::loom::pnr::ResolvedPnrConfigView &config) {
  if (llvm::Error error = registerApplicationSystemPnrCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      applicationDescriptor.reference(), config.canonicalViewBytes(),
      config.digest());
}

std::vector<CandidateGeneratorWorkUnitSummary>
rootCompleteSystemPnrCandidateGeneratorWorkSummary(
    const ::loom::pnr::SystemPnrGenerationAccounting &accounting) {
  const std::array<std::uint64_t, pnrCandidateGeneratorWorkUnits.size()>
      consumed = {accounting.seedAttemptSlots,
                  accounting.initializerAssignmentAttempts,
                  accounting.endpointExpansionSlots,
                  accounting.negotiationIterationSlots,
                  accounting.calibrationProposalSlots,
                  accounting.annealingBaseProposalSlots,
                  accounting.annealingMovableProposalSlots,
                  accounting.exactRepairRegionDecisions,
                  accounting.exactRepairSolverCalls};
  std::vector<CandidateGeneratorWorkUnitSummary> result;
  result.reserve(consumed.size());
  for (std::size_t ordinal = 0; ordinal != consumed.size(); ++ordinal)
    result.push_back({CandidateGeneratorWorkUnitRef(ordinal), consumed[ordinal],
                      consumed[ordinal]});
  return result;
}

} // namespace loom::dse
