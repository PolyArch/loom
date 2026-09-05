#include "ResourceTimeAdjacentMappingSelection.h"

#include "JointHardwareReopenInternal.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/MappingDebugLog.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "PnR/System/SystemMappingMigration.h"

#include "llvm/ADT/STLExtras.h"

#include <array>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace loom::dse::joint_reopen_detail {
namespace {

ResourceTimeSpectrumFunnelResult
structuralSelectionFailure(std::uint64_t partitionMatchingCandidates,
                           std::uint64_t preservationMatchingCandidates) {
  std::string diagnostic;
  if (partitionMatchingCandidates == 0)
    diagnostic = "no generated Mapping realizes the exact partition intent";
  else if (preservationMatchingCandidates == 0)
    diagnostic = "no partition-matching Mapping preserves the exact "
                 "cone-external System selections";
  else
    diagnostic = "no eligible Mapping established the requested schedule";
  return ResourceTimeSpectrumFunnelResult{
      ResourceTimeSpectrumVerification{IncompleteResourceTimeSpectrum{
          ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
          std::move(diagnostic), 0}},
      ResourceTimeSpectrumFunnelAccounting{}};
}

llvm::Error
validateVerifierResult(const ResourceTimeSpectrumFunnelResult &result,
                       const ArtifactRootReference &dataflowReference,
                       const ArtifactRootReference &fabricReference,
                       llvm::ArrayRef<ArtifactRootReference> candidates) {
  if (result.accounting.independentlyImportedMappings > candidates.size())
    return invalid("resource-time Mapping verifier imported a foreign "
                   "candidate");
  if (const auto *incomplete =
          std::get_if<IncompleteResourceTimeSpectrum>(&result.verification)) {
    if (incomplete->independentlyImportedMappingCount > candidates.size() ||
        result.accounting.independentlyImportedMappings !=
            incomplete->independentlyImportedMappingCount)
      return invalid("resource-time Mapping verifier retained a foreign "
                     "incomplete candidate");
    return llvm::Error::success();
  }
  const auto &verified =
      std::get<VerifiedResourceTimeSpectrum>(result.verification);
  if (verified.dataflow != dataflowReference ||
      verified.fabric != fabricReference)
    return invalid("resource-time Mapping verifier changed an immutable "
                   "owner");
  if (verified.scenarios.empty())
    return invalid("resource-time Mapping verifier returned no scenario");
  if (result.accounting.verifiedScenarios != verified.scenarios.size())
    return invalid("resource-time Mapping verifier scenario accounting does "
                   "not match its result");
  std::set<ArtifactRootReference, decltype(&artifactRootReferenceLess)>
      verifiedMappings(&artifactRootReferenceLess);
  for (const VerifiedResourceTimeSpectrumScenario &scenario :
       verified.scenarios) {
    if (scenario.systemMappings.empty())
      return invalid("resource-time Mapping verifier scenario has no "
                     "Mapping candidate");
    std::set<ArtifactRootReference, decltype(&artifactRootReferenceLess)>
        scenarioMappings(&artifactRootReferenceLess);
    for (const ArtifactRootReference &mapping : scenario.systemMappings) {
      if (!llvm::is_contained(candidates, mapping))
        return invalid("resource-time Mapping verifier scenario names a "
                       "foreign candidate");
      if (!scenarioMappings.insert(mapping).second)
        return invalid("resource-time Mapping verifier scenario repeats a "
                       "Mapping candidate");
      verifiedMappings.insert(mapping);
    }
  }
  if (result.accounting.independentlyImportedMappings !=
      verifiedMappings.size())
    return invalid("resource-time Mapping verifier import accounting does "
                   "not match its scenarios");
  return llvm::Error::success();
}

bool cancelled(const ResourceTimeSpectrumFunnelResult &result) {
  const auto *incomplete =
      std::get_if<IncompleteResourceTimeSpectrum>(&result.verification);
  return incomplete &&
         incomplete->reason ==
             ResourceTimeSpectrumIncompleteReason::CancelledOrTimeout;
}

} // namespace

llvm::Expected<ResourceTimePartitionMappingSelection>
selectResourceTimePartitionMapping(
    JointDesignExecution &execution,
    const ArtifactRootReference &dataflowReference,
    const ArtifactRootReference &fabricReference,
    llvm::ArrayRef<pnr::SystemBindingPartitionIntent> partitions,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> reopenedRoots,
    const mapping::SystemMappingView *requiredParentMapping,
    llvm::ArrayRef<DsePlanIncompleteReason> prerequisiteIncompleteReasons,
    PreMappingSpectrumEndpoint spectrumEndpoint,
    JointResourceTimeMappingRepairSide side,
    JointResourceTimeMappingVerifier mappingVerifier,
    const ArtifactStore &artifacts) {
  if (partitions.empty())
    return invalid("resource-time Mapping selection has no partition intent");
  for (const pnr::SystemBindingPartitionIntent &partition : partitions) {
    if (partition.root.artifact != dataflowReference.artifact)
      return invalid("resource-time partition intent has a foreign Dataflow "
                     "owner");
    if (partition.partitionCount == 0)
      return invalid("resource-time partition intent has no resource");
  }
  auto dataflowArtifact =
      ::dataflow::importCanonicalDataflow(dataflowReference, artifacts);
  if (!dataflowArtifact)
    return dataflowArtifact.takeError();
  auto dataflow = dataflowArtifact->view();
  if (!dataflow)
    return dataflow.takeError();

  const std::vector<ArtifactRootReference> mappings = mappingRoots(execution);
  std::uint64_t partitionMatchingCandidates = 0;
  std::uint64_t preservationMatchingCandidates = 0;
  std::uint64_t acceptedCandidates = 0;
  std::vector<ArtifactRootReference> eligibleMappings;
  const std::optional<PreMappingSpectrumClass> requestedClass =
      spectrumClassForEndpoint(spectrumEndpoint);
  const auto emitSelection =
      [&](const std::optional<ArtifactRootReference> &selected,
          const std::optional<ResourceTimeSpectrumFunnelResult> &spectrum) {
        llvm::StringRef disposition = "proof_not_established";
        if (selected) {
          disposition = "verified";
        } else if (spectrum) {
          if (const auto *incomplete =
                  std::get_if<IncompleteResourceTimeSpectrum>(
                      &spectrum->verification))
            disposition = resourceTimeSpectrumIncompleteReasonSpelling(
                incomplete->reason);
        }
        mapping_debug::emit(
            mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
            mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
              fields["operation"] = "resource_time_partition_mapping_selection";
              fields["mode"] = side == JointResourceTimeMappingRepairSide::Cold
                                   ? "cold"
                                   : "incremental";
              fields["candidate_count"] = mappings.size();
              fields["requested_root_count"] = partitions.size();
              fields["partition_matching_candidates"] =
                  partitionMatchingCandidates;
              fields["preservation_matching_candidates"] =
                  preservationMatchingCandidates;
              fields["accepted_candidates"] = acceptedCandidates;
              fields["disposition"] = disposition;
              fields["selected_mapping"] =
                  selected ? llvm::json::Value(
                                 formatArtifactIdentityHex(selected->artifact))
                           : llvm::json::Value(nullptr);
            });
      };
  for (const ArtifactRootReference &reference : mappings) {
    auto imported = mapping::importSystemMapping(reference, artifacts);
    if (!imported)
      return imported.takeError();
    if (imported->view().dataflowIdentity() != dataflowReference.artifact)
      return invalid("resource-time Mapping selection found a foreign "
                     "Dataflow owner");
    if (imported->view().fabricIdentity() != fabricReference.artifact)
      return invalid("resource-time Mapping selection found a foreign "
                     "Fabric owner");
    auto contexts = mapping::projectSystemExecutionContexts(
        *dataflow, imported->view().executionBindings());
    if (!contexts)
      return contexts.takeError();
    bool matches = true;
    for (const pnr::SystemBindingPartitionIntent &partition : partitions) {
      auto resources =
          pnr::projectResourceTimeMappingResources(*contexts, partition.root);
      if (!resources)
        return resources.takeError();
      if (resources->size() != partition.partitionCount) {
        matches = false;
        break;
      }
    }
    if (!matches)
      continue;
    ++partitionMatchingCandidates;
    if (requiredParentMapping) {
      auto preserved = pnr::preservesSystemMappingMigrationCone(
          *requiredParentMapping, imported->view(), reopenedRoots, artifacts);
      if (!preserved)
        return preserved.takeError();
      if (!*preserved)
        continue;
    }
    ++preservationMatchingCandidates;
    eligibleMappings.push_back(reference);
  }
  std::vector<DsePlanIncompleteReason> executionIncompleteReasons(
      prerequisiteIncompleteReasons.begin(),
      prerequisiteIncompleteReasons.end());
  if (const auto *incomplete =
          std::get_if<IncompleteDsePlanExecution>(&execution.planExecution))
    executionIncompleteReasons.push_back(incomplete->reason());
  if (!executionIncompleteReasons.empty()) {
    execution.summary.selectedMapping.reset();
    execution.summary.selectedPlanOrdinal.reset();
    emitSelection(std::nullopt, std::nullopt);
    return ResourceTimePartitionMappingSelection{
        std::nullopt, std::nullopt, std::move(eligibleMappings),
        std::move(executionIncompleteReasons)};
  }
  std::optional<ResourceTimeSpectrumFunnelResult> singletonRejection;
  for (const ArtifactRootReference &reference : eligibleMappings) {
    const std::array singleton = {reference};
    auto verification = mappingVerifier(side, singleton);
    if (!verification)
      return verification.takeError();
    if (llvm::Error error = validateVerifierResult(
            *verification, dataflowReference, fabricReference, singleton))
      return std::move(error);
    if (cancelled(*verification)) {
      execution.summary.selectedMapping.reset();
      execution.summary.selectedPlanOrdinal.reset();
      std::optional<ResourceTimeSpectrumFunnelResult> spectrum(
          std::move(*verification));
      emitSelection(std::nullopt, spectrum);
      return ResourceTimePartitionMappingSelection{
          std::nullopt, std::move(spectrum), std::move(eligibleMappings), {}};
    }
    if (!resourceTimeSpectrumAdmitsMappingClass(*verification, reference,
                                                requestedClass)) {
      singletonRejection = std::move(*verification);
      continue;
    }
    ++acceptedCandidates;
    execution.summary.selectedMapping = reference;
    execution.summary.selectedPlanOrdinal = 0;
    std::optional<ArtifactRootReference> selected(reference);
    std::optional<ResourceTimeSpectrumFunnelResult> selectedSpectrum(
        std::move(*verification));
    emitSelection(selected, selectedSpectrum);
    return ResourceTimePartitionMappingSelection{
        selected, std::move(selectedSpectrum), std::move(eligibleMappings), {}};
  }
  execution.summary.selectedMapping.reset();
  execution.summary.selectedPlanOrdinal.reset();
  std::optional<ResourceTimeSpectrumFunnelResult> spectrum;
  if (eligibleMappings.size() == 1) {
    spectrum = std::move(singletonRejection);
  } else if (!eligibleMappings.empty()) {
    auto verification = mappingVerifier(side, eligibleMappings);
    if (!verification)
      return verification.takeError();
    if (llvm::Error error =
            validateVerifierResult(*verification, dataflowReference,
                                   fabricReference, eligibleMappings))
      return std::move(error);
    spectrum = std::move(*verification);
  } else {
    spectrum = structuralSelectionFailure(partitionMatchingCandidates,
                                          preservationMatchingCandidates);
  }
  emitSelection(std::nullopt, spectrum);
  return ResourceTimePartitionMappingSelection{
      std::nullopt, std::move(spectrum), std::move(eligibleMappings), {}};
}

} // namespace loom::dse::joint_reopen_detail
