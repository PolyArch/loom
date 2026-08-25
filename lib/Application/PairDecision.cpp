#include "Application/Build.h"
#include "BuildInternal.h"

#include "DSE/Promotion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::application {

namespace build_detail {

constexpr llvm::StringLiteral preAdmissionManifestJoinOwner =
    "application_build";
constexpr llvm::StringLiteral preAdmissionManifestJoinContract =
    "pre_mapping_owner_verified_v1";

ApplicationObjectiveObservation
unsupportedObjective(ApplicationObjectiveDimension dimension) {
  return {dimension, std::nullopt, ApplicationObjectiveEvidence::Unsupported, 0,
          false};
}

std::vector<ApplicationObjectiveObservation> makeUnsupportedObjectiveVector() {
  std::vector<ApplicationObjectiveObservation> result;
  result.reserve(11);
  for (std::uint8_t ordinal = 0; ordinal != 11; ++ordinal)
    result.push_back(unsupportedObjective(
        static_cast<ApplicationObjectiveDimension>(ordinal)));
  return result;
}

void setObjective(ApplicationObjectiveObservation &observation,
                  std::uint64_t value, ApplicationObjectiveEvidence evidence,
                  std::uint16_t confidencePermille = 1000,
                  bool outOfDistribution = false) {
  observation.value = value;
  observation.evidence = evidence;
  observation.confidencePermille = confidencePermille;
  observation.outOfDistribution = outOfDistribution;
}

ApplicationPairDecisionDisposition mapIncompleteReasonToPairDisposition(
    const dse::DsePlanIncompleteReason &reason) {
  if (const auto *candidate =
          std::get_if<dse::CandidateGeneratorIncompleteReason>(&reason)) {
    switch (*candidate) {
    case dse::CandidateGeneratorIncompleteReason::CancelledOrTimeout:
      return ApplicationPairDecisionDisposition::CancelledOrTimeout;
    case dse::CandidateGeneratorIncompleteReason::SemanticLimitReached:
      return ApplicationPairDecisionDisposition::BudgetExhausted;
    case dse::CandidateGeneratorIncompleteReason::ProviderUnavailable:
    case dse::CandidateGeneratorIncompleteReason::ExecutionFailed:
      return ApplicationPairDecisionDisposition::ImplementationFailure;
    case dse::CandidateGeneratorIncompleteReason::Unsupported:
      return ApplicationPairDecisionDisposition::UnsupportedSemantic;
    case dse::CandidateGeneratorIncompleteReason::ProofNotEstablished:
      return ApplicationPairDecisionDisposition::MappingProofNotEstablished;
    }
  }
  if (const auto *promotion =
          std::get_if<dse::PromotionAcquisitionIncompleteReason>(&reason)) {
    switch (*promotion) {
    case dse::PromotionAcquisitionIncompleteReason::CancelledOrTimeout:
      return ApplicationPairDecisionDisposition::CancelledOrTimeout;
    case dse::PromotionAcquisitionIncompleteReason::SemanticWorkLimit:
      return ApplicationPairDecisionDisposition::BudgetExhausted;
    case dse::PromotionAcquisitionIncompleteReason::ProviderUnavailable:
      return ApplicationPairDecisionDisposition::ImplementationFailure;
    case dse::PromotionAcquisitionIncompleteReason::Unsupported:
    case dse::PromotionAcquisitionIncompleteReason::ObjectiveUnavailable:
      return ApplicationPairDecisionDisposition::UnsupportedSemantic;
    }
  }
  if (const auto *selection =
          std::get_if<dse::IncompleteSelectionReason>(&reason)) {
    switch (*selection) {
    case dse::IncompleteSelectionReason::CancelledOrTimeoutEvidence:
      return ApplicationPairDecisionDisposition::CancelledOrTimeout;
    case dse::IncompleteSelectionReason::MissingEvidence:
    case dse::IncompleteSelectionReason::UnsupportedEvidence:
    case dse::IncompleteSelectionReason::NonComparableEvidence:
    case dse::IncompleteSelectionReason::ObjectiveUnavailable:
      return ApplicationPairDecisionDisposition::MappingProofNotEstablished;
    case dse::IncompleteSelectionReason::ExecutionFailedEvidence:
      return ApplicationPairDecisionDisposition::ImplementationFailure;
    }
  }
  return ApplicationPairDecisionDisposition::MappingProofNotEstablished;
}

ApplicationPairDecisionRecord deriveApplicationPairDecision(
    const PreparedApplicationBuild &prepared,
    const std::vector<ApplicationMappingCandidateOutcome> &outcomes,
    const dse::JointDesignExecution &execution,
    llvm::ArrayRef<ApplicationPairQualityInvocationRecord> qualityInvocations) {
  const dse::JointDesignExecutionSummary &summary = execution.summary;
  const auto invocationRunKey = execution.invocationRunKey();
  ApplicationPairDecisionRecord result;
  result.selectedObjective = makeUnsupportedObjectiveVector();
  result.portfolioInput = prepared.portfolioInput;
  if (result.portfolioInput)
    result.portfolioExecutionBinding =
        ApplicationPortfolioExecutionBinding::DeclaredOnly;
  result.invocationRunKey =
      invocationRunKey ? invocationRunKey : prepared.preMappingInvocationRunKey;
  result.manifestJoinStatus =
      invocationRunKey
          ? ApplicationPairManifestJoinStatus::OwnerScopedPlanningClosure
      : prepared.preMappingInvocationRunKey
          ? ApplicationPairManifestJoinStatus::OwnerScopedPlanningClosure
      : summary.attempts.empty()
          ? ApplicationPairManifestJoinStatus::OwnerVerifiedPreAdmission
          : ApplicationPairManifestJoinStatus::Missing;
  if (result.manifestJoinStatus ==
      ApplicationPairManifestJoinStatus::OwnerVerifiedPreAdmission) {
    result.manifestJoinOwner = preAdmissionManifestJoinOwner.str();
    result.manifestJoinContract = preAdmissionManifestJoinContract.str();
    result.manifestJoinOwnerVerified = true;
  }
  result.sourceProgram = prepared.preMappingSourceProgram;
  result.fabric = prepared.preMappingFabric;
  result.workload = prepared.preMappingWorkload;
  result.runtimeInput = prepared.preMappingRuntimeInput;
  auto identity = deriveApplicationPairIdentity(
      prepared.preMappingSourceProgram, prepared.preMappingFabric,
      prepared.preMappingWorkload, prepared.preMappingRuntimeInput);
  if (!identity) {
    // The roots were already admitted by the application preparation owner.
    // Keep a deterministic zero-free record only if an internal corruption is
    // encountered; the caller will retain the diagnostic error separately.
    result.detail = llvm::toString(identity.takeError());
  } else {
    result.pairIdentity = *identity;
  }

  result.hostOnlyBaseline = makeUnsupportedObjectiveVector();
  if (prepared.preMappingSourceHostOnlyWork) {
    setObjective(result.hostOnlyBaseline[static_cast<std::size_t>(
                     ApplicationObjectiveDimension::HostOnlyWork)],
                 *prepared.preMappingSourceHostOnlyWork,
                 ApplicationObjectiveEvidence::RuntimeMeasured);
    result.hostOnlyBaselineComplete = true;
  }
  for (const dse::PreMappingCandidatePlanningRecord &planning :
       prepared.candidateInventory) {
    if (result.hostOnlyBaselineComplete)
      break;
    if (!planning.estimatedRuntimePicoseconds)
      continue;
    setObjective(result.hostOnlyBaseline[static_cast<std::size_t>(
                     ApplicationObjectiveDimension::HostOnlyWork)],
                 *planning.estimatedRuntimePicoseconds,
                 ApplicationObjectiveEvidence::Analytic, 250, true);
    result.hostOnlyBaselineComplete = true;
    break;
  }
  result.planningRecordCount = prepared.candidateInventory.size();
  result.qualityObjectiveDimensionLabels =
      summary.qualityObjectiveDimensionLabels;
  result.qualityDisposition = summary.qualityDisposition;
  result.qualityIncompleteCandidate = summary.qualityIncompleteCandidate;
  result.qualityObservations = summary.qualityObservations;
  result.hardwarePromotionObjectiveDimensionLabels =
      summary.hardwarePromotionObjectiveDimensionLabels;
  result.hardwarePromotionObservations = summary.hardwarePromotionObservations;
  result.qualityInvocations.assign(qualityInvocations.begin(),
                                   qualityInvocations.end());
  result.candidates.reserve(prepared.candidateInventory.size());
  for (std::size_t ordinal = 0; ordinal != prepared.candidateInventory.size();
       ++ordinal) {
    const dse::PreMappingCandidatePlanningRecord &planning =
        prepared.candidateInventory[ordinal];
    if (!planning.candidateIdentity) {
      ++result.nonCandidatePlanningRecordCount;
      continue;
    }
    ApplicationPairCandidateRecord candidate;
    candidate.planningRecordOrdinal = ordinal;
    candidate.candidateIdentity = planning.candidateIdentity;
    candidate.structuredProgram = planning.structuredProgram;
    candidate.canonicalDataflow = planning.canonicalDataflow;
    if (planning.projection)
      candidate.planningProjectionIdentity = planning.projection->identity;
    if (planning.materializedProjection)
      candidate.materializedProjectionIdentity =
          planning.materializedProjection->identity;
    candidate.planningDisposition = planning.disposition;
    candidate.scheduleIntent = planning.scheduleIntent;
    candidate.planningIncompleteReason = planning.incompleteReason;
    candidate.verifiedSpectrum = planning.verifiedSpectrum;
    candidate.objective = makeUnsupportedObjectiveVector();
    if (planning.estimatedRuntimePicoseconds) {
      setObjective(candidate.objective[static_cast<std::size_t>(
                       ApplicationObjectiveDimension::HostOnlyWork)],
                   *planning.estimatedRuntimePicoseconds,
                   ApplicationObjectiveEvidence::Analytic, 250, true);
    }
    for (const ApplicationMappingCandidateOutcome &outcome : outcomes) {
      if (outcome.preMappingCandidateRecordOrdinal != ordinal)
        continue;
      candidate.enteredMapping = true;
      const bool isSelectedOutcome =
          summary.selectedPlanOrdinal &&
          *summary.selectedPlanOrdinal == outcome.planOrdinal &&
          summary.selectedMapping &&
          llvm::is_contained(outcome.systemMappings, *summary.selectedMapping);
      ApplicationPairMappingObservation mappingObservation{
          outcome.planOrdinal,
          outcome.resourceTimeScheduleHintDigest,
          outcome.system,
          outcome.disposition,
          outcome.runtimeDisposition,
          outcome.incompleteReason,
          outcome.systemMappings,
          outcome.runtimeEvidence,
          outcome.oracleEvidence,
          outcome.dfgCycles,
          outcome.cgraCycles,
          outcome.resourceCoreCost,
          std::nullopt};
      if (outcome.resourceTimeSpectrum) {
        if (const auto *verification =
                std::get_if<dse::VerifiedResourceTimeSpectrum>(
                    &outcome.resourceTimeSpectrum->verification)) {
          for (const dse::VerifiedResourceTimeSpectrumScenario &scenario :
               verification->scenarios) {
            if (!candidate.verifiedSpectrum)
              candidate.verifiedSpectrum = scenario.spectrumClass;
            if (!mappingObservation.verifiedSpectrum)
              mappingObservation.verifiedSpectrum = scenario.spectrumClass;
          }
        }
      }
      candidate.mappingObservations.push_back(std::move(mappingObservation));
      if (isSelectedOutcome) {
        candidate.selected = true;
        candidate.planOrdinal = outcome.planOrdinal;
      }
      const auto setObservedDimension =
          [&](ApplicationObjectiveDimension dimension,
              const std::optional<std::uint64_t> &raw) {
            if (raw) {
              setObjective(
                  candidate.objective[static_cast<std::size_t>(dimension)],
                  *raw, ApplicationObjectiveEvidence::RuntimeMeasured);
              return;
            }
          };
      if (isSelectedOutcome) {
        setObservedDimension(ApplicationObjectiveDimension::DfgCycles,
                             outcome.dfgCycles);
        setObservedDimension(ApplicationObjectiveDimension::CgraCycles,
                             outcome.cgraCycles);
        setObservedDimension(ApplicationObjectiveDimension::ResourceCoreCost,
                             outcome.resourceCoreCost);
      }
    }
    // The current JointDesign summary owns invocation-wide Mapping work, not
    // a candidate-local split. Keep this dimension explicitly unsupported on
    // non-selected candidates instead of attributing the whole invocation to
    // each plan.
    candidate.objective[static_cast<std::size_t>(
        ApplicationObjectiveDimension::MappingWork)] =
        unsupportedObjective(ApplicationObjectiveDimension::MappingWork);
    result.candidates.push_back(std::move(candidate));
  }

  const ApplicationPairCandidateRecord *selected = nullptr;
  for (const ApplicationPairCandidateRecord &candidate : result.candidates)
    if (candidate.selected) {
      selected = &candidate;
      break;
    }
  if (selected) {
    result.selectedCandidateIdentity = selected->candidateIdentity;
    for (const ApplicationMappingCandidateOutcome &outcome : outcomes)
      if (outcome.preMappingCandidateRecordOrdinal ==
              selected->planningRecordOrdinal &&
          summary.selectedPlanOrdinal &&
          outcome.planOrdinal == *summary.selectedPlanOrdinal &&
          summary.selectedMapping &&
          llvm::is_contained(outcome.systemMappings,
                             *summary.selectedMapping)) {
        result.selectedSystem = outcome.system;
        result.selectedSystemMapping = summary.selectedMapping;
        const auto &objective = selected->objective;
        const auto dfg = objective[static_cast<std::size_t>(
            ApplicationObjectiveDimension::DfgCycles)];
        const auto cgra = objective[static_cast<std::size_t>(
            ApplicationObjectiveDimension::CgraCycles)];
        if (dfg.value) {
          setObjective(result.hostOnlyBaseline[static_cast<std::size_t>(
                           ApplicationObjectiveDimension::DfgCycles)],
                       *dfg.value,
                       ApplicationObjectiveEvidence::RuntimeMeasured);
        }
        result.selectedObjective = objective;
        setObjective(result.selectedObjective[static_cast<std::size_t>(
                         ApplicationObjectiveDimension::MappingWork)],
                     summary.techMappingDispatchCount +
                         summary.spatialPnrDispatchCount +
                         summary.systemPnrDispatchCount,
                     ApplicationObjectiveEvidence::RuntimeMeasured);
        result.finalApplicationQorComplete = result.hostOnlyBaselineComplete &&
                                             dfg.value.has_value() &&
                                             cgra.value.has_value();
        if (result.portfolioInput &&
            outcome.runtimeDisposition ==
                ApplicationMappingRuntimeDisposition::Completed &&
            !outcome.runtimeEvidence.empty())
          result.portfolioExecutionBinding =
              ApplicationPortfolioExecutionBinding::CanonicalSimulation;
        if (result.portfolioInput &&
            result.portfolioInput->input.profile.warmupSamples == 0 &&
            result.portfolioInput->input.profile.measuredSamples == 1 &&
            outcome.runtimeDisposition ==
                ApplicationMappingRuntimeDisposition::Completed &&
            !outcome.runtimeEvidence.empty() &&
            !outcome.oracleEvidence.empty() && outcome.dfgCycles &&
            outcome.cgraCycles)
          result.portfolioExecutionBinding =
              ApplicationPortfolioExecutionBinding::
                  CanonicalSimulationAndOracle;
        const bool portfolioExecutionComplete =
            !result.portfolioInput || result.portfolioExecutionBinding ==
                                          ApplicationPortfolioExecutionBinding::
                                              CanonicalSimulationAndOracle;
        result.finalApplicationQorComplete =
            result.finalApplicationQorComplete && portfolioExecutionComplete;
        if (outcome.runtimeDisposition !=
            ApplicationMappingRuntimeDisposition::Completed) {
          const auto setRuntimeDetail = [&](llvm::StringRef fallback) {
            result.detail = outcome.incompleteReason
                                ? dse::toString(*outcome.incompleteReason).str()
                                : fallback.str();
          };
          switch (outcome.runtimeDisposition) {
          case ApplicationMappingRuntimeDisposition::Unsupported:
            result.disposition =
                ApplicationPairDecisionDisposition::UnsupportedSemantic;
            setRuntimeDetail("selected application runtime is unsupported");
            break;
          case ApplicationMappingRuntimeDisposition::ProofNotEstablished:
          case ApplicationMappingRuntimeDisposition::NotRequested:
            result.disposition =
                ApplicationPairDecisionDisposition::MappingProofNotEstablished;
            setRuntimeDetail(
                "selected application runtime proof was not established");
            break;
          case ApplicationMappingRuntimeDisposition::ExecutionFailed:
            result.disposition =
                ApplicationPairDecisionDisposition::ImplementationFailure;
            setRuntimeDetail("selected application runtime execution failed");
            break;
          case ApplicationMappingRuntimeDisposition::CancelledOrTimeout:
            result.disposition =
                ApplicationPairDecisionDisposition::CancelledOrTimeout;
            setRuntimeDetail(
                "selected application runtime was cancelled or timed out");
            break;
          case ApplicationMappingRuntimeDisposition::Completed:
            llvm_unreachable("completed runtime disposition handled above");
          }
        } else if (!portfolioExecutionComplete) {
          result.disposition =
              ApplicationPairDecisionDisposition::MappingProofNotEstablished;
          result.detail = "selected Mapping lacks completed runtime and "
                          "comparison evidence";
        } else if (outcome.system != prepared.preMappingFabric) {
          result.disposition =
              ApplicationPairDecisionDisposition::HardwareDseAlternative;
        } else if (dfg.value && cgra.value && *cgra.value < *dfg.value) {
          result.disposition =
              ApplicationPairDecisionDisposition::VerifiedAcceleration;
        } else if (dfg.value && cgra.value) {
          result.disposition = ApplicationPairDecisionDisposition::
              VerifiedFeasibleButNotBeneficial;
        } else {
          result.disposition =
              ApplicationPairDecisionDisposition::MappingProofNotEstablished;
          result.detail =
              "selected Mapping lacks measured DFG or CGRA cycle evidence";
        }
        break;
      }
    if (!result.selectedSystemMapping && !result.detail)
      result.detail =
          "selected JointDesign checkpoint has no exact application Mapping "
          "outcome";
  }
  if (!selected) {
    const bool exactHardwareIncompatibility =
        !summary.attempts.empty() &&
        llvm::all_of(summary.attempts,
                     [](const auto &attempt) {
                       return attempt.disposition ==
                                  dse::JointDesignAttemptDisposition::
                                      ProvenNoFeasibleCandidate &&
                              attempt.systemMappings.empty();
                     }) &&
        !summary.jointFrontierTruncated;
    if (exactHardwareIncompatibility) {
      result.disposition =
          ApplicationPairDecisionDisposition::ExactHardwareIncompatible;
      result.detail = "all admitted System candidates published typed "
                      "ProvenNoFeasibleCandidate witnesses";
    }
    for (const dse::JointDesignAttemptRecord &attempt : summary.attempts)
      if (attempt.incompleteReason) {
        result.disposition =
            mapIncompleteReasonToPairDisposition(*attempt.incompleteReason);
        result.detail = dse::toString(*attempt.incompleteReason).str();
        break;
      }
    if (result.disposition ==
            ApplicationPairDecisionDisposition::ImplementationFailure &&
        summary.declaredWorkExhausted) {
      result.disposition = ApplicationPairDecisionDisposition::BudgetExhausted;
      result.detail =
          "declared joint-design work was exhausted without a selected Mapping";
    }
    if (summary.declaredWorkExhausted &&
        result.disposition ==
            ApplicationPairDecisionDisposition::MappingProofNotEstablished) {
      result.disposition = ApplicationPairDecisionDisposition::BudgetExhausted;
      result.detail =
          "declared joint-design work was exhausted before Mapping proof";
    }
    if (result.disposition ==
            ApplicationPairDecisionDisposition::ImplementationFailure &&
        !result.detail)
      result.detail =
          "joint design summary has no selected Mapping checkpoint or typed "
          "incomplete witness";
  }
  return result;
}

ApplicationPairDecisionRecord makePreparationPairDecision(
    const std::optional<ArtifactRootReference> &sourceProgram,
    const std::optional<ArtifactRootReference> &fabric,
    const std::optional<ArtifactRootReference> &workload,
    const std::optional<ArtifactRootReference> &runtimeInput,
    llvm::ArrayRef<dse::PreMappingCandidatePlanningRecord> inventory,
    ApplicationPairDecisionDisposition disposition, llvm::StringRef detail,
    std::optional<std::uint64_t> sourceHostOnlyWork,
    std::optional<std::array<std::uint8_t, 32>> invocationRunKey,
    bool ownerVerifiedPreAdmission,
    std::optional<SelectedApplicationInput> portfolioInput) {
  ApplicationPairDecisionRecord result;
  result.selectedObjective = makeUnsupportedObjectiveVector();
  result.portfolioInput = std::move(portfolioInput);
  if (result.portfolioInput)
    result.portfolioExecutionBinding =
        ApplicationPortfolioExecutionBinding::DeclaredOnly;
  result.invocationRunKey = std::move(invocationRunKey);
  if (result.invocationRunKey)
    result.manifestJoinStatus =
        ApplicationPairManifestJoinStatus::OwnerScopedPlanningClosure;
  result.disposition = disposition;
  if (!result.invocationRunKey) {
    if (ownerVerifiedPreAdmission) {
      result.manifestJoinStatus =
          ApplicationPairManifestJoinStatus::OwnerVerifiedPreAdmission;
      result.manifestJoinOwner = preAdmissionManifestJoinOwner.str();
      result.manifestJoinContract = preAdmissionManifestJoinContract.str();
      result.manifestJoinOwnerVerified = true;
    } else {
      // A root-bearing invocation must carry the canonical run key. The only
      // valid keyless boundary is the explicit owner-scoped exception above.
      result.manifestJoinStatus = ApplicationPairManifestJoinStatus::Missing;
    }
  }
  result.detail = detail.str();
  result.sourceProgram = sourceProgram;
  result.fabric = fabric;
  result.workload = workload;
  result.runtimeInput = runtimeInput;
  result.planningRecordCount = inventory.size();
  result.hostOnlyBaseline = makeUnsupportedObjectiveVector();
  if (sourceHostOnlyWork) {
    setObjective(result.hostOnlyBaseline[static_cast<std::size_t>(
                     ApplicationObjectiveDimension::HostOnlyWork)],
                 *sourceHostOnlyWork,
                 ApplicationObjectiveEvidence::RuntimeMeasured);
    result.hostOnlyBaselineComplete = true;
  }
  result.candidates.reserve(inventory.size());
  for (std::size_t ordinal = 0; ordinal != inventory.size(); ++ordinal) {
    const auto &record = inventory[ordinal];
    if (!result.hostOnlyBaselineComplete &&
        record.estimatedRuntimePicoseconds) {
      setObjective(result.hostOnlyBaseline[static_cast<std::size_t>(
                       ApplicationObjectiveDimension::HostOnlyWork)],
                   *record.estimatedRuntimePicoseconds,
                   ApplicationObjectiveEvidence::Analytic, 250, true);
      result.hostOnlyBaselineComplete = true;
    }
    if (!record.candidateIdentity) {
      ++result.nonCandidatePlanningRecordCount;
      continue;
    }
    ApplicationPairCandidateRecord candidate;
    candidate.planningRecordOrdinal = ordinal;
    candidate.candidateIdentity = record.candidateIdentity;
    candidate.structuredProgram = record.structuredProgram;
    candidate.canonicalDataflow = record.canonicalDataflow;
    if (record.projection)
      candidate.planningProjectionIdentity = record.projection->identity;
    if (record.materializedProjection)
      candidate.materializedProjectionIdentity =
          record.materializedProjection->identity;
    candidate.planningDisposition = record.disposition;
    candidate.scheduleIntent = record.scheduleIntent;
    candidate.planningIncompleteReason = record.incompleteReason;
    candidate.verifiedSpectrum = record.verifiedSpectrum;
    candidate.objective = makeUnsupportedObjectiveVector();
    if (record.estimatedRuntimePicoseconds)
      setObjective(candidate.objective[static_cast<std::size_t>(
                       ApplicationObjectiveDimension::HostOnlyWork)],
                   *record.estimatedRuntimePicoseconds,
                   ApplicationObjectiveEvidence::Analytic, 250, true);
    result.candidates.push_back(std::move(candidate));
  }
  if (sourceProgram && fabric && workload && runtimeInput) {
    auto identity = deriveApplicationPairIdentity(*sourceProgram, *fabric,
                                                  *workload, *runtimeInput);
    if (identity)
      result.pairIdentity = *identity;
    else
      result.detail = llvm::toString(identity.takeError());
  }
  return result;
}

ApplicationPairDecisionRecord makePreAdmissionFailurePairDecision(
    std::optional<SelectedApplicationInput> portfolioInput,
    const ArtifactRootReference &requestedSystem,
    ApplicationPairDecisionDisposition disposition, llvm::StringRef detail) {
  ApplicationPairDecisionRecord decision;
  decision.selectedObjective = makeUnsupportedObjectiveVector();
  decision.portfolioInput = std::move(portfolioInput);
  decision.manifestJoinStatus =
      ApplicationPairManifestJoinStatus::OwnerVerifiedPreAdmission;
  decision.manifestJoinOwner = preAdmissionManifestJoinOwner.str();
  decision.manifestJoinContract = preAdmissionManifestJoinContract.str();
  decision.manifestJoinOwnerVerified = true;
  decision.fabric = requestedSystem;
  decision.disposition = disposition;
  decision.detail = "owner_verified_pre_admission: " + detail.str();
  decision.hostOnlyBaseline = makeUnsupportedObjectiveVector();
  return decision;
}

} // namespace build_detail

llvm::StringRef toString(ApplicationPairManifestJoinStatus value) {
  switch (value) {
  case ApplicationPairManifestJoinStatus::Exact:
    return "exact";
  case ApplicationPairManifestJoinStatus::OwnerScopedPlanningClosure:
    return "owner_scoped_planning_closure";
  case ApplicationPairManifestJoinStatus::OwnerVerifiedPreAdmission:
    return "owner_verified_pre_admission";
  case ApplicationPairManifestJoinStatus::Missing:
    return "missing";
  }
  llvm_unreachable("unknown application pair manifest join status");
}

llvm::StringRef toString(ApplicationPortfolioExecutionBinding value) {
  switch (value) {
  case ApplicationPortfolioExecutionBinding::NotSelected:
    return "not_selected";
  case ApplicationPortfolioExecutionBinding::DeclaredOnly:
    return "declared_only";
  case ApplicationPortfolioExecutionBinding::CanonicalSimulation:
    return "canonical_simulation";
  case ApplicationPortfolioExecutionBinding::CanonicalSimulationAndOracle:
    return "canonical_simulation_and_oracle";
  }
  llvm_unreachable("unknown application portfolio execution binding");
}

ApplicationPairDecisionRecord makeUnsupportedPortfolioProfilePairDecision(
    SelectedApplicationInput selection,
    const ArtifactRootReference &requestedSystem, llvm::StringRef detail) {
  return build_detail::makePreparationPairDecision(
      std::nullopt, requestedSystem, std::nullopt, std::nullopt, {},
      ApplicationPairDecisionDisposition::UnsupportedSemantic, detail,
      std::nullopt, std::nullopt, true, std::move(selection));
}

} // namespace loom::application
