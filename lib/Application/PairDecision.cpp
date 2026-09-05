#include "Application/Build.h"
#include "BuildInternal.h"

#include "DSE/Promotion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
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

constexpr std::uint16_t analyticConfidencePermille = 250;
constexpr std::uint16_t calibratedConfidencePermille = 500;
constexpr std::uint64_t partsPerMillion = 1000000;

/// Expresses measured CGRA cycles in the funnel's picosecond basis and states
/// the funnel's relative prediction error against them. The prediction stays
/// untouched; both values are absent when the basis, the measurement, or the
/// 64-bit domain is missing.
void setFunnelExactMeasurement(ApplicationPairMappingObservation &observation,
                               std::uint64_t analyticClockPeriodPicoseconds) {
  if (!observation.cgraCycles || analyticClockPeriodPicoseconds == 0)
    return;
  const unsigned __int128 measured =
      static_cast<unsigned __int128>(*observation.cgraCycles) *
      analyticClockPeriodPicoseconds;
  if (measured > std::numeric_limits<std::uint64_t>::max())
    return;
  observation.measuredMakespanPicoseconds =
      static_cast<std::uint64_t>(measured);
  if (!observation.predictedMakespanPicoseconds || measured == 0)
    return;
  const unsigned __int128 predicted = *observation.predictedMakespanPicoseconds;
  const unsigned __int128 difference =
      predicted > measured ? predicted - measured : measured - predicted;
  const unsigned __int128 error = difference * partsPerMillion / measured;
  if (error <= std::numeric_limits<std::uint64_t>::max())
    observation.predictionErrorPartsPerMillion =
        static_cast<std::uint64_t>(error);
}

/// Pre-Mapping analytic dimensions of one candidate: the projection is exact
/// structural provenance of the ownership, not a calibrated prediction.
void setProjectedObjectiveDimensions(
    std::vector<ApplicationObjectiveObservation> &objective,
    const dse::PreMappingCandidateProjection &projection) {
  setObjective(objective[static_cast<std::size_t>(
                   ApplicationObjectiveDimension::HostResidualWork)],
               projection.hostDynamicLeafExecutions,
               ApplicationObjectiveEvidence::Analytic,
               analyticConfidencePermille, true);
  if (projection.estimatedCutTrafficBytes)
    setObjective(objective[static_cast<std::size_t>(
                     ApplicationObjectiveDimension::CutTransferWork)],
                 *projection.estimatedCutTrafficBytes,
                 ApplicationObjectiveEvidence::Analytic,
                 analyticConfidencePermille, true);
  setObjective(objective[static_cast<std::size_t>(
                   ApplicationObjectiveDimension::LaunchSynchronizationWork)],
               projection.launchSynchronizationCost,
               ApplicationObjectiveEvidence::Analytic,
               analyticConfidencePermille, true);
}

std::optional<long double>
decimalMeasure(const ResolvedObjectiveScalar &value) {
  const auto *decimal = std::get_if<ResolvedObjectiveDecimal>(&value);
  if (!decimal)
    return std::nullopt;
  return static_cast<long double>(decimal->coefficient) *
         std::pow(10.0L, static_cast<long double>(decimal->base10Exponent));
}

std::optional<std::uint64_t> scaledUnit(long double value,
                                        long double unitsPerBase) {
  const long double scaled = std::round(value * unitsPerBase);
  if (!(scaled >= 0.0L) ||
      scaled >
          static_cast<long double>(std::numeric_limits<std::uint64_t>::max()))
    return std::nullopt;
  return static_cast<std::uint64_t>(scaled);
}

/// Physical dimensions of the selected Mapping from its completed calibrated
/// FPA observation: area, power, and the energy of one measured CGRA execution
/// at the predicted limiting clock. Absent or incomplete observations leave
/// the dimensions unsupported.
void setCalibratedPhysicalDimensions(
    std::vector<ApplicationObjectiveObservation> &objective,
    const dse::JointDesignExecutionSummary &summary,
    const ArtifactRootReference &selectedMapping,
    std::optional<std::uint64_t> cgraCycles) {
  const std::vector<std::string> &labels =
      summary.qualityObjectiveDimensionLabels;
  for (const dse::JointDesignQualityObservation &observation :
       summary.qualityObservations) {
    if (observation.candidate != selectedMapping ||
        observation.incompleteReason ||
        observation.provenance.calibratedModelSupport !=
            dse::JointDesignCalibratedModelSupport::InDomain)
      continue;
    const std::vector<ResolvedObjectiveScalar> &measures =
        observation.provenance.rawMeasures;
    if (measures.size() != labels.size())
      continue;
    const auto measure =
        [&](llvm::StringRef label) -> std::optional<long double> {
      for (std::size_t ordinal = 0; ordinal != labels.size(); ++ordinal)
        if (labels[ordinal] == label)
          return decimalMeasure(measures[ordinal]);
      return std::nullopt;
    };
    const auto frequencyHz = measure("limiting_clock_frequency");
    const auto areaSquareMeters = measure("total_area");
    const auto dynamicWatts = measure("dynamic_power");
    const auto leakageWatts = measure("leakage_power");
    if (areaSquareMeters)
      if (auto area = scaledUnit(*areaSquareMeters, 1e12L))
        setObjective(objective[static_cast<std::size_t>(
                         ApplicationObjectiveDimension::Area)],
                     *area, ApplicationObjectiveEvidence::Calibrated,
                     calibratedConfidencePermille);
    if (!dynamicWatts || !leakageWatts)
      return;
    const long double powerWatts = *dynamicWatts + *leakageWatts;
    if (auto power = scaledUnit(powerWatts, 1e6L))
      setObjective(objective[static_cast<std::size_t>(
                       ApplicationObjectiveDimension::Power)],
                   *power, ApplicationObjectiveEvidence::Calibrated,
                   calibratedConfidencePermille);
    if (frequencyHz && *frequencyHz > 0.0L && cgraCycles)
      if (auto energy = scaledUnit(
              powerWatts * static_cast<long double>(*cgraCycles) / *frequencyHz,
              1e12L))
        setObjective(objective[static_cast<std::size_t>(
                         ApplicationObjectiveDimension::Energy)],
                     *energy, ApplicationObjectiveEvidence::Calibrated,
                     calibratedConfidencePermille);
    return;
  }
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

ApplicationPairDecisionDisposition
mapResourceTimeFrontierReasonToPairDisposition(
    dse::ResourceTimeFrontierIncompleteReason reason) {
  switch (reason) {
  case dse::ResourceTimeFrontierIncompleteReason::BudgetExhausted:
    return ApplicationPairDecisionDisposition::BudgetExhausted;
  case dse::ResourceTimeFrontierIncompleteReason::CancelledOrTimeout:
    return ApplicationPairDecisionDisposition::CancelledOrTimeout;
  case dse::ResourceTimeFrontierIncompleteReason::ProofNotEstablished:
    return ApplicationPairDecisionDisposition::MappingProofNotEstablished;
  case dse::ResourceTimeFrontierIncompleteReason::Unsupported:
    return ApplicationPairDecisionDisposition::UnsupportedSemantic;
  }
  llvm_unreachable("unknown resource-time frontier incomplete reason");
}

std::optional<ApplicationPairDecisionDisposition>
mapRuntimeDispositionToPairDisposition(
    ApplicationMappingRuntimeDisposition disposition) {
  switch (disposition) {
  case ApplicationMappingRuntimeDisposition::Completed:
    return std::nullopt;
  case ApplicationMappingRuntimeDisposition::Unsupported:
    return ApplicationPairDecisionDisposition::UnsupportedSemantic;
  case ApplicationMappingRuntimeDisposition::ProofNotEstablished:
  case ApplicationMappingRuntimeDisposition::NotRequested:
    return ApplicationPairDecisionDisposition::MappingProofNotEstablished;
  case ApplicationMappingRuntimeDisposition::ExecutionFailed:
    return ApplicationPairDecisionDisposition::ImplementationFailure;
  case ApplicationMappingRuntimeDisposition::CancelledOrTimeout:
    return ApplicationPairDecisionDisposition::CancelledOrTimeout;
  }
  llvm_unreachable("unknown application runtime disposition");
}

std::optional<ApplicationPairDecisionDisposition>
mapQualityDispositionToPairDisposition(
    dse::JointDesignQualityDisposition disposition) {
  switch (disposition) {
  case dse::JointDesignQualityDisposition::Unsupported:
    return ApplicationPairDecisionDisposition::UnsupportedSemantic;
  case dse::JointDesignQualityDisposition::ProofNotEstablished:
    return ApplicationPairDecisionDisposition::MappingProofNotEstablished;
  case dse::JointDesignQualityDisposition::ExecutionFailed:
    return ApplicationPairDecisionDisposition::ImplementationFailure;
  case dse::JointDesignQualityDisposition::CancelledOrTimeout:
    return ApplicationPairDecisionDisposition::CancelledOrTimeout;
  case dse::JointDesignQualityDisposition::NotRequested:
  case dse::JointDesignQualityDisposition::Complete:
    return std::nullopt;
  }
  llvm_unreachable("unknown joint quality disposition");
}

std::optional<ApplicationPairDecisionDisposition>
classifyResourceTimeSelectionOutcome(
    const std::optional<dse::ResourceTimeSpectrumFunnelResult> &spectrum,
    std::optional<dse::PreMappingSpectrumClass> requestedClass,
    const std::optional<ArtifactRootReference> &runtimeMapping) {
  if (!requestedClass)
    return std::nullopt;
  const auto reason = resourceTimeSelectionIncompleteReason(
      spectrum, requestedClass, runtimeMapping);
  if (!reason)
    return std::nullopt;
  return mapIncompleteReasonToPairDisposition(
      dse::DsePlanIncompleteReason{*reason});
}

void retainPrioritizedIncompleteReason(
    std::optional<dse::DsePlanIncompleteReason> &retained,
    dse::DsePlanIncompleteReason reason) {
  if (!retained) {
    retained = std::move(reason);
    return;
  }
  const std::array causes = {mapIncompleteReasonToPairDisposition(*retained),
                             mapIncompleteReasonToPairDisposition(reason)};
  if (causes.front() != causes.back() &&
      prioritizeIncompletePairDisposition(causes, false) == causes.back())
    retained = std::move(reason);
}

ApplicationIncrementalMappingOutcome deriveIncrementalMappingOutcome(
    const ApplicationIncrementalMappingObservation &observation) {
  const std::optional<dse::PreMappingSpectrumClass> requestedClass =
      dse::spectrumClassForEndpoint(observation.spectrumEndpoint);
  const auto spectrumAdmits =
      [&](const std::optional<dse::ResourceTimeSpectrumFunnelResult> &spectrum,
          const std::optional<ArtifactRootReference> &mapping) {
        if (!mapping || !spectrum)
          return false;
        return mapping && spectrum &&
               dse::resourceTimeSpectrumAdmitsMappingClass(*spectrum, *mapping,
                                                           requestedClass);
      };
  const auto runtimeEvidenceClosed =
      [](llvm::ArrayRef<ArtifactRootReference> runtimeEvidence,
         llvm::ArrayRef<ArtifactRootReference> oracleEvidence) {
        return !runtimeEvidence.empty() && !oracleEvidence.empty() &&
               llvm::all_of(oracleEvidence, [&](const auto &oracle) {
                 return llvm::is_contained(runtimeEvidence, oracle);
               });
      };
  const auto retainedEligibleMapping =
      [](const std::optional<ArtifactRootReference> &mapping,
         llvm::ArrayRef<ArtifactRootReference> candidates,
         llvm::ArrayRef<ArtifactRootReference> eligible) {
        return mapping && llvm::is_contained(candidates, *mapping) &&
               llvm::is_contained(eligible, *mapping);
      };
  const bool coldVerified =
      observation.coldExecutionIncompleteReasons.empty() &&
      retainedEligibleMapping(observation.coldMapping,
                              observation.coldMappingCandidates,
                              observation.coldEligibleMappings) &&
      observation.coldRuntimeDisposition ==
          ApplicationMappingRuntimeDisposition::Completed &&
      runtimeEvidenceClosed(observation.coldRuntimeEvidence,
                            observation.coldOracleEvidence) &&
      spectrumAdmits(observation.coldSelectionSpectrum,
                     observation.coldMapping);
  const bool incrementalVerified =
      observation.incrementalExecutionIncompleteReasons.empty() &&
      retainedEligibleMapping(observation.childMapping,
                              observation.incrementalMappingCandidates,
                              observation.incrementalEligibleMappings) &&
      observation.incrementalRuntimeDisposition ==
          ApplicationMappingRuntimeDisposition::Completed &&
      runtimeEvidenceClosed(observation.incrementalRuntimeEvidence,
                            observation.incrementalOracleEvidence) &&
      spectrumAdmits(observation.incrementalSelectionSpectrum,
                     observation.childMapping);
  if (coldVerified && incrementalVerified)
    return {dse::JointDesignAttemptDisposition::Verified, std::nullopt, true};

  std::optional<dse::DsePlanIncompleteReason> retained;
  const auto retainSide =
      [&](const std::optional<ArtifactRootReference> &mapping,
          llvm::ArrayRef<dse::DsePlanIncompleteReason> executionReasons,
          const std::optional<dse::ResourceTimeSpectrumFunnelResult> &spectrum,
          ApplicationMappingRuntimeDisposition runtimeDisposition) {
        for (const dse::DsePlanIncompleteReason &reason : executionReasons)
          retainPrioritizedIncompleteReason(retained, reason);
        if (!executionReasons.empty())
          return;
        const auto spectrumReason = resourceTimeSelectionIncompleteReason(
            spectrum, requestedClass, mapping);
        if (spectrumReason)
          retainPrioritizedIncompleteReason(retained, *spectrumReason);
        else if (!spectrumAdmits(spectrum, mapping))
          retainPrioritizedIncompleteReason(
              retained,
              dse::CandidateGeneratorIncompleteReason::ProofNotEstablished);
        if (mapping)
          if (auto runtimeReason =
                  resourceTimeRuntimeIncompleteReason(runtimeDisposition))
            retainPrioritizedIncompleteReason(retained, *runtimeReason);
      };
  retainSide(
      observation.coldMapping, observation.coldExecutionIncompleteReasons,
      observation.coldSelectionSpectrum, observation.coldRuntimeDisposition);
  retainSide(observation.childMapping,
             observation.incrementalExecutionIncompleteReasons,
             observation.incrementalSelectionSpectrum,
             observation.incrementalRuntimeDisposition);
  if (!retained)
    retainPrioritizedIncompleteReason(
        retained, dse::CandidateGeneratorIncompleteReason::ProofNotEstablished);
  return {dse::JointDesignAttemptDisposition::Incomplete, std::move(retained),
          false};
}

ApplicationPairDecisionDisposition prioritizeIncompletePairDisposition(
    llvm::ArrayRef<ApplicationPairDecisionDisposition> causes,
    bool declaredWorkExhausted) {
  const auto priority = [](ApplicationPairDecisionDisposition disposition) {
    switch (disposition) {
    case ApplicationPairDecisionDisposition::BudgetExhausted:
      return 0U;
    case ApplicationPairDecisionDisposition::UnsupportedSemantic:
      return 1U;
    case ApplicationPairDecisionDisposition::MappingProofNotEstablished:
      return 2U;
    case ApplicationPairDecisionDisposition::ImplementationFailure:
      return 3U;
    case ApplicationPairDecisionDisposition::CancelledOrTimeout:
      return 4U;
    case ApplicationPairDecisionDisposition::VerifiedAcceleration:
    case ApplicationPairDecisionDisposition::VerifiedFeasibleButNotBeneficial:
    case ApplicationPairDecisionDisposition::NoPromisingCandidate:
    case ApplicationPairDecisionDisposition::ExactHardwareIncompatible:
    case ApplicationPairDecisionDisposition::HardwareDseAlternative:
      llvm_unreachable("complete disposition is not an incomplete cause");
    }
    llvm_unreachable("unknown application pair disposition");
  };
  if (causes.empty())
    return declaredWorkExhausted
               ? ApplicationPairDecisionDisposition::BudgetExhausted
               : ApplicationPairDecisionDisposition::ImplementationFailure;
  return *llvm::max_element(causes, [&](auto lhs, auto rhs) {
    return priority(lhs) < priority(rhs);
  });
}

ApplicationPairDecisionDisposition classifyPreMappingNoFeasibleOutcome(
    const dse::CompletedPreMappingNoFeasibleCandidate &outcome) {
  // A representable candidate the exact Fabric refused is the decision; a
  // coordinate whose schedule intent had no applicable candidate does not
  // turn that hardware refusal into an unsupported program.
  const bool fabricRefused = exactFabricOwnershipRejection(outcome) != nullptr;
  std::vector<ApplicationPairDecisionDisposition> causes;
  for (const dse::PreMappingCandidatePlanningRecord &record :
       outcome.candidateInventory) {
    if (record.incompleteReason) {
      causes.push_back(
          mapIncompleteReasonToPairDisposition(*record.incompleteReason));
      continue;
    }
    switch (record.disposition) {
    case dse::PreMappingCandidatePlanningDisposition::ExactGateRejected:
      break;
    case dse::PreMappingCandidatePlanningDisposition::Unsupported:
      if (!fabricRefused)
        causes.push_back(
            ApplicationPairDecisionDisposition::UnsupportedSemantic);
      break;
    case dse::PreMappingCandidatePlanningDisposition::Unknown:
    case dse::PreMappingCandidatePlanningDisposition::Retained:
    case dse::PreMappingCandidatePlanningDisposition::HeuristicPruned:
      causes.push_back(
          ApplicationPairDecisionDisposition::MappingProofNotEstablished);
      break;
    case dse::PreMappingCandidatePlanningDisposition::CancelledOrTimeout:
      causes.push_back(ApplicationPairDecisionDisposition::CancelledOrTimeout);
      break;
    case dse::PreMappingCandidatePlanningDisposition::CoordinateBudget:
    case dse::PreMappingCandidatePlanningDisposition::
        ProgramMaterializationBudget:
    case dse::PreMappingCandidatePlanningDisposition::AnalyticEvaluationBudget:
    case dse::PreMappingCandidatePlanningDisposition::FunctionalReplayBudget:
    case dse::PreMappingCandidatePlanningDisposition::DataflowPromotionBudget:
    case dse::PreMappingCandidatePlanningDisposition::MappingPairBudget:
      causes.push_back(ApplicationPairDecisionDisposition::BudgetExhausted);
      break;
    }
  }
  if (!outcome.completeness.domainComplete ||
      !outcome.completeness.budgetComplete)
    causes.push_back(ApplicationPairDecisionDisposition::BudgetExhausted);
  if (!outcome.completeness.providerComplete ||
      !outcome.completeness.evidenceComplete ||
      !outcome.completeness.selectionComplete)
    causes.push_back(
        ApplicationPairDecisionDisposition::MappingProofNotEstablished);
  if (!causes.empty())
    return prioritizeIncompletePairDisposition(causes, false);
  // The program then keeps its verified host path, and the decision names
  // the incompatible hardware rather than an absent candidate.
  return fabricRefused
             ? ApplicationPairDecisionDisposition::ExactHardwareIncompatible
             : ApplicationPairDecisionDisposition::NoPromisingCandidate;
}

const dse::StructuredOwnershipCandidateRejectionRecord *
exactFabricOwnershipRejection(
    const dse::CompletedPreMappingNoFeasibleCandidate &outcome) {
  // A refusal explains the outcome only when no Dataflow candidate was ever
  // planned; once candidates exist, their own records own the reason.
  if (llvm::any_of(outcome.candidateInventory,
                   [](const dse::PreMappingCandidatePlanningRecord &record) {
                     return record.structuredProgram.has_value();
                   }))
    return nullptr;
  const dse::StructuredOwnershipCandidateRejectionRecord *first = nullptr;
  for (const dse::StructuredOwnershipFinalizationRejection &finalization :
       outcome.finalizationRejections) {
    const dse::StructuredOwnershipCandidateRejectionRecord *rejection =
        &finalization.rejection;
    if (rejection->kind != frontend::SpatialOwnershipCandidateRejectionKind::
                               ExactFabricInadmissible)
      continue;
    if (rejection->memoryContract)
      return rejection;
    if (!first)
      first = rejection;
  }
  return first;
}

ApplicationPairDecisionRecord deriveApplicationPairDecision(
    const PreparedApplicationBuild &prepared,
    const std::vector<ApplicationMappingCandidateOutcome> &outcomes,
    const dse::JointDesignExecution &execution,
    llvm::ArrayRef<ApplicationIncrementalMappingObservation>
        incrementalMappingObservations,
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
                 ApplicationObjectiveEvidence::Analytic,
                 analyticConfidencePermille, true);
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
  for (const ApplicationIncrementalMappingObservation &observation :
       incrementalMappingObservations) {
    ++result.resourceTimeMappingRepairAttemptCount;
    if (observation.spectrumEndpoint !=
        prepared.resourceTimePolicy.spectrumEndpoint) {
      retainPrioritizedIncompleteReason(
          result.resourceTimeMappingRepairIncompleteReason,
          dse::CandidateGeneratorIncompleteReason::ExecutionFailed);
      continue;
    }
    const ApplicationIncrementalMappingOutcome outcome =
        deriveIncrementalMappingOutcome(observation);
    if (outcome.verified) {
      ++result.resourceTimeMappingRepairVerifiedCount;
      continue;
    }
    if (outcome.incompleteReason)
      retainPrioritizedIncompleteReason(
          result.resourceTimeMappingRepairIncompleteReason,
          *outcome.incompleteReason);
    else
      retainPrioritizedIncompleteReason(
          result.resourceTimeMappingRepairIncompleteReason,
          dse::CandidateGeneratorIncompleteReason::ProofNotEstablished);
  }
  const std::optional<dse::PreMappingSpectrumClass> requestedSpectrumClass =
      dse::spectrumClassForEndpoint(
          prepared.resourceTimePolicy.spectrumEndpoint);
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
                   ApplicationObjectiveEvidence::Analytic,
                   analyticConfidencePermille, true);
    }
    if (planning.projection)
      setProjectedObjectiveDimensions(candidate.objective,
                                      *planning.projection);
    for (const ApplicationMappingCandidateOutcome &outcome : outcomes) {
      if (outcome.preMappingCandidateRecordOrdinal != ordinal)
        continue;
      candidate.enteredMapping = true;
      if (!candidate.planOrdinal)
        candidate.planOrdinal = outcome.planOrdinal;
      const bool isSelectedOutcome =
          summary.selectedPlanOrdinal &&
          *summary.selectedPlanOrdinal == outcome.planOrdinal &&
          summary.selectedMapping &&
          outcome.runtimeMapping == summary.selectedMapping;
      ApplicationPairMappingObservation mappingObservation{
          outcome.planOrdinal,
          outcome.resourceTimeScheduleHintDigest,
          outcome.system,
          outcome.disposition,
          outcome.runtimeDisposition,
          outcome.incompleteReason,
          outcome.systemMappings,
          outcome.runtimeMapping,
          outcome.runtimeEvidence,
          outcome.oracleEvidence,
          outcome.dfgCycles,
          outcome.cgraCycles,
          outcome.resourceCoreCost,
          std::nullopt,
          std::nullopt};
      mappingObservation.hardwareMutationRepairRecord =
          outcome.hardwareMutationRepairRecord;
      for (const dse::ResourceTimeCandidateFunnelEvaluation &evaluation :
           prepared.resourceTimeFunnel.evaluations) {
        if (evaluation.candidateIdentity != *planning.candidateIdentity)
          continue;
        mappingObservation.physicalModelSupport =
            evaluation.physicalModelSupport;
        if (evaluation.bestHint) {
          mappingObservation.predictedMakespanPicoseconds =
              evaluation.bestHint->estimatedMakespanPicoseconds;
          mappingObservation.predictedSupport = evaluation.bestHint->support;
        }
        break;
      }
      if (outcome.system == prepared.preMappingFabric)
        setFunnelExactMeasurement(
            mappingObservation,
            prepared.preMappingFabricAnalyticClockPeriodPicoseconds);
      if (outcome.resourceTimeSpectrum) {
        if (const auto *verification =
                std::get_if<dse::VerifiedResourceTimeSpectrum>(
                    &outcome.resourceTimeSpectrum->verification)) {
          for (const dse::VerifiedResourceTimeSpectrumScenario &scenario :
               verification->scenarios) {
            if (!candidate.verifiedSpectrum ||
                (requestedSpectrumClass &&
                 scenario.spectrumClass == *requestedSpectrumClass))
              candidate.verifiedSpectrum = scenario.spectrumClass;
            if ((!outcome.runtimeMapping ||
                 llvm::is_contained(scenario.systemMappings,
                                    *outcome.runtimeMapping)) &&
                (!mappingObservation.verifiedSpectrum ||
                 (requestedSpectrumClass &&
                  scenario.spectrumClass == *requestedSpectrumClass)))
              mappingObservation.verifiedSpectrum = scenario.spectrumClass;
          }
        } else
          mappingObservation.resourceTimeSpectrumIncompleteReason =
              std::get<dse::IncompleteResourceTimeSpectrum>(
                  outcome.resourceTimeSpectrum->verification)
                  .reason;
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

  ApplicationFunnelExactComparison &comparison = result.funnelExactComparison;
  comparison.analyticClockPeriodPicoseconds =
      prepared.preMappingFabricAnalyticClockPeriodPicoseconds;
  const ApplicationPairMappingObservation *bestPredicted = nullptr;
  const ApplicationPairMappingObservation *bestMeasured = nullptr;
  for (const ApplicationPairCandidateRecord &candidate : result.candidates)
    for (const ApplicationPairMappingObservation &observation :
         candidate.mappingObservations) {
      ++comparison.mappedCandidates;
      if (observation.predictionErrorPartsPerMillion) {
        ++comparison.predictionErrorCandidates;
        if (!comparison.maximumPredictionErrorPartsPerMillion ||
            *observation.predictionErrorPartsPerMillion >
                *comparison.maximumPredictionErrorPartsPerMillion)
          comparison.maximumPredictionErrorPartsPerMillion =
              observation.predictionErrorPartsPerMillion;
      }
      if (observation.predictedMakespanPicoseconds)
        ++comparison.predictedFeasibleCandidates;
      if (observation.mappingDisposition ==
          dse::JointDesignAttemptDisposition::Verified)
        ++comparison.verifiedCandidates;
      if (observation.cgraCycles)
        ++comparison.measuredCandidates;
      if (observation.physicalModelSupport ==
          dse::ResourceTimeEstimateSupport::OutOfDomain)
        ++comparison.outOfDistributionCandidates;
      if (!observation.predictedMakespanPicoseconds || !observation.cgraCycles)
        continue;
      if (!bestPredicted || *observation.predictedMakespanPicoseconds <
                                *bestPredicted->predictedMakespanPicoseconds)
        bestPredicted = &observation;
      if (!bestMeasured || *observation.cgraCycles < *bestMeasured->cgraCycles)
        bestMeasured = &observation;
    }
  if (bestPredicted && bestMeasured)
    comparison.bestRankingMatch =
        bestPredicted->system == bestMeasured->system &&
        bestPredicted->scheduleHintDigest == bestMeasured->scheduleHintDigest;

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
          outcome.planOrdinal < prepared.mappingAlternatives.size() &&
          summary.selectedMapping &&
          outcome.runtimeMapping == summary.selectedMapping &&
          ((!requestedSpectrumClass &&
            outcome.resourceTimeScheduleHintDigest ==
                prepared.mappingAlternatives[outcome.planOrdinal]
                    .resourceTimeScheduleHintDigest) ||
           (requestedSpectrumClass && outcome.resourceTimeSpectrum &&
            dse::resourceTimeSpectrumAdmitsMappingClass(
                *outcome.resourceTimeSpectrum, *summary.selectedMapping,
                requestedSpectrumClass)))) {
        result.selectedSystem = outcome.system;
        result.selectedSystemMapping = summary.selectedMapping;
        if (!result.selectedScheduleHintDigest)
          result.selectedScheduleHintDigest =
              outcome.resourceTimeScheduleHintDigest;
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
        setCalibratedPhysicalDimensions(result.selectedObjective, summary,
                                        *summary.selectedMapping,
                                        outcome.cgraCycles);
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
            if (const auto &refusal = outcome.runtimeMemoryContractRefusal)
              result.detail =
                  ("exact CGRA execution provider does not model the " +
                   dataflow::memoryContractClassSpelling(
                       refusal->contractClass) +
                   " memory contract of actor " +
                   llvm::Twine(refusal->actor.entity.value()))
                      .str();
            else
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
        !summary.jointFrontierTruncated &&
        prepared.preMappingCompleteness.exactComplete() &&
        !prepared.resourceTimeFunnel.truncated &&
        !prepared.resourceTimeFunnel.incompleteReason;
    if (exactHardwareIncompatibility) {
      result.disposition =
          ApplicationPairDecisionDisposition::ExactHardwareIncompatible;
      result.detail = "all admitted System candidates published typed "
                      "ProvenNoFeasibleCandidate witnesses";
    } else {
      std::vector<ApplicationPairDecisionDisposition> incompleteCauses;
      if (prepared.resourceTimeFunnel.incompleteReason)
        incompleteCauses.push_back(
            mapResourceTimeFrontierReasonToPairDisposition(
                *prepared.resourceTimeFunnel.incompleteReason));
      for (const dse::JointDesignAttemptRecord &attempt : summary.attempts)
        if (attempt.incompleteReason)
          incompleteCauses.push_back(
              mapIncompleteReasonToPairDisposition(*attempt.incompleteReason));
      for (const ApplicationMappingCandidateOutcome &outcome : outcomes) {
        std::optional<ApplicationPairDecisionDisposition> spectrumDisposition;
        if (outcome.runtimeDisposition ==
                ApplicationMappingRuntimeDisposition::Completed ||
            (outcome.runtimeDisposition ==
                 ApplicationMappingRuntimeDisposition::NotRequested &&
             !outcome.systemMappings.empty()))
          spectrumDisposition = classifyResourceTimeSelectionOutcome(
              outcome.resourceTimeSpectrum, requestedSpectrumClass,
              outcome.runtimeMapping);
        if (spectrumDisposition)
          incompleteCauses.push_back(*spectrumDisposition);
        if (outcome.disposition ==
                dse::JointDesignAttemptDisposition::Incomplete &&
            outcome.incompleteReason)
          incompleteCauses.push_back(
              mapIncompleteReasonToPairDisposition(*outcome.incompleteReason));
        if (outcome.runtimeDisposition !=
            ApplicationMappingRuntimeDisposition::Completed) {
          if (outcome.runtimeDisposition !=
                  ApplicationMappingRuntimeDisposition::NotRequested ||
              (outcome.disposition ==
                   dse::JointDesignAttemptDisposition::Verified &&
               !outcome.systemMappings.empty() && !spectrumDisposition))
            if (auto disposition = mapRuntimeDispositionToPairDisposition(
                    outcome.runtimeDisposition))
              incompleteCauses.push_back(*disposition);
          continue;
        }
      }
      for (const ApplicationPairQualityInvocationRecord &invocation :
           qualityInvocations)
        if (auto disposition = mapQualityDispositionToPairDisposition(
                invocation.qualityDisposition))
          incompleteCauses.push_back(*disposition);
      if (qualityInvocations.empty())
        if (auto disposition = mapQualityDispositionToPairDisposition(
                summary.qualityDisposition))
          incompleteCauses.push_back(*disposition);
      result.disposition = prioritizeIncompletePairDisposition(
          incompleteCauses, summary.declaredWorkExhausted ||
                                summary.jointFrontierTruncated ||
                                prepared.resourceTimeFunnel.truncated);
      if (!result.detail) {
        switch (result.disposition) {
        case ApplicationPairDecisionDisposition::CancelledOrTimeout:
          result.detail = "application Mapping or runtime execution was "
                          "cancelled or timed out";
          break;
        case ApplicationPairDecisionDisposition::ImplementationFailure:
          result.detail = "application Mapping or runtime provider failed";
          break;
        case ApplicationPairDecisionDisposition::MappingProofNotEstablished:
          result.detail = "application Mapping, spectrum, or runtime proof was "
                          "not established";
          break;
        case ApplicationPairDecisionDisposition::UnsupportedSemantic:
          result.detail = "application Mapping or runtime semantics were "
                          "unsupported";
          break;
        case ApplicationPairDecisionDisposition::BudgetExhausted:
          result.detail = "bounded application Mapping work was exhausted";
          break;
        case ApplicationPairDecisionDisposition::VerifiedAcceleration:
        case ApplicationPairDecisionDisposition::
            VerifiedFeasibleButNotBeneficial:
        case ApplicationPairDecisionDisposition::NoPromisingCandidate:
        case ApplicationPairDecisionDisposition::ExactHardwareIncompatible:
        case ApplicationPairDecisionDisposition::HardwareDseAlternative:
          llvm_unreachable("complete disposition has no incomplete detail");
        }
      }
    }
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
