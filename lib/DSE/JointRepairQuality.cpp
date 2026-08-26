#include "DSE/JointHardwareReopen.h"

#include "Common/ArtifactStore.h"
#include "Evaluation/Evidence.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "joint_repair_quality_invalid: " + message);
}

JointDesignQualityIncompleteReason
incompleteReason(JointDesignQualityDisposition disposition) {
  switch (disposition) {
  case JointDesignQualityDisposition::Unsupported:
    return JointDesignQualityIncompleteReason::Unsupported;
  case JointDesignQualityDisposition::ExecutionFailed:
    return JointDesignQualityIncompleteReason::ExecutionFailed;
  case JointDesignQualityDisposition::CancelledOrTimeout:
    return JointDesignQualityIncompleteReason::CancelledOrTimeout;
  case JointDesignQualityDisposition::NotRequested:
  case JointDesignQualityDisposition::Complete:
  case JointDesignQualityDisposition::ProofNotEstablished:
    return JointDesignQualityIncompleteReason::ProofNotEstablished;
  }
  llvm_unreachable("unknown repair quality disposition");
}

llvm::Error
validateEvidence(const std::optional<ArtifactRootReference> &evidence,
                 const ArtifactStore &artifacts) {
  if (!evidence)
    return llvm::Error::success();
  if (evidence->schemaIdentity !=
          evaluation::EvaluationEvidence::artifactSchema.identity ||
      evidence->schemaVersion !=
          evaluation::EvaluationEvidence::artifactSchema.version)
    return invalid("repair quality acquisition returned foreign Evidence");
  auto stored = artifacts.get(*evidence);
  if (!stored)
    return stored.takeError();
  return llvm::Error::success();
}

llvm::Error validateEvidenceSet(
    llvm::ArrayRef<ArtifactRootReference> evidence,
    const ArtifactStore &artifacts) {
  if (!llvm::is_sorted(evidence, artifactRootReferenceLess) ||
      std::adjacent_find(evidence.begin(), evidence.end()) != evidence.end())
    return invalid("repair quality Evidence set is not canonical");
  for (const ArtifactRootReference &reference : evidence)
    if (llvm::Error error = validateEvidence(reference, artifacts))
      return error;
  return llvm::Error::success();
}

bool sameObservation(const JointDesignQualityObservation &lhs,
                     const JointDesignQualityObservation &rhs) {
  return lhs.objectiveCodes == rhs.objectiveCodes &&
         lhs.incompleteReason == rhs.incompleteReason &&
         lhs.evidence == rhs.evidence && lhs.provenance == rhs.provenance;
}

llvm::Expected<const JointDesignPair *>
mappingDomain(const JointDesignExecution &execution,
              const ArtifactRootReference &mapping) {
  const JointDesignPair *result = nullptr;
  for (const JointMappedPair &pair : execution.mappedPairs) {
    if (!llvm::is_contained(pair.systemMappings, mapping))
      continue;
    if (result && !(*result == pair.pair))
      return invalid("one repair Mapping appears in conflicting pair domains");
    result = &pair.pair;
  }
  if (!result)
    return invalid("repair quality observation names a foreign Mapping");
  return result;
}

} // namespace

llvm::Expected<std::uint64_t> deriveApplicationRuntimeResourceCoreCost(
    const JointDesignExecution &execution,
    const ArtifactRootReference &mapping, const ArtifactStore &artifacts) {
  const JointDesignPair *domain = nullptr;
  for (const JointMappedPair &pair : execution.mappedPairs) {
    if (!llvm::is_contained(pair.systemMappings, mapping))
      continue;
    if (domain && !(*domain == pair.pair))
      return invalid("application runtime Mapping crossed pair domains");
    domain = &pair.pair;
  }
  if (!domain)
    return invalid("application runtime Mapping lost its pair domain");
  auto system = fabric::importEntireFabricRoot(domain->system, artifacts);
  if (!system)
    return system.takeError();
  auto systemView = fabric::requireSystemRoot(system->view());
  if (!systemView)
    return systemView.takeError();
  return static_cast<std::uint64_t>(
      systemView->artifact().accCoreOccurrences().size());
}

llvm::Expected<JointRepairQualitySelectionOutcome>
selectJointRepairMappingByQuality(
    llvm::ArrayRef<JointDesignExecution> executions,
    const JointBoundedQualityPolicy &quality, const ArtifactStore &artifacts) {
  if (executions.empty())
    return invalid("repair quality selection has an invalid execution set");
  if (!quality.objectiveProgram || !quality.acquire ||
      quality.finalTotalOrdering >=
          quality.objectiveProgram->totalOrderingCount())
    return invalid("repair quality selection has an incomplete policy");

  struct RecordedObservation final {
    ArtifactRootReference candidate;
    JointDesignPair domain;
    const JointDesignQualityObservation *observation = nullptr;
  };
  struct RecordedEvidence final {
    ArtifactRootReference evidence;
    JointDesignPair domain;
  };
  std::vector<RecordedObservation> recordedObservations;
  std::vector<RecordedEvidence> recordedEvidence;
  const auto recordEvidenceDomain =
      [&](const ArtifactRootReference &evidence,
          const JointDesignPair &domain) -> llvm::Error {
    const auto recorded = llvm::find_if(
        recordedEvidence, [&](const RecordedEvidence &candidate) {
          return candidate.evidence == evidence;
        });
    if (recorded == recordedEvidence.end()) {
      recordedEvidence.push_back({evidence, domain});
      return llvm::Error::success();
    }
    if (!(recorded->domain == domain))
      return invalid("one repair Evidence crossed pair quality domains");
    return llvm::Error::success();
  };
  std::vector<ArtifactRootReference> candidates;
  std::vector<CandidateObjectiveVector> objectives;
  std::vector<std::size_t> executionOrdinals;
  std::optional<JointRepairQualityIncomplete> firstIncomplete;
  for (std::size_t ordinal = 0; ordinal != executions.size(); ++ordinal) {
    const JointDesignExecution &execution = executions[ordinal];
    if (execution.summary.qualityObjectiveDimensionLabels !=
        quality.objectiveDimensionLabels)
      return invalid("repair quality labels changed after acquisition");
    for (const JointDesignQualityObservation &observation :
         execution.summary.qualityObservations) {
      if (llvm::count_if(execution.summary.qualityObservations,
                         [&](const auto &candidate) {
                           return candidate.candidate == observation.candidate;
                         }) != 1)
        return invalid("repair quality summary has duplicate Mapping "
                       "observations");
      auto domain = mappingDomain(execution, observation.candidate);
      if (!domain)
        return domain.takeError();
      if (llvm::Error error = validateJointDesignQualityProvenanceDomain(
              quality, observation.provenance,
              !observation.incompleteReason.has_value()))
        return std::move(error);
      if (llvm::Error error = validateEvidence(observation.evidence, artifacts))
        return std::move(error);
      if (llvm::Error error = validateEvidenceSet(
              observation.provenance.supportingEvidence, artifacts))
        return std::move(error);
      if (llvm::Error error = validateEvidenceSet(
              observation.provenance.verificationEvidence, artifacts))
        return std::move(error);
      if (observation.evidence)
        if (llvm::Error error =
                recordEvidenceDomain(*observation.evidence, **domain))
          return std::move(error);
      for (const ArtifactRootReference &reference :
           observation.provenance.supportingEvidence)
        if (llvm::Error error = recordEvidenceDomain(reference, **domain))
          return std::move(error);
      for (const ArtifactRootReference &reference :
           observation.provenance.verificationEvidence)
        if (llvm::Error error = recordEvidenceDomain(reference, **domain))
          return std::move(error);
      for (const ArtifactRootReference &verification :
           observation.provenance.verificationEvidence)
        if (!llvm::is_contained(observation.provenance.supportingEvidence,
                                verification))
          return invalid("repair quality verification Evidence is outside "
                         "its supporting Evidence");
      if (observation.provenance.spatialFifoFeedback &&
          observation.provenance.spatialFifoFeedback->parentMapping !=
              observation.candidate)
        return invalid("repair quality FIFO feedback names a foreign Mapping");
      if (observation.provenance.spatialOperandQueueFeedback &&
          observation.provenance.spatialOperandQueueFeedback->parentMapping &&
          *observation.provenance.spatialOperandQueueFeedback->parentMapping !=
              observation.candidate)
        return invalid(
            "repair quality operand feedback names a foreign Mapping");
      if (observation.provenance.spatialTransportFeedback &&
          observation.provenance.spatialTransportFeedback->parentMapping &&
          *observation.provenance.spatialTransportFeedback->parentMapping !=
              observation.candidate)
        return invalid(
            "repair quality transport feedback names a foreign Mapping");
      if (observation.incompleteReason) {
        if (!observation.objectiveCodes.empty())
          return invalid("incomplete repair quality observation retained "
                         "objective codes");
      } else {
        auto objective = quality.objectiveProgram->adoptVectorCodes(
            observation.objectiveCodes);
        if (!objective)
          return objective.takeError();
        if (llvm::Error error = validateJointDesignQualityObjective(
                *quality.objectiveProgram, observation.provenance,
                observation.objectiveCodes))
          return std::move(error);
      }
      const auto duplicate = llvm::find_if(
          recordedObservations, [&](const RecordedObservation &recorded) {
            return recorded.candidate == observation.candidate;
          });
      if (duplicate != recordedObservations.end()) {
        if (!(duplicate->domain == **domain))
          return invalid("one repair Mapping crossed pair quality domains");
        if (!sameObservation(*duplicate->observation, observation))
          return invalid("repair quality observations assigned conflicting "
                         "provenance to one Mapping");
      } else {
        recordedObservations.push_back(
            {observation.candidate, **domain, &observation});
      }
    }
    for (const JointMappedPair &pair : execution.mappedPairs)
      for (const ArtifactRootReference &mapping : pair.systemMappings)
        if (!llvm::any_of(execution.summary.qualityObservations,
                          [&](const auto &observation) {
                            return observation.candidate == mapping;
                          }))
          return invalid("repair Mapping has no observation");

    if (execution.summary.qualityDisposition !=
        JointDesignQualityDisposition::Complete) {
      if (execution.summary.selectedMapping)
        return invalid("incomplete repair quality retained a selected "
                       "Mapping");
      const JointDesignQualityIncompleteReason reason =
          incompleteReason(execution.summary.qualityDisposition);
      IncompleteJointDesignQuality incomplete{
          reason, execution.summary.qualityIncompleteCandidate, std::nullopt};
      if (execution.summary.qualityIncompleteCandidate) {
        const auto recorded = llvm::find_if(
            execution.summary.qualityObservations,
            [&](const JointDesignQualityObservation &observation) {
              return observation.candidate ==
                     *execution.summary.qualityIncompleteCandidate;
            });
        if (recorded == execution.summary.qualityObservations.end())
          return invalid("repair quality incomplete candidate has no "
                         "observation");
        if (recorded->incompleteReason != reason ||
            !recorded->objectiveCodes.empty())
          return invalid("repair quality summary changed its incomplete "
                         "observation");
        incomplete.evidence = recorded->evidence;
        incomplete.provenance = recorded->provenance;
      } else if (!execution.summary.qualityObservations.empty()) {
        return invalid("repair quality incomplete summary lost its exact "
                       "candidate");
      }
      if (!firstIncomplete)
        firstIncomplete =
            JointRepairQualityIncomplete{ordinal, std::move(incomplete)};
      continue;
    }

    if (!execution.summary.selectedMapping)
      return invalid("complete repair quality has no selected Mapping");
    const auto recorded = llvm::find_if(
        execution.summary.qualityObservations, [&](const auto &observation) {
          return observation.candidate == *execution.summary.selectedMapping;
        });
    if (recorded == execution.summary.qualityObservations.end() ||
        recorded->incompleteReason)
      return invalid("repair quality summary has no complete selected "
                     "observation");
    auto objective =
        quality.objectiveProgram->adoptVectorCodes(recorded->objectiveCodes);
    if (!objective)
      return objective.takeError();
    const auto existing =
        llvm::find(candidates, *execution.summary.selectedMapping);
    if (existing != candidates.end())
      continue;
    candidates.push_back(*execution.summary.selectedMapping);
    objectives.push_back(
        {*execution.summary.selectedMapping, std::move(*objective)});
    executionOrdinals.push_back(ordinal);
  }

  if (firstIncomplete)
    return JointRepairQualitySelectionOutcome{std::move(*firstIncomplete)};
  if (candidates.empty())
    return invalid("repair quality selection has no complete Mapping");

  auto candidateSet =
      CandidateSet::get(mapping::mappingArtifactSchema, candidates);
  if (!candidateSet)
    return candidateSet.takeError();
  auto pareto =
      applyCandidateSelection(*candidateSet, candidates, objectives,
                              ParetoSelection{quality.paretoDimensions},
                              quality.objectiveProgram.get());
  if (!pareto)
    return pareto.takeError();
  auto selected =
      applyCandidateSelection(*candidateSet, *pareto, objectives,
                              TopKSelection{quality.finalTotalOrdering, 1},
                              quality.objectiveProgram.get());
  if (!selected)
    return selected.takeError();
  if (selected->size() != 1)
    return invalid("repair quality selection did not produce one winner");
  const auto winner = llvm::find(candidates, selected->front());
  if (winner == candidates.end())
    return invalid("repair quality winner has no execution owner");
  const std::size_t winnerOrdinal =
      static_cast<std::size_t>(std::distance(candidates.begin(), winner));
  return JointRepairQualitySelectionOutcome{JointRepairQualitySelection{
      executionOrdinals[winnerOrdinal], selected->front()}};
}

} // namespace loom::dse
