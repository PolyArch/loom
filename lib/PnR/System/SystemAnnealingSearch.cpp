#include "PnR/System/SystemAnnealingSearch.h"

#include "Common/MappingDebugLog.h"
#include "PnR/System/SystemPnrProblem.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <optional>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>

using namespace loom;
using namespace loom::pnr;

char SystemActionTransitionFailure::ID;

void SystemActionTransitionFailure::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code SystemActionTransitionFailure::convertToErrorCode() const {
  return std::make_error_code(
      kind_ == SystemActionTransitionFailureKind::WorkLimit
          ? std::errc::resource_unavailable_try_again
          : std::errc::invalid_argument);
}

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "invalid System annealing search: " + message);
}

llvm::Error checkedAdd(std::uint64_t amount, std::uint64_t &target,
                       const llvm::Twine &subject) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - target)
    return invalid(subject + " count overflows u64");
  target += amount;
  return llvm::Error::success();
}

llvm::Expected<std::uint64_t>
checkedMultiply(std::uint64_t lhs, std::uint64_t rhs, llvm::StringRef subject) {
  if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
    return invalid(subject + " count overflows u64");
  return lhs * rhs;
}

struct TransitionFailureObservation final {
  bool consumed = false;
  std::optional<SystemUpstreamReopenWitness> reopenWitness;
};

llvm::Expected<TransitionFailureObservation>
consumeTransitionFailure(llvm::Error error) {
  TransitionFailureObservation observation;
  llvm::Error remaining = llvm::handleErrors(
      std::move(error),
      [&](const SystemActionTransitionFailure &failure) -> llvm::Error {
        observation.consumed = true;
        observation.reopenWitness = failure.reopenWitness();
        return llvm::Error::success();
      });
  if (remaining)
    return std::move(remaining);
  return observation;
}

llvm::Error scheduleUpstreamReopenActions(
    const FrozenSystemPnrProblem &problem,
    const SystemUpstreamReopenWitness &witness,
    std::vector<SystemExecutionBindingReopenAction> &pending,
    SystemAnnealingStatistics &statistics) {
  if (statistics.upstreamReopenWitnessCount ==
      std::numeric_limits<std::uint64_t>::max())
    return invalid("upstream reopen witness count overflows u64");
  ++statistics.upstreamReopenWitnessCount;
  bool hasAlternative = false;
  for (PnrIndex graphDecision : witness.graphDecisions) {
    if (graphDecision >= problem.graphDecisions().size())
      return invalid("upstream reopen witness names a foreign graph decision");
    hasAlternative |=
        problem.graphChoiceCatalogOrdinals(graphDecision).size() > 1;
  }
  if (!hasAlternative)
    return llvm::Error::success();
  SystemExecutionBindingReopenAction action{witness.capacityCell,
                                            witness.graphDecisions};
  if (!llvm::is_contained(pending, action))
    pending.push_back(std::move(action));
  llvm::sort(pending, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.capacityCell, lhs.graphDecisions) >
           std::tie(rhs.capacityCell, rhs.graphDecisions);
  });
  mapping_debug::emit(
      mapping_debug::Level::Decision, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::MappingFailure, [&](llvm::json::Object &fields) {
        fields["operation"] = "upstream_reopen_witness";
        fields["capacity_ref"] = witness.capacityCell;
        llvm::json::Array decisions;
        for (PnrIndex decision : witness.graphDecisions)
          decisions.push_back(decision);
        fields["graph_decisions"] = std::move(decisions);
        fields["pending_actions"] = static_cast<std::uint64_t>(pending.size());
      });
  return llvm::Error::success();
}

llvm::Error accountProbe(const SystemActionProbeAccounting &work,
                         SystemAnnealingStatistics &statistics,
                         llvm::StringRef scope) {
  if (llvm::Error error =
          checkedAdd(work.assignmentAttempts, statistics.assignmentAttempts,
                     scope + " assignment attempt"))
    return error;
  if (llvm::Error error =
          checkedAdd(work.endpointExpansions, statistics.endpointExpansions,
                     scope + " endpoint expansion"))
    return error;
  return checkedAdd(work.negotiationIterations,
                    statistics.negotiationIterations,
                    scope + " negotiation iteration");
}

void encodeSystemAction(llvm::json::Object &fields,
                        const SystemMappingAction &action) {
  std::visit(
      [&](const auto &domainAction) {
        using DomainAction = std::decay_t<decltype(domainAction)>;
        if constexpr (std::is_same_v<DomainAction,
                                     SystemExecutionBindingAction>) {
          fields["action_domain"] = "execution_binding";
          fields["action_kind"] = "execution_binding";
          fields["decision"] = domainAction.decision;
          fields["choice"] = domainAction.choice;
        } else if constexpr (std::is_same_v<
                                 DomainAction,
                                 SystemExecutionBindingReopenAction>) {
          fields["action_domain"] = "execution_binding";
          fields["action_kind"] = "execution_binding_reopen";
          fields["capacity_ref"] = domainAction.capacityCell;
          llvm::json::Array graphDecisions;
          for (PnrIndex decision : domainAction.graphDecisions)
            graphDecisions.push_back(decision);
          fields["graph_decisions"] = std::move(graphDecisions);
        } else if constexpr (std::is_same_v<DomainAction,
                                            SystemTransportRoutingAction>) {
          fields["action_domain"] = "routing";
          std::visit(
              [&](const auto &choice) {
                using Choice = std::decay_t<decltype(choice)>;
                if constexpr (std::is_same_v<Choice,
                                             SystemWholeLegRoutingAction>) {
                  fields["action_kind"] = "whole_leg";
                  fields["logical_leg"] = choice.leg;
                } else if constexpr (std::is_same_v<
                                         Choice,
                                         SystemSingleSinkRoutingAction>) {
                  fields["action_kind"] = "single_sink";
                  fields["logical_leg"] = choice.leg;
                  fields["sink_obligation"] = choice.sinkObligation;
                } else if constexpr (std::is_same_v<
                                         Choice,
                                         SystemRootedSubtreeRoutingAction>) {
                  fields["action_kind"] = "rooted_subtree";
                  fields["logical_leg"] = choice.leg;
                  fields["root_endpoint"] = choice.rootEndpoint;
                } else if constexpr (std::is_same_v<
                                         Choice,
                                         SystemWitnessRegionRoutingAction>) {
                  fields["action_kind"] = "witness_region";
                  fields["witness_kind"] =
                      static_cast<std::uint64_t>(choice.witnessKind);
                  fields["witness_ordinal"] = choice.witnessOrdinal;
                } else {
                  fields["action_kind"] = "global";
                }
              },
              domainAction);
        } else {
          fields["action_domain"] = "resource";
          std::visit(
              [&](const auto &choice) {
                using Choice = std::decay_t<decltype(choice)>;
                if constexpr (std::is_same_v<Choice,
                                             SystemServiceTargetAction>) {
                  fields["action_kind"] = "service_target";
                  fields["context"] = choice.context;
                  fields["subject"] = choice.subject;
                  fields["choice"] = choice.choice;
                } else if constexpr (std::is_same_v<
                                         Choice,
                                         SystemInstructionUsePatternAction>) {
                  fields["action_kind"] = "instruction_use_pattern";
                  fields["use"] = choice.use;
                  fields["choice"] = choice.choice;
                } else {
                  fields["action_kind"] = "service_use_pattern";
                  fields["use"] = choice.use;
                  fields["choice"] = choice.choice;
                }
              },
              domainAction);
        }
      },
      action);
}

llvm::StringRef differenceSign(dse::ObjectiveDifferenceSign sign) {
  switch (sign) {
  case dse::ObjectiveDifferenceSign::Negative:
    return "negative";
  case dse::ObjectiveDifferenceSign::Zero:
    return "zero";
  case dse::ObjectiveDifferenceSign::Positive:
    return "positive";
  }
  llvm_unreachable("unknown objective difference sign");
}

void emitSystemActionEvent(
    mapping_debug::Event event, const SystemMappingAction &action,
    llvm::StringRef scope, std::uint64_t seedAttemptOrdinal,
    std::uint64_t proposalSlot, std::optional<std::uint64_t> temperatureLevel,
    std::optional<std::uint64_t> temperature, llvm::StringRef outcome = {},
    std::optional<dse::ObjectiveSignedDifference> difference = std::nullopt,
    const SystemActionMutationRecord *mutation = nullptr) {
  mapping_debug::emit(
      mapping_debug::Level::Decision, mapping_debug::Stage::SystemPnr, event,
      [&](llvm::json::Object &fields) {
        fields["search_scope"] = scope;
        fields["seed_attempt"] = seedAttemptOrdinal;
        fields["proposal_slot"] = proposalSlot;
        if (temperatureLevel)
          fields["temperature_level"] = *temperatureLevel;
        if (temperature)
          fields["temperature"] = *temperature;
        if (!outcome.empty())
          fields["outcome"] = outcome;
        encodeSystemAction(fields, action);
        if (difference &&
            mapping_debug::enabled(mapping_debug::Level::Detail)) {
          fields["energy_difference_sign"] = differenceSign(difference->sign);
          fields["energy_difference_high"] = difference->magnitude.high;
          fields["energy_difference_low"] = difference->magnitude.low;
        }
        if (mutation) {
          fields["transaction_outcome"] =
              outcome == "accepted" ? "commit" : "rollback";
          fields["changed_thread_decisions"] =
              static_cast<std::uint64_t>(mutation->threadDecisions.size());
          fields["changed_graph_decisions"] =
              static_cast<std::uint64_t>(mutation->graphDecisions.size());
          fields["affected_service_legs"] =
              static_cast<std::uint64_t>(mutation->serviceLegs.size());
          fields["affected_service_targets"] =
              static_cast<std::uint64_t>(mutation->serviceTargets.size());
          fields["affected_instruction_uses"] = static_cast<std::uint64_t>(
              mutation->instructionResourceUses.size());
          fields["affected_service_uses"] =
              static_cast<std::uint64_t>(mutation->serviceResourceUses.size());
          fields["capacity_overuse_before"] = mutation->capacityOveruseBefore;
          fields["capacity_overuse_after"] = mutation->capacityOveruseAfter;
          fields["recurrence_minimum_initiation_interval_before"] =
              mutation->recurrenceMinimumInitiationIntervalBefore;
          fields["recurrence_minimum_initiation_interval_after"] =
              mutation->recurrenceMinimumInitiationIntervalAfter;
          fields["resource_minimum_initiation_interval_before"] =
              mutation->resourceMinimumInitiationIntervalBefore;
          fields["resource_minimum_initiation_interval_after"] =
              mutation->resourceMinimumInitiationIntervalAfter;
          fields["transport_bit_cycle_demand_before"] =
              mutation->transportBitCycleDemandBefore;
          fields["transport_bit_cycle_demand_after"] =
              mutation->transportBitCycleDemandAfter;
          fields["progress_before"] =
              static_cast<std::uint64_t>(mutation->progressBefore);
          fields["progress_after"] =
              static_cast<std::uint64_t>(mutation->progressAfter);
        }
      });
}

} // namespace

llvm::Expected<SystemAnnealingStatistics>
SystemAnnealingSearchScratch::run(SystemCandidateStateHandle &candidate,
                                  std::uint64_t seedAttemptOrdinal,
                                  ExecutionControlView executionControl) {
  const FrozenSystemPnrProblem &problem = candidate->problem();
  const ResolvedPnrPolicyConfig &policy = problem.config().policy();
  if (seedAttemptOrdinal >= policy.search.initializer.seedAttemptCount)
    return invalid("seed attempt ordinal is outside the fixed slot set");
  auto evaluated = problem.objectiveProgram().evaluate(*candidate);
  if (!evaluated)
    return evaluated.takeError();
  dse::ObjectiveVector currentObjective = std::move(*evaluated);
  SystemCandidateStateHandle bestCapacityClosed;
  std::optional<dse::ObjectiveVector> bestCapacityClosedObjective;
  const auto considerCapacityClosed =
      [&](const SystemCandidateStateHandle &proposed,
          const dse::ObjectiveVector &objective) -> llvm::Error {
    if (proposed->capacityOveruse() != 0)
      return llvm::Error::success();
    if (bestCapacityClosedObjective) {
      auto comparison = problem.objectiveProgram().compareSelectedRank(
          objective, {}, *bestCapacityClosedObjective, {});
      if (!comparison)
        return comparison.takeError();
      if (*comparison >= 0)
        return llvm::Error::success();
    }
    bestCapacityClosed = proposed;
    bestCapacityClosedObjective = objective;
    return llvm::Error::success();
  };
  if (llvm::Error error = considerCapacityClosed(candidate, currentObjective))
    return std::move(error);

  SystemAnnealingStatistics statistics;
  if (candidate->capacityOveruse() == 0 &&
      policy.search.completionGoal ==
          ResolvedPnrCompletionGoal::FirstVerifiedCandidate) {
    statistics.completionGoalReached = true;
    if (llvm::Error error = candidate->verify())
      return std::move(error);
    return statistics;
  }
  const auto finishInterrupted =
      [&]() -> llvm::Expected<SystemAnnealingStatistics> {
    statistics.interrupted = true;
    if (bestCapacityClosed)
      candidate = bestCapacityClosed;
    if (llvm::Error error = candidate->verify())
      return std::move(error);
    return statistics;
  };
  pendingReopenActions_.clear();
  const ResolvedPnrAnnealingPolicy &annealing = policy.search.annealing;
  if (annealing.calibrationProposalCount >
      positiveCalibrationDeltas_.max_size())
    return invalid("calibration sample capacity exceeds host size_t");
  positiveCalibrationDeltas_.clear();
  positiveCalibrationDeltas_.reserve(
      static_cast<std::size_t>(annealing.calibrationProposalCount));
  DeterministicPnrRandomStream calibrationStream =
      DeterministicPnrRandomStream::create(policy.determinism.masterSeed,
                                           seedAttemptOrdinal,
                                           PnrRandomStreamPurpose::Calibration);
  if (llvm::Error error = actionDomain_.rebuild(*candidate))
    return std::move(error);
  for (std::uint64_t slot = 0; slot < annealing.calibrationProposalCount;
       ++slot) {
    if (executionControl.stopRequested())
      return finishInterrupted();
    if (llvm::Error error = checkedAdd(1, statistics.calibrationProposalSlots,
                                       "calibration proposal slot"))
      return std::move(error);
    auto action = proposeSystemAction(policy.search.actionProposal,
                                      actionDomain_.view(), calibrationStream);
    if (!action)
      return action.takeError();
    if (!*action)
      continue;
    emitSystemActionEvent(mapping_debug::Event::ActionProposal, **action,
                          "calibration", seedAttemptOrdinal, slot, std::nullopt,
                          std::nullopt);
    SystemActionProbeAccounting work;
    auto probe = probeSystemAction(candidate, currentObjective, **action, work);
    if (llvm::Error error = accountProbe(work, statistics, "calibration"))
      return std::move(error);
    if (!probe) {
      auto observed = consumeTransitionFailure(probe.takeError());
      if (!observed)
        return observed.takeError();
      if (!observed->consumed)
        return invalid("Action failure had no classification");
      if (observed->reopenWitness)
        if (llvm::Error error = scheduleUpstreamReopenActions(
                problem, *observed->reopenWitness, pendingReopenActions_,
                statistics))
          return std::move(error);
      emitSystemActionEvent(mapping_debug::Event::ActionOutcome, **action,
                            "calibration", seedAttemptOrdinal, slot,
                            std::nullopt, std::nullopt, "transition_failure");
      continue;
    }
    if (probe->reopenWitness)
      if (llvm::Error error =
              scheduleUpstreamReopenActions(problem, *probe->reopenWitness,
                                            pendingReopenActions_, statistics))
        return std::move(error);
    if (probe->energyDifference.sign == dse::ObjectiveDifferenceSign::Positive)
      positiveCalibrationDeltas_.push_back(probe->energyDifference.magnitude);
    emitSystemActionEvent(mapping_debug::Event::ActionOutcome, **action,
                          "calibration", seedAttemptOrdinal, slot, std::nullopt,
                          std::nullopt, "discarded", probe->energyDifference,
                          &probe->mutation);
  }

  auto initialTemperature =
      calibrateAnnealingTemperature(annealing, positiveCalibrationDeltas_);
  if (!initialTemperature)
    return initialTemperature.takeError();
  auto schedule =
      AnnealingTemperatureSchedule::create(annealing, *initialTemperature);
  if (!schedule)
    return schedule.takeError();
  DeterministicPnrRandomStream proposalStream =
      DeterministicPnrRandomStream::create(
          policy.determinism.masterSeed, seedAttemptOrdinal,
          PnrRandomStreamPurpose::ActionProposal);
  DeterministicPnrRandomStream acceptanceStream =
      DeterministicPnrRandomStream::create(policy.determinism.masterSeed,
                                           seedAttemptOrdinal,
                                           PnrRandomStreamPurpose::Acceptance);
  std::uint64_t temperatureLevel = 0;
  do {
    if (executionControl.stopRequested())
      return finishInterrupted();
    if (llvm::Error error = actionDomain_.rebuild(*candidate))
      return std::move(error);
    bool domainCurrent = true;
    const std::uint64_t movable = actionDomain_.selectableMovableDecisionCount(
        policy.search.actionProposal);
    auto proposalCount = annealingProposalsPerLevel(annealing, movable);
    if (!proposalCount)
      return proposalCount.takeError();
    auto movableSlots =
        checkedMultiply(annealing.proposalsPerMovableDecision, movable,
                        "movable-decision proposal slot");
    if (!movableSlots)
      return movableSlots.takeError();
    for (std::uint64_t slot = 0; slot < *proposalCount; ++slot) {
      if (executionControl.stopRequested())
        return finishInterrupted();
      std::uint64_t &slotDomain =
          slot < annealing.proposalsPerLevelBase
              ? statistics.annealingBaseProposalSlots
              : statistics.annealingMovableProposalSlots;
      if (llvm::Error error =
              checkedAdd(1, slotDomain, "annealing domain slot"))
        return std::move(error);
      if (!domainCurrent) {
        if (llvm::Error error = actionDomain_.rebuild(*candidate))
          return std::move(error);
        domainCurrent = true;
      }
      std::optional<SystemMappingAction> selectedAction;
      const bool upstreamReopen = !pendingReopenActions_.empty();
      if (upstreamReopen) {
        selectedAction = pendingReopenActions_.back();
        pendingReopenActions_.pop_back();
        if (statistics.upstreamReopenActionProposalCount ==
            std::numeric_limits<std::uint64_t>::max())
          return invalid("upstream reopen Action count overflows u64");
        ++statistics.upstreamReopenActionProposalCount;
      } else {
        auto action = proposeSystemAction(policy.search.actionProposal,
                                          actionDomain_.view(), proposalStream);
        if (!action)
          return action.takeError();
        if (!*action)
          continue;
        selectedAction = std::move(**action);
      }
      const SystemMappingAction &action = *selectedAction;
      emitSystemActionEvent(mapping_debug::Event::ActionProposal, action,
                            upstreamReopen ? "upstream_reopen" : "annealing",
                            seedAttemptOrdinal, slot, temperatureLevel,
                            schedule->temperature());
      SystemActionProbeAccounting work;
      auto probe = probeSystemAction(candidate, currentObjective, action, work);
      if (llvm::Error error = accountProbe(work, statistics, "annealing"))
        return std::move(error);
      if (!probe) {
        auto observed = consumeTransitionFailure(probe.takeError());
        if (!observed)
          return observed.takeError();
        if (!observed->consumed)
          return invalid("Action failure had no classification");
        if (observed->reopenWitness)
          if (llvm::Error error = scheduleUpstreamReopenActions(
                  problem, *observed->reopenWitness, pendingReopenActions_,
                  statistics))
            return std::move(error);
        emitSystemActionEvent(mapping_debug::Event::ActionOutcome, action,
                              upstreamReopen ? "upstream_reopen" : "annealing",
                              seedAttemptOrdinal, slot, temperatureLevel,
                              schedule->temperature(), "transition_failure");
        continue;
      }
      auto accepted = acceptAnnealingDelta(
          probe->energyDifference, schedule->temperature(), acceptanceStream);
      if (!accepted)
        return accepted.takeError();
      if (*accepted) {
        if (statistics.mutationOracleVerificationCount ==
            std::numeric_limits<std::uint64_t>::max())
          return invalid("mutation oracle verification count overflows u64");
        ++statistics.mutationOracleVerificationCount;
        if (llvm::Error error = probe->candidate->verify())
          return llvm::joinErrors(
              invalid("accepted mutation diverged from its full oracle"),
              std::move(error));
        if (llvm::Error error =
                considerCapacityClosed(probe->candidate, probe->objective))
          return std::move(error);
        candidate = std::move(probe->candidate);
        currentObjective = std::move(probe->objective);
        if (statistics.acceptedActionCount ==
            std::numeric_limits<std::uint64_t>::max())
          return invalid("accepted Action count overflows u64");
        ++statistics.acceptedActionCount;
        if (upstreamReopen) {
          if (statistics.upstreamReopenAcceptedActionCount ==
              std::numeric_limits<std::uint64_t>::max())
            return invalid(
                "accepted upstream reopen Action count overflows u64");
          ++statistics.upstreamReopenAcceptedActionCount;
        }
        pendingReopenActions_.clear();
        domainCurrent = false;
      }
      if (probe->reopenWitness)
        if (llvm::Error error = scheduleUpstreamReopenActions(
                problem, *probe->reopenWitness, pendingReopenActions_,
                statistics))
          return std::move(error);
      emitSystemActionEvent(mapping_debug::Event::ActionOutcome, action,
                            upstreamReopen ? "upstream_reopen" : "annealing",
                            seedAttemptOrdinal, slot, temperatureLevel,
                            schedule->temperature(),
                            *accepted ? "accepted" : "rejected",
                            probe->energyDifference, &probe->mutation);
      if (*accepted && candidate->capacityOveruse() == 0 &&
          policy.search.completionGoal ==
              ResolvedPnrCompletionGoal::FirstVerifiedCandidate) {
        statistics.completionGoalReached = true;
        return statistics;
      }
    }
    ++temperatureLevel;
  } while (schedule->advanceAfterCompletedLevel());

  if (executionControl.stopRequested())
    return finishInterrupted();

  if (bestCapacityClosed)
    candidate = std::move(bestCapacityClosed);
  if (llvm::Error error = candidate->verify())
    return std::move(error);
  return statistics;
}
