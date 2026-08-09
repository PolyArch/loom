#include "PnR/System/SystemAnnealingSearch.h"

#include "Common/MappingDebugLog.h"
#include "PnR/System/SystemPnrProblem.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <optional>
#include <system_error>
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

llvm::Expected<bool> consumeTransitionFailure(llvm::Error error) {
  bool consumed = false;
  llvm::Error remaining = llvm::handleErrors(
      std::move(error),
      [&](const SystemActionTransitionFailure &) -> llvm::Error {
        consumed = true;
        return llvm::Error::success();
      });
  if (remaining)
    return std::move(remaining);
  return consumed;
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
    std::optional<dse::ObjectiveSignedDifference> difference = std::nullopt) {
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
      });
}

} // namespace

llvm::Expected<SystemAnnealingStatistics>
SystemAnnealingSearchScratch::run(SystemCandidateStateHandle &candidate,
                                  std::uint64_t seedAttemptOrdinal) {
  const FrozenSystemPnrProblem &problem = candidate->problem();
  const ResolvedPnrPolicyConfig &policy = problem.config().policy();
  if (seedAttemptOrdinal >= policy.search.initializer.seedAttemptCount)
    return invalid("seed attempt ordinal is outside the fixed slot set");
  auto evaluated = problem.objectiveProgram().evaluate(*candidate);
  if (!evaluated)
    return evaluated.takeError();
  dse::ObjectiveVector currentObjective = std::move(*evaluated);

  SystemAnnealingStatistics statistics;
  const ResolvedPnrAnnealingPolicy &annealing = policy.search.annealing;
  statistics.calibrationProposalSlots = annealing.calibrationProposalCount;
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
      auto consumed = consumeTransitionFailure(probe.takeError());
      if (!consumed)
        return consumed.takeError();
      if (!*consumed)
        return invalid("Action failure had no classification");
      emitSystemActionEvent(mapping_debug::Event::ActionOutcome, **action,
                            "calibration", seedAttemptOrdinal, slot,
                            std::nullopt, std::nullopt, "transition_failure");
      continue;
    }
    if (probe->energyDifference.sign == dse::ObjectiveDifferenceSign::Positive)
      positiveCalibrationDeltas_.push_back(probe->energyDifference.magnitude);
    emitSystemActionEvent(mapping_debug::Event::ActionOutcome, **action,
                          "calibration", seedAttemptOrdinal, slot, std::nullopt,
                          std::nullopt, "discarded", probe->energyDifference);
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
    if (llvm::Error error = actionDomain_.rebuild(*candidate))
      return std::move(error);
    bool domainCurrent = true;
    const std::uint64_t movable = actionDomain_.movableDecisionCount();
    auto proposalCount = annealingProposalsPerLevel(annealing, movable);
    if (!proposalCount)
      return proposalCount.takeError();
    auto movableSlots =
        checkedMultiply(annealing.proposalsPerMovableDecision, movable,
                        "movable-decision proposal slot");
    if (!movableSlots)
      return movableSlots.takeError();
    if (llvm::Error error = checkedAdd(annealing.proposalsPerLevelBase,
                                       statistics.annealingBaseProposalSlots,
                                       "base proposal slot"))
      return std::move(error);
    if (llvm::Error error =
            checkedAdd(*movableSlots, statistics.annealingMovableProposalSlots,
                       "movable-decision proposal slot"))
      return std::move(error);
    for (std::uint64_t slot = 0; slot < *proposalCount; ++slot) {
      if (!domainCurrent) {
        if (llvm::Error error = actionDomain_.rebuild(*candidate))
          return std::move(error);
        domainCurrent = true;
      }
      auto action = proposeSystemAction(policy.search.actionProposal,
                                        actionDomain_.view(), proposalStream);
      if (!action)
        return action.takeError();
      if (!*action)
        continue;
      emitSystemActionEvent(mapping_debug::Event::ActionProposal, **action,
                            "annealing", seedAttemptOrdinal, slot,
                            temperatureLevel, schedule->temperature());
      SystemActionProbeAccounting work;
      auto probe =
          probeSystemAction(candidate, currentObjective, **action, work);
      if (llvm::Error error = accountProbe(work, statistics, "annealing"))
        return std::move(error);
      if (!probe) {
        auto consumed = consumeTransitionFailure(probe.takeError());
        if (!consumed)
          return consumed.takeError();
        if (!*consumed)
          return invalid("Action failure had no classification");
        emitSystemActionEvent(mapping_debug::Event::ActionOutcome, **action,
                              "annealing", seedAttemptOrdinal, slot,
                              temperatureLevel, schedule->temperature(),
                              "transition_failure");
        continue;
      }
      auto accepted = acceptAnnealingDelta(
          probe->energyDifference, schedule->temperature(), acceptanceStream);
      if (!accepted)
        return accepted.takeError();
      if (*accepted) {
        candidate = std::move(probe->candidate);
        currentObjective = std::move(probe->objective);
        if (statistics.acceptedActionCount ==
            std::numeric_limits<std::uint64_t>::max())
          return invalid("accepted Action count overflows u64");
        ++statistics.acceptedActionCount;
        domainCurrent = false;
      }
      emitSystemActionEvent(
          mapping_debug::Event::ActionOutcome, **action, "annealing",
          seedAttemptOrdinal, slot, temperatureLevel, schedule->temperature(),
          *accepted ? "accepted" : "rejected", probe->energyDifference);
    }
    ++temperatureLevel;
  } while (schedule->advanceAfterCompletedLevel());

  if (llvm::Error error = candidate->verify())
    return std::move(error);
  return statistics;
}
