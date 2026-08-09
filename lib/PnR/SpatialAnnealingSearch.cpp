#include "PnR/SpatialAnnealingSearch.h"

#include "Common/MappingDebugLog.h"

#include "llvm/Support/Error.h"

#include <limits>
#include <optional>
#include <system_error>
#include <type_traits>
#include <utility>

using namespace loom;
using namespace loom::pnr;

namespace {

llvm::Error searchError(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "invalid Spatial annealing search: %s", message.str().c_str());
}

llvm::Error addCount(std::uint64_t &target, std::uint64_t amount,
                     llvm::StringRef subject) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - target)
    return searchError(subject + " count overflows u64");
  target += amount;
  return llvm::Error::success();
}

llvm::Expected<std::uint64_t>
multiplyCount(std::uint64_t lhs, std::uint64_t rhs, llvm::StringRef subject) {
  if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
    return searchError(subject + " count overflows u64");
  return lhs * rhs;
}

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

void encodeSpatialAction(llvm::json::Object &fields,
                         const SpatialMappingAction &action) {
  std::visit(
      [&](const auto &domainAction) {
        using DomainAction = std::decay_t<decltype(domainAction)>;
        if constexpr (std::is_same_v<DomainAction,
                                     SpatialRealizationBindingAction>) {
          fields["action_domain"] = "realization";
          std::visit(
              [&](const auto &choice) {
                using Choice = std::decay_t<decltype(choice)>;
                if constexpr (std::is_same_v<Choice,
                                             SpatialComputeBindingAction>) {
                  fields["action_kind"] = "compute_binding";
                  fields["realization"] = choice.realization;
                  fields["placement"] = choice.placement;
                  fields["instruction_context"] = choice.instructionContext;
                } else {
                  fields["action_kind"] = "memory_binding";
                  fields["realization"] = choice.realization;
                  fields["placement"] = choice.placement;
                }
              },
              domainAction);
        } else if constexpr (std::is_same_v<
                                 DomainAction,
                                 SpatialTransportRoutingAction>) {
          fields["action_domain"] = "routing";
          std::visit(
              [&](const auto &choice) {
                using Choice = std::decay_t<decltype(choice)>;
                if constexpr (std::is_same_v<Choice,
                                             SpatialWholeNetRoutingAction>) {
                  fields["action_kind"] = "whole_net";
                  fields["logical_net"] = choice.logicalNet;
                } else if constexpr (std::is_same_v<
                                         Choice,
                                         SpatialSingleSinkRoutingAction>) {
                  fields["action_kind"] = "single_sink";
                  fields["logical_net"] = choice.logicalNet;
                  fields["sink_obligation"] = choice.sinkObligation;
                } else if constexpr (std::is_same_v<
                                         Choice,
                                         SpatialRootedSubtreeRoutingAction>) {
                  fields["action_kind"] = "rooted_subtree";
                  fields["logical_net"] = choice.logicalNet;
                  fields["root_endpoint"] = choice.rootEndpoint;
                } else if constexpr (std::is_same_v<
                                         Choice,
                                         SpatialWitnessRegionRoutingAction>) {
                  fields["action_kind"] = "witness_region";
                  fields["witness_kind"] = static_cast<std::uint64_t>(
                      choice.witnessKind);
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
                                             SpatialPortAttachmentAction>) {
                  fields["action_kind"] = "port_attachment";
                  fields["demand"] = choice.demand;
                  fields["attachment_option"] = choice.attachmentOption;
                } else if constexpr (std::is_same_v<
                                         Choice,
                                         SpatialGraphBoundaryAttachmentAction>) {
                  fields["action_kind"] = "graph_boundary_attachment";
                  fields["boundary"] = choice.boundary;
                  fields["attachment_option"] = choice.attachmentOption;
                } else if constexpr (std::is_same_v<
                                         Choice,
                                         SpatialMemoryOperationPlanAction>) {
                  fields["action_kind"] = "memory_operation_plan";
                  fields["actor"] = choice.actor;
                  fields["plan"] = choice.plan;
                } else if constexpr (std::is_same_v<
                                         Choice,
                                         SpatialLogicalMemoryBindingAction>) {
                  fields["action_kind"] = "logical_memory_binding";
                  fields["binding"] = choice.binding;
                  fields["target"] = choice.target;
                  fields["physical_offset_bytes"] =
                      choice.physicalOffsetBytes;
                } else if constexpr (std::is_same_v<
                                         Choice,
                                         SpatialMemoryUseDispatchAction>) {
                  fields["action_kind"] = "memory_use_dispatch";
                  fields["use"] = choice.use;
                  fields["dispatch_option"] = choice.dispatchOption;
                } else {
                  fields["action_kind"] = "memory_exposure";
                  fields["exposure"] = choice.exposure;
                  fields["exposure_option"] = choice.exposureOption;
                }
              },
              domainAction);
        }
      },
      action);
}

void encodeSpatialActionEndpoints(llvm::json::Object &fields,
                                  const SpatialCandidateState &candidate,
                                  const SpatialMappingAction &action) {
  const auto *resource =
      std::get_if<SpatialResourceAllocationAction>(&action);
  if (!resource)
    return;
  const FrozenSpatialPortIndex &ports = candidate.problem().ports();
  std::visit(
      [&](const auto &choice) {
        using Choice = std::decay_t<decltype(choice)>;
        if constexpr (std::is_same_v<Choice,
                                     SpatialGraphBoundaryAttachmentAction>) {
          if (choice.boundary >= ports.graphBoundaries().size() ||
              choice.attachmentOption >= ports.attachmentOptions().size())
            return;
          const FrozenSpatialGraphBoundary &boundary =
              ports.graphBoundaries()[choice.boundary];
          const PnrIndex current =
              candidate.graphBoundaryAttachment(choice.boundary);
          if (current >= ports.attachmentOptions().size())
            return;
          fields["logical_net"] = boundary.logicalNet;
          fields["payload_width_bits"] = boundary.payloadWidthBits;
          fields["current_attachment_option"] = current;
          fields["current_endpoint"] =
              ports.attachmentOptions()[current].endpoint;
          fields["proposed_endpoint"] =
              ports.attachmentOptions()[choice.attachmentOption].endpoint;
        } else if constexpr (std::is_same_v<Choice,
                                            SpatialPortAttachmentAction>) {
          if (choice.demand >= ports.portDemands().size() ||
              choice.attachmentOption >= ports.attachmentOptions().size())
            return;
          const FrozenSpatialPortDemand &demand =
              ports.portDemands()[choice.demand];
          const PnrIndex current = candidate.portAttachment(choice.demand);
          if (current >= ports.attachmentOptions().size())
            return;
          fields["logical_net"] = demand.logicalNet;
          fields["payload_width_bits"] = demand.payloadWidthBits;
          fields["current_attachment_option"] = current;
          fields["current_endpoint"] =
              ports.attachmentOptions()[current].endpoint;
          fields["proposed_endpoint"] =
              ports.attachmentOptions()[choice.attachmentOption].endpoint;
        }
      },
      *resource);
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

void emitSpatialActionEvent(
    loom::mapping_debug::Event event, const SpatialMappingAction &action,
    llvm::StringRef scope, std::uint64_t seedAttemptOrdinal,
    std::uint64_t proposalSlot,
    std::optional<std::uint64_t> temperatureLevel,
    std::optional<std::uint64_t> temperature,
    const SpatialCandidateState *candidate = nullptr,
    llvm::StringRef outcome = {},
    std::optional<dse::ObjectiveSignedDifference> difference = std::nullopt) {
  loom::mapping_debug::emit(
      loom::mapping_debug::Level::Decision,
      loom::mapping_debug::Stage::SpatialPnr, event,
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
        encodeSpatialAction(fields, action);
        if (candidate)
          encodeSpatialActionEndpoints(fields, *candidate, action);
        if (difference && loom::mapping_debug::enabled(
                              loom::mapping_debug::Level::Detail)) {
          fields["energy_difference_sign"] =
              differenceSign(difference->sign);
          fields["energy_difference_high"] = difference->magnitude.high;
          fields["energy_difference_low"] = difference->magnitude.low;
        }
      });
}

} // namespace

llvm::Expected<bool>
SpatialAnnealingSearchScratch::consumeTransitionFailure(llvm::Error failure) {
  bool consumed = false;
  llvm::Error unhandled = llvm::handleErrors(
      std::move(failure),
      [&](const SpatialActionTransitionFailure &) -> llvm::Error {
        consumed = true;
        return llvm::Error::success();
      });
  if (unhandled)
    return std::move(unhandled);
  return consumed;
}

llvm::Expected<SpatialAnnealingStatistics>
SpatialAnnealingSearchScratch::run(SpatialCandidateState &candidate,
                                   std::uint64_t seedAttemptOrdinal) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const ResolvedPnrPolicyConfig &policy = problem.config().policy();
  if (seedAttemptOrdinal >= policy.search.initializer.seedAttemptCount)
    return searchError("seed attempt ordinal is outside the fixed slot set");
  if (llvm::Error error = actionDomain_.prepare(problem))
    return std::move(error);
  if (llvm::Error error = actionExecutor_.prepare(candidate))
    return std::move(error);

  const ResolvedPnrAnnealingPolicy &annealing = policy.search.annealing;
  if (annealing.calibrationProposalCount >
      positiveCalibrationDeltas_.max_size())
    return searchError("calibration sample capacity exceeds host size_t");
  positiveCalibrationDeltas_.clear();
  positiveCalibrationDeltas_.reserve(
      static_cast<std::size_t>(annealing.calibrationProposalCount));

  SpatialAnnealingStatistics statistics;
  statistics.calibrationProposalSlots = annealing.calibrationProposalCount;
  DeterministicPnrRandomStream calibrationStream =
      DeterministicPnrRandomStream::create(policy.determinism.masterSeed,
                                           seedAttemptOrdinal,
                                           PnrRandomStreamPurpose::Calibration);
  if (llvm::Error error = actionDomain_.rebuild(candidate))
    return std::move(error);
  for (std::uint64_t slot = 0; slot < annealing.calibrationProposalCount;
       ++slot) {
    auto action = proposeSpatialAction(policy.search.actionProposal,
                                       actionDomain_.view(), calibrationStream);
    if (!action)
      return action.takeError();
    if (!*action)
      continue;
    emitSpatialActionEvent(loom::mapping_debug::Event::ActionProposal,
                           **action, "calibration", seedAttemptOrdinal, slot,
                           std::nullopt, std::nullopt, &candidate);

    auto probe = actionExecutor_.probe(candidate, **action);
    if (!probe) {
      auto consumed = consumeTransitionFailure(probe.takeError());
      if (!consumed)
        return consumed.takeError();
      if (!*consumed)
        return searchError("Action failure had no failure classification");
      if (llvm::Error error =
              addCount(statistics.calibrationTransitionFailureCount, 1,
                       "calibration transition failure"))
        return std::move(error);
      emitSpatialActionEvent(loom::mapping_debug::Event::ActionOutcome,
                             **action, "calibration", seedAttemptOrdinal,
                             slot, std::nullopt, std::nullopt,
                             nullptr, "transition_failure");
      continue;
    }
    if (llvm::Error error =
            addCount(statistics.calibrationProbeCount, 1, "calibration probe"))
      return std::move(error);
    const dse::ObjectiveSignedDifference difference =
        probe->energyDifference();
    if (difference.sign ==
        dse::ObjectiveDifferenceSign::Positive)
      positiveCalibrationDeltas_.push_back(difference.magnitude);
    if (llvm::Error error = probe->discard())
      return std::move(error);
    emitSpatialActionEvent(loom::mapping_debug::Event::ActionOutcome,
                           **action, "calibration", seedAttemptOrdinal, slot,
                           std::nullopt, std::nullopt, nullptr, "discarded",
                           difference);
  }

  auto initialTemperature =
      calibrateAnnealingTemperature(annealing, positiveCalibrationDeltas_);
  if (!initialTemperature)
    return initialTemperature.takeError();
  statistics.initialTemperature = *initialTemperature;
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
  do {
    if (llvm::Error error = actionDomain_.rebuild(candidate))
      return std::move(error);
    bool domainCurrent = true;
    const std::uint64_t movableDecisionCount =
        actionDomain_.movableDecisionCount();
    auto proposalCount =
        annealingProposalsPerLevel(annealing, movableDecisionCount);
    if (!proposalCount)
      return proposalCount.takeError();
    auto movableProposalCount =
        multiplyCount(annealing.proposalsPerMovableDecision,
                      movableDecisionCount, "movable-decision proposal slot");
    if (!movableProposalCount)
      return movableProposalCount.takeError();
    if (annealing.proposalsPerLevelBase >
            std::numeric_limits<std::uint64_t>::max() - *movableProposalCount ||
        annealing.proposalsPerLevelBase + *movableProposalCount !=
            *proposalCount)
      return searchError(
          "proposal work projection disagrees with the search domain");
    if (llvm::Error error =
            addCount(statistics.annealingBaseProposalSlots,
                     annealing.proposalsPerLevelBase, "base proposal slot"))
      return std::move(error);
    if (llvm::Error error =
            addCount(statistics.annealingMovableProposalSlots,
                     *movableProposalCount, "movable-decision proposal slot"))
      return std::move(error);
    if (llvm::Error error = addCount(statistics.temperatureLevelCount, 1,
                                     "annealing temperature level"))
      return std::move(error);
    if (schedule->isFinalLevel())
      if (llvm::Error error = addCount(statistics.minimumTemperatureLevelCount,
                                       1, "minimum-temperature level"))
        return std::move(error);
    if (llvm::Error error = addCount(statistics.annealingProposalSlots,
                                     *proposalCount, "annealing proposal slot"))
      return std::move(error);

    for (std::uint64_t slot = 0; slot < *proposalCount; ++slot) {
      if (!domainCurrent) {
        if (llvm::Error error = actionDomain_.rebuild(candidate))
          return std::move(error);
        domainCurrent = true;
      }
      auto action = proposeSpatialAction(policy.search.actionProposal,
                                         actionDomain_.view(), proposalStream);
      if (!action)
        return action.takeError();
      if (!*action)
        continue;
      const std::uint64_t temperatureLevel =
          statistics.temperatureLevelCount - 1;
      emitSpatialActionEvent(loom::mapping_debug::Event::ActionProposal,
                             **action, "annealing", seedAttemptOrdinal, slot,
                             temperatureLevel, schedule->temperature(),
                             &candidate);

      auto probe = actionExecutor_.probe(candidate, **action);
      if (!probe) {
        auto consumed = consumeTransitionFailure(probe.takeError());
        if (!consumed)
          return consumed.takeError();
        if (!*consumed)
          return searchError("Action failure had no failure classification");
        if (llvm::Error error =
                addCount(statistics.annealingTransitionFailureCount, 1,
                         "annealing transition failure"))
          return std::move(error);
        emitSpatialActionEvent(loom::mapping_debug::Event::ActionOutcome,
                               **action, "annealing", seedAttemptOrdinal,
                               slot, temperatureLevel, schedule->temperature(),
                               nullptr, "transition_failure");
        continue;
      }
      if (llvm::Error error =
              addCount(statistics.annealingProbeCount, 1, "annealing probe"))
        return std::move(error);
      const dse::ObjectiveSignedDifference difference =
          probe->energyDifference();
      auto resolution =
          probe->resolve(schedule->temperature(), acceptanceStream);
      if (!resolution)
        return resolution.takeError();
      std::uint64_t &count = resolution->accepted
                                 ? statistics.acceptedActionCount
                                 : statistics.rejectedActionCount;
      if (llvm::Error error = addCount(
              count, 1,
              resolution->accepted ? "accepted Action" : "rejected Action"))
        return std::move(error);
      if (resolution->accepted)
        domainCurrent = false;
      emitSpatialActionEvent(
          loom::mapping_debug::Event::ActionOutcome, **action, "annealing",
          seedAttemptOrdinal, slot, temperatureLevel, schedule->temperature(),
          nullptr, resolution->accepted ? "accepted" : "rejected",
          difference);
    }
  } while (schedule->advanceAfterCompletedLevel());

  if (statistics.minimumTemperatureLevelCount != 1)
    return searchError(
        "annealing schedule did not execute one minimum-temperature level");
  if (llvm::Error error = candidate.verify())
    return std::move(error);
  statistics.endpointExpansions = actionExecutor_.endpointExpansionCount();
  statistics.negotiationIterations =
      actionExecutor_.negotiationIterationCount();
  return statistics;
}

std::size_t SpatialAnnealingSearchScratch::retainedStorageBytes() const {
  return actionDomain_.retainedStorageBytes() +
         actionExecutor_.retainedStorageBytes() +
         retainedBytes(positiveCalibrationDeltas_);
}
