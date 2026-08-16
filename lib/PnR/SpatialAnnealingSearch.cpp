#include "PnR/SpatialAnnealingSearch.h"

#include "Common/MappingDebugLog.h"
#include "PnR/MappingObjective.h"
#include "PnR/SpatialCanonicalSeed.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <limits>
#include <optional>
#include <system_error>
#include <type_traits>
#include <utility>

using namespace loom;
using namespace loom::pnr;

namespace {

enum class SpatialSearchScope : std::uint8_t {
  Calibration,
  Annealing,
};

enum class SpatialActionOutcome : std::uint8_t {
  TransitionFailure,
  Discarded,
  Accepted,
  Rejected,
  SemanticNoop,
  CachedInactive,
};

llvm::StringRef spelling(SpatialSearchScope scope) {
  switch (scope) {
  case SpatialSearchScope::Calibration:
    return "calibration";
  case SpatialSearchScope::Annealing:
    return "annealing";
  }
  llvm_unreachable("unknown Spatial search scope");
}

llvm::StringRef spelling(SpatialActionOutcome outcome) {
  switch (outcome) {
  case SpatialActionOutcome::TransitionFailure:
    return "transition_failure";
  case SpatialActionOutcome::Discarded:
    return "discarded";
  case SpatialActionOutcome::Accepted:
    return "accepted";
  case SpatialActionOutcome::Rejected:
    return "rejected";
  case SpatialActionOutcome::SemanticNoop:
    return "semantic_noop";
  case SpatialActionOutcome::CachedInactive:
    return "cached_inactive";
  }
  llvm_unreachable("unknown Spatial Action outcome");
}

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
        } else if constexpr (std::is_same_v<DomainAction,
                                            SpatialTransportRoutingAction>) {
          fields["action_domain"] = "routing";
          std::visit(
              [&](const auto &choice) {
                using Choice = std::decay_t<decltype(choice)>;
                if constexpr (std::is_same_v<Choice,
                                             SpatialWholeNetRoutingAction>) {
                  fields["action_kind"] = "whole_net";
                  fields["logical_net"] = choice.logicalNet;
                  fields["disposition"] =
                      static_cast<std::uint8_t>(choice.disposition);
                  if (choice.disposition ==
                      SpatialWholeNetDispositionKind::RegisterFifo)
                    fields["register_fifo_transfer"] =
                        choice.registerFifoTransfer;
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
                                             SpatialPortAttachmentAction>) {
                  fields["action_kind"] = "port_attachment";
                  fields["demand"] = choice.demand;
                  fields["attachment_option"] = choice.attachmentOption;
                } else if constexpr (
                    std::is_same_v<Choice,
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
                  fields["physical_offset_bytes"] = choice.physicalOffsetBytes;
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
  const auto *resource = std::get_if<SpatialResourceAllocationAction>(&action);
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
    SpatialSearchScope scope, std::uint64_t seedAttemptOrdinal,
    std::uint64_t proposalSlot, std::optional<std::uint64_t> temperatureLevel,
    std::optional<std::uint64_t> temperature,
    const SpatialCandidateState *candidate = nullptr,
    std::optional<SpatialActionOutcome> outcome = std::nullopt,
    std::optional<dse::ObjectiveSignedDifference> difference = std::nullopt) {
  loom::mapping_debug::emit(
      loom::mapping_debug::Level::Decision,
      loom::mapping_debug::Stage::SpatialPnr, event,
      [&](llvm::json::Object &fields) {
        fields["search_scope"] = spelling(scope);
        fields["seed_attempt"] = seedAttemptOrdinal;
        fields["proposal_slot"] = proposalSlot;
        if (temperatureLevel)
          fields["temperature_level"] = *temperatureLevel;
        if (temperature)
          fields["temperature"] = *temperature;
        if (outcome)
          fields["outcome"] = spelling(*outcome);
        encodeSpatialAction(fields, action);
        if (candidate)
          encodeSpatialActionEndpoints(fields, *candidate, action);
        if (difference &&
            loom::mapping_debug::enabled(loom::mapping_debug::Level::Detail)) {
          fields["energy_difference_sign"] = differenceSign(difference->sign);
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
SpatialAnnealingSearchScratch::run(SpatialPathFinderSeed &seed,
                                   ExecutionControlView executionControl) {
  return run(seed.candidate, seed.attemptOrdinal, executionControl);
}

llvm::Expected<SpatialAnnealingStatistics> SpatialAnnealingSearchScratch::run(
    SpatialCandidateStateHandle &candidateHandle,
    std::uint64_t seedAttemptOrdinal, ExecutionControlView executionControl) {
  if (!candidateHandle)
    return searchError("candidate owner is null");
  SpatialCandidateState &candidate = *candidateHandle;
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
  inactiveActionKeys_.clear();

  const auto actionIsInactive = [&](const SpatialActionKey &key) {
    return llvm::binary_search(inactiveActionKeys_, key);
  };
  const auto rememberInactiveAction = [&](const SpatialActionKey &key) {
    const auto insertion = llvm::lower_bound(inactiveActionKeys_, key);
    if (insertion == inactiveActionKeys_.end() || !(*insertion == key))
      inactiveActionKeys_.insert(insertion, key);
  };
  const auto reserveInactiveActionCapacity = [&]() -> llvm::Error {
    const SpatialActionProposalDomain domain = actionDomain_.view();
    std::size_t count = domain.realizationChoices.size();
    if (domain.transportChoices.size() > inactiveActionKeys_.max_size() - count)
      return searchError("inactive Action cache capacity exceeds host size_t");
    count += domain.transportChoices.size();
    if (domain.resourceChoices.size() > inactiveActionKeys_.max_size() - count)
      return searchError("inactive Action cache capacity exceeds host size_t");
    count += domain.resourceChoices.size();
    if (inactiveActionKeys_.capacity() < count)
      inactiveActionKeys_.reserve(count);
    return llvm::Error::success();
  };

  SpatialAnnealingStatistics statistics;
  auto exactClosure = spatialMappingViolationsAreZero(candidate);
  if (!exactClosure)
    return exactClosure.takeError();
  SpatialCandidateStateHandle bestFeasibleIncumbent;
  std::optional<dse::ObjectiveVector> bestFeasibleObjective;
  if (*exactClosure) {
    auto snapshot = candidate.cloneFullyRouted();
    if (!snapshot)
      return snapshot.takeError();
    bestFeasibleIncumbent = std::move(*snapshot);
    bestFeasibleObjective = actionExecutor_.currentObjective();
    statistics.exactClosureReached = true;
    loom::mapping_debug::emit(loom::mapping_debug::Level::Summary,
                              loom::mapping_debug::Stage::SpatialPnr,
                              loom::mapping_debug::Event::Statistics,
                              [&](llvm::json::Object &fields) {
                                fields["operation"] = "annealing_incumbent";
                                fields["seed_attempt"] = seedAttemptOrdinal;
                                fields["reason"] = "feasible_on_entry";
                              });
  }
  const auto finishInterrupted =
      [&]() -> llvm::Expected<SpatialAnnealingStatistics> {
    statistics.interrupted = true;
    statistics.endpointExpansions = actionExecutor_.endpointExpansionCount();
    statistics.negotiationIterations =
        actionExecutor_.negotiationIterationCount();
    if (bestFeasibleIncumbent) {
      if (llvm::Error error = bestFeasibleIncumbent->verify())
        return std::move(error);
      candidateHandle = std::move(bestFeasibleIncumbent);
      statistics.bestFeasibleIncumbentRestored = true;
    }
    if (llvm::Error error = candidateHandle->verify())
      return std::move(error);
    return statistics;
  };
  DeterministicPnrRandomStream calibrationStream =
      DeterministicPnrRandomStream::create(policy.determinism.masterSeed,
                                           seedAttemptOrdinal,
                                           PnrRandomStreamPurpose::Calibration);
  if (llvm::Error error = actionDomain_.rebuild(candidate))
    return std::move(error);
  if (llvm::Error error = reserveInactiveActionCapacity())
    return std::move(error);
  for (std::uint64_t slot = 0; slot < annealing.calibrationProposalCount;
       ++slot) {
    if (executionControl.stopRequested())
      return finishInterrupted();
    if (llvm::Error error = addCount(statistics.calibrationProposalSlots, 1,
                                     "calibration proposal slot"))
      return std::move(error);
    auto action =
        actionDomain_.propose(policy.search.actionProposal, calibrationStream);
    if (!action)
      return action.takeError();
    if (!*action)
      continue;
    const SpatialActionKey actionKey = spatialActionKey(**action);
    emitSpatialActionEvent(loom::mapping_debug::Event::ActionProposal, **action,
                           SpatialSearchScope::Calibration, seedAttemptOrdinal,
                           slot, std::nullopt, std::nullopt, &candidate);
    if (actionIsInactive(actionKey)) {
      if (llvm::Error error = addCount(statistics.cachedInactiveActionCount, 1,
                                       "cached inactive Action"))
        return std::move(error);
      emitSpatialActionEvent(loom::mapping_debug::Event::ActionOutcome,
                             **action, SpatialSearchScope::Calibration,
                             seedAttemptOrdinal, slot, std::nullopt,
                             std::nullopt, nullptr,
                             SpatialActionOutcome::CachedInactive);
      continue;
    }

    auto probe = actionExecutor_.probe(candidate, **action);
    if (!probe) {
      auto consumed = consumeTransitionFailure(probe.takeError());
      if (!consumed)
        return consumed.takeError();
      if (!*consumed)
        return searchError("Action failure had no failure classification");
      rememberInactiveAction(actionKey);
      if (llvm::Error error =
              addCount(statistics.calibrationTransitionFailureCount, 1,
                       "calibration transition failure"))
        return std::move(error);
      emitSpatialActionEvent(loom::mapping_debug::Event::ActionOutcome,
                             **action, SpatialSearchScope::Calibration,
                             seedAttemptOrdinal, slot, std::nullopt,
                             std::nullopt, nullptr,
                             SpatialActionOutcome::TransitionFailure);
      continue;
    }
    if (llvm::Error error =
            addCount(statistics.calibrationProbeCount, 1, "calibration probe"))
      return std::move(error);
    const dse::ObjectiveSignedDifference difference = probe->energyDifference();
    if (probe->isSemanticNoop()) {
      rememberInactiveAction(actionKey);
      if (llvm::Error error = addCount(statistics.semanticNoopActionCount, 1,
                                       "semantic no-op Action"))
        return std::move(error);
      if (llvm::Error error = probe->discard())
        return std::move(error);
      emitSpatialActionEvent(loom::mapping_debug::Event::ActionOutcome,
                             **action, SpatialSearchScope::Calibration,
                             seedAttemptOrdinal, slot, std::nullopt,
                             std::nullopt, nullptr,
                             SpatialActionOutcome::SemanticNoop, difference);
      continue;
    }
    if (difference.sign == dse::ObjectiveDifferenceSign::Positive)
      positiveCalibrationDeltas_.push_back(difference.magnitude);
    if (llvm::Error error = probe->discard())
      return std::move(error);
    emitSpatialActionEvent(loom::mapping_debug::Event::ActionOutcome, **action,
                           SpatialSearchScope::Calibration, seedAttemptOrdinal,
                           slot, std::nullopt, std::nullopt, nullptr,
                           SpatialActionOutcome::Discarded, difference);
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

  loom::mapping_debug::emit(
      loom::mapping_debug::Level::Summary,
      loom::mapping_debug::Stage::SpatialPnr,
      loom::mapping_debug::Event::Statistics, [&](llvm::json::Object &fields) {
        fields["operation"] = "annealing_policy";
        fields["seed_attempt"] = seedAttemptOrdinal;
        fields["initial_temperature"] = statistics.initialTemperature;
        fields["minimum_temperature"] = annealing.minimumTemperature;
        fields["cooling_numerator"] = annealing.coolingRatio.numerator;
        fields["cooling_denominator"] = annealing.coolingRatio.denominator;
        fields["calibration_slots"] = statistics.calibrationProposalSlots;
        fields["calibration_probes"] = statistics.calibrationProbeCount;
        fields["semantic_noop_actions"] = statistics.semanticNoopActionCount;
        fields["cached_inactive_actions"] =
            statistics.cachedInactiveActionCount;
      });

  DeterministicPnrRandomStream proposalStream =
      DeterministicPnrRandomStream::create(
          policy.determinism.masterSeed, seedAttemptOrdinal,
          PnrRandomStreamPurpose::ActionProposal);
  DeterministicPnrRandomStream acceptanceStream =
      DeterministicPnrRandomStream::create(policy.determinism.masterSeed,
                                           seedAttemptOrdinal,
                                           PnrRandomStreamPurpose::Acceptance);
  do {
    if (executionControl.stopRequested())
      return finishInterrupted();
    if (llvm::Error error = actionDomain_.rebuild(candidate))
      return std::move(error);
    bool domainCurrent = true;
    const std::uint64_t levelProbeBegin = statistics.annealingProbeCount;
    const std::uint64_t levelAcceptedBegin = statistics.acceptedActionCount;
    const std::uint64_t levelRejectedBegin = statistics.rejectedActionCount;
    const std::uint64_t levelNoopBegin = statistics.semanticNoopActionCount;
    const std::uint64_t levelCachedBegin = statistics.cachedInactiveActionCount;
    const std::uint64_t levelFailureBegin =
        statistics.annealingTransitionFailureCount;
    const std::uint64_t levelHeuristicHitBegin =
        actionExecutor_.heuristicCacheHitCount();
    const std::uint64_t levelHeuristicBuildBegin =
        actionExecutor_.heuristicBuildCount();
    const std::uint64_t movableDecisionCount =
        actionDomain_.selectableMovableDecisionCount(
            policy.search.actionProposal);
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
    if (llvm::Error error = addCount(statistics.temperatureLevelCount, 1,
                                     "annealing temperature level"))
      return std::move(error);
    if (schedule->isFinalLevel())
      if (llvm::Error error = addCount(statistics.minimumTemperatureLevelCount,
                                       1, "minimum-temperature level"))
        return std::move(error);
    for (std::uint64_t slot = 0; slot < *proposalCount; ++slot) {
      if (executionControl.stopRequested())
        return finishInterrupted();
      if (llvm::Error error = addCount(statistics.annealingProposalSlots, 1,
                                       "annealing proposal slot"))
        return std::move(error);
      std::uint64_t &slotDomain =
          slot < annealing.proposalsPerLevelBase
              ? statistics.annealingBaseProposalSlots
              : statistics.annealingMovableProposalSlots;
      if (llvm::Error error = addCount(slotDomain, 1, "annealing domain slot"))
        return std::move(error);
      if (!domainCurrent) {
        if (llvm::Error error = actionDomain_.rebuild(candidate))
          return std::move(error);
        domainCurrent = true;
      }
      auto action =
          actionDomain_.propose(policy.search.actionProposal, proposalStream);
      if (!action)
        return action.takeError();
      if (!*action)
        continue;
      const SpatialActionKey actionKey = spatialActionKey(**action);
      const std::uint64_t temperatureLevel =
          statistics.temperatureLevelCount - 1;
      emitSpatialActionEvent(loom::mapping_debug::Event::ActionProposal,
                             **action, SpatialSearchScope::Annealing,
                             seedAttemptOrdinal, slot, temperatureLevel,
                             schedule->temperature(), &candidate);
      if (actionIsInactive(actionKey)) {
        if (llvm::Error error = addCount(statistics.cachedInactiveActionCount,
                                         1, "cached inactive Action"))
          return std::move(error);
        emitSpatialActionEvent(loom::mapping_debug::Event::ActionOutcome,
                               **action, SpatialSearchScope::Annealing,
                               seedAttemptOrdinal, slot, temperatureLevel,
                               schedule->temperature(), nullptr,
                               SpatialActionOutcome::CachedInactive);
        continue;
      }

      auto probe = actionExecutor_.probe(candidate, **action);
      if (!probe) {
        auto consumed = consumeTransitionFailure(probe.takeError());
        if (!consumed)
          return consumed.takeError();
        if (!*consumed)
          return searchError("Action failure had no failure classification");
        rememberInactiveAction(actionKey);
        if (llvm::Error error =
                addCount(statistics.annealingTransitionFailureCount, 1,
                         "annealing transition failure"))
          return std::move(error);
        emitSpatialActionEvent(loom::mapping_debug::Event::ActionOutcome,
                               **action, SpatialSearchScope::Annealing,
                               seedAttemptOrdinal, slot, temperatureLevel,
                               schedule->temperature(), nullptr,
                               SpatialActionOutcome::TransitionFailure);
        continue;
      }
      if (llvm::Error error =
              addCount(statistics.annealingProbeCount, 1, "annealing probe"))
        return std::move(error);
      const dse::ObjectiveSignedDifference difference =
          probe->energyDifference();
      if (probe->isSemanticNoop()) {
        rememberInactiveAction(actionKey);
        if (llvm::Error error = addCount(statistics.semanticNoopActionCount, 1,
                                         "semantic no-op Action"))
          return std::move(error);
        if (llvm::Error error = probe->discard())
          return std::move(error);
        emitSpatialActionEvent(loom::mapping_debug::Event::ActionOutcome,
                               **action, SpatialSearchScope::Annealing,
                               seedAttemptOrdinal, slot, temperatureLevel,
                               schedule->temperature(), nullptr,
                               SpatialActionOutcome::SemanticNoop, difference);
        continue;
      }
      auto proposedClosure = spatialMappingViolationsAreZero(candidate);
      if (!proposedClosure)
        return proposedClosure.takeError();
      const dse::ObjectiveVector proposedObjective = probe->objective();
      auto resolution =
          probe->resolve(schedule->temperature(), acceptanceStream);
      if (!resolution)
        return resolution.takeError();
      const bool accepted = resolution->accepted;
      std::uint64_t &count = accepted ? statistics.acceptedActionCount
                                      : statistics.rejectedActionCount;
      if (llvm::Error error = addCount(
              count, 1, accepted ? "accepted Action" : "rejected Action"))
        return std::move(error);
      if (accepted) {
        if (difference.sign == dse::ObjectiveDifferenceSign::Positive)
          if (llvm::Error error =
                  addCount(statistics.acceptedWorseningActionCount, 1,
                           "accepted worsening Action"))
            return std::move(error);
        domainCurrent = false;
        inactiveActionKeys_.clear();
        if (*proposedClosure) {
          bool improvesBest = !bestFeasibleObjective;
          if (bestFeasibleObjective) {
            auto comparison = problem.objectiveProgram().compareSelectedRank(
                proposedObjective, {}, *bestFeasibleObjective, {});
            if (!comparison)
              return comparison.takeError();
            improvesBest = *comparison < 0;
          }
          if (improvesBest) {
            auto snapshot = candidate.cloneFullyRouted();
            if (!snapshot)
              return snapshot.takeError();
            auto snapshotObjective =
                problem.objectiveProgram().evaluate(**snapshot);
            if (!snapshotObjective)
              return snapshotObjective.takeError();
            if (snapshotObjective->codes() != proposedObjective.codes())
              return searchError(
                  "best feasible snapshot changed its objective");
            bestFeasibleIncumbent = std::move(*snapshot);
            bestFeasibleObjective = std::move(*snapshotObjective);
          }
        }
      }
      emitSpatialActionEvent(loom::mapping_debug::Event::ActionOutcome,
                             **action, SpatialSearchScope::Annealing,
                             seedAttemptOrdinal, slot, temperatureLevel,
                             schedule->temperature(), nullptr,
                             accepted ? SpatialActionOutcome::Accepted
                                      : SpatialActionOutcome::Rejected,
                             difference);
      statistics.exactClosureReached = static_cast<bool>(bestFeasibleIncumbent);
    }
    loom::mapping_debug::emit(
        loom::mapping_debug::Level::Summary,
        loom::mapping_debug::Stage::SpatialPnr,
        loom::mapping_debug::Event::Statistics,
        [&](llvm::json::Object &fields) {
          llvm::json::Array violationValues;
          for (std::uint32_t ordinal = 0;
               ordinal != resolvedPnrViolationKindCount; ++ordinal) {
            auto value = spatialMappingViolationValue(
                candidate, static_cast<ResolvedPnrViolationKind>(ordinal));
            if (!value) {
              llvm::consumeError(value.takeError());
              violationValues.push_back(nullptr);
            } else {
              violationValues.push_back(*value);
            }
          }
          llvm::json::Array objectiveCodes;
          for (std::uint64_t code : actionExecutor_.currentObjective().codes())
            objectiveCodes.push_back(code);
          const SpatialActionProposalDomain domain = actionDomain_.view();
          fields["operation"] = "annealing_level";
          fields["seed_attempt"] = seedAttemptOrdinal;
          fields["temperature_level"] = statistics.temperatureLevelCount - 1;
          fields["temperature"] = schedule->temperature();
          fields["proposal_slots"] = *proposalCount;
          fields["probes"] = statistics.annealingProbeCount - levelProbeBegin;
          fields["accepted_actions"] =
              statistics.acceptedActionCount - levelAcceptedBegin;
          fields["rejected_actions"] =
              statistics.rejectedActionCount - levelRejectedBegin;
          fields["semantic_noop_actions"] =
              statistics.semanticNoopActionCount - levelNoopBegin;
          fields["cached_inactive_actions"] =
              statistics.cachedInactiveActionCount - levelCachedBegin;
          fields["transition_failures"] =
              statistics.annealingTransitionFailureCount - levelFailureBegin;
          fields["heuristic_cache_hits"] =
              actionExecutor_.heuristicCacheHitCount() - levelHeuristicHitBegin;
          fields["heuristic_builds"] =
              actionExecutor_.heuristicBuildCount() - levelHeuristicBuildBegin;
          fields["heuristic_cache_entries"] =
              actionExecutor_.heuristicCacheEntryCount();
          fields["heuristic_cache_evictions"] =
              actionExecutor_.heuristicCacheEvictionCount();
          fields["heuristic_cache_retained_bytes"] =
              actionExecutor_.heuristicCacheRetainedBytes();
          fields["inactive_cache_size"] = inactiveActionKeys_.size();
          fields["movable_decisions"] = movableDecisionCount;
          fields["realization_choices"] = domain.realizationChoices.size();
          fields["realization_choices_examined"] =
              actionDomain_.examinedRealizationChoiceCount();
          fields["realization_choices_pruned_by_fixed_relations"] =
              actionDomain_.fixedRelationPrunedRealizationChoiceCount();
          fields["routing_choices"] = domain.transportChoices.size();
          fields["resource_choices"] = domain.resourceChoices.size();
          fields["atomic_capacity_overuse"] = candidate.atomicCapacityOveruse();
          fields["route_capacity_overuse"] = candidate.routeCapacityOveruse();
          fields["tag_resident_capacity_overuse"] =
              candidate.tagResidentCapacityOveruse();
          fields["violation_values"] = std::move(violationValues);
          fields["objective_codes"] = std::move(objectiveCodes);
          fields["exact_closure"] = statistics.exactClosureReached;
        });
  } while (schedule->advanceAfterCompletedLevel());

  if (executionControl.stopRequested())
    return finishInterrupted();

  if (statistics.minimumTemperatureLevelCount != 1)
    return searchError(
        "annealing schedule did not execute one minimum-temperature level");
  if (llvm::Error error = candidate.verify())
    return std::move(error);
  statistics.endpointExpansions = actionExecutor_.endpointExpansionCount();
  statistics.negotiationIterations =
      actionExecutor_.negotiationIterationCount();
  if (bestFeasibleIncumbent) {
    auto restoredObjective =
        problem.objectiveProgram().evaluate(*bestFeasibleIncumbent);
    if (!restoredObjective)
      return restoredObjective.takeError();
    if (!bestFeasibleObjective ||
        restoredObjective->codes() != bestFeasibleObjective->codes())
      return searchError("best feasible incumbent objective changed");
    if (llvm::Error error = bestFeasibleIncumbent->verify())
      return std::move(error);
    candidateHandle = std::move(bestFeasibleIncumbent);
    statistics.bestFeasibleIncumbentRestored = true;
    loom::mapping_debug::emit(loom::mapping_debug::Level::Summary,
                              loom::mapping_debug::Stage::SpatialPnr,
                              loom::mapping_debug::Event::Statistics,
                              [&](llvm::json::Object &fields) {
                                fields["operation"] = "annealing_incumbent";
                                fields["seed_attempt"] = seedAttemptOrdinal;
                                fields["reason"] = "best_feasible_restored";
                                fields["accepted_worsening_actions"] =
                                    statistics.acceptedWorseningActionCount;
                              });
  }
  return statistics;
}

std::size_t SpatialAnnealingSearchScratch::retainedStorageBytes() const {
  return actionDomain_.retainedStorageBytes() +
         actionExecutor_.retainedStorageBytes() +
         retainedBytes(positiveCalibrationDeltas_) +
         retainedBytes(inactiveActionKeys_);
}
