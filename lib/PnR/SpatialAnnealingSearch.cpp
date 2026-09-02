#include "PnR/SpatialAnnealingSearch.h"

#include "Common/MappingDebugLog.h"
#include "PnR/MappingObjective.h"
#include "PnR/SpatialCanonicalSeed.h"

#include "SpatialLocalTransferIndex.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
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
  LocalTransferAdoption,
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
  case SpatialSearchScope::LocalTransferAdoption:
    return "local_transfer_adoption";
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

llvm::Expected<bool>
spatialMappingIsExactRepairReady(const SpatialCandidateState &candidate) {
  for (std::uint32_t ordinal = 0; ordinal != resolvedPnrViolationKindCount;
       ++ordinal) {
    const auto kind = static_cast<ResolvedPnrViolationKind>(ordinal);
    if (kind == ResolvedPnrViolationKind::CapacityOveruse)
      continue;
    auto value = spatialMappingViolationValue(candidate, kind);
    if (!value)
      return value.takeError();
    if (*value != 0)
      return false;
  }
  return true;
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
                } else if constexpr (std::is_same_v<
                                         Choice, SpatialGlobalRoutingAction>) {
                  fields["action_kind"] = "global";
                } else {
                  fields["action_kind"] = "physical_tag";
                  fields["logical_net"] = choice.logicalNet;
                  fields["segment_ordinal"] = choice.segmentOrdinal;
                  fields["tag_width_bits"] = choice.value.getBitWidth();
                  llvm::SmallString<32> tagValue;
                  choice.value.toStringUnsigned(tagValue, 10);
                  fields["tag_value"] = std::string(tagValue);
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

/// One local-transfer adoption probe: the register-FIFO whole-net Action and,
/// when the option needs it, the single endpoint relocation it is coupled to.
void emitLocalTransferAdoptionEvent(
    loom::mapping_debug::Event event, std::uint64_t seedAttemptOrdinal,
    std::uint64_t probeOrdinal, PnrIndex logicalNet,
    const SpatialLocalTransferAdoption &adoption,
    std::optional<SpatialActionOutcome> outcome = std::nullopt,
    std::optional<dse::ObjectiveSignedDifference> difference = std::nullopt) {
  loom::mapping_debug::emit(
      loom::mapping_debug::Level::Decision,
      loom::mapping_debug::Stage::SpatialPnr, event,
      [&](llvm::json::Object &fields) {
        fields["search_scope"] =
            spelling(SpatialSearchScope::LocalTransferAdoption);
        fields["seed_attempt"] = seedAttemptOrdinal;
        fields["proposal_slot"] = probeOrdinal;
        if (outcome)
          fields["outcome"] = spelling(*outcome);
        encodeSpatialAction(
            fields,
            SpatialTransportRoutingAction{SpatialWholeNetRoutingAction{
                logicalNet, SpatialWholeNetDispositionKind::RegisterFifo,
                adoption.option}});
        if (adoption.relocation) {
          fields["relocation_realization"] = adoption.relocation->realization;
          fields["relocation_placement"] = adoption.relocation->placement;
          fields["relocation_instruction_context"] =
              adoption.relocation->instructionContext;
        }
        if (difference &&
            loom::mapping_debug::enabled(loom::mapping_debug::Level::Detail)) {
          fields["energy_difference_sign"] = differenceSign(difference->sign);
          fields["energy_difference_high"] = difference->magnitude.high;
          fields["energy_difference_low"] = difference->magnitude.low;
        }
      });
}

} // namespace

llvm::Expected<std::optional<SpatialActionTransitionFailureKind>>
SpatialAnnealingSearchScratch::consumeTransitionFailure(llvm::Error failure) {
  std::optional<SpatialActionTransitionFailureKind> consumed;
  llvm::Error unhandled = llvm::handleErrors(
      std::move(failure),
      [&](const SpatialActionTransitionFailure &transition) -> llvm::Error {
        consumed = transition.kind();
        return llvm::Error::success();
      });
  if (unhandled)
    return std::move(unhandled);
  return consumed;
}

llvm::Error SpatialAnnealingSearchScratch::adoptAdmittedLocalTransfers(
    SpatialCandidateState &candidate, std::uint64_t seedAttemptOrdinal,
    SpatialAnnealingStatistics &statistics,
    ExecutionControlView executionControl,
    SpatialPnrWorkLedgerView workLedger) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const auto &localTransfers = problem.localTransfers();
  const std::size_t logicalNetCount = problem.transfers().logicalNets().size();
  std::uint64_t consideredNets = 0;
  std::uint64_t probes = 0;
  std::uint64_t adopted = 0;
  std::uint64_t relocated = 0;
  const auto emitSweep = [&](llvm::StringRef status) {
    loom::mapping_debug::emit(
        loom::mapping_debug::Level::Summary,
        loom::mapping_debug::Stage::SpatialPnr,
        loom::mapping_debug::Event::Statistics,
        [&](llvm::json::Object &fields) {
          fields["operation"] = "local_transfer_adoption_sweep";
          fields["seed_attempt"] = seedAttemptOrdinal;
          fields["status"] = status;
          fields["admitted_option_count"] = localTransfers.options().size();
          fields["candidate_nets"] = consideredNets;
          fields["probes"] = probes;
          fields["adopted"] = adopted;
          fields["relocated"] = relocated;
          std::uint64_t selected = 0;
          for (PnrIndex logicalNet = 0; logicalNet < logicalNetCount;
               ++logicalNet)
            selected += candidate.usesRegisterFifo(logicalNet);
          fields["register_fifo_transfers"] = selected;
        });
  };
  auto feasible = spatialMappingViolationsAreZero(candidate);
  if (!feasible)
    return feasible.takeError();
  if (!*feasible) {
    emitSweep("skipped_violation");
    return llvm::Error::success();
  }
  if (localTransfers.options().empty()) {
    emitSweep("no_admitted_option");
    return llvm::Error::success();
  }
  if (llvm::Error error = actionDomain_.prepare(problem))
    return error;
  if (llvm::Error error =
          actionExecutor_.prepare(candidate, workLedger, executionControl))
    return error;
  if (llvm::Error error = actionDomain_.rebuild(candidate))
    return error;

  // A relocation may drop the pairing of another net incident to the moved
  // realization, so after a pass that relocated an endpoint a second,
  // relocation-free pass adopts every pairing that became resident; it only
  // adds pairings and therefore terminates.
  bool interrupted = false;
  for (const bool relocationsAdmitted : {true, false}) {
    if (interrupted || (!relocationsAdmitted && relocated == 0))
      break;
    for (PnrIndex logicalNet = 0; logicalNet < logicalNetCount && !interrupted;
         ++logicalNet) {
      if (candidate.usesRegisterFifo(logicalNet) ||
          localTransfers.domains()[logicalNet].optionCount == 0)
        continue;
      if (llvm::Error error = detail::enumerateSpatialLocalTransferAdoptions(
              problem, candidate.computeBindingSelections(),
              candidate.registerFifoTransferSelections(), logicalNet,
              relocationsAdmitted
                  ? actionDomain_.view().realizationChoices
                  : llvm::ArrayRef<SpatialRealizationBindingAction>(),
              adoptionAlternatives_))
        return error;
      if (adoptionAlternatives_.empty())
        continue;
      consideredNets += relocationsAdmitted;
      for (const SpatialLocalTransferAdoption &adoption :
           adoptionAlternatives_) {
        if (executionControl.stopRequested()) {
          interrupted = true;
          break;
        }
        llvm::SmallVector<SpatialMappingAction, 2> actions;
        if (adoption.relocation)
          actions.push_back(
              SpatialRealizationBindingAction{*adoption.relocation});
        actions.push_back(
            SpatialTransportRoutingAction{SpatialWholeNetRoutingAction{
                logicalNet, SpatialWholeNetDispositionKind::RegisterFifo,
                adoption.option}});
        if (llvm::Error error =
                addCount(statistics.plannedLocalTransferAdoptionProbes, 1,
                         "planned local-transfer adoption probe"))
          return error;
        if (llvm::Error error =
                workLedger.plan(SpatialPnrWorkKind::LocalTransferAdoptionProbe))
          return error;
        emitLocalTransferAdoptionEvent(
            loom::mapping_debug::Event::ActionProposal, seedAttemptOrdinal,
            probes, logicalNet, adoption);
        auto probe = actionExecutor_.probeBatch(
            candidate, actions, SpatialActionExecutionContext::Search);
        if (llvm::Error error = addCount(statistics.localTransferAdoptionProbes,
                                         1, "local-transfer adoption probe"))
          return error;
        if (llvm::Error error = workLedger.consume(
                SpatialPnrWorkKind::LocalTransferAdoptionProbe))
          return error;
        const std::uint64_t probeOrdinal = probes++;
        if (!probe) {
          auto consumed = consumeTransitionFailure(probe.takeError());
          if (!consumed)
            return consumed.takeError();
          if (!*consumed)
            return searchError(
                "adoption failure had no failure classification");
          if (**consumed == SpatialActionTransitionFailureKind::Interrupted) {
            interrupted = true;
            break;
          }
          emitLocalTransferAdoptionEvent(
              loom::mapping_debug::Event::ActionOutcome, seedAttemptOrdinal,
              probeOrdinal, logicalNet, adoption,
              SpatialActionOutcome::TransitionFailure);
          continue;
        }
        const dse::ObjectiveSignedDifference difference =
            probe->energyDifference();
        if (probe->isSemanticNoop()) {
          if (llvm::Error error = probe->discard())
            return error;
          emitLocalTransferAdoptionEvent(
              loom::mapping_debug::Event::ActionOutcome, seedAttemptOrdinal,
              probeOrdinal, logicalNet, adoption,
              SpatialActionOutcome::SemanticNoop, difference);
          continue;
        }
        auto comparison = problem.objectiveProgram().compareSelectedRank(
            probe->objective(), {}, actionExecutor_.currentObjective(), {});
        if (!comparison)
          return comparison.takeError();
        if (*comparison > 0) {
          if (llvm::Error error = probe->discard())
            return error;
          emitLocalTransferAdoptionEvent(
              loom::mapping_debug::Event::ActionOutcome, seedAttemptOrdinal,
              probeOrdinal, logicalNet, adoption,
              SpatialActionOutcome::Rejected, difference);
          continue;
        }
        if (llvm::Error error = probe->commit())
          return error;
        ++adopted;
        relocated += adoption.relocation.has_value();
        emitLocalTransferAdoptionEvent(
            loom::mapping_debug::Event::ActionOutcome, seedAttemptOrdinal,
            probeOrdinal, logicalNet, adoption, SpatialActionOutcome::Accepted,
            difference);
        if (llvm::Error error = actionDomain_.rebuild(candidate))
          return error;
        break;
      }
    }
  }
  auto closed = spatialMappingViolationsAreZero(candidate);
  if (!closed)
    return closed.takeError();
  if (!*closed)
    return searchError("local-transfer adoption left a Mapping violation");
  if (llvm::Error error = addCount(statistics.adoptedLocalTransfers, adopted,
                                   "adopted local transfer"))
    return error;
  if (llvm::Error error = addCount(statistics.relocatedLocalTransfers,
                                   relocated, "relocated local transfer"))
    return error;
  emitSweep(interrupted ? "interrupted" : "completed");
  return llvm::Error::success();
}

llvm::Expected<SpatialAnnealingStatistics>
SpatialAnnealingSearchScratch::run(SpatialPathFinderSeed &seed,
                                   ExecutionControlView executionControl,
                                   SpatialPnrWorkLedgerView workLedger) {
  return run(seed.candidate, seed.attemptOrdinal, executionControl, workLedger);
}

llvm::Expected<SpatialAnnealingStatistics>
SpatialAnnealingSearchScratch::run(SpatialCandidateStateHandle &candidateHandle,
                                   std::uint64_t seedAttemptOrdinal,
                                   ExecutionControlView executionControl,
                                   SpatialPnrWorkLedgerView workLedger) {
  if (!candidateHandle)
    return searchError("candidate owner is null");
  SpatialCandidateState &candidate = *candidateHandle;
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const ResolvedPnrPolicyConfig &policy = problem.config().policy();
  if (seedAttemptOrdinal >= policy.search.initializer.seedAttemptCount)
    return searchError("seed attempt ordinal is outside the fixed slot set");

  SpatialAnnealingStatistics statistics;
  auto exactClosure = spatialMappingViolationsAreZero(candidate);
  if (!exactClosure)
    return exactClosure.takeError();
  if (*exactClosure && policy.search.completionGoal ==
                           ResolvedPnrCompletionGoal::FirstVerifiedCandidate) {
    statistics.exactClosureReached = true;
    statistics.completionGoalReached = true;
    loom::mapping_debug::emit(loom::mapping_debug::Level::Summary,
                              loom::mapping_debug::Stage::SpatialPnr,
                              loom::mapping_debug::Event::Statistics,
                              [&](llvm::json::Object &fields) {
                                fields["operation"] = "annealing_incumbent";
                                fields["seed_attempt"] = seedAttemptOrdinal;
                                fields["reason"] = "completion_goal_on_entry";
                              });
    if (llvm::Error error = adoptAdmittedLocalTransfers(
            candidate, seedAttemptOrdinal, statistics, executionControl,
            workLedger))
      return std::move(error);
    statistics.endpointExpansions = actionExecutor_.endpointExpansionCount();
    statistics.negotiationIterations =
        actionExecutor_.negotiationIterationCount();
    if (llvm::Error error = candidate.verify())
      return std::move(error);
    emitProvisionalHandshakeProjectionStatistics(
        actionExecutor_.handshakeProjectionStatistics(), seedAttemptOrdinal);
    return statistics;
  }
  if (llvm::Error error = actionDomain_.prepare(problem))
    return std::move(error);
  if (llvm::Error error =
          actionExecutor_.prepare(candidate, workLedger, executionControl))
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

  std::shared_ptr<const SpatialFullyRoutedSnapshot> bestSelectedRankIncumbent;
  std::optional<dse::ObjectiveVector> bestSelectedRankObjective;
  std::shared_ptr<const SpatialFullyRoutedSnapshot> bestFeasibleIncumbent;
  std::optional<dse::ObjectiveVector> bestFeasibleObjective;
  const auto captureIncumbent = [&](const dse::ObjectiveVector &objective,
                                    bool feasible) -> llvm::Error {
    bool improvesSelectedRank = !bestSelectedRankObjective;
    if (bestSelectedRankObjective) {
      auto comparison = problem.objectiveProgram().compareSelectedRank(
          objective, {}, *bestSelectedRankObjective, {});
      if (!comparison)
        return comparison.takeError();
      improvesSelectedRank = *comparison < 0;
    }
    bool improvesFeasible = feasible && !bestFeasibleObjective;
    if (feasible && bestFeasibleObjective) {
      auto comparison = problem.objectiveProgram().compareSelectedRank(
          objective, {}, *bestFeasibleObjective, {});
      if (!comparison)
        return comparison.takeError();
      improvesFeasible = *comparison < 0;
    }
    if (!improvesSelectedRank && !improvesFeasible)
      return llvm::Error::success();
    if (candidate.unroutedObligationCount() != 0) {
      if (feasible)
        return searchError("a feasible incumbent is incompletely routed");
      return llvm::Error::success();
    }
    auto snapshot = candidate.snapshotFullyRouted();
    if (!snapshot)
      return snapshot.takeError();
    if (statistics.incumbentSnapshotCount ==
        std::numeric_limits<std::uint64_t>::max())
      return searchError("search incumbent snapshot count overflows u64");
    ++statistics.incumbentSnapshotCount;
    auto shared = std::make_shared<const SpatialFullyRoutedSnapshot>(
        std::move(*snapshot));
    if (improvesSelectedRank) {
      bestSelectedRankIncumbent = shared;
      bestSelectedRankObjective = objective;
    }
    if (improvesFeasible) {
      bestFeasibleIncumbent = std::move(shared);
      bestFeasibleObjective = objective;
    }
    return llvm::Error::success();
  };
  // Materializes one stored incumbent snapshot into the returned candidate.
  // The materialized candidate must reproduce the objective observed when the
  // snapshot was captured; a mismatch means a derived-state owner diverged.
  const auto restoreIncumbent =
      [&](const SpatialFullyRoutedSnapshot &snapshot,
          const std::optional<dse::ObjectiveVector> &expected,
          llvm::StringRef mismatch) -> llvm::Error {
    auto restored = SpatialCandidateState::materializeFullyRouted(snapshot);
    if (!restored)
      return restored.takeError();
    auto restoredObjective = problem.objectiveProgram().evaluate(**restored);
    if (!restoredObjective)
      return restoredObjective.takeError();
    if (!expected || restoredObjective->codes() != expected->codes())
      return searchError(mismatch);
    candidateHandle = std::move(*restored);
    return llvm::Error::success();
  };
  if (llvm::Error error =
          captureIncumbent(actionExecutor_.currentObjective(), *exactClosure))
    return std::move(error);
  if (*exactClosure) {
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
  // The restored incumbent is a fresh candidate object: the sweep re-prepares
  // the executor on it, so its routing work is added to the annealing totals
  // captured before the restore.
  const auto adoptRestoredIncumbentLocalTransfers = [&]() -> llvm::Error {
    if (llvm::Error error = adoptAdmittedLocalTransfers(
            *candidateHandle, seedAttemptOrdinal, statistics, executionControl,
            workLedger))
      return error;
    if (llvm::Error error =
            addCount(statistics.endpointExpansions,
                     actionExecutor_.endpointExpansionCount(),
                     "endpoint expansion"))
      return error;
    return addCount(statistics.negotiationIterations,
                    actionExecutor_.negotiationIterationCount(),
                    "negotiation iteration");
  };
  const auto finishInterrupted =
      [&]() -> llvm::Expected<SpatialAnnealingStatistics> {
    statistics.interrupted = true;
    statistics.endpointExpansions = actionExecutor_.endpointExpansionCount();
    statistics.negotiationIterations =
        actionExecutor_.negotiationIterationCount();
    if (bestFeasibleIncumbent) {
      if (llvm::Error error = restoreIncumbent(
              *bestFeasibleIncumbent, bestFeasibleObjective,
              "search incumbent snapshot changed its objective"))
        return std::move(error);
      statistics.bestFeasibleIncumbentRestored = true;
    } else if (bestSelectedRankIncumbent) {
      if (llvm::Error error = restoreIncumbent(
              *bestSelectedRankIncumbent, bestSelectedRankObjective,
              "search incumbent snapshot changed its objective"))
        return std::move(error);
      statistics.bestSelectedRankIncumbentRestored = true;
    }
    if (llvm::Error error = candidateHandle->verify())
      return std::move(error);
    emitProvisionalHandshakeProjectionStatistics(
        actionExecutor_.handshakeProjectionStatistics(), seedAttemptOrdinal);
    return statistics;
  };
  const auto finishAtCompletionGoal =
      [&]() -> llvm::Expected<SpatialAnnealingStatistics> {
    if (!bestFeasibleIncumbent)
      return searchError("completion goal has no feasible incumbent");
    statistics.completionGoalReached = true;
    statistics.endpointExpansions = actionExecutor_.endpointExpansionCount();
    statistics.negotiationIterations =
        actionExecutor_.negotiationIterationCount();
    if (llvm::Error error =
            restoreIncumbent(*bestFeasibleIncumbent, bestFeasibleObjective,
                             "search incumbent snapshot changed its objective"))
      return std::move(error);
    statistics.bestFeasibleIncumbentRestored = true;
    loom::mapping_debug::emit(loom::mapping_debug::Level::Summary,
                              loom::mapping_debug::Stage::SpatialPnr,
                              loom::mapping_debug::Event::Statistics,
                              [&](llvm::json::Object &fields) {
                                fields["operation"] = "annealing_incumbent";
                                fields["seed_attempt"] = seedAttemptOrdinal;
                                fields["reason"] = "completion_goal_reached";
                              });
    if (llvm::Error error = adoptRestoredIncumbentLocalTransfers())
      return std::move(error);
    emitProvisionalHandshakeProjectionStatistics(
        actionExecutor_.handshakeProjectionStatistics(), seedAttemptOrdinal);
    return statistics;
  };
  const auto finishAtRepairReadyHandoff =
      [&]() -> llvm::Expected<SpatialAnnealingStatistics> {
    if (llvm::Error error = candidate.verify())
      return std::move(error);
    statistics.repairReadyHandoff = true;
    statistics.endpointExpansions = actionExecutor_.endpointExpansionCount();
    statistics.negotiationIterations =
        actionExecutor_.negotiationIterationCount();
    loom::mapping_debug::emit(loom::mapping_debug::Level::Summary,
                              loom::mapping_debug::Stage::SpatialPnr,
                              loom::mapping_debug::Event::Statistics,
                              [&](llvm::json::Object &fields) {
                                fields["operation"] = "annealing_incumbent";
                                fields["seed_attempt"] = seedAttemptOrdinal;
                                fields["reason"] = "exact_repair_handoff";
                                fields["temperature_levels"] =
                                    statistics.temperatureLevelCount;
                              });
    emitProvisionalHandshakeProjectionStatistics(
        actionExecutor_.handshakeProjectionStatistics(), seedAttemptOrdinal);
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
    if (llvm::Error error = addCount(statistics.plannedCalibrationProposalSlots,
                                     1, "planned calibration proposal slot"))
      return std::move(error);
    if (llvm::Error error =
            workLedger.plan(SpatialPnrWorkKind::CalibrationProposal))
      return std::move(error);
    auto action =
        actionDomain_.propose(policy.search.actionProposal, calibrationStream);
    if (!action)
      return action.takeError();
    if (llvm::Error error = addCount(statistics.calibrationProposalSlots, 1,
                                     "calibration proposal slot"))
      return std::move(error);
    if (llvm::Error error =
            workLedger.consume(SpatialPnrWorkKind::CalibrationProposal))
      return std::move(error);
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
    if (isIdentitySpatialAction(candidate, **action)) {
      // A proven identity re-selection is the semantic no-op the probe would
      // have discovered after the full transaction and closure. It draws no
      // acceptance randomness either way, so pruning it here keeps the search
      // trajectory identical while skipping the transaction.
      rememberInactiveAction(actionKey);
      if (llvm::Error error = addCount(statistics.semanticNoopActionCount, 1,
                                       "semantic no-op Action"))
        return std::move(error);
      emitSpatialActionEvent(loom::mapping_debug::Event::ActionOutcome,
                             **action, SpatialSearchScope::Calibration,
                             seedAttemptOrdinal, slot, std::nullopt,
                             std::nullopt, nullptr,
                             SpatialActionOutcome::SemanticNoop);
      continue;
    }

    auto probe = actionExecutor_.probe(candidate, **action);
    if (!probe) {
      auto consumed = consumeTransitionFailure(probe.takeError());
      if (!consumed)
        return consumed.takeError();
      if (!*consumed)
        return searchError("Action failure had no failure classification");
      if (**consumed == SpatialActionTransitionFailureKind::Interrupted)
        return finishInterrupted();
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
        fields["temperature_level_limit"] = annealing.temperatureLevelLimit;
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
    bool pendingDomainChangesValid = false;
    std::vector<std::pair<SpatialCandidateScratch::DecisionKind, PnrIndex>>
        pendingDomainDecisions;
    std::vector<PnrIndex> pendingDomainNets;
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
    const std::uint64_t levelForwardHeuristicQueryBegin =
        actionExecutor_.forwardHeuristicQueryCount();
    const std::uint64_t levelForwardHeuristicUnreachableBegin =
        actionExecutor_.forwardHeuristicUnreachableCount();
    const std::uint64_t levelArcCostValidationScanBegin =
        actionExecutor_.arcCostValidationScanCount();
    const std::uint64_t levelPhysicalTimingValidationScanBegin =
        actionExecutor_.physicalTimingValidationScanCount();
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
      if (!domainCurrent) {
        if (pendingDomainChangesValid) {
          if (llvm::Error error = actionDomain_.applyCommitted(
                  candidate, pendingDomainDecisions, pendingDomainNets))
            return std::move(error);
        } else if (llvm::Error error = actionDomain_.rebuild(candidate)) {
          return std::move(error);
        }
        pendingDomainChangesValid = false;
        domainCurrent = true;
      }
      const bool baseSlot = slot < annealing.proposalsPerLevelBase;
      std::uint64_t &plannedDomain =
          baseSlot ? statistics.plannedAnnealingBaseProposalSlots
                   : statistics.plannedAnnealingMovableProposalSlots;
      const SpatialPnrWorkKind workKind =
          baseSlot ? SpatialPnrWorkKind::AnnealingBaseProposal
                   : SpatialPnrWorkKind::AnnealingMovableProposal;
      if (llvm::Error error =
              addCount(plannedDomain, 1, "planned annealing domain slot"))
        return std::move(error);
      if (llvm::Error error = workLedger.plan(workKind))
        return std::move(error);
      auto action =
          actionDomain_.propose(policy.search.actionProposal, proposalStream);
      if (!action)
        return action.takeError();
      if (llvm::Error error = addCount(statistics.annealingProposalSlots, 1,
                                       "annealing proposal slot"))
        return std::move(error);
      std::uint64_t &slotDomain =
          baseSlot ? statistics.annealingBaseProposalSlots
                   : statistics.annealingMovableProposalSlots;
      if (llvm::Error error = addCount(slotDomain, 1, "annealing domain slot"))
        return std::move(error);
      if (llvm::Error error = workLedger.consume(workKind))
        return std::move(error);
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
      if (isIdentitySpatialAction(candidate, **action)) {
        rememberInactiveAction(actionKey);
        if (llvm::Error error = addCount(statistics.semanticNoopActionCount, 1,
                                         "semantic no-op Action"))
          return std::move(error);
        emitSpatialActionEvent(loom::mapping_debug::Event::ActionOutcome,
                               **action, SpatialSearchScope::Annealing,
                               seedAttemptOrdinal, slot, temperatureLevel,
                               schedule->temperature(), nullptr,
                               SpatialActionOutcome::SemanticNoop);
        continue;
      }

      auto probe = actionExecutor_.probe(candidate, **action);
      if (!probe) {
        auto consumed = consumeTransitionFailure(probe.takeError());
        if (!consumed)
          return consumed.takeError();
        if (!*consumed)
          return searchError("Action failure had no failure classification");
        if (**consumed == SpatialActionTransitionFailureKind::Interrupted)
          return finishInterrupted();
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
        pendingDomainChangesValid = actionExecutor_.hasCommittedChanges();
        if (pendingDomainChangesValid) {
          pendingDomainDecisions.assign(
              actionExecutor_.committedDecisionChanges().begin(),
              actionExecutor_.committedDecisionChanges().end());
          pendingDomainNets.assign(
              actionExecutor_.committedLogicalNetChanges().begin(),
              actionExecutor_.committedLogicalNetChanges().end());
        }
        inactiveActionKeys_.clear();
        if (llvm::Error error =
                captureIncumbent(proposedObjective, *proposedClosure))
          return std::move(error);
      }
      emitSpatialActionEvent(loom::mapping_debug::Event::ActionOutcome,
                             **action, SpatialSearchScope::Annealing,
                             seedAttemptOrdinal, slot, temperatureLevel,
                             schedule->temperature(), nullptr,
                             accepted ? SpatialActionOutcome::Accepted
                                      : SpatialActionOutcome::Rejected,
                             difference);
      statistics.exactClosureReached = static_cast<bool>(bestFeasibleIncumbent);
      if (accepted && *proposedClosure &&
          policy.search.completionGoal ==
              ResolvedPnrCompletionGoal::FirstVerifiedCandidate)
        return finishAtCompletionGoal();
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
          const std::uint64_t heuristicCacheHits =
              actionExecutor_.heuristicCacheHitCount() - levelHeuristicHitBegin;
          const std::uint64_t heuristicBuilds =
              actionExecutor_.heuristicBuildCount() - levelHeuristicBuildBegin;
          fields["heuristic_cache_hits"] = heuristicCacheHits;
          fields["heuristic_builds"] = heuristicBuilds;
          const std::uint64_t forwardHeuristicQueries =
              actionExecutor_.forwardHeuristicQueryCount() -
              levelForwardHeuristicQueryBegin;
          const std::uint64_t forwardHeuristicUnreachable =
              actionExecutor_.forwardHeuristicUnreachableCount() -
              levelForwardHeuristicUnreachableBegin;
          fields["forward_heuristic_queries"] = forwardHeuristicQueries;
          fields["forward_heuristic_unreachable_queries"] =
              forwardHeuristicUnreachable;
          if (forwardHeuristicQueries != 0)
            fields["forward_heuristic_unreachable_ratio"] =
                static_cast<double>(forwardHeuristicUnreachable) /
                static_cast<double>(forwardHeuristicQueries);
          else
            fields["forward_heuristic_unreachable_ratio"] = nullptr;
          const std::uint64_t heuristicLookups =
              heuristicBuilds > std::numeric_limits<std::uint64_t>::max() -
                                    heuristicCacheHits
                  ? std::numeric_limits<std::uint64_t>::max()
                  : heuristicCacheHits + heuristicBuilds;
          fields["heuristic_cache_lookups"] = heuristicLookups;
          if (heuristicLookups != 0)
            fields["heuristic_cache_hit_ratio"] =
                static_cast<double>(heuristicCacheHits) /
                static_cast<double>(heuristicLookups);
          else
            fields["heuristic_cache_hit_ratio"] = nullptr;
          fields["heuristic_cache_entries"] =
              actionExecutor_.heuristicCacheEntryCount();
          fields["heuristic_cache_evictions"] =
              actionExecutor_.heuristicCacheEvictionCount();
          fields["heuristic_cache_retained_bytes"] =
              actionExecutor_.heuristicCacheRetainedBytes();
          fields["arc_cost_validation_scans"] =
              actionExecutor_.arcCostValidationScanCount() -
              levelArcCostValidationScanBegin;
          fields["physical_timing_validation_scans"] =
              actionExecutor_.physicalTimingValidationScanCount() -
              levelPhysicalTimingValidationScanBegin;
          fields["inactive_cache_size"] = inactiveActionKeys_.size();
          const HandshakeActiveDemandStatistics handshakeStatistics =
              candidate.handshake().materializationStatistics();
          fields["handshake_transaction_closures"] =
              handshakeStatistics.transactionClosureCount;
          fields["handshake_inserted_arcs"] =
              handshakeStatistics.transactionInsertedArcCount;
          fields["handshake_removed_arcs"] =
              handshakeStatistics.transactionRemovedArcCount;
          fields["handshake_affected_nodes"] =
              handshakeStatistics.transactionAffectedNodeCount;
          fields["handshake_affected_rank_span"] =
              handshakeStatistics.transactionAffectedRankSpan;
          fields["handshake_materialized_nodes"] =
              handshakeStatistics.materializedNodeCount;
          fields["handshake_materialized_arcs"] =
              handshakeStatistics.materializedArcCount;
          fields["handshake_materialization_work"] =
              handshakeStatistics.deterministicWork;
          fields["incumbent_snapshots"] = statistics.incumbentSnapshotCount;
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
    auto repairReady = spatialMappingIsExactRepairReady(candidate);
    if (!repairReady)
      return repairReady.takeError();
    if (*repairReady &&
        policy.search.completionGoal ==
            ResolvedPnrCompletionGoal::FirstVerifiedCandidate &&
        policy.search.exactRepair.kind != ResolvedPnrExactRepairKind::Disabled)
      return finishAtRepairReadyHandoff();
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
    if (llvm::Error error =
            restoreIncumbent(*bestFeasibleIncumbent, bestFeasibleObjective,
                             "best feasible incumbent objective changed"))
      return std::move(error);
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
                                fields["incumbent_snapshots"] =
                                    statistics.incumbentSnapshotCount;
                              });
    if (llvm::Error error = adoptRestoredIncumbentLocalTransfers())
      return std::move(error);
  } else if (bestSelectedRankIncumbent) {
    if (llvm::Error error = restoreIncumbent(
            *bestSelectedRankIncumbent, bestSelectedRankObjective,
            "best selected-rank incumbent objective changed"))
      return std::move(error);
    statistics.bestSelectedRankIncumbentRestored = true;
    loom::mapping_debug::emit(loom::mapping_debug::Level::Summary,
                              loom::mapping_debug::Stage::SpatialPnr,
                              loom::mapping_debug::Event::Statistics,
                              [&](llvm::json::Object &fields) {
                                fields["operation"] = "annealing_incumbent";
                                fields["seed_attempt"] = seedAttemptOrdinal;
                                fields["reason"] =
                                    "best_selected_rank_restored";
                                fields["accepted_worsening_actions"] =
                                    statistics.acceptedWorseningActionCount;
                                fields["incumbent_snapshots"] =
                                    statistics.incumbentSnapshotCount;
                              });
  }
  emitProvisionalHandshakeProjectionStatistics(
      actionExecutor_.handshakeProjectionStatistics(), seedAttemptOrdinal);
  return statistics;
}

std::size_t SpatialAnnealingSearchScratch::retainedStorageBytes() const {
  return actionDomain_.retainedStorageBytes() +
         actionExecutor_.retainedStorageBytes() +
         retainedBytes(positiveCalibrationDeltas_) +
         retainedBytes(inactiveActionKeys_) +
         retainedBytes(adoptionAlternatives_);
}
