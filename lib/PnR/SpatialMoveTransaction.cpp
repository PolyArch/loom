#include "PnR/SpatialCandidateState.h"

#include "SpatialBindingRelationModel.h"
#include "SpatialCandidateStateInternal.h"
#include "SpatialMemoryConstraintModel.h"
#include "SpatialRouteConstraintModel.h"

#include "Common/MappingDebugLog.h"
#include "Fabric/Identity/FabricRefText.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

using namespace loom::pnr;

namespace {

using detail::attachmentTraversal;
using detail::candidateError;
using detail::computePlacementFragments;
using detail::memoryPlanFragments;
using detail::rangeContains;

llvm::Error replaceContribution(std::uint64_t oldValue, std::uint64_t newValue,
                                std::uint64_t &total, llvm::StringRef subject) {
  if (oldValue > total)
    return candidateError(subject + " is inconsistent with its total");
  const std::uint64_t base = total - oldValue;
  if (newValue > std::numeric_limits<std::uint64_t>::max() - base)
    return candidateError(subject + " total overflows u64");
  total = base + newValue;
  return llvm::Error::success();
}

llvm::StringRef signalKind(loom::fabric::HandshakeSignalKind kind) {
  switch (kind) {
  case loom::fabric::HandshakeSignalKind::Valid:
    return "valid";
  case loom::fabric::HandshakeSignalKind::Ready:
    return "ready";
  }
  llvm_unreachable("unknown handshake signal kind");
}

llvm::StringRef ownerKind(loom::fabric::FabricHandshakeOwnerKind kind) {
  using Kind = loom::fabric::FabricHandshakeOwnerKind;
  switch (kind) {
  case Kind::PointConnection:
    return "point_connection";
  case Kind::PeOccurrence:
    return "pe_occurrence";
  case Kind::FuOccurrence:
    return "fu_occurrence";
  case Kind::MemoryOccurrence:
    return "memory_occurrence";
  case Kind::SwitchOccurrence:
    return "switch_occurrence";
  case Kind::FifoOccurrence:
    return "fifo_occurrence";
  case Kind::BoundaryOccurrence:
    return "boundary_occurrence";
  case Kind::TransferPattern:
    return "transfer_pattern";
  }
  llvm_unreachable("unknown handshake owner kind");
}

void appendOwnerFields(llvm::json::Object &fields,
                       const loom::fabric::FabricHandshakeOwner &owner) {
  fields["owner_kind"] = ownerKind(owner.kind());
  std::visit(
      [&](const auto &payload) {
        using Payload = std::decay_t<decltype(payload)>;
        if constexpr (std::is_same_v<
                          Payload,
                          loom::fabric::FabricPointConnectionPayload>) {
          fields["source_endpoint_ref"] =
              loom::fabric::printFabricRef(payload.source);
          fields["destination_endpoint_ref"] =
              loom::fabric::printFabricRef(payload.destination);
        } else {
          fields["owner_ref"] = loom::fabric::printFabricRef(payload);
        }
      },
      owner.payload());
}

void emitHandshakeCycle(const FrozenSpatialPnrProblem &problem,
                        const HandshakeCandidateState &handshake,
                        llvm::ArrayRef<PnrIndex> witness) {
  loom::mapping_debug::emit(
      loom::mapping_debug::Level::Decision,
      loom::mapping_debug::Stage::SpatialPnr,
      loom::mapping_debug::Event::MappingFailure,
      [&](llvm::json::Object &fields) {
        fields["operation"] = "selected_handshake_cycle";
        fields["witness_arc_count"] = witness.size();
        const std::size_t sampleCount =
            loom::mapping_debug::enabled(loom::mapping_debug::Level::Detail)
                ? witness.size()
                : std::min<std::size_t>(witness.size(), 8);
        fields["witness_arc_sample_count"] = sampleCount;
        fields["witness_arc_omitted_count"] = witness.size() - sampleCount;
        llvm::json::Array arcs;
        const auto potentialArcs = problem.handshake().arcs();
        const auto signals = problem.handshake().nodeSignals();
        for (PnrIndex arc : witness.take_front(sampleCount)) {
          llvm::json::Object entry;
          entry["arc_ref"] = arc;
          if (arc < potentialArcs.size()) {
            const FrozenSpatialHandshakeArc &record = potentialArcs[arc];
            entry["source_node"] = record.source;
            entry["destination_node"] = record.destination;
            if (record.source < signals.size() && signals[record.source]) {
              entry["source_endpoint_ref"] = loom::fabric::printFabricRef(
                  signals[record.source]->endpoint);
              entry["source_signal"] =
                  signalKind(signals[record.source]->signal);
            }
            if (record.destination < signals.size() &&
                signals[record.destination]) {
              entry["destination_endpoint_ref"] = loom::fabric::printFabricRef(
                  signals[record.destination]->endpoint);
              entry["destination_signal"] =
                  signalKind(signals[record.destination]->signal);
            }
            llvm::json::Array contributions;
            std::size_t contributionCount = 0;
            const auto fragments = problem.handshake().fragments();
            const auto fragmentOwners =
                problem.handshake().fragmentOwnerOrdinals();
            const auto owners = problem.handshake().owners();
            for (auto [fragmentOrdinal, fragment] :
                 llvm::enumerate(fragments)) {
              if (handshake.fragmentRefcount(
                      static_cast<PnrIndex>(fragmentOrdinal)) == 0)
                continue;
              const auto fragmentArcs =
                  problem.handshake().fragmentArcOrdinals().slice(
                      fragment.contributionOffset,
                      fragment.contributionCount);
              if (!llvm::binary_search(fragmentArcs, arc))
                continue;
              ++contributionCount;
              if (!loom::mapping_debug::enabled(
                      loom::mapping_debug::Level::Detail) &&
                  contributions.size() == 4)
                continue;
              llvm::json::Object contribution;
              contribution["fragment_ref"] = fragmentOrdinal;
              contribution["fragment_refcount"] = handshake.fragmentRefcount(
                  static_cast<PnrIndex>(fragmentOrdinal));
              const PnrIndex owner = fragmentOwners[fragmentOrdinal];
              contribution["owner_ordinal"] = owner;
              appendOwnerFields(contribution, owners[owner]);
              contributions.push_back(std::move(contribution));
            }
            entry["active_contribution_count"] = contributionCount;
            entry["active_contribution_omitted_count"] =
                contributionCount - contributions.size();
            entry["active_contributions"] = std::move(contributions);
          }
          arcs.push_back(std::move(entry));
        }
        fields["witness_arcs"] = std::move(arcs);
      });
}

} // namespace

SpatialMoveTransaction::SpatialMoveTransaction(
    SpatialCandidateStateHandle state, SpatialCandidateScratch &scratch)
    : state_(std::move(state)), scratch_(&scratch),
      initialUnroutedObligationCount_(state_->unroutedObligationCount_),
      initialAtomicCapacityOveruse_(state_->atomicCapacityOveruse_) {
  state_->activeTransaction_ = this;
  scratch_->activeTransaction_ = this;
}

SpatialMoveTransaction::SpatialMoveTransaction(
    SpatialMoveTransaction &&other) noexcept
    : state_(std::move(other.state_)), scratch_(other.scratch_),
      closed_(other.closed_), cycle_(other.cycle_),
      routeDeltasCollected_(other.routeDeltasCollected_),
      tagDeltasCollected_(other.tagDeltasCollected_),
      routeViolationApplied_(other.routeViolationApplied_),
      initialUnroutedObligationCount_(other.initialUnroutedObligationCount_),
      initialAtomicCapacityOveruse_(other.initialAtomicCapacityOveruse_) {
  other.scratch_ = nullptr;
  if (state_)
    state_->activeTransaction_ = this;
  if (scratch_)
    scratch_->activeTransaction_ = this;
}

SpatialMoveTransaction::~SpatialMoveTransaction() {
  if (scratch_)
    rollback();
}

llvm::Error SpatialMoveTransaction::ensureCollecting() const {
  if (!scratch_)
    return candidateError("move is no longer active");
  if (closed_)
    return candidateError("move is already closed");
  return llvm::Error::success();
}

void SpatialMoveTransaction::recordCompute(PnrIndex realization) {
  if (scratch_->computeJournalMarks_[realization] == scratch_->decisionEpoch_)
    return;
  scratch_->computeJournalMarks_[realization] = scratch_->decisionEpoch_;
  const auto old = state_->computeBindings_[realization];
  scratch_->decisionDeltas_.push_back(
      {SpatialCandidateScratch::DecisionKind::ComputeBinding, realization,
       old.placement, old.instructionContext,
       state_->bindingRelationChoices_[realization]});
}

void SpatialMoveTransaction::recordMemory(PnrIndex realization) {
  if (scratch_->memoryJournalMarks_[realization] == scratch_->decisionEpoch_)
    return;
  scratch_->memoryJournalMarks_[realization] = scratch_->decisionEpoch_;
  scratch_->decisionDeltas_.push_back(
      {SpatialCandidateScratch::DecisionKind::MemoryBinding, realization,
       state_->memoryBindings_[realization].placement, 0,
       state_->bindingRelationChoices_[state_->problem_->bindingRelations()
                                           .computeDecisionCount() +
                                       realization]});
}

void SpatialMoveTransaction::recordPort(PnrIndex demand) {
  if (scratch_->portJournalMarks_[demand] == scratch_->decisionEpoch_)
    return;
  scratch_->portJournalMarks_[demand] = scratch_->decisionEpoch_;
  scratch_->decisionDeltas_.push_back(
      {SpatialCandidateScratch::DecisionKind::PortAttachment, demand,
       state_->portAttachments_[demand], 0,
       state_->bindingRelationChoices_[state_->problem_->bindingRelations()
                                           .portDecisionOffset() +
                                       demand]});
}

void SpatialMoveTransaction::recordBoundary(PnrIndex boundary) {
  if (scratch_->boundaryJournalMarks_[boundary] == scratch_->decisionEpoch_)
    return;
  scratch_->boundaryJournalMarks_[boundary] = scratch_->decisionEpoch_;
  scratch_->decisionDeltas_.push_back(
      {SpatialCandidateScratch::DecisionKind::GraphBoundaryAttachment, boundary,
       state_->graphBoundaryAttachments_[boundary], 0,
       state_->bindingRelationChoices_[state_->problem_->bindingRelations()
                                           .graphBoundaryDecisionOffset() +
                                       boundary]});
}

void SpatialMoveTransaction::recordMemoryPlan(PnrIndex actor) {
  if (scratch_->memoryPlanJournalMarks_[actor] == scratch_->decisionEpoch_)
    return;
  scratch_->memoryPlanJournalMarks_[actor] = scratch_->decisionEpoch_;
  scratch_->decisionDeltas_.push_back(
      {SpatialCandidateScratch::DecisionKind::MemoryOperationPlan, actor,
       state_->memoryOperationPlans_[actor], 0});
}

void SpatialMoveTransaction::recordLogicalMemory(PnrIndex binding) {
  if (scratch_->logicalMemoryJournalMarks_[binding] == scratch_->decisionEpoch_)
    return;
  scratch_->logicalMemoryJournalMarks_[binding] = scratch_->decisionEpoch_;
  const auto old = state_->logicalMemoryBindings_[binding];
  scratch_->decisionDeltas_.push_back(
      {SpatialCandidateScratch::DecisionKind::LogicalMemoryBinding, binding,
       old.target, 0, 0, old.physicalOffsetBytes});
}

void SpatialMoveTransaction::recordMemoryDispatch(PnrIndex use) {
  if (scratch_->memoryDispatchJournalMarks_[use] == scratch_->decisionEpoch_)
    return;
  scratch_->memoryDispatchJournalMarks_[use] = scratch_->decisionEpoch_;
  scratch_->decisionDeltas_.push_back(
      {SpatialCandidateScratch::DecisionKind::MemoryUseDispatch, use,
       state_->memoryUseDispatches_[use], 0});
}

void SpatialMoveTransaction::recordMemoryExposure(PnrIndex exposure) {
  if (scratch_->memoryExposureJournalMarks_[exposure] ==
      scratch_->decisionEpoch_)
    return;
  scratch_->memoryExposureJournalMarks_[exposure] = scratch_->decisionEpoch_;
  scratch_->decisionDeltas_.push_back(
      {SpatialCandidateScratch::DecisionKind::MemoryExposure, exposure,
       state_->memoryExposureSelections_[exposure], 0});
}

void SpatialMoveTransaction::markCompute(PnrIndex realization) {
  if (scratch_->affectedComputeMarks_[realization] !=
      scratch_->affectedEpoch_) {
    scratch_->affectedComputeMarks_[realization] = scratch_->affectedEpoch_;
    scratch_->affectedComputes_.push_back(realization);
  }
  markBindingRelations(realization);
}

void SpatialMoveTransaction::markMemory(PnrIndex realization) {
  if (scratch_->affectedMemoryMarks_[realization] != scratch_->affectedEpoch_) {
    scratch_->affectedMemoryMarks_[realization] = scratch_->affectedEpoch_;
    scratch_->affectedMemories_.push_back(realization);
  }
  markBindingRelations(
      state_->problem_->bindingRelations().computeDecisionCount() +
      realization);
  const auto offsets =
      state_->problem_->ports().memoryRealizationDemandOffsets();
  for (PnrIndex demand :
       state_->problem_->ports().memoryRealizationDemands().slice(
           offsets[realization],
           offsets[realization + 1] - offsets[realization]))
    markPort(demand);
  const auto &record =
      state_->problem_->realizations().memoryRealizations()[realization];
  const auto &memory = state_->problem_->memory();
  for (PnrIndex local = 0; local < record.actorCount; ++local) {
    const PnrIndex actor = record.actorOffset + local;
    markMemoryPlan(actor);
    for (PnrIndex use = memory.actorUseOffsets()[actor];
         use != memory.actorUseOffsets()[actor + 1]; ++use)
      markMemoryDispatch(use);
  }
}

void SpatialMoveTransaction::markPort(PnrIndex demand) {
  if (scratch_->affectedPortMarks_[demand] != scratch_->affectedEpoch_) {
    scratch_->affectedPortMarks_[demand] = scratch_->affectedEpoch_;
    scratch_->affectedPorts_.push_back(demand);
  }
  markBindingRelations(
      state_->problem_->bindingRelations().portDecisionOffset() + demand);
  markNet(state_->problem_->ports().portDemands()[demand].logicalNet);
}

void SpatialMoveTransaction::markBoundary(PnrIndex boundary) {
  if (scratch_->affectedBoundaryMarks_[boundary] != scratch_->affectedEpoch_) {
    scratch_->affectedBoundaryMarks_[boundary] = scratch_->affectedEpoch_;
    scratch_->affectedBoundaries_.push_back(boundary);
  }
  markBindingRelations(
      state_->problem_->bindingRelations().graphBoundaryDecisionOffset() +
      boundary);
  markNet(state_->problem_->ports().graphBoundaries()[boundary].logicalNet);
}

void SpatialMoveTransaction::markMemoryPlan(PnrIndex actor) {
  if (scratch_->affectedMemoryPlanMarks_[actor] == scratch_->affectedEpoch_)
    return;
  scratch_->affectedMemoryPlanMarks_[actor] = scratch_->affectedEpoch_;
  scratch_->affectedMemoryPlans_.push_back(actor);
}

void SpatialMoveTransaction::markLogicalMemory(PnrIndex binding) {
  if (scratch_->affectedLogicalMemoryMarks_[binding] !=
      scratch_->affectedEpoch_) {
    scratch_->affectedLogicalMemoryMarks_[binding] = scratch_->affectedEpoch_;
    scratch_->affectedLogicalMemories_.push_back(binding);
  }
  const auto &memory = state_->problem_->memory();
  for (PnrIndex use :
       memory.bindingUses().slice(memory.bindingUseOffsets()[binding],
                                  memory.bindingUseOffsets()[binding + 1] -
                                      memory.bindingUseOffsets()[binding]))
    markMemoryDispatch(use);
  for (PnrIndex exposure : memory.bindingExposures().slice(
           memory.bindingExposureOffsets()[binding],
           memory.bindingExposureOffsets()[binding + 1] -
               memory.bindingExposureOffsets()[binding]))
    markMemoryExposure(exposure);
}

void SpatialMoveTransaction::markMemoryDispatch(PnrIndex use) {
  if (scratch_->affectedMemoryDispatchMarks_[use] != scratch_->affectedEpoch_) {
    scratch_->affectedMemoryDispatchMarks_[use] = scratch_->affectedEpoch_;
    scratch_->affectedMemoryDispatches_.push_back(use);
  }
  const PnrIndex group =
      state_->problem_->memory().rootedUseServiceGroups()[use];
  if (group != getInvalidPnrIndex())
    markMemoryServiceGroup(group);
}

void SpatialMoveTransaction::markMemoryServiceGroup(PnrIndex group) {
  if (scratch_->affectedMemoryServiceGroupMarks_[group] ==
      scratch_->affectedEpoch_)
    return;
  scratch_->affectedMemoryServiceGroupMarks_[group] = scratch_->affectedEpoch_;
  scratch_->affectedMemoryServiceGroups_.push_back(group);
}

void SpatialMoveTransaction::markMemoryExposure(PnrIndex exposure) {
  if (scratch_->affectedMemoryExposureMarks_[exposure] ==
      scratch_->affectedEpoch_)
    return;
  scratch_->affectedMemoryExposureMarks_[exposure] = scratch_->affectedEpoch_;
  scratch_->affectedMemoryExposures_.push_back(exposure);
}

void SpatialMoveTransaction::markNet(PnrIndex logicalNet) {
  if (scratch_->affectedNetMarks_[logicalNet] == scratch_->affectedEpoch_)
    return;
  scratch_->affectedNetMarks_[logicalNet] = scratch_->affectedEpoch_;
  scratch_->affectedNets_.push_back(logicalNet);
}

void SpatialMoveTransaction::markBindingRelations(PnrIndex decision) {
  for (PnrIndex relation :
       state_->problem_->bindingRelations().decisionRelations(decision)) {
    if (scratch_->affectedBindingRelationMarks_[relation] ==
        scratch_->affectedEpoch_)
      continue;
    scratch_->affectedBindingRelationMarks_[relation] =
        scratch_->affectedEpoch_;
    scratch_->affectedBindingRelations_.push_back(relation);
  }
}

llvm::Error
SpatialMoveTransaction::changeFragments(llvm::ArrayRef<PnrIndex> oldFragments,
                                        llvm::ArrayRef<PnrIndex> newFragments) {
  if (oldFragments == newFragments)
    return llvm::Error::success();
  if (llvm::Error error =
          scratch_->handshakeTransaction_->removeFragments(oldFragments))
    return error;
  return scratch_->handshakeTransaction_->addFragments(newFragments);
}

llvm::Error
SpatialMoveTransaction::changeTraversal(std::optional<PnrIndex> oldTraversal,
                                        std::optional<PnrIndex> newTraversal) {
  if (oldTraversal == newTraversal)
    return llvm::Error::success();
  if (oldTraversal)
    if (llvm::Error error =
            scratch_->handshakeTransaction_->removeTraversalUses(*oldTraversal,
                                                                 1))
      return error;
  if (newTraversal)
    return scratch_->handshakeTransaction_->addTraversalUses(*newTraversal, 1);
  return llvm::Error::success();
}

llvm::Error SpatialMoveTransaction::setComputeBinding(
    PnrIndex realization, PnrIndex placement, PnrIndex instructionContext) {
  if (llvm::Error error = ensureCollecting())
    return error;
  const auto realizations =
      state_->problem_->realizations().computeRealizations();
  if (realization >= realizations.size())
    return candidateError("compute realization is out of range");
  const auto &record = realizations[realization];
  if (!rangeContains(record.placementOffset, record.placementCount, placement))
    return candidateError("new compute placement is outside its domain");
  const auto &placementRecord =
      state_->problem_->realizations().computePlacements()[placement];
  if (!rangeContains(placementRecord.contextOffset,
                     placementRecord.contextCount, instructionContext))
    return candidateError("new instruction context is outside its domain");
  const auto old = state_->computeBindings_[realization];
  if (old.placement == placement &&
      old.instructionContext == instructionContext)
    return llvm::Error::success();
  const auto relationChoice =
      state_->problem_->bindingRelations().computeChoiceOrdinal(
          realization, placement, instructionContext);
  if (!relationChoice)
    return candidateError("new compute binding has no relation-domain choice");

  recordCompute(realization);
  markCompute(realization);
  if (old.placement != placement) {
    const auto offsets =
        state_->problem_->ports().computeRealizationDemandOffsets();
    for (PnrIndex demand :
         state_->problem_->ports().computeRealizationDemands().slice(
             offsets[realization],
             offsets[realization + 1] - offsets[realization]))
      markPort(demand);
    if (llvm::Error error =
            changeFragments(computePlacementFragments(
                                state_->problem_->handshake(), old.placement),
                            computePlacementFragments(
                                state_->problem_->handshake(), placement)))
      return error;
  }
  const auto overuse =
      state_->problem_->capacity().computeInstructionContextOveruse();
  if (llvm::Error error = replaceContribution(
          overuse[old.instructionContext], overuse[instructionContext],
          state_->atomicCapacityOveruse_, "compute capacity overuse"))
    return error;
  const auto envelopeOffsets =
      state_->problem_->capacity().computeInstructionContextEnvelopeOffsets();
  if (llvm::Error error = state_->replaceResourceTimeEnvelopeSlice(
          envelopeOffsets[old.instructionContext],
          envelopeOffsets[old.instructionContext + 1] -
              envelopeOffsets[old.instructionContext],
          envelopeOffsets[instructionContext],
          envelopeOffsets[instructionContext + 1] -
              envelopeOffsets[instructionContext]))
    return error;
  state_->computeBindings_[realization] = {placement, instructionContext};
  state_->bindingRelationChoices_[realization] = *relationChoice;
  return llvm::Error::success();
}

llvm::Error SpatialMoveTransaction::setMemoryBinding(PnrIndex realization,
                                                     PnrIndex placement) {
  if (llvm::Error error = ensureCollecting())
    return error;
  const auto realizations =
      state_->problem_->realizations().memoryRealizations();
  if (realization >= realizations.size())
    return candidateError("memory realization is out of range");
  const auto &record = realizations[realization];
  if (!rangeContains(record.placementOffset, record.placementCount, placement))
    return candidateError("new memory placement is outside its domain");
  if (state_->memoryBindings_[realization].placement == placement)
    return llvm::Error::success();
  const auto relationChoice =
      state_->problem_->bindingRelations().memoryChoiceOrdinal(realization,
                                                               placement);
  if (!relationChoice)
    return candidateError("new memory binding has no relation-domain choice");
  recordMemory(realization);
  markMemory(realization);
  state_->memoryBindings_[realization].placement = placement;
  state_->bindingRelationChoices_[state_->problem_->bindingRelations()
                                      .computeDecisionCount() +
                                  realization] = *relationChoice;
  return llvm::Error::success();
}

llvm::Error
SpatialMoveTransaction::setPortAttachment(PnrIndex demand,
                                          PnrIndex attachmentOption) {
  if (llvm::Error error = ensureCollecting())
    return error;
  if (demand >= state_->portAttachments_.size() ||
      attachmentOption >= state_->problem_->ports().attachmentOptions().size())
    return candidateError("new PortDemand attachment is out of range");
  const auto relationChoice =
      state_->problem_->bindingRelations().portAttachmentChoiceOrdinal(
          demand, attachmentOption);
  if (!relationChoice)
    return candidateError(
        "new PortDemand attachment has no relation-domain choice");

  const PnrIndex old = state_->portAttachments_[demand];
  state_->portAttachments_[demand] = attachmentOption;
  if (llvm::Error error = state_->validatePortAttachment(demand)) {
    state_->portAttachments_[demand] = old;
    return error;
  }
  state_->portAttachments_[demand] = old;
  if (old == attachmentOption)
    return llvm::Error::success();

  recordPort(demand);
  markPort(demand);
  if (llvm::Error error = changeTraversal(
          attachmentTraversal(state_->problem_->ports(), old),
          attachmentTraversal(state_->problem_->ports(), attachmentOption)))
    return error;
  state_->portAttachments_[demand] = attachmentOption;
  state_->bindingRelationChoices_
      [state_->problem_->bindingRelations().portDecisionOffset() + demand] =
      *relationChoice;
  return llvm::Error::success();
}

llvm::Error
SpatialMoveTransaction::setGraphBoundaryAttachment(PnrIndex boundary,
                                                   PnrIndex attachmentOption) {
  if (llvm::Error error = ensureCollecting())
    return error;
  if (boundary >= state_->graphBoundaryAttachments_.size() ||
      attachmentOption >= state_->problem_->ports().attachmentOptions().size())
    return candidateError("new graph-boundary attachment is out of range");
  const auto &record = state_->problem_->ports().graphBoundaries()[boundary];
  if (!rangeContains(record.attachmentOptionOffset,
                     record.attachmentOptionCount, attachmentOption))
    return candidateError(
        "new graph-boundary attachment is outside its domain");
  const auto relationChoice =
      state_->problem_->bindingRelations().graphBoundaryAttachmentChoiceOrdinal(
          boundary, attachmentOption);
  if (!relationChoice)
    return candidateError(
        "new graph-boundary attachment has no relation-domain choice");
  const PnrIndex old = state_->graphBoundaryAttachments_[boundary];
  if (old == attachmentOption)
    return llvm::Error::success();
  recordBoundary(boundary);
  markBoundary(boundary);
  state_->graphBoundaryAttachments_[boundary] = attachmentOption;
  state_->bindingRelationChoices_[state_->problem_->bindingRelations()
                                      .graphBoundaryDecisionOffset() +
                                  boundary] = *relationChoice;
  return llvm::Error::success();
}

llvm::Error SpatialMoveTransaction::setMemoryOperationPlan(PnrIndex actor,
                                                           PnrIndex plan) {
  if (llvm::Error error = ensureCollecting())
    return error;
  if (actor >= state_->memoryOperationPlans_.size() ||
      plan >= state_->problem_->handshake().memoryOperationPlans().size())
    return candidateError("new memory operation plan is out of range");
  const PnrIndex old = state_->memoryOperationPlans_[actor];
  state_->memoryOperationPlans_[actor] = plan;
  if (llvm::Error error = state_->validateMemoryOperationPlan(actor)) {
    state_->memoryOperationPlans_[actor] = old;
    return error;
  }
  state_->memoryOperationPlans_[actor] = old;
  if (old == plan)
    return llvm::Error::success();
  recordMemoryPlan(actor);
  markMemoryPlan(actor);
  if (llvm::Error error = changeFragments(
          memoryPlanFragments(state_->problem_->handshake(), old),
          memoryPlanFragments(state_->problem_->handshake(), plan)))
    return error;
  const auto overuse =
      state_->problem_->capacity().memoryOperationPlanOveruse();
  if (llvm::Error error = replaceContribution(overuse[old], overuse[plan],
                                              state_->atomicCapacityOveruse_,
                                              "memory capacity overuse"))
    return error;
  const auto planEnvelopes =
      state_->problem_->capacity().memoryOperationPlanEnvelopes();
  if (llvm::Error error = state_->replaceResourceTimeEnvelope(
          planEnvelopes[old], planEnvelopes[plan]))
    return error;
  state_->memoryOperationPlans_[actor] = plan;
  return llvm::Error::success();
}

llvm::Error SpatialMoveTransaction::setLogicalMemoryBinding(
    PnrIndex binding, PnrIndex target, std::uint64_t physicalOffsetBytes) {
  if (llvm::Error error = ensureCollecting())
    return error;
  if (binding >= state_->logicalMemoryBindings_.size() ||
      target >= state_->problem_->memory().bindingTargets().size())
    return candidateError("new logical memory binding is out of range");

  const SpatialLogicalMemoryBindingSelection replacement{target,
                                                         physicalOffsetBytes};
  const auto old = state_->logicalMemoryBindings_[binding];
  if (old.target == replacement.target &&
      old.physicalOffsetBytes == replacement.physicalOffsetBytes)
    return llvm::Error::success();

  state_->logicalMemoryBindings_[binding] = replacement;
  if (llvm::Error error = state_->validateLogicalMemoryBinding(binding)) {
    state_->logicalMemoryBindings_[binding] = old;
    return error;
  }
  state_->logicalMemoryBindings_[binding] = old;

  recordLogicalMemory(binding);
  markLogicalMemory(binding);
  state_->logicalMemoryBindings_[binding] = replacement;
  return llvm::Error::success();
}

llvm::Error
SpatialMoveTransaction::setMemoryUseDispatch(PnrIndex use,
                                             PnrIndex dispatchOption) {
  if (llvm::Error error = ensureCollecting())
    return error;
  if (use >= state_->memoryUseDispatches_.size() ||
      dispatchOption >= state_->problem_->memory().dispatchOptions().size())
    return candidateError("new memory dispatch is out of range");
  auto domain = state_->memoryDispatchDomain(use);
  if (!domain)
    return domain.takeError();
  if (!rangeContains((*domain)->optionOffset, (*domain)->optionCount,
                     dispatchOption))
    return candidateError("new memory dispatch is outside its domain");

  const PnrIndex old = state_->memoryUseDispatches_[use];
  if (old == dispatchOption)
    return llvm::Error::success();
  recordMemoryDispatch(use);
  markMemoryDispatch(use);
  if (llvm::Error error =
          state_->changeMemoryServiceUsage(use, old, dispatchOption))
    return error;
  state_->memoryUseDispatches_[use] = dispatchOption;
  return llvm::Error::success();
}

llvm::Error
SpatialMoveTransaction::setMemoryExposureSelection(PnrIndex exposure,
                                                   PnrIndex exposureOption) {
  if (llvm::Error error = ensureCollecting())
    return error;
  if (exposure >= state_->memoryExposureSelections_.size() ||
      exposureOption >= state_->problem_->memory().exposureOptions().size())
    return candidateError("new memory exposure selection is out of range");
  const PnrIndex old = state_->memoryExposureSelections_[exposure];
  if (old == exposureOption)
    return llvm::Error::success();

  recordMemoryExposure(exposure);
  markMemoryExposure(exposure);
  state_->changeMemoryExposureUsage(exposure, old, exposureOption);
  state_->memoryExposureSelections_[exposure] = exposureOption;
  return llvm::Error::success();
}

llvm::Expected<RouteTreeTransaction *>
SpatialMoveTransaction::routeTransaction(PnrIndex logicalNet) {
  if (llvm::Error error = ensureCollecting())
    return std::move(error);
  if (logicalNet >= state_->routeTrees_.size())
    return candidateError("logical net is out of range");
  if (!scratch_->routeTransactions_[logicalNet]) {
    auto transaction = state_->routeTrees_[logicalNet]->beginTransaction(
        *scratch_->routeScratch_[logicalNet]);
    if (!transaction)
      return transaction.takeError();
    scratch_->routeTransactions_[logicalNet].emplace(std::move(*transaction));
    scratch_->touchedRoutes_.push_back(logicalNet);
    markNet(logicalNet);
  }
  return &*scratch_->routeTransactions_[logicalNet];
}

llvm::Error SpatialMoveTransaction::bindRouteSource(PnrIndex logicalNet,
                                                    PnrIndex endpoint) {
  if (logicalNet >= state_->routeTrees_.size())
    return candidateError("logical net is out of range");
  if (endpoint != state_->logicalNetSourceEndpoint(logicalNet))
    return candidateError(
        "route source does not match the selected logical attachment");
  auto transaction = routeTransaction(logicalNet);
  if (!transaction)
    return transaction.takeError();
  return (*transaction)->bindSource(endpoint);
}

llvm::Error SpatialMoveTransaction::bindRouteSink(PnrIndex logicalNet,
                                                  PnrIndex sinkObligation,
                                                  PnrIndex endpoint) {
  if (logicalNet >= state_->routeTrees_.size())
    return candidateError("logical net is out of range");
  const auto &net = state_->problem_->transfers().logicalNets()[logicalNet];
  if (sinkObligation >= net.sinkCount)
    return candidateError("route sink obligation is out of range");
  if (endpoint != state_->logicalNetSinkEndpoint(logicalNet, sinkObligation))
    return candidateError(
        "route sink does not match the selected logical attachment");
  auto transaction = routeTransaction(logicalNet);
  if (!transaction)
    return transaction.takeError();
  return (*transaction)->bindSink(sinkObligation, endpoint);
}

llvm::Error SpatialMoveTransaction::attachRoutePath(
    PnrIndex logicalNet, PnrIndex attachmentEndpoint,
    llvm::ArrayRef<PnrIndex> forwardArcs, PnrIndex sinkObligation) {
  if (logicalNet >= state_->routeTrees_.size())
    return candidateError("logical net is out of range");
  const auto &net = state_->problem_->transfers().logicalNets()[logicalNet];
  if (sinkObligation >= net.sinkCount)
    return candidateError("route sink obligation is out of range");
  const std::uint32_t payloadWidth = state_->logicalNetPayloadWidth(logicalNet);
  for (PnrIndex arc : forwardArcs) {
    if (arc >= state_->problem_->routing().routingArcs().size())
      return candidateError("route path contains an out-of-range arc");
    if (state_->problem_->routing().routingArcs()[arc].payloadCapacityBits <
        payloadWidth)
      return candidateError("route path cannot carry its payload width");
  }
  auto transaction = routeTransaction(logicalNet);
  if (!transaction)
    return transaction.takeError();
  return (*transaction)
      ->attachPath(attachmentEndpoint, forwardArcs, sinkObligation);
}

llvm::Error SpatialMoveTransaction::ripUpRouteSink(PnrIndex logicalNet,
                                                   PnrIndex sinkObligation) {
  auto transaction = routeTransaction(logicalNet);
  if (!transaction)
    return transaction.takeError();
  return (*transaction)->ripUpSink(sinkObligation);
}

llvm::Error
SpatialMoveTransaction::ripUpRouteSubtree(PnrIndex logicalNet,
                                          PnrIndex subtreeRootEndpoint) {
  auto transaction = routeTransaction(logicalNet);
  if (!transaction)
    return transaction.takeError();
  return (*transaction)->ripUpSubtree(subtreeRootEndpoint);
}

llvm::Error SpatialMoveTransaction::ripUpWholeRoute(PnrIndex logicalNet) {
  auto transaction = routeTransaction(logicalNet);
  if (!transaction)
    return transaction.takeError();
  return (*transaction)->ripUpWholeNet();
}

llvm::Error SpatialMoveTransaction::validateAffectedState() const {
  for (PnrIndex realization : scratch_->affectedComputes_)
    if (llvm::Error error = state_->validateComputeBinding(realization))
      return error;
  for (PnrIndex realization : scratch_->affectedMemories_)
    if (llvm::Error error = state_->validateMemoryBinding(realization))
      return error;
  for (PnrIndex demand : scratch_->affectedPorts_)
    if (llvm::Error error = state_->validatePortAttachment(demand))
      return error;
  for (PnrIndex boundary : scratch_->affectedBoundaries_)
    if (llvm::Error error = state_->validateGraphBoundaryAttachment(boundary))
      return error;
  for (PnrIndex actor : scratch_->affectedMemoryPlans_)
    if (llvm::Error error = state_->validateMemoryOperationPlan(actor))
      return error;
  for (PnrIndex binding : scratch_->affectedLogicalMemories_) {
    if (llvm::Error error = state_->validateLogicalMemoryBinding(binding))
      return error;
    if (llvm::Error error =
            state_->validateLogicalMemoryBindingOverlap(binding))
      return error;
  }
  for (PnrIndex use : scratch_->affectedMemoryDispatches_)
    if (llvm::Error error = state_->validateMemoryUseDispatch(use))
      return error;
  for (PnrIndex group : scratch_->affectedMemoryServiceGroups_)
    if (state_->memoryServiceGroupActivePatternCounts_[group] > 1)
      return candidateError(
          "one memory service-use group selects multiple UsePatterns");
  for (PnrIndex exposure : scratch_->affectedMemoryExposures_)
    if (llvm::Error error = state_->validateMemoryExposureSelection(exposure))
      return error;
  for (PnrIndex relation : scratch_->affectedBindingRelations_)
    if (llvm::Error error = state_->verifyBindingRelation(relation))
      return error;
  if (!scratch_->affectedLogicalMemories_.empty())
    if (llvm::Error error = state_->problem_->memoryConstraints().verify(
            state_->logicalMemoryBindings_,
            *scratch_->memoryConstraintScratch_))
      return error;

  for (PnrIndex logicalNet : scratch_->affectedNets_) {
    const auto &net = state_->problem_->transfers().logicalNets()[logicalNet];
    const std::uint32_t payloadWidth =
        state_->logicalNetPayloadWidth(logicalNet);
    for (PnrIndex sink = 0; sink < net.sinkCount; ++sink) {
      const auto binding = state_->problem_->transfers()
                               .logicalNetSinkBindings()[net.sinkOffset + sink];
      if (state_->terminalPayloadWidth(binding) != payloadWidth)
        return candidateError("logical-net terminal widths disagree");
    }

    const RouteTreeState &route = *state_->routeTrees_[logicalNet];
    if (route.isUnrouted())
      continue;
    if (route.sourceEndpoint() != state_->logicalNetSourceEndpoint(logicalNet))
      return candidateError(
          "route source disagrees with its selected attachment");
    for (PnrIndex sink = 0; sink < net.sinkCount; ++sink)
      if (route.sinkEndpoint(sink) !=
          state_->logicalNetSinkEndpoint(logicalNet, sink))
        return candidateError(
            "route sink disagrees with its selected attachment");
  }
  if (llvm::Error error = scratch_->routeConstraintScratch_->verifyAffected(
          *state_, scratch_->affectedNets_))
    return error;
  return llvm::Error::success();
}

llvm::Expected<bool> SpatialMoveTransaction::close() {
  if (!scratch_)
    return candidateError("move is no longer active");
  if (closed_)
    return !cycle_;
  if (llvm::Error error = collectRouteTraversalDeltas())
    return std::move(error);
  if (llvm::Error error = state_->tagAssignments_.stageRouteUpdates(
          state_->routeTrees_, scratch_->routeTransactions_,
          scratch_->touchedRoutes_, scratch_->tagScratch_))
    return std::move(error);
  tagDeltasCollected_ = true;
  if (llvm::Error error = validateAffectedState())
    return std::move(error);
  auto closure = scratch_->handshakeTransaction_->close();
  if (!closure)
    return closure.takeError();
  closed_ = true;
  cycle_ = !*closure;
  if (cycle_)
    emitHandshakeCycle(state_->problem(), state_->handshake(), cycleWitness());
  return !cycle_;
}

llvm::ArrayRef<PnrIndex> SpatialMoveTransaction::cycleWitness() const {
  if (!scratch_ || !scratch_->handshakeTransaction_)
    return {};
  return scratch_->handshakeTransaction_->cycleWitness();
}

llvm::ArrayRef<PnrIndex>
SpatialMoveTransaction::touchedRouteTraversals() const {
  if (!scratch_ || !closed_)
    return {};
  return scratch_->touchedTraversals_;
}

bool SpatialMoveTransaction::hasSemanticChange() const {
  assert(scratch_ && closed_ && "move semantic comparison requires close");
  for (const SpatialCandidateScratch::DecisionDelta &delta :
       scratch_->decisionDeltas_) {
    switch (delta.kind) {
    case SpatialCandidateScratch::DecisionKind::ComputeBinding: {
      const auto current = state_->computeBindings_[delta.index];
      if (current.placement != delta.oldValue0 ||
          current.instructionContext != delta.oldValue1)
        return true;
      break;
    }
    case SpatialCandidateScratch::DecisionKind::MemoryBinding:
      if (state_->memoryBindings_[delta.index].placement != delta.oldValue0)
        return true;
      break;
    case SpatialCandidateScratch::DecisionKind::PortAttachment:
      if (state_->portAttachments_[delta.index] != delta.oldValue0)
        return true;
      break;
    case SpatialCandidateScratch::DecisionKind::GraphBoundaryAttachment:
      if (state_->graphBoundaryAttachments_[delta.index] != delta.oldValue0)
        return true;
      break;
    case SpatialCandidateScratch::DecisionKind::MemoryOperationPlan:
      if (state_->memoryOperationPlans_[delta.index] != delta.oldValue0)
        return true;
      break;
    case SpatialCandidateScratch::DecisionKind::LogicalMemoryBinding: {
      const auto current = state_->logicalMemoryBindings_[delta.index];
      if (current.target != delta.oldValue0 ||
          current.physicalOffsetBytes != delta.oldWideValue)
        return true;
      break;
    }
    case SpatialCandidateScratch::DecisionKind::MemoryUseDispatch:
      if (state_->memoryUseDispatches_[delta.index] != delta.oldValue0)
        return true;
      break;
    case SpatialCandidateScratch::DecisionKind::MemoryExposure:
      if (state_->memoryExposureSelections_[delta.index] != delta.oldValue0)
        return true;
      break;
    }
  }
  for (PnrIndex logicalNet : scratch_->touchedRoutes_)
    if (scratch_->routeTransactions_[logicalNet]->hasSemanticChange())
      return true;
  return false;
}

llvm::Error SpatialMoveTransaction::commit() {
  if (!scratch_)
    return candidateError("move is no longer active");
  auto closure = close();
  if (!closure)
    return closure.takeError();
  if (!*closure)
    return candidateError("cannot commit a selected handshake cycle");

  for (PnrIndex logicalNet : scratch_->touchedRoutes_)
    if (llvm::Error error = scratch_->routeTransactions_[logicalNet]->commit())
      return candidateError("prepared RouteTree commit failed: " +
                            llvm::toString(std::move(error)));
  if (llvm::Error error = scratch_->handshakeTransaction_->commit())
    return candidateError("closed handshake commit failed: " +
                          llvm::toString(std::move(error)));
  if (tagDeltasCollected_)
    state_->tagAssignments_.commit(scratch_->tagScratch_);
  acceptAppliedRouteResources();
  finish();
  return llvm::Error::success();
}

void SpatialMoveTransaction::rollback() noexcept {
  if (!scratch_)
    return;
  if (tagDeltasCollected_)
    state_->tagAssignments_.rollback(scratch_->tagScratch_);
  rollbackAppliedRouteResources();
  for (PnrIndex logicalNet : llvm::reverse(scratch_->touchedRoutes_))
    if (scratch_->routeTransactions_[logicalNet])
      scratch_->routeTransactions_[logicalNet]->rollback();
  if (scratch_->handshakeTransaction_)
    scratch_->handshakeTransaction_->rollback();

  for (const SpatialCandidateScratch::DecisionDelta &delta :
       llvm::reverse(scratch_->decisionDeltas_)) {
    switch (delta.kind) {
    case SpatialCandidateScratch::DecisionKind::ComputeBinding: {
      const auto current = state_->computeBindings_[delta.index];
      const auto offsets = state_->problem_->capacity()
                               .computeInstructionContextEnvelopeOffsets();
      if (llvm::Error error = state_->replaceResourceTimeEnvelopeSlice(
              offsets[current.instructionContext],
              offsets[current.instructionContext + 1] -
                  offsets[current.instructionContext],
              offsets[delta.oldValue1],
              offsets[delta.oldValue1 + 1] - offsets[delta.oldValue1])) {
        assert(false && "validated compute resource-time rollback failed");
        llvm::consumeError(std::move(error));
      }
      state_->computeBindings_[delta.index] = {delta.oldValue0,
                                               delta.oldValue1};
      state_->bindingRelationChoices_[delta.index] = delta.oldValue2;
      break;
    }
    case SpatialCandidateScratch::DecisionKind::MemoryBinding:
      state_->memoryBindings_[delta.index].placement = delta.oldValue0;
      state_->bindingRelationChoices_[state_->problem_->bindingRelations()
                                          .computeDecisionCount() +
                                      delta.index] = delta.oldValue2;
      break;
    case SpatialCandidateScratch::DecisionKind::PortAttachment:
      state_->portAttachments_[delta.index] = delta.oldValue0;
      state_->bindingRelationChoices_[state_->problem_->bindingRelations()
                                          .portDecisionOffset() +
                                      delta.index] = delta.oldValue2;
      break;
    case SpatialCandidateScratch::DecisionKind::GraphBoundaryAttachment:
      state_->graphBoundaryAttachments_[delta.index] = delta.oldValue0;
      state_->bindingRelationChoices_[state_->problem_->bindingRelations()
                                          .graphBoundaryDecisionOffset() +
                                      delta.index] = delta.oldValue2;
      break;
    case SpatialCandidateScratch::DecisionKind::MemoryOperationPlan: {
      const PnrIndex current = state_->memoryOperationPlans_[delta.index];
      const auto envelopes =
          state_->problem_->capacity().memoryOperationPlanEnvelopes();
      if (llvm::Error error = state_->replaceResourceTimeEnvelope(
              envelopes[current], envelopes[delta.oldValue0])) {
        assert(false && "validated memory resource-time rollback failed");
        llvm::consumeError(std::move(error));
      }
      state_->memoryOperationPlans_[delta.index] = delta.oldValue0;
      break;
    }
    case SpatialCandidateScratch::DecisionKind::LogicalMemoryBinding:
      state_->logicalMemoryBindings_[delta.index] = {delta.oldValue0,
                                                     delta.oldWideValue};
      break;
    case SpatialCandidateScratch::DecisionKind::MemoryUseDispatch: {
      const PnrIndex current = state_->memoryUseDispatches_[delta.index];
      if (llvm::Error error = state_->changeMemoryServiceUsage(
              delta.index, current, delta.oldValue0)) {
        assert(false && "validated memory service-use rollback failed");
        llvm::consumeError(std::move(error));
      }
      state_->memoryUseDispatches_[delta.index] = delta.oldValue0;
      break;
    }
    case SpatialCandidateScratch::DecisionKind::MemoryExposure: {
      const PnrIndex current = state_->memoryExposureSelections_[delta.index];
      state_->changeMemoryExposureUsage(delta.index, current, delta.oldValue0);
      state_->memoryExposureSelections_[delta.index] = delta.oldValue0;
      break;
    }
    }
  }
  state_->atomicCapacityOveruse_ = initialAtomicCapacityOveruse_;
  finish();
}

void SpatialMoveTransaction::finish() {
  state_->activeTransaction_ = nullptr;
  scratch_->activeTransaction_ = nullptr;
  scratch_->resetTransaction();
  scratch_ = nullptr;
  state_.reset();
}
