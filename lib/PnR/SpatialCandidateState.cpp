#include "PnR/SpatialCandidateState.h"

#include "SpatialBindingRelationModel.h"
#include "SpatialCandidateStateInternal.h"
#include "SpatialMemoryConstraintModel.h"
#include "SpatialOperandPairingPressure.h"
#include "SpatialPhysicalTiming.h"
#include "SpatialProgressAnalysis.h"
#include "SpatialRecurrenceTimingInternal.h"
#include "SpatialRouteConstraintModel.h"
#include "SpatialSwitchHandshakeProjection.h"
#include "StaticSchedulePressure.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace loom::pnr;

namespace {

using detail::attachmentTraversal;
using detail::candidateError;
using detail::computePlacementFragments;
using detail::memoryPlanFragments;
using detail::rangeContains;

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

llvm::Error increment(PnrIndex &value, PnrIndex amount,
                      llvm::StringRef subject) {
  if (amount > std::numeric_limits<PnrIndex>::max() - value)
    return candidateError(subject + " count overflows PnrIndex");
  value += amount;
  return llvm::Error::success();
}

void advanceEpoch(std::uint64_t &epoch,
                  llvm::ArrayRef<std::vector<std::uint64_t> *> marks) {
  if (++epoch != 0)
    return;
  for (std::vector<std::uint64_t> *values : marks)
    std::fill(values->begin(), values->end(), 0);
  epoch = 1;
}

std::vector<llvm::ArrayRef<std::optional<llvm::APInt>>>
tagValueViews(const SpatialTagAssignmentState &assignments,
              std::size_t logicalNetCount) {
  std::vector<llvm::ArrayRef<std::optional<llvm::APInt>>> result;
  result.reserve(logicalNetCount);
  for (PnrIndex logicalNet = 0; logicalNet != logicalNetCount; ++logicalNet)
    result.push_back(assignments.values(logicalNet));
  return result;
}

llvm::Expected<HandshakeCandidateStateHandle> createInitialHandshakeState(
    const FrozenSpatialPnrProblemHandle &problem,
    llvm::ArrayRef<SpatialComputeBindingSelection> computeBindings,
    llvm::ArrayRef<PnrIndex> portAttachments,
    llvm::ArrayRef<PnrIndex> memoryOperationPlans,
    llvm::ArrayRef<PnrIndex> registerFifoTransfers) {
  std::vector<PnrIndex> selectedFragments;
  for (const SpatialComputeBindingSelection &binding : computeBindings)
    llvm::append_range(
        selectedFragments,
        computePlacementFragments(problem->handshake(), binding.placement));
  for (PnrIndex plan : memoryOperationPlans)
    llvm::append_range(selectedFragments,
                       memoryPlanFragments(problem->handshake(), plan));

  std::vector<PnrIndex> traversalUses(problem->routing().traversals().size(),
                                      0);
  const auto addTraversalUse = [&](PnrIndex traversal) -> llvm::Error {
    if (traversal >= traversalUses.size())
      return candidateError("initial handshake traversal is out of range");
    return increment(traversalUses[traversal], 1, "handshake traversal");
  };
  for (PnrIndex option : portAttachments)
    if (std::optional<PnrIndex> traversal =
            attachmentTraversal(problem->ports(), option))
      if (llvm::Error error = addTraversalUse(*traversal))
        return std::move(error);
  for (PnrIndex option : registerFifoTransfers) {
    if (option == getInvalidPnrIndex())
      continue;
    if (option >= problem->localTransfers().options().size())
      return candidateError("initial register-FIFO transfer is out of range");
    const auto &transfer = problem->localTransfers().options()[option];
    if (llvm::Error error = addTraversalUse(transfer.writeTraversal))
      return std::move(error);
    if (llvm::Error error = addTraversalUse(transfer.readTraversal))
      return std::move(error);
  }
  auto handshakeOwner = std::shared_ptr<const FrozenSpatialHandshakeIndex>(
      problem, &problem->handshake());
  return HandshakeCandidateState::create(std::move(handshakeOwner),
                                         selectedFragments, traversalUses);
}

llvm::Expected<bool> projectHandshakeSelections(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<SpatialComputeBindingSelection> computeBindings,
    llvm::ArrayRef<PnrIndex> portAttachments,
    llvm::ArrayRef<PnrIndex> memoryOperationPlans,
    llvm::ArrayRef<PnrIndex> registerFifoTransfers,
    llvm::ArrayRef<const RouteTreeState *> routes,
    llvm::ArrayRef<llvm::ArrayRef<std::optional<llvm::APInt>>> tagValues) {
  std::vector<PnrIndex> selectedFragments;
  for (const SpatialComputeBindingSelection &binding : computeBindings)
    llvm::append_range(
        selectedFragments,
        computePlacementFragments(problem.handshake(), binding.placement));
  for (PnrIndex plan : memoryOperationPlans)
    llvm::append_range(selectedFragments,
                       memoryPlanFragments(problem.handshake(), plan));

  std::vector<PnrIndex> traversalUses(problem.routing().traversals().size(), 0);
  const auto addTraversalUse = [&](PnrIndex traversal) -> llvm::Error {
    if (traversal >= traversalUses.size())
      return candidateError("projected handshake traversal is out of range");
    return increment(traversalUses[traversal], 1, "handshake traversal");
  };
  for (PnrIndex option : portAttachments)
    if (std::optional<PnrIndex> traversal =
            attachmentTraversal(problem.ports(), option))
      if (llvm::Error error = addTraversalUse(*traversal))
        return std::move(error);
  for (PnrIndex option : registerFifoTransfers) {
    if (option == getInvalidPnrIndex())
      continue;
    if (option >= problem.localTransfers().options().size())
      return candidateError("projected register-FIFO transfer is out of range");
    const auto &transfer = problem.localTransfers().options()[option];
    if (llvm::Error error = addTraversalUse(transfer.writeTraversal))
      return std::move(error);
    if (llvm::Error error = addTraversalUse(transfer.readTraversal))
      return std::move(error);
  }

  for (const RouteTreeState *route : routes) {
    if (!route || &route->routingGraph() != &problem.routing())
      return candidateError(
          "projected RouteTree does not belong to the frozen routing graph");
    for (const RouteTreeNode &node : route->nodeStorage()) {
      if (!node.isActive() || node.parentArc == getInvalidPnrIndex())
        continue;
      if (node.parentArc >= problem.routing().routingArcs().size())
        return candidateError("projected RouteTree arc is out of range");
      const PnrIndex traversal =
          problem.routing().routingArcs()[node.parentArc].traversal;
      if (traversal >= traversalUses.size())
        return candidateError("projected RouteTree traversal is out of range");
      if (llvm::Error error =
              increment(traversalUses[traversal], 1, "handshake traversal"))
        return std::move(error);
    }
  }
  auto switchFragments = detail::deriveSpatialTemporalSwitchHandshakeFragments(
      problem, routes, tagValues);
  if (!switchFragments)
    return switchFragments.takeError();
  llvm::append_range(selectedFragments, *switchFragments);
  return independentlyVerifyHandshakeProjectionAcyclic(
      problem.handshake(), selectedFragments, traversalUses);
}

} // namespace

SpatialCandidateScratch::~SpatialCandidateScratch() {
  if (activeTransaction_)
    activeTransaction_->rollback();
}

SpatialCandidateScratch::SpatialCandidateScratch() = default;

llvm::Error
SpatialCandidateScratch::prepare(const FrozenSpatialPnrProblem &problem) {
  if (activeTransaction_)
    return candidateError("cannot prepare scratch during a move");
  preparedProblem_ = nullptr;

  const std::size_t netCount = problem.transfers().logicalNets().size();
  routeTransactions_.clear();
  routeTransactions_.reserve(netCount);
  routeScratch_.clear();
  routeScratch_.reserve(netCount);
  for (std::size_t net = 0; net < netCount; ++net) {
    routeTransactions_.emplace_back(std::nullopt);
    routeScratch_.push_back(std::make_unique<RouteTreeTransactionScratch>());
  }
  if (llvm::Error error = tagScratch_.prepare(problem))
    return error;
  if (llvm::Error error = handshakeScratch_.prepare(problem.handshake()))
    return error;
  if (!routeConstraintScratch_)
    routeConstraintScratch_ =
        std::make_unique<detail::SpatialRouteConstraintScratch>();
  if (llvm::Error error = routeConstraintScratch_->prepare(problem))
    return error;
  if (!memoryConstraintScratch_)
    memoryConstraintScratch_ =
        std::make_unique<detail::SpatialMemoryConstraintScratch>();
  if (llvm::Error error =
          problem.memoryConstraints().prepareScratch(*memoryConstraintScratch_))
    return error;

  const auto &realizations = problem.realizations();
  const auto &ports = problem.ports();
  const std::size_t computeCount = realizations.computeRealizations().size();
  const std::size_t memoryCount = realizations.memoryRealizations().size();
  const std::size_t portCount = ports.portDemands().size();
  const std::size_t boundaryCount = ports.graphBoundaries().size();
  const std::size_t memoryPlanCount = realizations.memoryActors().size();
  const std::size_t logicalMemoryCount =
      problem.memory().logicalBindings().size();
  const std::size_t memoryDispatchCount = problem.memory().rootedUses().size();
  const std::size_t memoryServiceGroupCount =
      problem.memory().serviceUseGroups().size();
  const std::size_t memoryExposureCount = problem.memory().exposures().size();
  const std::size_t traversalCount = problem.routing().traversals().size();
  const std::size_t bindingRelationCount =
      problem.bindingRelations().relations().relations().size();

  computeJournalMarks_.assign(computeCount, 0);
  memoryJournalMarks_.assign(memoryCount, 0);
  portJournalMarks_.assign(portCount, 0);
  boundaryJournalMarks_.assign(boundaryCount, 0);
  memoryPlanJournalMarks_.assign(memoryPlanCount, 0);
  logicalMemoryJournalMarks_.assign(logicalMemoryCount, 0);
  memoryDispatchJournalMarks_.assign(memoryDispatchCount, 0);
  memoryExposureJournalMarks_.assign(memoryExposureCount, 0);
  registerFifoTransferJournalMarks_.assign(netCount, 0);
  decisionDeltas_.clear();
  decisionDeltas_.reserve(computeCount + memoryCount + portCount +
                          boundaryCount + memoryPlanCount + logicalMemoryCount +
                          memoryDispatchCount + memoryExposureCount + netCount);

  affectedComputeMarks_.assign(computeCount, 0);
  affectedMemoryMarks_.assign(memoryCount, 0);
  affectedPortMarks_.assign(portCount, 0);
  affectedBoundaryMarks_.assign(boundaryCount, 0);
  affectedMemoryPlanMarks_.assign(memoryPlanCount, 0);
  affectedLogicalMemoryMarks_.assign(logicalMemoryCount, 0);
  affectedMemoryDispatchMarks_.assign(memoryDispatchCount, 0);
  affectedMemoryServiceGroupMarks_.assign(memoryServiceGroupCount, 0);
  affectedMemoryExposureMarks_.assign(memoryExposureCount, 0);
  affectedNetMarks_.assign(netCount, 0);
  affectedBindingRelationMarks_.assign(bindingRelationCount, 0);
  affectedComputes_.reserve(computeCount);
  affectedMemories_.reserve(memoryCount);
  affectedPorts_.reserve(portCount);
  affectedBoundaries_.reserve(boundaryCount);
  affectedMemoryPlans_.reserve(memoryPlanCount);
  affectedLogicalMemories_.reserve(logicalMemoryCount);
  affectedMemoryDispatches_.reserve(memoryDispatchCount);
  affectedMemoryServiceGroups_.reserve(memoryServiceGroupCount);
  affectedMemoryExposures_.reserve(memoryExposureCount);
  affectedNets_.reserve(netCount);
  affectedBindingRelations_.reserve(bindingRelationCount);

  touchedRoutes_.reserve(netCount);
  routeViews_.reserve(netCount);
  tagValueViews_.reserve(netCount);
  physicalTimingChangedNets_.reserve(netCount);
  physicalTimingOldWorstArrivals_.reserve(netCount);
  physicalTimingOldNegativeSlacks_.reserve(netCount);
  physicalTimingRouteNodeArrivals_.reserve(
      problem.routing().routingEndpoints().size());
  physicalTimingRouteNodeWorklist_.reserve(
      problem.routing().routingEndpoints().size());
  traversalDeltaMarks_.assign(traversalCount, 0);
  traversalRemoved_.assign(traversalCount, 0);
  traversalAdded_.assign(traversalCount, 0);
  touchedTraversals_.reserve(traversalCount);
  progressRecordedRouteDeltaCounts_.assign(netCount, 0);
  progressRecordedRouteDeltaEpochs_.assign(netCount, 0);
  progressTerminalActive_.assign(netCount, 0);
  progressTraversalDeltas_.clear();
  progressTraversalDeltas_.reserve(traversalCount);
  progressDirtyNetMarks_.assign(netCount, 0);
  progressDirtyNets_.clear();
  progressDirtyNets_.reserve(netCount);
  progressDependencyJournalMarks_.assign(netCount, 0);
  progressDependencyDeltas_.clear();
  progressDependencyDeltas_.reserve(netCount);

  decisionEpoch_ = 0;
  affectedEpoch_ = 0;
  traversalEpoch_ = 0;
  progressRecordedRouteDeltaEpoch_ = 0;
  preparedProblem_ = &problem;
  resetTransaction();
  return llvm::Error::success();
}

std::size_t SpatialCandidateScratch::retainedStorageBytes() const {
  std::size_t bytes =
      retainedBytes(routeScratch_) + retainedBytes(routeTransactions_) +
      tagScratch_.retainedStorageBytes() +
      handshakeScratch_.retainedStorageBytes() +
      (routeConstraintScratch_ ? routeConstraintScratch_->retainedStorageBytes()
                               : 0) +
      (memoryConstraintScratch_
           ? memoryConstraintScratch_->retainedStorageBytes()
           : 0);
  for (const auto &scratch : routeScratch_)
    bytes += scratch->retainedRollbackStorageBytes();
  bytes +=
      retainedBytes(computeJournalMarks_) + retainedBytes(memoryJournalMarks_) +
      retainedBytes(portJournalMarks_) + retainedBytes(boundaryJournalMarks_) +
      retainedBytes(memoryPlanJournalMarks_) +
      retainedBytes(logicalMemoryJournalMarks_) +
      retainedBytes(memoryDispatchJournalMarks_) +
      retainedBytes(memoryExposureJournalMarks_) +
      retainedBytes(registerFifoTransferJournalMarks_) +
      retainedBytes(decisionDeltas_) + retainedBytes(affectedComputeMarks_) +
      retainedBytes(affectedMemoryMarks_) + retainedBytes(affectedPortMarks_) +
      retainedBytes(affectedBoundaryMarks_) +
      retainedBytes(affectedMemoryPlanMarks_) +
      retainedBytes(affectedLogicalMemoryMarks_) +
      retainedBytes(affectedMemoryDispatchMarks_) +
      retainedBytes(affectedMemoryServiceGroupMarks_) +
      retainedBytes(affectedMemoryExposureMarks_) +
      retainedBytes(affectedNetMarks_) +
      retainedBytes(affectedBindingRelationMarks_) +
      retainedBytes(affectedComputes_) + retainedBytes(affectedMemories_) +
      retainedBytes(affectedPorts_) + retainedBytes(affectedBoundaries_) +
      retainedBytes(affectedMemoryPlans_) + retainedBytes(affectedNets_) +
      retainedBytes(affectedLogicalMemories_) +
      retainedBytes(affectedMemoryDispatches_) +
      retainedBytes(affectedMemoryServiceGroups_) +
      retainedBytes(affectedMemoryExposures_) +
      retainedBytes(affectedBindingRelations_) + retainedBytes(touchedRoutes_) +
      retainedBytes(routeViews_) + retainedBytes(tagValueViews_) +
      retainedBytes(physicalTimingChangedNets_) +
      retainedBytes(physicalTimingOldWorstArrivals_) +
      retainedBytes(physicalTimingOldNegativeSlacks_) +
      retainedBytes(physicalTimingRouteNodeArrivals_) +
      retainedBytes(physicalTimingRouteNodeWorklist_) +
      retainedBytes(oldSwitchHandshakeFragments_) +
      retainedBytes(newSwitchHandshakeFragments_) +
      retainedBytes(removedSwitchHandshakeFragments_) +
      retainedBytes(addedSwitchHandshakeFragments_) +
      retainedBytes(traversalDeltaMarks_) + retainedBytes(traversalRemoved_) +
      retainedBytes(traversalAdded_) + retainedBytes(touchedTraversals_) +
      retainedBytes(progressRecordedRouteDeltaCounts_) +
      retainedBytes(progressRecordedRouteDeltaEpochs_) +
      retainedBytes(progressTerminalActive_) +
      retainedBytes(progressTraversalDeltas_) +
      retainedBytes(progressDirtyNetMarks_) +
      retainedBytes(progressDirtyNets_) +
      retainedBytes(progressDependencyJournalMarks_) +
      retainedBytes(progressDependencyDeltas_);
  return bytes;
}

void SpatialCandidateScratch::beginTransaction() {
  resetTransaction();
  advanceEpoch(decisionEpoch_,
               {&computeJournalMarks_, &memoryJournalMarks_, &portJournalMarks_,
                &boundaryJournalMarks_, &memoryPlanJournalMarks_,
                &logicalMemoryJournalMarks_, &memoryDispatchJournalMarks_,
                &memoryExposureJournalMarks_,
                &registerFifoTransferJournalMarks_,
                &progressDependencyJournalMarks_});
  advanceEpoch(
      affectedEpoch_,
      {&affectedComputeMarks_, &affectedMemoryMarks_, &affectedPortMarks_,
       &affectedBoundaryMarks_, &affectedMemoryPlanMarks_,
       &affectedLogicalMemoryMarks_, &affectedMemoryDispatchMarks_,
       &affectedMemoryServiceGroupMarks_, &affectedMemoryExposureMarks_,
       &affectedNetMarks_, &affectedBindingRelationMarks_});
  advanceEpoch(traversalEpoch_, {&traversalDeltaMarks_});
  advanceProgressRouteDeltaEpoch();
}

void SpatialCandidateScratch::advanceProgressRouteDeltaEpoch() {
  if (++progressRecordedRouteDeltaEpoch_ == 0) {
    std::fill(progressRecordedRouteDeltaEpochs_.begin(),
              progressRecordedRouteDeltaEpochs_.end(), 0);
    progressRecordedRouteDeltaEpoch_ = 1;
  }
}

void SpatialCandidateScratch::resetTransaction() {
  for (PnrIndex net : touchedRoutes_) {
    routeTransactions_[net].reset();
  }
  touchedRoutes_.clear();
  routeViews_.clear();
  tagValueViews_.clear();
  physicalTimingChangedNets_.clear();
  physicalTimingOldWorstArrivals_.clear();
  physicalTimingOldNegativeSlacks_.clear();
  oldSwitchHandshakeFragments_.clear();
  newSwitchHandshakeFragments_.clear();
  removedSwitchHandshakeFragments_.clear();
  addedSwitchHandshakeFragments_.clear();
  switchHandshakeBaselineCaptured_ = false;
  touchedTraversals_.clear();
  for (PnrIndex logicalNet : progressDirtyNets_)
    progressDirtyNetMarks_[logicalNet] = 0;
  progressDirtyNets_.clear();
  progressTraversalDeltas_.clear();
  progressDependencyDeltas_.clear();
  decisionDeltas_.clear();
  affectedComputes_.clear();
  affectedMemories_.clear();
  affectedPorts_.clear();
  affectedBoundaries_.clear();
  affectedMemoryPlans_.clear();
  affectedLogicalMemories_.clear();
  affectedMemoryDispatches_.clear();
  affectedMemoryServiceGroups_.clear();
  affectedMemoryExposures_.clear();
  affectedNets_.clear();
  affectedBindingRelations_.clear();
  handshakeTransaction_.reset();
  resourceFullyAppliedRouteCount_ = 0;
  resourcePartiallyAppliedDeltaCount_ = 0;
}

llvm::Expected<SpatialCandidateStateHandle>
SpatialCandidateState::create(FrozenSpatialPnrProblemHandle problem,
                              SpatialCandidateInitialization initialization) {
  if (!problem)
    return candidateError("FrozenSpatialPnrProblem owner is null");
  const auto &realizations = problem->realizations();
  const auto &ports = problem->ports();
  if (initialization.computeBindings.size() !=
          realizations.computeRealizations().size() ||
      initialization.memoryBindings.size() !=
          realizations.memoryRealizations().size() ||
      initialization.portAttachments.size() != ports.portDemands().size() ||
      initialization.graphBoundaryAttachments.size() !=
          ports.graphBoundaries().size() ||
      initialization.memoryOperationPlans.size() !=
          realizations.memoryActors().size() ||
      initialization.logicalMemoryBindings.size() !=
          problem->memory().logicalBindings().size() ||
      initialization.memoryUseDispatches.size() !=
          problem->memory().rootedUses().size() ||
      initialization.memoryExposureSelections.size() !=
          problem->memory().exposures().size() ||
      (!initialization.registerFifoTransfers.empty() &&
       initialization.registerFifoTransfers.size() !=
           problem->transfers().logicalNets().size()))
    return candidateError(
        "initialization dimensions do not match the frozen problem");

  std::vector<SpatialComputeBindingSelection> computeBindings(
      initialization.computeBindings.begin(),
      initialization.computeBindings.end());
  std::vector<SpatialMemoryBindingSelection> memoryBindings(
      initialization.memoryBindings.begin(),
      initialization.memoryBindings.end());
  std::vector<PnrIndex> portAttachments(initialization.portAttachments.begin(),
                                        initialization.portAttachments.end());
  std::vector<PnrIndex> graphBoundaryAttachments(
      initialization.graphBoundaryAttachments.begin(),
      initialization.graphBoundaryAttachments.end());
  const detail::SpatialBindingRelationModel &bindingRelations =
      problem->bindingRelations();
  std::vector<PnrIndex> bindingRelationChoices;
  bindingRelationChoices.reserve(bindingRelations.decisionCount());
  for (auto [realization, binding] : llvm::enumerate(computeBindings)) {
    const auto choice = bindingRelations.computeChoiceOrdinal(
        static_cast<PnrIndex>(realization), binding.placement,
        binding.instructionContext);
    if (!choice)
      return candidateError(
          "compute binding has no frozen relation-domain choice");
    bindingRelationChoices.push_back(*choice);
  }
  for (auto [realization, binding] : llvm::enumerate(memoryBindings)) {
    const auto choice = bindingRelations.memoryChoiceOrdinal(
        static_cast<PnrIndex>(realization), binding.placement);
    if (!choice)
      return candidateError(
          "memory binding has no frozen relation-domain choice");
    bindingRelationChoices.push_back(*choice);
  }
  for (auto [demand, attachment] : llvm::enumerate(portAttachments)) {
    const auto choice = bindingRelations.portAttachmentChoiceOrdinal(
        static_cast<PnrIndex>(demand), attachment);
    if (!choice)
      return candidateError(
          "PortAttachment has no frozen relation-domain choice");
    bindingRelationChoices.push_back(*choice);
  }
  for (auto [boundary, attachment] :
       llvm::enumerate(graphBoundaryAttachments)) {
    const auto choice = bindingRelations.graphBoundaryAttachmentChoiceOrdinal(
        static_cast<PnrIndex>(boundary), attachment);
    if (!choice)
      return candidateError(
          "graph-boundary attachment has no frozen relation-domain choice");
    bindingRelationChoices.push_back(*choice);
  }
  std::vector<PnrIndex> memoryOperationPlans(
      initialization.memoryOperationPlans.begin(),
      initialization.memoryOperationPlans.end());
  std::vector<SpatialLogicalMemoryBindingSelection> logicalMemoryBindings(
      initialization.logicalMemoryBindings.begin(),
      initialization.logicalMemoryBindings.end());
  std::vector<PnrIndex> memoryUseDispatches(
      initialization.memoryUseDispatches.begin(),
      initialization.memoryUseDispatches.end());
  std::vector<PnrIndex> memoryExposureSelections(
      initialization.memoryExposureSelections.begin(),
      initialization.memoryExposureSelections.end());
  std::vector<PnrIndex> registerFifoTransfers(
      problem->transfers().logicalNets().size(), getInvalidPnrIndex());
  if (!initialization.registerFifoTransfers.empty())
    registerFifoTransfers.assign(initialization.registerFifoTransfers.begin(),
                                 initialization.registerFifoTransfers.end());

  auto handshake =
      createInitialHandshakeState(problem, computeBindings, portAttachments,
                                  memoryOperationPlans, registerFifoTransfers);
  if (!handshake)
    return handshake.takeError();

  auto routingOwner = std::shared_ptr<const FrozenSpatialRoutingGraph>(
      problem, &problem->routing());
  std::vector<RouteTreeStateHandle> routeTrees;
  routeTrees.reserve(problem->transfers().logicalNets().size());
  for (const FrozenSpatialLogicalNet &net :
       problem->transfers().logicalNets()) {
    auto routeTree = RouteTreeState::create(routingOwner, net.sinkCount);
    if (!routeTree)
      return routeTree.takeError();
    routeTrees.push_back(std::move(*routeTree));
  }
  auto routeResources = SpatialRouteResourceState::create(*problem);
  if (!routeResources)
    return routeResources.takeError();
  for (PnrIndex logicalNet = 0; logicalNet < registerFifoTransfers.size();
       ++logicalNet) {
    const PnrIndex selected = registerFifoTransfers[logicalNet];
    if (selected == getInvalidPnrIndex())
      continue;
    if (selected >= problem->localTransfers().options().size())
      return candidateError("initial register-FIFO transfer is out of range");
    const auto &option = problem->localTransfers().options()[selected];
    if (llvm::Error error = routeResources->applyTraversalDelta(
            logicalNet, option.writeTraversal, 0, 1))
      return std::move(error);
    if (llvm::Error error = routeResources->applyTraversalDelta(
            logicalNet, option.readTraversal, 0, 1))
      return std::move(error);
  }
  auto tagAssignments = SpatialTagAssignmentState::create(*problem, routeTrees);
  if (!tagAssignments)
    return tagAssignments.takeError();
  std::uint64_t unroutedObligationCount = 0;
  for (auto [logicalNet, net] :
       llvm::enumerate(problem->transfers().logicalNets())) {
    if (registerFifoTransfers[logicalNet] != getInvalidPnrIndex())
      continue;
    if (net.sinkCount >
        std::numeric_limits<std::uint64_t>::max() - unroutedObligationCount)
      return candidateError("unrouted obligation count overflows u64");
    unroutedObligationCount += net.sinkCount;
  }

  std::vector<const RouteTreeState *> routePointers;
  routePointers.reserve(routeTrees.size());
  for (const RouteTreeStateHandle &route : routeTrees)
    routePointers.push_back(route.get());
  std::vector<std::uint64_t> logicalNetWorstArrivalDelayQuanta;
  std::vector<std::uint64_t> logicalNetNegativeSlackQuanta;
  auto physicalTiming = detail::projectSpatialPhysicalTiming(
      *problem, routePointers, registerFifoTransfers, portAttachments,
      graphBoundaryAttachments, &logicalNetWorstArrivalDelayQuanta,
      &logicalNetNegativeSlackQuanta);
  if (!physicalTiming)
    return physicalTiming.takeError();

  auto operandIngressPressure = detail::measureSpatialOperandIngressPressure(
      *problem, portAttachments, registerFifoTransfers);
  if (!operandIngressPressure)
    return operandIngressPressure.takeError();

  auto candidate = SpatialCandidateStateHandle(new SpatialCandidateState(
      std::move(problem), std::move(computeBindings), std::move(memoryBindings),
      std::move(bindingRelationChoices), std::move(portAttachments),
      std::move(graphBoundaryAttachments), std::move(memoryOperationPlans),
      std::move(logicalMemoryBindings), std::move(memoryUseDispatches),
      std::move(memoryExposureSelections), std::move(registerFifoTransfers),
      std::move(routeTrees), std::move(*handshake), std::move(*routeResources),
      std::move(*tagAssignments), unroutedObligationCount, 0, 0,
      *operandIngressPressure,
      std::move(logicalNetWorstArrivalDelayQuanta),
      std::move(logicalNetNegativeSlackQuanta),
      physicalTiming->worstArrivalDelayQuanta,
      physicalTiming->totalNegativeSlackQuanta));
  for (PnrIndex index = 0; index < candidate->computeBindings_.size(); ++index)
    if (llvm::Error error = candidate->validateComputeBinding(index))
      return std::move(error);
  for (PnrIndex index = 0; index < candidate->memoryBindings_.size(); ++index)
    if (llvm::Error error = candidate->validateMemoryBinding(index))
      return std::move(error);
  for (PnrIndex index = 0; index < candidate->portAttachments_.size(); ++index)
    if (llvm::Error error = candidate->validatePortAttachment(index))
      return std::move(error);
  for (PnrIndex index = 0; index < candidate->graphBoundaryAttachments_.size();
       ++index)
    if (llvm::Error error = candidate->validateGraphBoundaryAttachment(index))
      return std::move(error);
  for (PnrIndex index = 0; index < candidate->memoryOperationPlans_.size();
       ++index)
    if (llvm::Error error = candidate->validateMemoryOperationPlan(index))
      return std::move(error);
  if (llvm::Error error = candidate->rebuildMemoryExposureUsage())
    return std::move(error);
  if (llvm::Error error = candidate->rebuildMemoryServiceUsage())
    return std::move(error);
  if (llvm::Error error = candidate->verifyMemorySelections())
    return std::move(error);
  if (llvm::Error error = candidate->rebuildResourceTimeEnvelopeSelections())
    return std::move(error);
  for (PnrIndex index = 0; index < candidate->routeTrees_.size(); ++index)
    if (llvm::Error error = candidate->validateLogicalNet(index))
      return std::move(error);
  if (llvm::Error error = candidate->verifyRegisterFifoTransfers())
    return std::move(error);
  auto capacityOveruse = candidate->recomputeAtomicCapacityOveruse();
  if (!capacityOveruse)
    return capacityOveruse.takeError();
  candidate->atomicCapacityOveruse_ = *capacityOveruse;
  auto schedulePressure = detail::measureStaticSchedulePressure(*candidate);
  if (!schedulePressure)
    return schedulePressure.takeError();
  candidate->staticSchedulePressure_ = *schedulePressure;
  auto recurrenceTiming =
      detail::projectSpatialRecurrenceTiming(*candidate, routePointers);
  if (!recurrenceTiming)
    return recurrenceTiming.takeError();
  candidate->recurrenceTiming_ = std::move(*recurrenceTiming);
  auto progressState = SpatialProgressState::create(*candidate);
  if (!progressState)
    return progressState.takeError();
  candidate->progressState_ = std::move(*progressState);
  if (llvm::Error error = candidate->verify())
    return std::move(error);
  return candidate;
}

llvm::Expected<SpatialCandidateStateHandle>
SpatialCandidateState::cloneFullyRouted() const {
  if (activeTransaction_)
    return candidateError("cannot snapshot an active candidate transaction");
  if (unroutedObligationCount_ != 0)
    return candidateError("cannot snapshot an incompletely routed candidate");

  const SpatialCandidateInitialization initialization{
      computeBindings_,       memoryBindings_,
      portAttachments_,       graphBoundaryAttachments_,
      memoryOperationPlans_,  logicalMemoryBindings_,
      memoryUseDispatches_,   memoryExposureSelections_,
      registerFifoTransfers_,
  };
  auto cloned = create(problem_, initialization);
  if (!cloned)
    return cloned.takeError();

  SpatialCandidateScratch scratch;
  if (llvm::Error error = scratch.prepare(*problem_))
    return std::move(error);
  auto move = (*cloned)->beginMove(scratch);
  if (!move)
    return move.takeError();

  const FrozenSpatialRoutingGraph &routing = problem_->routing();
  const auto arcs = routing.routingArcs();
  const auto arcSources = routing.arcSources();
  std::vector<PnrIndex> reversePath;
  for (PnrIndex logicalNet = 0; logicalNet != routeTrees_.size();
       ++logicalNet) {
    if (usesRegisterFifo(logicalNet))
      continue;
    const RouteTreeState &sourceTree = routeTree(logicalNet);
    const auto source = sourceTree.sourceEndpoint();
    if (!source)
      return candidateError("fully routed snapshot lacks a route source");
    if (llvm::Error error = move->bindRouteSource(logicalNet, *source))
      return std::move(error);

    const PnrIndex sinkCount =
        problem_->transfers().logicalNets()[logicalNet].sinkCount;
    for (PnrIndex sink = 0; sink != sinkCount; ++sink) {
      const auto endpoint = sourceTree.sinkEndpoint(sink);
      if (!endpoint)
        return candidateError("fully routed snapshot lacks a route sink");
      if (llvm::Error error = move->bindRouteSink(logicalNet, sink, *endpoint))
        return std::move(error);

      auto slot = sourceTree.findNode(*endpoint);
      if (!slot)
        return candidateError(
            "fully routed snapshot sink is absent from its RouteTree");
      reversePath.clear();
      for (std::size_t depth = 0;; ++depth) {
        if (depth > sourceTree.nodeStorage().size())
          return candidateError("snapshot RouteTree contains a cycle");
        const RouteTreeNode &node = sourceTree.node(*slot);
        if (node.parentArc == getInvalidPnrIndex())
          break;
        if (node.parentArc >= arcs.size() ||
            node.parentArc >= arcSources.size())
          return candidateError("snapshot RouteTree arc is out of range");
        reversePath.push_back(node.parentArc);
        slot = sourceTree.findNode(arcSources[node.parentArc]);
        if (!slot)
          return candidateError("snapshot RouteTree parent is absent");
      }

      std::vector<PnrIndex> forwardPath(reversePath.rbegin(),
                                        reversePath.rend());
      PnrIndex attachment = *source;
      std::size_t pathBegin = 0;
      const RouteTreeState &targetTree = (*cloned)->routeTree(logicalNet);
      for (auto [index, arc] : llvm::enumerate(forwardPath)) {
        if (targetTree.findNode(arcs[arc].target)) {
          attachment = arcs[arc].target;
          pathBegin = index + 1;
        }
      }
      if (llvm::Error error = move->attachRoutePath(
              logicalNet, attachment,
              llvm::ArrayRef<PnrIndex>(forwardPath).drop_front(pathBegin),
              sink))
        return std::move(error);
    }
  }

  if (llvm::Error error = move->commit())
    return std::move(error);
  if (llvm::Error error = (*cloned)->verify())
    return std::move(error);
  return std::move(*cloned);
}

const SpatialComputeBindingSelection &
SpatialCandidateState::computeBinding(PnrIndex realization) const {
  assert(realization < computeBindings_.size());
  return computeBindings_[realization];
}

const SpatialMemoryBindingSelection &
SpatialCandidateState::memoryBinding(PnrIndex realization) const {
  assert(realization < memoryBindings_.size());
  return memoryBindings_[realization];
}

PnrIndex SpatialCandidateState::portAttachment(PnrIndex demand) const {
  assert(demand < portAttachments_.size());
  return portAttachments_[demand];
}

PnrIndex
SpatialCandidateState::graphBoundaryAttachment(PnrIndex boundary) const {
  assert(boundary < graphBoundaryAttachments_.size());
  return graphBoundaryAttachments_[boundary];
}

PnrIndex SpatialCandidateState::memoryOperationPlan(PnrIndex actor) const {
  assert(actor < memoryOperationPlans_.size());
  return memoryOperationPlans_[actor];
}

const SpatialLogicalMemoryBindingSelection &
SpatialCandidateState::logicalMemoryBinding(PnrIndex binding) const {
  assert(binding < logicalMemoryBindings_.size());
  return logicalMemoryBindings_[binding];
}

PnrIndex SpatialCandidateState::memoryUseDispatch(PnrIndex use) const {
  assert(use < memoryUseDispatches_.size());
  return memoryUseDispatches_[use];
}

PnrIndex
SpatialCandidateState::memoryExposureSelection(PnrIndex exposure) const {
  assert(exposure < memoryExposureSelections_.size());
  return memoryExposureSelections_[exposure];
}

PnrIndex
SpatialCandidateState::registerFifoTransfer(PnrIndex logicalNet) const {
  assert(logicalNet < registerFifoTransfers_.size());
  return registerFifoTransfers_[logicalNet];
}

llvm::Error
SpatialCandidateState::validateComputeBinding(PnrIndex realization) const {
  const auto realizations = problem_->realizations().computeRealizations();
  if (realization >= realizations.size())
    return candidateError("compute realization is out of range");
  const auto &record = realizations[realization];
  const auto binding = computeBindings_[realization];
  if (!rangeContains(record.placementOffset, record.placementCount,
                     binding.placement))
    return candidateError("compute binding selects a foreign placement domain");
  const auto &placement =
      problem_->realizations().computePlacements()[binding.placement];
  if (placement.realization != realization ||
      !rangeContains(placement.contextOffset, placement.contextCount,
                     binding.instructionContext))
    return candidateError(
        "compute binding selects a foreign instruction context");
  return llvm::Error::success();
}

llvm::Error
SpatialCandidateState::validateRegisterFifoTransfer(PnrIndex logicalNet) const {
  if (logicalNet >= registerFifoTransfers_.size() ||
      logicalNet >= problem_->localTransfers().domains().size())
    return candidateError("register-FIFO transfer net is out of range");
  const PnrIndex selected = registerFifoTransfers_[logicalNet];
  if (selected == getInvalidPnrIndex())
    return llvm::Error::success();
  const auto &domain = problem_->localTransfers().domains()[logicalNet];
  if (!rangeContains(domain.optionOffset, domain.optionCount, selected) ||
      selected >= problem_->localTransfers().options().size())
    return candidateError(
        "register-FIFO transfer selects an option outside its net domain");
  const auto &option = problem_->localTransfers().options()[selected];
  if (option.logicalNet != logicalNet ||
      option.producerRealization >= computeBindings_.size() ||
      option.consumerRealization >= computeBindings_.size() ||
      computeBindings_[option.producerRealization].placement !=
          option.producerPlacement ||
      computeBindings_[option.consumerRealization].placement !=
          option.consumerPlacement)
    return candidateError(
        "register-FIFO transfer disagrees with selected compute placements");
  if (problem_->routeConstraints().netHasConstraints(logicalNet))
    return candidateError(
        "register-FIFO transfer bypasses an explicit route constraint");
  return llvm::Error::success();
}

llvm::Error SpatialCandidateState::verifyRegisterFifoTransfers() const {
  using FifoKey =
      std::pair<::loom::fabric::FabricEntityId, ::loom::fabric::FabricOrdinal>;
  std::set<FifoKey> claimed;
  for (PnrIndex logicalNet = 0; logicalNet < registerFifoTransfers_.size();
       ++logicalNet) {
    if (llvm::Error error = validateRegisterFifoTransfer(logicalNet))
      return error;
    const PnrIndex selected = registerFifoTransfers_[logicalNet];
    if (selected == getInvalidPnrIndex())
      continue;
    if (!routeTrees_[logicalNet]->isUnrouted())
      return candidateError(
          "register-FIFO transfer also owns an external RouteTree");
    const auto &option = problem_->localTransfers().options()[selected];
    if (!claimed.emplace(option.pe.id(), option.registerFifo).second)
      return candidateError(
          "register-FIFO transfer resource has multiple owners");
  }
  return llvm::Error::success();
}

llvm::Error
SpatialCandidateState::validateMemoryBinding(PnrIndex realization) const {
  const auto realizations = problem_->realizations().memoryRealizations();
  if (realization >= realizations.size())
    return candidateError("memory realization is out of range");
  const auto &record = realizations[realization];
  const PnrIndex placement = memoryBindings_[realization].placement;
  if (!rangeContains(record.placementOffset, record.placementCount,
                     placement) ||
      problem_->realizations().memoryPlacements()[placement].realization !=
          realization)
    return candidateError("memory binding selects a foreign placement domain");
  return llvm::Error::success();
}

llvm::Error
SpatialCandidateState::validatePortAttachment(PnrIndex demand) const {
  const auto demands = problem_->ports().portDemands();
  if (demand >= demands.size())
    return candidateError("PortDemand is out of range");
  const auto &record = demands[demand];
  PnrIndex placement = getInvalidPnrIndex();
  PnrIndex placementOffset = 0;
  if (record.kind == FrozenSpatialPortDemandKind::Compute) {
    if (record.realization >= computeBindings_.size())
      return candidateError("compute PortDemand owner is out of range");
    placement = computeBindings_[record.realization].placement;
    placementOffset = problem_->realizations()
                          .computeRealizations()[record.realization]
                          .placementOffset;
  } else {
    if (record.realization >= memoryBindings_.size())
      return candidateError("memory PortDemand owner is out of range");
    placement = memoryBindings_[record.realization].placement;
    placementOffset = problem_->realizations()
                          .memoryRealizations()[record.realization]
                          .placementOffset;
  }
  if (placement < placementOffset ||
      placement - placementOffset >= record.placementDomainCount)
    return candidateError("PortDemand placement selection is out of range");
  const PnrIndex domainIndex =
      record.placementDomainOffset + placement - placementOffset;
  const auto &domain = problem_->ports().placementDomains()[domainIndex];
  const PnrIndex option = portAttachments_[demand];
  if (domain.placement != placement ||
      !rangeContains(domain.attachmentOptionOffset,
                     domain.attachmentOptionCount, option))
    return candidateError(
        "PortDemand selects an attachment outside its placement domain");
  return llvm::Error::success();
}

llvm::Error SpatialCandidateState::validateGraphBoundaryAttachment(
    PnrIndex boundary) const {
  const auto boundaries = problem_->ports().graphBoundaries();
  if (boundary >= boundaries.size())
    return candidateError("graph boundary is out of range");
  const auto &record = boundaries[boundary];
  if (!rangeContains(record.attachmentOptionOffset,
                     record.attachmentOptionCount,
                     graphBoundaryAttachments_[boundary]))
    return candidateError(
        "graph boundary selects an attachment outside its domain");
  return llvm::Error::success();
}

llvm::Error
SpatialCandidateState::validateMemoryOperationPlan(PnrIndex actor) const {
  const auto &realizations = problem_->realizations();
  if (actor >= realizations.memoryActors().size())
    return candidateError("memory actor is out of range");
  const PnrIndex owner = realizations.memoryActorRealizations()[actor];
  if (owner >= memoryBindings_.size())
    return candidateError("memory actor owner is out of range");
  const auto &realization = realizations.memoryRealizations()[owner];
  if (!rangeContains(realization.actorOffset, realization.actorCount, actor))
    return candidateError("memory actor owner projection is inconsistent");
  const PnrIndex placement = memoryBindings_[owner].placement;
  const auto offsets = problem_->handshake().memoryPlacementDomainOffsets();
  if (placement + 1 >= offsets.size())
    return candidateError("memory placement has no operation-plan domain");
  const PnrIndex domainIndex =
      offsets[placement] + actor - realization.actorOffset;
  if (domainIndex >= problem_->handshake().memoryOperationDomains().size())
    return candidateError("memory operation-plan domain is out of range");
  const auto &domain =
      problem_->handshake().memoryOperationDomains()[domainIndex];
  if (domain.placement != placement || domain.actor != actor ||
      !rangeContains(domain.planOffset, domain.planCount,
                     memoryOperationPlans_[actor]))
    return candidateError(
        "memory actor selects a plan outside its placement domain");
  return llvm::Error::success();
}

PnrIndex SpatialCandidateState::terminalEndpoint(
    FrozenSpatialTerminalBinding binding) const {
  const PnrIndex option =
      binding.kind == FrozenSpatialTerminalBindingKind::PortDemand
          ? portAttachments_[binding.index]
          : graphBoundaryAttachments_[binding.index];
  return problem_->ports().attachmentOptions()[option].endpoint;
}

std::uint32_t SpatialCandidateState::terminalPayloadWidth(
    FrozenSpatialTerminalBinding binding) const {
  return binding.kind == FrozenSpatialTerminalBindingKind::PortDemand
             ? problem_->ports().portDemands()[binding.index].payloadWidthBits
             : problem_->ports()
                   .graphBoundaries()[binding.index]
                   .payloadWidthBits;
}

PnrIndex
SpatialCandidateState::logicalNetSourceEndpoint(PnrIndex logicalNet) const {
  assert(logicalNet < problem_->transfers().logicalNets().size());
  return terminalEndpoint(
      problem_->transfers().logicalNetSourceBindings()[logicalNet]);
}

PnrIndex
SpatialCandidateState::logicalNetSinkEndpoint(PnrIndex logicalNet,
                                              PnrIndex sinkObligation) const {
  assert(logicalNet < problem_->transfers().logicalNets().size());
  const auto &net = problem_->transfers().logicalNets()[logicalNet];
  assert(sinkObligation < net.sinkCount);
  return terminalEndpoint(
      problem_->transfers()
          .logicalNetSinkBindings()[net.sinkOffset + sinkObligation]);
}

std::uint32_t
SpatialCandidateState::logicalNetPayloadWidth(PnrIndex logicalNet) const {
  assert(logicalNet < problem_->transfers().logicalNets().size());
  return problem_->transfers().logicalNets()[logicalNet].payloadWidthBits;
}

const RouteTreeState &
SpatialCandidateState::routeTree(PnrIndex logicalNet) const {
  assert(logicalNet < routeTrees_.size());
  return *routeTrees_[logicalNet];
}

llvm::Expected<SpatialCandidateRouteProjection>
SpatialCandidateState::projectVerifiedRoutes(
    llvm::ArrayRef<const RouteTreeState *> routes,
    SpatialTagAssignmentSummary *tagSummary) const {
  if (routes.size() != routeTrees_.size())
    return candidateError("projected route count does not match the candidate");
  std::uint64_t unrouted = 0;
  bool routeTerminalsCompatible = true;
  const auto logicalNets = problem_->transfers().logicalNets();
  for (PnrIndex logicalNet = 0; logicalNet < routes.size(); ++logicalNet) {
    if (!routes[logicalNet] ||
        &routes[logicalNet]->routingGraph() != &problem_->routing())
      return candidateError(
          "projected RouteTree does not belong to the candidate");
    if (usesRegisterFifo(logicalNet)) {
      if (!routes[logicalNet]->isUnrouted())
        return candidateError(
            "projected register-FIFO net also has an external route");
      continue;
    }
    if (!routes[logicalNet]->isUnrouted()) {
      const RouteTreeState &route = *routes[logicalNet];
      routeTerminalsCompatible &=
          route.sourceEndpoint() == logicalNetSourceEndpoint(logicalNet);
      for (PnrIndex sink = 0; sink < logicalNets[logicalNet].sinkCount; ++sink)
        routeTerminalsCompatible &= route.sinkEndpoint(sink) ==
                                    logicalNetSinkEndpoint(logicalNet, sink);
      continue;
    }
    if (logicalNets[logicalNet].sinkCount >
        std::numeric_limits<std::uint64_t>::max() - unrouted)
      return candidateError(
          "projected unrouted obligation count overflows u64");
    unrouted += logicalNets[logicalNet].sinkCount;
  }

  auto routeResources = SpatialRouteResourceState::projectVerifiedRoutes(
      *problem_, routes, registerFifoTransfers_);
  if (!routeResources)
    return routeResources.takeError();
  auto physicalTiming = detail::projectSpatialPhysicalTiming(
      *problem_, routes, registerFifoTransfers_, portAttachments_,
      graphBoundaryAttachments_);
  if (!physicalTiming)
    return physicalTiming.takeError();
  auto recurrenceTiming = detail::projectSpatialRecurrenceTiming(*this, routes);
  if (!recurrenceTiming)
    return recurrenceTiming.takeError();
  auto tags =
      tagAssignments_.projectVerifiedRoutes(routes, tagSummary != nullptr);
  if (!tags)
    return tags.takeError();
  const std::uint64_t tagResidentCapacityOveruse =
      tags->residentCapacityOveruse;
  const std::uint64_t tagUnassignedCount = tags->unassignedCount;
  const std::uint64_t tagConflictCount = tags->conflictCount;
  if (tagSummary)
    *tagSummary = std::move(*tags);
  const auto tagValues = tagValueViews(tagAssignments_, routes.size());
  auto handshakeAcyclic = projectHandshakeSelections(
      *problem_, computeBindings_, portAttachments_, memoryOperationPlans_,
      registerFifoTransfers_, routes, tagValues);
  if (!handshakeAcyclic)
    return handshakeAcyclic.takeError();
  std::uint64_t hardProgressViolation = 0;
  switch (problem_->progressBasis().kind) {
  case ::loom::mapping::MappingDataflowProgressBasisKind::Acyclic:
  case ::loom::mapping::MappingDataflowProgressBasisKind::InitializedFeedback: {
    hardProgressViolation = progressState_.hardProgressViolation();
    break;
  }
  case ::loom::mapping::MappingDataflowProgressBasisKind::Cyclic:
    return candidateError(
        "cyclic Dataflow basis requires a typed progress breaker");
  }

  return SpatialCandidateRouteProjection{
      unrouted,
      routeResources->totalCapacityOveruseRaw(),
      tagResidentCapacityOveruse,
      tagUnassignedCount,
      tagConflictCount,
      hardProgressViolation,
      routeResources->totalSelectedTraversalClaim(),
      routeResources->routeReleaseLatencyCycles(),
      routeResources->routeMinimumInitiationIntervalCycles(),
      std::move(*recurrenceTiming),
      routeResources->transportBitCycleDemand(),
      physicalTiming->worstArrivalDelayQuanta,
      physicalTiming->totalNegativeSlackQuanta,
      routeTerminalsCompatible,
      *handshakeAcyclic};
}

llvm::Expected<SpatialTagAssignmentSummary>
SpatialCandidateState::summarizeCurrentTagAssignments() const {
  return tagAssignments_.summarizeCurrentState(true);
}

llvm::Error
SpatialCandidateState::validateLogicalNet(PnrIndex logicalNet) const {
  if (logicalNet >= problem_->transfers().logicalNets().size())
    return candidateError("logical net is out of range");
  const auto &net = problem_->transfers().logicalNets()[logicalNet];
  const std::uint32_t payloadWidth = logicalNetPayloadWidth(logicalNet);
  for (PnrIndex sink = 0; sink < net.sinkCount; ++sink) {
    const auto binding =
        problem_->transfers().logicalNetSinkBindings()[net.sinkOffset + sink];
    if (terminalPayloadWidth(binding) != payloadWidth)
      return candidateError("logical-net terminal widths disagree");
  }

  const RouteTreeState &route = *routeTrees_[logicalNet];
  if (llvm::Error error = route.verify())
    return error;
  if (route.isUnrouted())
    return llvm::Error::success();
  if (route.sourceEndpoint() != logicalNetSourceEndpoint(logicalNet))
    return candidateError(
        "route source disagrees with its selected attachment");
  for (PnrIndex sink = 0; sink < net.sinkCount; ++sink)
    if (route.sinkEndpoint(sink) != logicalNetSinkEndpoint(logicalNet, sink))
      return candidateError(
          "route sink disagrees with its selected attachment");
  for (const RouteTreeNode &node : route.nodeStorage()) {
    if (!node.isActive() || node.parentArc == getInvalidPnrIndex())
      continue;
    if (node.parentArc >= problem_->routing().routingArcs().size() ||
        problem_->routing().routingArcs()[node.parentArc].payloadCapacityBits <
            payloadWidth)
      return candidateError("route traversal cannot carry its payload width");
  }
  return llvm::Error::success();
}

llvm::Error SpatialCandidateState::verifyHandshakeProjection() const {
  const auto &index = problem_->handshake();
  std::vector<PnrIndex> expectedFragments(index.fragments().size(), 0);
  std::vector<PnrIndex> expectedTraversals(
      index.traversalFragmentOffsets().size() - 1, 0);
  const auto addFragments =
      [&](llvm::ArrayRef<PnrIndex> fragments) -> llvm::Error {
    for (PnrIndex fragment : fragments) {
      if (fragment >= expectedFragments.size())
        return candidateError("handshake fragment projection is out of range");
      if (llvm::Error error =
              increment(expectedFragments[fragment], 1, "handshake fragment"))
        return error;
    }
    return llvm::Error::success();
  };

  if (llvm::Error error = addFragments(index.fixedFragments()))
    return error;
  for (const SpatialComputeBindingSelection &binding : computeBindings_)
    if (llvm::Error error =
            addFragments(computePlacementFragments(index, binding.placement)))
      return error;
  for (PnrIndex plan : memoryOperationPlans_)
    if (llvm::Error error = addFragments(memoryPlanFragments(index, plan)))
      return error;
  for (PnrIndex option : portAttachments_)
    if (std::optional<PnrIndex> traversal =
            attachmentTraversal(problem_->ports(), option))
      if (llvm::Error error = increment(expectedTraversals[*traversal], 1,
                                        "handshake traversal"))
        return error;
  for (PnrIndex option : registerFifoTransfers_) {
    if (option == getInvalidPnrIndex())
      continue;
    if (option >= problem_->localTransfers().options().size())
      return candidateError("register-FIFO handshake option is out of range");
    const auto &transfer = problem_->localTransfers().options()[option];
    if (transfer.writeTraversal >= expectedTraversals.size() ||
        transfer.readTraversal >= expectedTraversals.size())
      return candidateError(
          "register-FIFO handshake traversal is out of range");
    if (llvm::Error error =
            increment(expectedTraversals[transfer.writeTraversal], 1,
                      "handshake traversal"))
      return error;
    if (llvm::Error error =
            increment(expectedTraversals[transfer.readTraversal], 1,
                      "handshake traversal"))
      return error;
  }
  for (const RouteTreeStateHandle &route : routeTrees_)
    for (const RouteTreeNode &node : route->nodeStorage()) {
      if (!node.isActive() || node.parentArc == getInvalidPnrIndex())
        continue;
      const PnrIndex traversal =
          problem_->routing().routingArcs()[node.parentArc].traversal;
      if (llvm::Error error = increment(expectedTraversals[traversal], 1,
                                        "handshake traversal"))
        return error;
    }

  const auto traversalOffsets = index.traversalFragmentOffsets();
  for (PnrIndex traversal = 0; traversal < expectedTraversals.size();
       ++traversal) {
    if (expectedTraversals[traversal] == 0)
      continue;
    if (llvm::Error error = addFragments(index.traversalFragments().slice(
            traversalOffsets[traversal],
            traversalOffsets[traversal + 1] - traversalOffsets[traversal])))
      return error;
  }
  for (const auto &group : index.allTraversalGroups()) {
    bool active = true;
    for (PnrIndex traversal : index.allTraversalGroupWitnesses().slice(
             group.witnessOffset, group.witnessCount))
      active &= expectedTraversals[traversal] != 0;
    if (active)
      if (llvm::Error error =
              addFragments(llvm::ArrayRef<PnrIndex>(&group.fragment, 1)))
        return error;
  }
  std::vector<const RouteTreeState *> rawRoutes;
  rawRoutes.reserve(routeTrees_.size());
  for (const RouteTreeStateHandle &route : routeTrees_)
    rawRoutes.push_back(route.get());
  const auto tagValues = tagValueViews(tagAssignments_, rawRoutes.size());
  auto switchFragments = detail::deriveSpatialTemporalSwitchHandshakeFragments(
      *problem_, rawRoutes, tagValues);
  if (!switchFragments)
    return switchFragments.takeError();
  if (llvm::Error error = addFragments(*switchFragments))
    return error;

  if (llvm::Error error = handshake_->verify())
    return error;
  for (PnrIndex fragment = 0; fragment < expectedFragments.size(); ++fragment)
    if (handshake_->fragmentRefcount(fragment) != expectedFragments[fragment])
      return candidateError(
          "selected handshake fragments diverge from candidate decisions");
  for (PnrIndex traversal = 0; traversal < expectedTraversals.size();
       ++traversal)
    if (handshake_->traversalRefcount(traversal) !=
        expectedTraversals[traversal])
      return candidateError(
          "selected handshake traversals diverge from candidate decisions");
  return llvm::Error::success();
}

llvm::Expected<std::uint64_t>
SpatialCandidateState::recomputeAtomicCapacityOveruse() const {
  const auto compute = problem_->capacity().computeInstructionContextOveruse();
  const auto memory = problem_->capacity().memoryOperationPlanOveruse();
  const auto dispatch = problem_->capacity().memoryDispatchOptionOveruse();
  const auto dispatchPatterns =
      problem_->capacity().memoryDispatchOptionPatterns();
  std::uint64_t total = 0;
  for (const SpatialComputeBindingSelection &binding : computeBindings_) {
    if (binding.instructionContext >= compute.size())
      return candidateError(
          "compute binding has no capacity-envelope projection");
    if (compute[binding.instructionContext] >
        std::numeric_limits<std::uint64_t>::max() - total)
      return candidateError("capacity overuse total overflows u64");
    total += compute[binding.instructionContext];
  }
  for (PnrIndex plan : memoryOperationPlans_) {
    if (plan >= memory.size())
      return candidateError(
          "memory operation plan has no capacity-envelope projection");
    if (memory[plan] > std::numeric_limits<std::uint64_t>::max() - total)
      return candidateError("capacity overuse total overflows u64");
    total += memory[plan];
  }
  llvm::DenseSet<std::pair<PnrIndex, PnrIndex>> servicePatterns;
  servicePatterns.reserve(memoryUseDispatches_.size() * 2);
  for (PnrIndex use = 0; use < memoryUseDispatches_.size(); ++use) {
    const PnrIndex option = memoryUseDispatches_[use];
    if (option >= dispatch.size())
      return candidateError(
          "memory dispatch has no capacity-envelope projection");
    if (option >= dispatchPatterns.size())
      return candidateError("memory dispatch has no UsePattern projection");
    const PnrIndex pattern = dispatchPatterns[option];
    if (pattern == getInvalidPnrIndex())
      continue;
    if (use >= problem_->memory().rootedUseServiceGroups().size())
      return candidateError(
          "memory dispatch has no service-use group projection");
    const PnrIndex group = problem_->memory().rootedUseServiceGroups()[use];
    if (group == getInvalidPnrIndex())
      return candidateError(
          "memory service UsePattern has no owner-derived group");
    if (!servicePatterns.insert({group, pattern}).second)
      continue;
    if (dispatch[option] > std::numeric_limits<std::uint64_t>::max() - total)
      return candidateError("capacity overuse total overflows u64");
    total += dispatch[option];
  }
  llvm::DenseSet<std::pair<PnrIndex, PnrIndex>> providerBindings;
  providerBindings.reserve(memoryExposureSelections_.size() * 2);
  std::vector<std::uint64_t> providerCounts(
      problem_->memory().exposureProviders().size(), 0);
  for (PnrIndex exposure = 0; exposure < memoryExposureSelections_.size();
       ++exposure) {
    const PnrIndex option = memoryExposureSelections_[exposure];
    if (option >= problem_->memory().exposureOptions().size())
      return candidateError("memory exposure option is out of range");
    const PnrIndex provider =
        problem_->memory().exposureOptions()[option].provider;
    if (provider >= providerCounts.size())
      return candidateError("memory exposure provider is out of range");
    const auto key = std::make_pair(
        problem_->memory().exposures()[exposure].logicalBinding, provider);
    if (providerBindings.insert(key).second)
      ++providerCounts[provider];
  }
  for (PnrIndex provider = 0; provider < providerCounts.size(); ++provider) {
    const std::uint64_t capacity =
        problem_->memory().exposureProviders()[provider].maxExposedBindings;
    const std::uint64_t overuse = providerCounts[provider] > capacity
                                      ? providerCounts[provider] - capacity
                                      : 0;
    if (overuse > std::numeric_limits<std::uint64_t>::max() - total)
      return candidateError("capacity overuse total overflows u64");
    total += overuse;
  }
  return total;
}

llvm::Error SpatialCandidateState::verify() const {
  if (activeTransaction_)
    return candidateError("cannot verify during a move");
  if (!problem_ || !handshake_ ||
      computeBindings_.size() !=
          problem_->realizations().computeRealizations().size() ||
      memoryBindings_.size() !=
          problem_->realizations().memoryRealizations().size() ||
      bindingRelationChoices_.size() !=
          problem_->bindingRelations().decisionCount() ||
      portAttachments_.size() != problem_->ports().portDemands().size() ||
      graphBoundaryAttachments_.size() !=
          problem_->ports().graphBoundaries().size() ||
      memoryOperationPlans_.size() !=
          problem_->realizations().memoryActors().size() ||
      logicalMemoryBindings_.size() !=
          problem_->memory().logicalBindings().size() ||
      memoryUseDispatches_.size() != problem_->memory().rootedUses().size() ||
      memoryExposureSelections_.size() !=
          problem_->memory().exposures().size() ||
      registerFifoTransfers_.size() !=
          problem_->transfers().logicalNets().size() ||
      routeTrees_.size() != problem_->transfers().logicalNets().size())
    return candidateError("candidate shape does not match its frozen problem");
  for (PnrIndex index = 0; index < computeBindings_.size(); ++index)
    if (llvm::Error error = validateComputeBinding(index))
      return error;
  for (PnrIndex index = 0; index < memoryBindings_.size(); ++index)
    if (llvm::Error error = validateMemoryBinding(index))
      return error;
  if (llvm::Error error = verifyBindingRelations())
    return error;
  for (PnrIndex index = 0; index < portAttachments_.size(); ++index)
    if (llvm::Error error = validatePortAttachment(index))
      return error;
  for (PnrIndex index = 0; index < graphBoundaryAttachments_.size(); ++index)
    if (llvm::Error error = validateGraphBoundaryAttachment(index))
      return error;
  for (PnrIndex index = 0; index < memoryOperationPlans_.size(); ++index)
    if (llvm::Error error = validateMemoryOperationPlan(index))
      return error;
  if (llvm::Error error = verifyMemorySelections())
    return error;
  if (llvm::Error error =
          problem_->memoryConstraints().verify(logicalMemoryBindings_))
    return error;
  if (llvm::Error error = verifyRegisterFifoTransfers())
    return error;
  for (PnrIndex index = 0; index < routeTrees_.size(); ++index)
    if (llvm::Error error = validateLogicalNet(index))
      return error;
  detail::SpatialRouteConstraintScratch routeConstraints;
  if (llvm::Error error = routeConstraints.prepare(*problem_))
    return error;
  if (llvm::Error error = routeConstraints.verifyAll(*this))
    return error;
  std::uint64_t expectedUnroutedObligationCount = 0;
  const auto logicalNets = problem_->transfers().logicalNets();
  for (PnrIndex index = 0; index < routeTrees_.size(); ++index) {
    if (usesRegisterFifo(index))
      continue;
    if (!routeTrees_[index]->isUnrouted())
      continue;
    const std::uint64_t sinkCount = logicalNets[index].sinkCount;
    if (sinkCount > std::numeric_limits<std::uint64_t>::max() -
                        expectedUnroutedObligationCount)
      return candidateError("unrouted obligation count overflows u64");
    expectedUnroutedObligationCount += sinkCount;
  }
  if (unroutedObligationCount_ != expectedUnroutedObligationCount)
    return candidateError(
        "unrouted obligation count diverges from RouteTree state");
  auto expectedCapacityOveruse = recomputeAtomicCapacityOveruse();
  if (!expectedCapacityOveruse)
    return expectedCapacityOveruse.takeError();
  if (atomicCapacityOveruse_ != *expectedCapacityOveruse)
    return candidateError(
        "capacity overuse diverges from selected resource envelopes");
  auto expectedSchedulePressure = detail::measureStaticSchedulePressure(*this);
  if (!expectedSchedulePressure)
    return expectedSchedulePressure.takeError();
  if (staticSchedulePressure_ != *expectedSchedulePressure)
    return candidateError(
        "static schedule pressure diverges from selected bindings");
  auto expectedOperandIngressPressure =
      detail::measureSpatialOperandIngressPressure(*problem_, portAttachments_,
                                                   registerFifoTransfers_);
  if (!expectedOperandIngressPressure)
    return expectedOperandIngressPressure.takeError();
  if (sharedOperandIngressPressure_ != *expectedOperandIngressPressure)
    return candidateError(
        "shared operand ingress pressure diverges from selected attachments");
  if (llvm::Error error = verifyResourceTimeEnvelopeSelections())
    return error;
  if (llvm::Error error =
          routeResources_.verify(routeTrees_, registerFifoTransfers_))
    return error;
  if (llvm::Error error = progressState_.verify(*this))
    return error;
  std::vector<const RouteTreeState *> routes;
  routes.reserve(routeTrees_.size());
  for (const RouteTreeStateHandle &route : routeTrees_)
    routes.push_back(route.get());
  std::vector<std::uint64_t> expectedNetWorstArrivals;
  std::vector<std::uint64_t> expectedNetNegativeSlacks;
  auto physicalTiming = detail::projectSpatialPhysicalTiming(
      *problem_, routes, registerFifoTransfers_, portAttachments_,
      graphBoundaryAttachments_, &expectedNetWorstArrivals,
      &expectedNetNegativeSlacks);
  if (!physicalTiming)
    return physicalTiming.takeError();
  if (expectedNetWorstArrivals != logicalNetWorstArrivalDelayQuanta_ ||
      expectedNetNegativeSlacks != logicalNetNegativeSlackQuanta_ ||
      physicalTiming->worstArrivalDelayQuanta !=
          worstRouteArrivalDelayQuanta_ ||
      physicalTiming->totalNegativeSlackQuanta !=
          totalRouteNegativeSlackQuanta_)
    return candidateError(
        "cached physical timing diverges from selected routes");
  auto recurrenceTiming = detail::projectSpatialRecurrenceTiming(*this, routes);
  if (!recurrenceTiming)
    return recurrenceTiming.takeError();
  if (!(*recurrenceTiming == recurrenceTiming_))
    return candidateError(
        "cached recurrence timing diverges from selected Mapping");
  if (llvm::Error error = tagAssignments_.verify(routeTrees_))
    return error;
  return verifyHandshakeProjection();
}

llvm::Expected<SpatialMoveTransaction>
SpatialCandidateState::beginMove(SpatialCandidateScratch &scratch) & {
  if (activeTransaction_)
    return candidateError("candidate already has an active move");
  if (scratch.activeTransaction_)
    return candidateError("scratch already has an active move");
  if (scratch.preparedProblem_ != problem_.get())
    return candidateError(
        "scratch was prepared for a different frozen problem");
  if (scratch.computeJournalMarks_.size() != computeBindings_.size() ||
      scratch.memoryJournalMarks_.size() != memoryBindings_.size() ||
      scratch.portJournalMarks_.size() != portAttachments_.size() ||
      scratch.boundaryJournalMarks_.size() !=
          graphBoundaryAttachments_.size() ||
      scratch.memoryPlanJournalMarks_.size() != memoryOperationPlans_.size() ||
      scratch.logicalMemoryJournalMarks_.size() !=
          logicalMemoryBindings_.size() ||
      scratch.memoryDispatchJournalMarks_.size() !=
          memoryUseDispatches_.size() ||
      scratch.memoryExposureJournalMarks_.size() !=
          memoryExposureSelections_.size() ||
      scratch.routeTransactions_.size() != routeTrees_.size() ||
      scratch.progressTerminalActive_.size() != routeTrees_.size() ||
      scratch.traversalDeltaMarks_.size() !=
          problem_->routing().traversals().size() ||
      scratch.affectedBindingRelationMarks_.size() !=
          problem_->bindingRelations().relations().relations().size() ||
      !scratch.routeConstraintScratch_)
    return candidateError("scratch was not prepared for this candidate");

  scratch.beginTransaction();
  auto handshakeTransaction =
      handshake_->beginTransaction(scratch.handshakeScratch_);
  if (!handshakeTransaction)
    return handshakeTransaction.takeError();
  scratch.handshakeTransaction_.emplace(std::move(*handshakeTransaction));
  return SpatialMoveTransaction(shared_from_this(), scratch);
}
