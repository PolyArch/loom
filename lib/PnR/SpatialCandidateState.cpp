#include "PnR/SpatialCandidateState.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace loom::pnr;

namespace {

llvm::Error candidateError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid Spatial candidate state: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

bool rangeContains(PnrIndex offset, PnrIndex count, PnrIndex value) {
  return value >= offset && value - offset < count;
}

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

llvm::ArrayRef<PnrIndex>
computePlacementFragments(const FrozenSpatialHandshakeIndex &handshake,
                          PnrIndex placement) {
  const auto offsets = handshake.computePlacementFragmentOffsets();
  return handshake.computePlacementFragments().slice(
      offsets[placement], offsets[placement + 1] - offsets[placement]);
}

llvm::ArrayRef<PnrIndex>
memoryPlanFragments(const FrozenSpatialHandshakeIndex &handshake,
                    PnrIndex plan) {
  const auto record = handshake.memoryOperationPlans()[plan];
  return handshake.memoryPlanFragments().slice(record.fragmentOffset,
                                               record.fragmentCount);
}

std::optional<PnrIndex> attachmentTraversal(const FrozenSpatialPortIndex &ports,
                                            PnrIndex option) {
  return ports.attachmentOptions()[option].localTraversal;
}

llvm::Error addInitialHandshakeSelections(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<SpatialComputeBindingSelection> computeBindings,
    llvm::ArrayRef<PnrIndex> portAttachments,
    llvm::ArrayRef<PnrIndex> memoryOperationPlans,
    HandshakeCandidateState &handshake) {
  HandshakeCandidateScratch scratch;
  if (llvm::Error error = scratch.prepare(problem.handshake()))
    return error;
  auto transaction = handshake.beginTransaction(scratch);
  if (!transaction)
    return transaction.takeError();

  for (const SpatialComputeBindingSelection &binding : computeBindings)
    if (llvm::Error error = transaction->addFragments(
            computePlacementFragments(problem.handshake(), binding.placement)))
      return error;
  for (PnrIndex plan : memoryOperationPlans)
    if (llvm::Error error = transaction->addFragments(
            memoryPlanFragments(problem.handshake(), plan)))
      return error;
  for (PnrIndex option : portAttachments)
    if (std::optional<PnrIndex> traversal =
            attachmentTraversal(problem.ports(), option))
      if (llvm::Error error = transaction->addTraversalUses(*traversal, 1))
        return error;

  auto closure = transaction->close();
  if (!closure)
    return closure.takeError();
  if (!*closure)
    return candidateError(
        "initial selections close a combinational handshake cycle");
  return transaction->commit();
}

} // namespace

SpatialCandidateScratch::~SpatialCandidateScratch() {
  if (activeTransaction_)
    activeTransaction_->rollback();
}

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
  if (llvm::Error error = handshakeScratch_.prepare(problem.handshake()))
    return error;

  const auto &realizations = problem.realizations();
  const auto &ports = problem.ports();
  const std::size_t computeCount = realizations.computeRealizations().size();
  const std::size_t memoryCount = realizations.memoryRealizations().size();
  const std::size_t portCount = ports.portDemands().size();
  const std::size_t boundaryCount = ports.graphBoundaries().size();
  const std::size_t memoryPlanCount = realizations.memoryActors().size();
  const std::size_t traversalCount = problem.routing().traversals().size();

  computeJournalMarks_.assign(computeCount, 0);
  memoryJournalMarks_.assign(memoryCount, 0);
  portJournalMarks_.assign(portCount, 0);
  boundaryJournalMarks_.assign(boundaryCount, 0);
  memoryPlanJournalMarks_.assign(memoryPlanCount, 0);
  decisionDeltas_.clear();
  decisionDeltas_.reserve(computeCount + memoryCount + portCount +
                          boundaryCount + memoryPlanCount);

  affectedComputeMarks_.assign(computeCount, 0);
  affectedMemoryMarks_.assign(memoryCount, 0);
  affectedPortMarks_.assign(portCount, 0);
  affectedBoundaryMarks_.assign(boundaryCount, 0);
  affectedMemoryPlanMarks_.assign(memoryPlanCount, 0);
  affectedNetMarks_.assign(netCount, 0);
  affectedComputes_.reserve(computeCount);
  affectedMemories_.reserve(memoryCount);
  affectedPorts_.reserve(portCount);
  affectedBoundaries_.reserve(boundaryCount);
  affectedMemoryPlans_.reserve(memoryPlanCount);
  affectedNets_.reserve(netCount);

  touchedRoutes_.reserve(netCount);
  traversalDeltaMarks_.assign(traversalCount, 0);
  traversalRemoved_.assign(traversalCount, 0);
  traversalAdded_.assign(traversalCount, 0);
  touchedTraversals_.reserve(traversalCount);

  decisionEpoch_ = 0;
  affectedEpoch_ = 0;
  traversalEpoch_ = 0;
  preparedProblem_ = &problem;
  resetTransaction();
  return llvm::Error::success();
}

std::size_t SpatialCandidateScratch::retainedStorageBytes() const {
  std::size_t bytes = retainedBytes(routeScratch_) +
                      retainedBytes(routeTransactions_) +
                      handshakeScratch_.retainedStorageBytes();
  for (const auto &scratch : routeScratch_)
    bytes += scratch->retainedLookupRollbackStorageBytes();
  bytes +=
      retainedBytes(computeJournalMarks_) + retainedBytes(memoryJournalMarks_) +
      retainedBytes(portJournalMarks_) + retainedBytes(boundaryJournalMarks_) +
      retainedBytes(memoryPlanJournalMarks_) + retainedBytes(decisionDeltas_) +
      retainedBytes(affectedComputeMarks_) +
      retainedBytes(affectedMemoryMarks_) + retainedBytes(affectedPortMarks_) +
      retainedBytes(affectedBoundaryMarks_) +
      retainedBytes(affectedMemoryPlanMarks_) +
      retainedBytes(affectedNetMarks_) + retainedBytes(affectedComputes_) +
      retainedBytes(affectedMemories_) + retainedBytes(affectedPorts_) +
      retainedBytes(affectedBoundaries_) + retainedBytes(affectedMemoryPlans_) +
      retainedBytes(affectedNets_) + retainedBytes(touchedRoutes_) +
      retainedBytes(traversalDeltaMarks_) + retainedBytes(traversalRemoved_) +
      retainedBytes(traversalAdded_) + retainedBytes(touchedTraversals_);
  return bytes;
}

void SpatialCandidateScratch::beginTransaction() {
  resetTransaction();
  advanceEpoch(decisionEpoch_,
               {&computeJournalMarks_, &memoryJournalMarks_, &portJournalMarks_,
                &boundaryJournalMarks_, &memoryPlanJournalMarks_});
  advanceEpoch(affectedEpoch_, {&affectedComputeMarks_, &affectedMemoryMarks_,
                                &affectedPortMarks_, &affectedBoundaryMarks_,
                                &affectedMemoryPlanMarks_, &affectedNetMarks_});
  advanceEpoch(traversalEpoch_, {&traversalDeltaMarks_});
}

void SpatialCandidateScratch::resetTransaction() {
  for (PnrIndex net : touchedRoutes_)
    routeTransactions_[net].reset();
  touchedRoutes_.clear();
  touchedTraversals_.clear();
  decisionDeltas_.clear();
  affectedComputes_.clear();
  affectedMemories_.clear();
  affectedPorts_.clear();
  affectedBoundaries_.clear();
  affectedMemoryPlans_.clear();
  affectedNets_.clear();
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
          realizations.memoryActors().size())
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
  std::vector<PnrIndex> memoryOperationPlans(
      initialization.memoryOperationPlans.begin(),
      initialization.memoryOperationPlans.end());

  auto handshakeOwner = std::shared_ptr<const FrozenSpatialHandshakeIndex>(
      problem, &problem->handshake());
  auto handshake = HandshakeCandidateState::create(std::move(handshakeOwner));
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
  std::uint64_t unroutedObligationCount = 0;
  for (const FrozenSpatialLogicalNet &net :
       problem->transfers().logicalNets()) {
    if (net.sinkCount >
        std::numeric_limits<std::uint64_t>::max() - unroutedObligationCount)
      return candidateError("unrouted obligation count overflows u64");
    unroutedObligationCount += net.sinkCount;
  }

  auto candidate = SpatialCandidateStateHandle(new SpatialCandidateState(
      std::move(problem), std::move(computeBindings), std::move(memoryBindings),
      std::move(portAttachments), std::move(graphBoundaryAttachments),
      std::move(memoryOperationPlans), std::move(routeTrees),
      std::move(*handshake), std::move(*routeResources),
      unroutedObligationCount));
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
  for (PnrIndex index = 0; index < candidate->routeTrees_.size(); ++index)
    if (llvm::Error error = candidate->validateLogicalNet(index))
      return std::move(error);
  if (llvm::Error error = addInitialHandshakeSelections(
          candidate->problem(), candidate->computeBindings_,
          candidate->portAttachments_, candidate->memoryOperationPlans_,
          *candidate->handshake_))
    return std::move(error);
  if (llvm::Error error = candidate->verify())
    return std::move(error);
  return candidate;
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
  return terminalPayloadWidth(
      problem_->transfers().logicalNetSourceBindings()[logicalNet]);
}

const RouteTreeState &
SpatialCandidateState::routeTree(PnrIndex logicalNet) const {
  assert(logicalNet < routeTrees_.size());
  return *routeTrees_[logicalNet];
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

llvm::Error SpatialCandidateState::verify() const {
  if (activeTransaction_)
    return candidateError("cannot verify during a move");
  if (!problem_ || !handshake_ ||
      computeBindings_.size() !=
          problem_->realizations().computeRealizations().size() ||
      memoryBindings_.size() !=
          problem_->realizations().memoryRealizations().size() ||
      portAttachments_.size() != problem_->ports().portDemands().size() ||
      graphBoundaryAttachments_.size() !=
          problem_->ports().graphBoundaries().size() ||
      memoryOperationPlans_.size() !=
          problem_->realizations().memoryActors().size() ||
      routeTrees_.size() != problem_->transfers().logicalNets().size())
    return candidateError("candidate shape does not match its frozen problem");
  for (PnrIndex index = 0; index < computeBindings_.size(); ++index)
    if (llvm::Error error = validateComputeBinding(index))
      return error;
  for (PnrIndex index = 0; index < memoryBindings_.size(); ++index)
    if (llvm::Error error = validateMemoryBinding(index))
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
  for (PnrIndex index = 0; index < routeTrees_.size(); ++index)
    if (llvm::Error error = validateLogicalNet(index))
      return error;
  std::uint64_t expectedUnroutedObligationCount = 0;
  const auto logicalNets = problem_->transfers().logicalNets();
  for (PnrIndex index = 0; index < routeTrees_.size(); ++index) {
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
  if (llvm::Error error = routeResources_.verify(routeTrees_))
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
      scratch.routeTransactions_.size() != routeTrees_.size() ||
      scratch.traversalDeltaMarks_.size() !=
          problem_->routing().traversals().size())
    return candidateError("scratch was not prepared for this candidate");

  scratch.beginTransaction();
  auto handshakeTransaction =
      handshake_->beginTransaction(scratch.handshakeScratch_);
  if (!handshakeTransaction)
    return handshakeTransaction.takeError();
  scratch.handshakeTransaction_.emplace(std::move(*handshakeTransaction));
  return SpatialMoveTransaction(shared_from_this(), scratch);
}

SpatialMoveTransaction::SpatialMoveTransaction(
    SpatialCandidateStateHandle state, SpatialCandidateScratch &scratch)
    : state_(std::move(state)), scratch_(&scratch),
      initialUnroutedObligationCount_(state_->unroutedObligationCount_) {
  state_->activeTransaction_ = this;
  scratch_->activeTransaction_ = this;
}

SpatialMoveTransaction::SpatialMoveTransaction(
    SpatialMoveTransaction &&other) noexcept
    : state_(std::move(other.state_)), scratch_(other.scratch_),
      closed_(other.closed_), cycle_(other.cycle_),
      routeDeltasCollected_(other.routeDeltasCollected_),
      routeViolationApplied_(other.routeViolationApplied_),
      initialUnroutedObligationCount_(other.initialUnroutedObligationCount_) {
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
       old.placement, old.instructionContext});
}

void SpatialMoveTransaction::recordMemory(PnrIndex realization) {
  if (scratch_->memoryJournalMarks_[realization] == scratch_->decisionEpoch_)
    return;
  scratch_->memoryJournalMarks_[realization] = scratch_->decisionEpoch_;
  scratch_->decisionDeltas_.push_back(
      {SpatialCandidateScratch::DecisionKind::MemoryBinding, realization,
       state_->memoryBindings_[realization].placement, 0});
}

void SpatialMoveTransaction::recordPort(PnrIndex demand) {
  if (scratch_->portJournalMarks_[demand] == scratch_->decisionEpoch_)
    return;
  scratch_->portJournalMarks_[demand] = scratch_->decisionEpoch_;
  scratch_->decisionDeltas_.push_back(
      {SpatialCandidateScratch::DecisionKind::PortAttachment, demand,
       state_->portAttachments_[demand], 0});
}

void SpatialMoveTransaction::recordBoundary(PnrIndex boundary) {
  if (scratch_->boundaryJournalMarks_[boundary] == scratch_->decisionEpoch_)
    return;
  scratch_->boundaryJournalMarks_[boundary] = scratch_->decisionEpoch_;
  scratch_->decisionDeltas_.push_back(
      {SpatialCandidateScratch::DecisionKind::GraphBoundaryAttachment, boundary,
       state_->graphBoundaryAttachments_[boundary], 0});
}

void SpatialMoveTransaction::recordMemoryPlan(PnrIndex actor) {
  if (scratch_->memoryPlanJournalMarks_[actor] == scratch_->decisionEpoch_)
    return;
  scratch_->memoryPlanJournalMarks_[actor] = scratch_->decisionEpoch_;
  scratch_->decisionDeltas_.push_back(
      {SpatialCandidateScratch::DecisionKind::MemoryOperationPlan, actor,
       state_->memoryOperationPlans_[actor], 0});
}

void SpatialMoveTransaction::markCompute(PnrIndex realization) {
  if (scratch_->affectedComputeMarks_[realization] !=
      scratch_->affectedEpoch_) {
    scratch_->affectedComputeMarks_[realization] = scratch_->affectedEpoch_;
    scratch_->affectedComputes_.push_back(realization);
  }
}

void SpatialMoveTransaction::markMemory(PnrIndex realization) {
  if (scratch_->affectedMemoryMarks_[realization] != scratch_->affectedEpoch_) {
    scratch_->affectedMemoryMarks_[realization] = scratch_->affectedEpoch_;
    scratch_->affectedMemories_.push_back(realization);
  }
  const auto offsets =
      state_->problem_->ports().memoryRealizationDemandOffsets();
  for (PnrIndex demand :
       state_->problem_->ports().memoryRealizationDemands().slice(
           offsets[realization],
           offsets[realization + 1] - offsets[realization]))
    markPort(demand);
  const auto &record =
      state_->problem_->realizations().memoryRealizations()[realization];
  for (PnrIndex local = 0; local < record.actorCount; ++local)
    markMemoryPlan(record.actorOffset + local);
}

void SpatialMoveTransaction::markPort(PnrIndex demand) {
  if (scratch_->affectedPortMarks_[demand] != scratch_->affectedEpoch_) {
    scratch_->affectedPortMarks_[demand] = scratch_->affectedEpoch_;
    scratch_->affectedPorts_.push_back(demand);
  }
  markNet(state_->problem_->ports().portDemands()[demand].logicalNet);
}

void SpatialMoveTransaction::markBoundary(PnrIndex boundary) {
  if (scratch_->affectedBoundaryMarks_[boundary] != scratch_->affectedEpoch_) {
    scratch_->affectedBoundaryMarks_[boundary] = scratch_->affectedEpoch_;
    scratch_->affectedBoundaries_.push_back(boundary);
  }
  markNet(state_->problem_->ports().graphBoundaries()[boundary].logicalNet);
}

void SpatialMoveTransaction::markMemoryPlan(PnrIndex actor) {
  if (scratch_->affectedMemoryPlanMarks_[actor] == scratch_->affectedEpoch_)
    return;
  scratch_->affectedMemoryPlanMarks_[actor] = scratch_->affectedEpoch_;
  scratch_->affectedMemoryPlans_.push_back(actor);
}

void SpatialMoveTransaction::markNet(PnrIndex logicalNet) {
  if (scratch_->affectedNetMarks_[logicalNet] == scratch_->affectedEpoch_)
    return;
  scratch_->affectedNetMarks_[logicalNet] = scratch_->affectedEpoch_;
  scratch_->affectedNets_.push_back(logicalNet);
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
  state_->computeBindings_[realization] = {placement, instructionContext};
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
  recordMemory(realization);
  markMemory(realization);
  state_->memoryBindings_[realization].placement = placement;
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
  const PnrIndex old = state_->graphBoundaryAttachments_[boundary];
  if (old == attachmentOption)
    return llvm::Error::success();
  recordBoundary(boundary);
  markBoundary(boundary);
  state_->graphBoundaryAttachments_[boundary] = attachmentOption;
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
  state_->memoryOperationPlans_[actor] = plan;
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
  return llvm::Error::success();
}

llvm::Expected<bool> SpatialMoveTransaction::close() {
  if (!scratch_)
    return candidateError("move is no longer active");
  if (closed_)
    return !cycle_;
  if (llvm::Error error = collectRouteTraversalDeltas())
    return std::move(error);
  if (llvm::Error error = validateAffectedState())
    return std::move(error);
  auto closure = scratch_->handshakeTransaction_->close();
  if (!closure)
    return closure.takeError();
  closed_ = true;
  cycle_ = !*closure;
  return !cycle_;
}

llvm::ArrayRef<PnrIndex> SpatialMoveTransaction::cycleWitness() const {
  if (!scratch_ || !scratch_->handshakeTransaction_)
    return {};
  return scratch_->handshakeTransaction_->cycleWitness();
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
  acceptAppliedRouteResources();
  finish();
  return llvm::Error::success();
}

void SpatialMoveTransaction::rollback() noexcept {
  if (!scratch_)
    return;
  rollbackAppliedRouteResources();
  for (PnrIndex logicalNet : llvm::reverse(scratch_->touchedRoutes_))
    if (scratch_->routeTransactions_[logicalNet])
      scratch_->routeTransactions_[logicalNet]->rollback();
  if (scratch_->handshakeTransaction_)
    scratch_->handshakeTransaction_->rollback();

  for (const SpatialCandidateScratch::DecisionDelta &delta :
       llvm::reverse(scratch_->decisionDeltas_)) {
    switch (delta.kind) {
    case SpatialCandidateScratch::DecisionKind::ComputeBinding:
      state_->computeBindings_[delta.index] = {delta.oldValue0,
                                               delta.oldValue1};
      break;
    case SpatialCandidateScratch::DecisionKind::MemoryBinding:
      state_->memoryBindings_[delta.index].placement = delta.oldValue0;
      break;
    case SpatialCandidateScratch::DecisionKind::PortAttachment:
      state_->portAttachments_[delta.index] = delta.oldValue0;
      break;
    case SpatialCandidateScratch::DecisionKind::GraphBoundaryAttachment:
      state_->graphBoundaryAttachments_[delta.index] = delta.oldValue0;
      break;
    case SpatialCandidateScratch::DecisionKind::MemoryOperationPlan:
      state_->memoryOperationPlans_[delta.index] = delta.oldValue0;
      break;
    }
  }
  finish();
}

void SpatialMoveTransaction::finish() {
  state_->activeTransaction_ = nullptr;
  scratch_->activeTransaction_ = nullptr;
  scratch_->resetTransaction();
  scratch_ = nullptr;
  state_.reset();
}
