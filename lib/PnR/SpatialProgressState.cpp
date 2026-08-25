#include "PnR/SpatialProgressState.h"

#include "PnR/RouteTreeState.h"
#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialPnrProblem.h"

#include "SpatialProgressAnalysis.h"
#include "SpatialProgressIndex.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

using namespace loom::pnr;

namespace {

constexpr llvm::StringLiteral candidateArtifact = "SpatialProgressState";
constexpr PnrCapacityContext logicalNetCountContext{
    candidateArtifact, "net_owner_refcounts", "logical_nets",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext ownerCountContext{
    candidateArtifact, "owner_logical_net_counts", "finite_buffer_owners",
    PnrCapacityMeasure::Count};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_progress_state_invalid: " + message);
}

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

std::size_t retainedSparseBytes(
    const std::vector<llvm::DenseMap<PnrIndex, PnrIndex>> &values) {
  std::size_t bytes = retainedBytes(values);
  for (const auto &value : values)
    bytes += value.getMemorySize();
  return bytes;
}

bool sparseRefcountsEqual(
    const std::vector<llvm::DenseMap<PnrIndex, PnrIndex>> &lhs,
    const std::vector<llvm::DenseMap<PnrIndex, PnrIndex>> &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    if (lhs[index].size() != rhs[index].size())
      return false;
    for (const auto &[key, value] : lhs[index]) {
      const auto found = rhs[index].find(key);
      if (found == rhs[index].end() || found->second != value)
        return false;
    }
  }
  return true;
}

llvm::Expected<const FrozenSpatialAttachmentOption *>
selectedAttachment(const SpatialCandidateState &candidate,
                   FrozenSpatialTerminalBinding terminal) {
  PnrIndex option = getInvalidPnrIndex();
  switch (terminal.kind) {
  case FrozenSpatialTerminalBindingKind::PortDemand:
    if (terminal.index >= candidate.problem().ports().portDemands().size())
      return invalid("terminal PortDemand is out of range");
    option = candidate.portAttachment(terminal.index);
    break;
  case FrozenSpatialTerminalBindingKind::GraphBoundary:
    if (terminal.index >= candidate.problem().ports().graphBoundaries().size())
      return invalid("terminal graph boundary is out of range");
    option = candidate.graphBoundaryAttachment(terminal.index);
    break;
  }
  if (option >= candidate.problem().ports().attachmentOptions().size())
    return invalid("selected terminal attachment is out of range");
  return &candidate.problem().ports().attachmentOptions()[option];
}

template <typename Callback>
llvm::Error forEachSelectedTraversal(const SpatialCandidateState &candidate,
                                     PnrIndex logicalNet,
                                     Callback &&callback) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const auto nets = problem.transfers().logicalNets();
  const auto sources = problem.transfers().logicalNetSourceBindings();
  const auto sinks = problem.transfers().logicalNetSinkBindings();
  if (logicalNet >= nets.size() || logicalNet >= sources.size())
    return invalid("logical net is out of range");
  const RouteTreeState &tree = candidate.routeTree(logicalNet);
  if (candidate.usesRegisterFifo(logicalNet) || !tree.isRouted())
    return llvm::Error::success();

  auto source = selectedAttachment(candidate, sources[logicalNet]);
  if (!source)
    return source.takeError();
  if ((*source)->localTraversal)
    if (llvm::Error error = callback(
            *(*source)->localTraversal,
            SpatialProgressRouteAnchor{SpatialProgressRouteAnchorKind::
                                           SourceAttachment,
                                       logicalNet, *(*source)->localTraversal,
                                       (*source)->endpoint,
                                       getInvalidPnrIndex()}))
      return error;

  for (const RouteTreeNode &node : tree.nodeStorage()) {
    if (!node.isActive() || node.parentArc == getInvalidPnrIndex())
      continue;
    if (node.parentArc >= problem.routing().routingArcs().size())
      return invalid("selected RouteTree parent arc is out of range");
    const PnrIndex traversal =
        problem.routing().routingArcs()[node.parentArc].traversal;
    if (llvm::Error error = callback(
            traversal,
            SpatialProgressRouteAnchor{SpatialProgressRouteAnchorKind::
                                           RouteTreeArc,
                                       logicalNet, traversal, node.endpoint,
                                       getInvalidPnrIndex()}))
      return error;
  }

  const FrozenSpatialLogicalNet &net = nets[logicalNet];
  if (net.sinkOffset > sinks.size() ||
      net.sinkCount > sinks.size() - net.sinkOffset)
    return invalid("logical net sink terminal range is inconsistent");
  for (PnrIndex sink = 0; sink < net.sinkCount; ++sink) {
    auto attachment = selectedAttachment(candidate, sinks[net.sinkOffset + sink]);
    if (!attachment)
      return attachment.takeError();
    if (!(*attachment)->localTraversal)
      continue;
    if (llvm::Error error = callback(
            *(*attachment)->localTraversal,
            SpatialProgressRouteAnchor{SpatialProgressRouteAnchorKind::
                                           SinkAttachment,
                                       logicalNet,
                                       *(*attachment)->localTraversal,
                                       (*attachment)->endpoint, sink}))
      return error;
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<SpatialProgressState>
SpatialProgressState::create(const SpatialCandidateState &candidate) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  auto logicalNetCount = checkedPnrIndex(
      logicalNetCountContext, problem.transfers().logicalNets().size());
  if (!logicalNetCount)
    return logicalNetCount.takeError();
  auto ownerCount = checkedPnrIndex(
      ownerCountContext, problem.progressIndex().finiteBufferOwners().size());
  if (!ownerCount)
    return ownerCount.takeError();
  const std::size_t logicalNetWordCount =
      (static_cast<std::size_t>(*logicalNetCount) + 63) / 64;
  if (logicalNetWordCount != 0 &&
      static_cast<std::size_t>(*ownerCount) >
          std::numeric_limits<std::size_t>::max() / logicalNetWordCount)
    return invalid("owner logical-net bitset exceeds native size_t");

  SpatialProgressState result(
      problem, *logicalNetCount, *ownerCount, logicalNetWordCount,
      std::vector<llvm::DenseMap<PnrIndex, PnrIndex>>(*logicalNetCount),
      std::vector<PnrIndex>(*ownerCount, 0),
      std::vector<std::uint64_t>(static_cast<std::size_t>(*ownerCount) *
                                     logicalNetWordCount,
                                 0),
      std::vector<std::uint64_t>((static_cast<std::size_t>(*ownerCount) + 63) /
                                     64,
                                 0),
      std::vector<std::uint64_t>(*logicalNetCount, 0));

  for (PnrIndex logicalNet = 0; logicalNet < *logicalNetCount; ++logicalNet) {
    if (llvm::Error error = forEachSelectedTraversal(
            candidate, logicalNet,
            [&](PnrIndex traversal,
                const SpatialProgressRouteAnchor &) -> llvm::Error {
              return result.applyTraversalDelta(logicalNet, traversal, 0, 1);
            }))
      return std::move(error);
    if (llvm::Error error =
            result.refreshLogicalNetRouteDependencies(candidate, logicalNet))
      return std::move(error);
  }
  return result;
}

PnrIndex SpatialProgressState::finiteBufferOwnerLogicalNetCount(
    PnrIndex owner) const {
  assert(owner < ownerLogicalNetCounts_.size());
  return ownerLogicalNetCounts_[owner];
}

bool SpatialProgressState::finiteBufferOwnerConflicts(PnrIndex owner) const {
  assert(owner < ownerCount_);
  return (conflictingOwnerBits_[owner / 64] &
          (std::uint64_t{1} << (owner % 64))) != 0;
}

bool SpatialProgressState::logicalNetSelectsFiniteBufferOwner(
    PnrIndex logicalNet, PnrIndex owner) const {
  assert(logicalNet < logicalNetCount_ && owner < ownerCount_);
  if (logicalNetWordCount_ == 0)
    return false;
  return (ownerLogicalNetBits_[static_cast<std::size_t>(owner) *
                                  logicalNetWordCount_ +
                              logicalNet / 64] &
          (std::uint64_t{1} << (logicalNet % 64))) != 0;
}

std::uint64_t SpatialProgressState::logicalNetRouteDependencyViolationCount(
    PnrIndex logicalNet) const {
  assert(logicalNet < netRouteDependencyViolationCounts_.size());
  return netRouteDependencyViolationCounts_[logicalNet];
}

std::size_t SpatialProgressState::retainedStorageBytes() const {
  return retainedSparseBytes(netOwnerRefcounts_) +
         retainedBytes(ownerLogicalNetCounts_) +
         retainedBytes(ownerLogicalNetBits_) +
         retainedBytes(conflictingOwnerBits_) +
         retainedBytes(netRouteDependencyViolationCounts_);
}

llvm::Error SpatialProgressState::applyTraversalDelta(
    PnrIndex logicalNet, PnrIndex traversal, PnrIndex removed,
    PnrIndex added) {
  if (!problem_ || logicalNet >= logicalNetCount_ ||
      traversal >= problem_->routing().traversals().size())
    return invalid("traversal delta index is out of range");
  const PnrIndex owner = problem_->progressIndex().traversalOwner(traversal);
  if (owner == getInvalidPnrIndex())
    return llvm::Error::success();
  if (owner >= ownerCount_)
    return invalid("traversal finite-buffer owner is out of range");

  auto &refcounts = netOwnerRefcounts_[logicalNet];
  const auto found = refcounts.find(owner);
  const PnrIndex refcount = found == refcounts.end() ? 0 : found->second;
  if (removed > refcount)
    return invalid("finite-buffer owner refcount removal underflows");
  const PnrIndex remaining = refcount - removed;
  if (added > std::numeric_limits<PnrIndex>::max() - remaining)
    return invalid("finite-buffer owner refcount addition overflows PnrIndex");
  const PnrIndex next = remaining + added;
  const bool activate = refcount == 0 && next != 0;
  const bool deactivate = refcount != 0 && next == 0;
  if (!activate && !deactivate) {
    if (next != 0)
      found->second = next;
    return llvm::Error::success();
  }

  const PnrIndex oldOwnerCount = ownerLogicalNetCounts_[owner];
  if (activate && oldOwnerCount == std::numeric_limits<PnrIndex>::max())
    return invalid("finite-buffer logical-net count overflows PnrIndex");
  if (deactivate && oldOwnerCount == 0)
    return invalid("finite-buffer logical-net count underflows");
  const PnrIndex nextOwnerCount =
      activate ? oldOwnerCount + 1 : oldOwnerCount - 1;
  if (oldOwnerCount == 1 && nextOwnerCount == 2 &&
      sharedFiniteBufferConflictCount_ ==
          std::numeric_limits<std::uint64_t>::max())
    return invalid("shared finite-buffer conflict count exceeds u64");
  if (oldOwnerCount == 2 && nextOwnerCount == 1 &&
      sharedFiniteBufferConflictCount_ == 0)
    return invalid("shared finite-buffer conflict count underflows");

  std::uint64_t &netWord =
      ownerLogicalNetBits_[static_cast<std::size_t>(owner) *
                               logicalNetWordCount_ +
                           logicalNet / 64];
  const std::uint64_t netMask = std::uint64_t{1} << (logicalNet % 64);
  std::uint64_t &conflictWord = conflictingOwnerBits_[owner / 64];
  const std::uint64_t conflictMask = std::uint64_t{1} << (owner % 64);
  if (activate) {
    refcounts.try_emplace(owner, next);
    netWord |= netMask;
  } else {
    refcounts.erase(owner);
    netWord &= ~netMask;
  }
  ownerLogicalNetCounts_[owner] = nextOwnerCount;
  if (oldOwnerCount == 1 && nextOwnerCount == 2) {
    ++sharedFiniteBufferConflictCount_;
    conflictWord |= conflictMask;
  } else if (oldOwnerCount == 2 && nextOwnerCount == 1) {
    --sharedFiniteBufferConflictCount_;
    conflictWord &= ~conflictMask;
  }
  return llvm::Error::success();
}

void SpatialProgressState::revertTraversalDelta(
    PnrIndex logicalNet, PnrIndex traversal, PnrIndex removed,
    PnrIndex added) noexcept {
  llvm::cantFail(
      applyTraversalDelta(logicalNet, traversal, added, removed));
}

llvm::Expected<std::uint64_t>
SpatialProgressState::projectLogicalNetRouteDependencies(
    const SpatialCandidateState &candidate, PnrIndex logicalNet) const {
  const auto nets = candidate.problem().transfers().logicalNets();
  if (&candidate.problem() != problem_ || logicalNet >= nets.size())
    return invalid("route dependency projection names a foreign logical net");
  std::uint64_t count = 0;
  for (PnrIndex dependent = 0; dependent < nets[logicalNet].sinkCount;
       ++dependent) {
    auto prerequisites = spatialSinkProgressDependencies(
        candidate.problem(), logicalNet, dependent);
    if (!prerequisites)
      return prerequisites.takeError();
    for (const FrozenSpatialProgressPrerequisite &prerequisite :
         *prerequisites) {
      auto satisfied = spatialRouteProgressDependencySatisfied(
          candidate, logicalNet, prerequisite, dependent);
      if (!satisfied)
        return satisfied.takeError();
      if (*satisfied)
        continue;
      if (count == std::numeric_limits<std::uint64_t>::max())
        return invalid("logical-net route dependency count exceeds u64");
      ++count;
    }
  }
  return count;
}

llvm::Error SpatialProgressState::refreshLogicalNetRouteDependencies(
    const SpatialCandidateState &candidate, PnrIndex logicalNet) {
  if (logicalNet >= netRouteDependencyViolationCounts_.size())
    return invalid("route dependency cache index is out of range");
  auto projected = projectLogicalNetRouteDependencies(candidate, logicalNet);
  if (!projected)
    return projected.takeError();
  const std::uint64_t old =
      netRouteDependencyViolationCounts_[logicalNet];
  if (old > routeDependencyViolationCount_)
    return invalid("route dependency cache exceeds its total");
  const std::uint64_t base = routeDependencyViolationCount_ - old;
  if (*projected > std::numeric_limits<std::uint64_t>::max() - base)
    return invalid("route dependency violation total exceeds u64");
  netRouteDependencyViolationCounts_[logicalNet] = *projected;
  routeDependencyViolationCount_ = base + *projected;
  return llvm::Error::success();
}

void SpatialProgressState::restoreLogicalNetRouteDependencyCount(
    PnrIndex logicalNet, std::uint64_t count) noexcept {
  assert(logicalNet < netRouteDependencyViolationCounts_.size());
  const std::uint64_t current =
      netRouteDependencyViolationCounts_[logicalNet];
  assert(current <= routeDependencyViolationCount_);
  const std::uint64_t base = routeDependencyViolationCount_ - current;
  assert(count <= std::numeric_limits<std::uint64_t>::max() - base);
  netRouteDependencyViolationCounts_[logicalNet] = count;
  routeDependencyViolationCount_ = base + count;
}

llvm::Expected<std::vector<SpatialFiniteBufferConflictWitness>>
SpatialProgressState::finiteBufferConflictWitnesses(
    const SpatialCandidateState &candidate) const {
  if (&candidate.problem() != problem_)
    return invalid("finite-buffer witness candidate is foreign");
  std::vector<SpatialFiniteBufferConflictWitness> witnesses;
  witnesses.reserve(sharedFiniteBufferConflictCount_);
  const auto owners = problem_->progressIndex().finiteBufferOwners();
  for (PnrIndex owner = 0; owner < ownerCount_; ++owner) {
    if (!finiteBufferOwnerConflicts(owner))
      continue;
    SpatialFiniteBufferConflictWitness witness;
    witness.ownerOrdinal = owner;
    witness.owner = owners[owner];
    const std::size_t bitOffset =
        static_cast<std::size_t>(owner) * logicalNetWordCount_;
    for (std::size_t word = 0; word < logicalNetWordCount_; ++word) {
      std::uint64_t bits = ownerLogicalNetBits_[bitOffset + word];
      while (bits != 0) {
        const unsigned bit = llvm::countr_zero(bits);
        const std::size_t logicalNet = word * 64 + bit;
        if (logicalNet >= logicalNetCount_)
          return invalid("finite-buffer owner bitset has an excess bit");
        witness.competingLogicalNets.push_back(
            static_cast<PnrIndex>(logicalNet));
        bits &= bits - 1;
      }
    }
    for (PnrIndex logicalNet : witness.competingLogicalNets) {
      if (llvm::Error error = forEachSelectedTraversal(
              candidate, logicalNet,
              [&](PnrIndex traversal,
                  const SpatialProgressRouteAnchor &anchor) -> llvm::Error {
                if (problem_->progressIndex().traversalOwner(traversal) ==
                    owner)
                  witness.routeAnchors.push_back(anchor);
                return llvm::Error::success();
              }))
        return std::move(error);
    }
    llvm::sort(witness.routeAnchors,
               [](const SpatialProgressRouteAnchor &lhs,
                  const SpatialProgressRouteAnchor &rhs) {
                 return std::tie(lhs.logicalNet, lhs.kind, lhs.sinkObligation,
                                 lhs.endpoint, lhs.traversal) <
                        std::tie(rhs.logicalNet, rhs.kind, rhs.sinkObligation,
                                 rhs.endpoint, rhs.traversal);
               });
    witnesses.push_back(std::move(witness));
  }
  if (witnesses.size() != sharedFiniteBufferConflictCount_)
    return invalid("finite-buffer witness count diverges from conflict state");
  return witnesses;
}

llvm::Error
SpatialProgressState::verify(const SpatialCandidateState &candidate) const {
  if (!problem_ || &candidate.problem() != problem_)
    return invalid("progress state is bound to a foreign candidate");
  auto expected = create(candidate);
  if (!expected)
    return expected.takeError();
  if (logicalNetCount_ != expected->logicalNetCount_ ||
      ownerCount_ != expected->ownerCount_ ||
      logicalNetWordCount_ != expected->logicalNetWordCount_ ||
      !sparseRefcountsEqual(netOwnerRefcounts_,
                            expected->netOwnerRefcounts_) ||
      ownerLogicalNetCounts_ != expected->ownerLogicalNetCounts_ ||
      ownerLogicalNetBits_ != expected->ownerLogicalNetBits_ ||
      conflictingOwnerBits_ != expected->conflictingOwnerBits_ ||
      netRouteDependencyViolationCounts_ !=
          expected->netRouteDependencyViolationCounts_ ||
      sharedFiniteBufferConflictCount_ !=
          expected->sharedFiniteBufferConflictCount_ ||
      routeDependencyViolationCount_ !=
          expected->routeDependencyViolationCount_)
    return invalid("incremental projection diverges from cold reconstruction");

  auto cold = spatialCandidateClosedWaitCount(candidate);
  if (!cold)
    return cold.takeError();
  if (*cold != hardProgressViolation())
    return invalid("incremental hard-progress total diverges from cold verifier");
  return llvm::Error::success();
}
