#include "PnR/SpatialProgressState.h"

#include "PnR/RouteTreeState.h"
#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialPnrProblem.h"

#include "Common/MappingDebugLog.h"

#include "SpatialProgressAnalysis.h"
#include "SpatialProgressIndex.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
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

void saturatingAdd(std::uint64_t &value, std::uint64_t added) {
  value = added > std::numeric_limits<std::uint64_t>::max() - value
              ? std::numeric_limits<std::uint64_t>::max()
              : value + added;
}

std::uint64_t elapsedNanoseconds(std::chrono::steady_clock::time_point begin) {
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - begin)
                           .count();
  return elapsed <= 0 ? 0 : static_cast<std::uint64_t>(elapsed);
}

class ProgressTimer final {
public:
  ProgressTimer(std::uint64_t *count, std::uint64_t *wallTimeNanoseconds)
      : count_(count), wallTimeNanoseconds_(wallTimeNanoseconds),
        begin_(count ? std::chrono::steady_clock::now()
                     : std::chrono::steady_clock::time_point{}) {}

  ~ProgressTimer() { finish(); }

  void finish() {
    if (!count_)
      return;
    saturatingAdd(*count_, 1);
    saturatingAdd(*wallTimeNanoseconds_, elapsedNanoseconds(begin_));
    count_ = nullptr;
    wallTimeNanoseconds_ = nullptr;
  }

private:
  std::uint64_t *count_;
  std::uint64_t *wallTimeNanoseconds_;
  std::chrono::steady_clock::time_point begin_;
};

void emitProgressStatistics(const SpatialProgressStatistics &statistics) {
  loom::mapping_debug::emit(
      loom::mapping_debug::Level::Summary,
      loom::mapping_debug::Stage::SpatialPnr,
      loom::mapping_debug::Event::Statistics, [&](llvm::json::Object &fields) {
        fields["statistics_kind"] = "spatial_progress_projection";
        fields["incremental_update_count"] = statistics.incrementalUpdateCount;
        fields["incremental_update_wall_time_nanoseconds"] =
            statistics.incrementalUpdateWallTimeNanoseconds;
        fields["cached_verification_count"] =
            statistics.cachedVerificationCount;
        fields["cold_verification_count"] = statistics.coldVerificationCount;
        fields["cold_verification_wall_time_nanoseconds"] =
            statistics.coldVerificationWallTimeNanoseconds;
        fields["cold_progress_scan_count"] = statistics.coldProgressScanCount;
        fields["cold_progress_scan_wall_time_nanoseconds"] =
            statistics.coldProgressScanWallTimeNanoseconds;
      });
}

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

std::string capacityProjectionDifference(
    const SpatialProgressNetCapacityProjection &incremental,
    const SpatialProgressNetCapacityProjection &cold) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << "incremental_owners=" << incremental.owners.size()
         << " cold_owners=" << cold.owners.size();
  const std::size_t common =
      std::min(incremental.owners.size(), cold.owners.size());
  for (std::size_t index = 0; index != common; ++index) {
    const auto &left = incremental.owners[index];
    const auto &right = cold.owners[index];
    if (left == right)
      continue;
    stream << " first_difference=" << index << " incremental={owner="
           << left.owner << ",channels=" << left.channelCount
           << ",feedback=" << left.initializedFeedbackChannelCount
           << ",repeated=" << left.repeatedWithinChannel
           << ",indeterminate=" << left.queueClassIndeterminate
           << ",classes=" << left.queueClasses.size() << "} cold={owner="
           << right.owner << ",channels=" << right.channelCount
           << ",feedback=" << right.initializedFeedbackChannelCount
           << ",repeated=" << right.repeatedWithinChannel
           << ",indeterminate=" << right.queueClassIndeterminate
           << ",classes=" << right.queueClasses.size() << '}';
    break;
  }
  stream.flush();
  return result;
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
  result.netCapacityProjections_.resize(*logicalNetCount);
  result.ownerChannelCounts_.assign(*ownerCount, 0);
  result.ownerInitializedFeedbackChannelCounts_.assign(*ownerCount, 0);
  result.ownerRepeatedChannelNetCounts_.assign(*ownerCount, 0);
  result.ownerIndeterminateQueueClassNetCounts_.assign(*ownerCount, 0);
  result.ownerQueueClassRefcounts_.resize(*ownerCount);
  result.traversalSelectionCounts_.assign(problem.routing().traversals().size(),
                                          0);
  result.ownerRouteAnchorCounts_.assign(*ownerCount, 0);
  result.capacityProofDebtOwnerBits_.assign(
      (static_cast<std::size_t>(*ownerCount) + 63) / 64, 0);

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
    auto previous =
        result.replaceLogicalNetCapacityProjection(candidate, logicalNet);
    if (!previous)
      return previous.takeError();
    if (!previous->owners.empty())
      return invalid("initial capacity proof projection was not empty");
  }
  result.statisticsEnabled_ =
      loom::mapping_debug::enabled(loom::mapping_debug::Level::Summary);
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

std::optional<PnrIndex>
SpatialProgressState::firstFiniteBufferConflictOwner() const {
  for (std::size_t word = 0; word < conflictingOwnerBits_.size(); ++word) {
    const std::uint64_t bits = conflictingOwnerBits_[word];
    if (bits == 0)
      continue;
    const std::size_t owner = word * 64 + llvm::countr_zero(bits);
    assert(owner < ownerCount_ && "conflicting-owner bitset has excess bits");
    return static_cast<PnrIndex>(owner);
  }
  return std::nullopt;
}

llvm::Error SpatialProgressState::enumerateFiniteBufferConflictOwners(
    std::vector<PnrIndex> &owners) const {
  owners.clear();
  owners.reserve(static_cast<std::size_t>(sharedFiniteBufferConflictCount_));
  for (std::size_t word = 0; word < conflictingOwnerBits_.size(); ++word) {
    std::uint64_t bits = conflictingOwnerBits_[word];
    while (bits != 0) {
      const unsigned bit = llvm::countr_zero(bits);
      const std::size_t owner = word * 64 + bit;
      if (owner >= ownerCount_)
        return invalid("conflicting-owner bitset has an excess bit");
      owners.push_back(static_cast<PnrIndex>(owner));
      bits &= bits - 1;
    }
  }
  if (owners.size() != sharedFiniteBufferConflictCount_)
    return invalid("conflicting-owner bitset disagrees with its total");
  return llvm::Error::success();
}

std::optional<PnrIndex>
SpatialProgressState::firstCapacityProofDebtOwner() const {
  for (std::size_t word = 0; word < capacityProofDebtOwnerBits_.size();
       ++word) {
    const std::uint64_t bits = capacityProofDebtOwnerBits_[word];
    if (bits == 0)
      continue;
    const std::size_t owner = word * 64 + llvm::countr_zero(bits);
    assert(owner < ownerCount_ && "capacity-debt bitset has excess bits");
    return static_cast<PnrIndex>(owner);
  }
  return std::nullopt;
}

llvm::Error SpatialProgressState::enumerateCapacityProofDebtOwners(
    std::vector<PnrIndex> &owners) const {
  owners.clear();
  owners.reserve(static_cast<std::size_t>(capacityProofDebtWitnessCount_));
  for (std::size_t word = 0; word < capacityProofDebtOwnerBits_.size();
       ++word) {
    std::uint64_t bits = capacityProofDebtOwnerBits_[word];
    while (bits != 0) {
      const unsigned bit = llvm::countr_zero(bits);
      const std::size_t owner = word * 64 + bit;
      if (owner >= ownerCount_)
        return invalid("capacity-debt bitset has an excess bit");
      owners.push_back(static_cast<PnrIndex>(owner));
      bits &= bits - 1;
    }
  }
  if (owners.size() != capacityProofDebtWitnessCount_)
    return invalid("capacity-debt bitset disagrees with its total");
  return llvm::Error::success();
}

std::optional<PnrIndex>
SpatialProgressState::firstCapacityShortfallOwner() const {
  for (PnrIndex owner = 0; owner < ownerCount_; ++owner)
    if (ownerCapacityShortfall(owner) != 0)
      return owner;
  return std::nullopt;
}

llvm::Error SpatialProgressState::enumerateCapacityShortfallOwners(
    std::vector<PnrIndex> &owners) const {
  owners.clear();
  owners.reserve(static_cast<std::size_t>(capacityShortfallOwnerCount_));
  for (PnrIndex owner = 0; owner < ownerCount_; ++owner)
    if (ownerCapacityShortfall(owner) != 0)
      owners.push_back(owner);
  if (owners.size() != capacityShortfallOwnerCount_)
    return invalid("capacity-shortfall owner inventory disagrees with its "
                   "total");
  return llvm::Error::success();
}

std::uint64_t SpatialProgressState::logicalNetRouteDependencyViolationCount(
    PnrIndex logicalNet) const {
  assert(logicalNet < netRouteDependencyViolationCounts_.size());
  return netRouteDependencyViolationCounts_[logicalNet];
}

std::size_t SpatialProgressState::retainedStorageBytes() const {
  std::size_t bytes = retainedSparseBytes(netOwnerRefcounts_) +
                      retainedBytes(ownerLogicalNetCounts_) +
                      retainedBytes(ownerLogicalNetBits_) +
                      retainedBytes(conflictingOwnerBits_) +
                      retainedBytes(netRouteDependencyViolationCounts_) +
                      retainedBytes(netCapacityProjections_) +
                      retainedBytes(ownerChannelCounts_) +
                      retainedBytes(ownerInitializedFeedbackChannelCounts_) +
                      retainedBytes(ownerRepeatedChannelNetCounts_) +
                      retainedBytes(ownerIndeterminateQueueClassNetCounts_) +
                      retainedBytes(ownerQueueClassRefcounts_) +
                      retainedBytes(traversalSelectionCounts_) +
                      retainedBytes(ownerRouteAnchorCounts_) +
                      retainedBytes(capacityProofDebtOwnerBits_);
  for (const SpatialProgressNetCapacityProjection &projection :
       netCapacityProjections_)
    bytes += retainedBytes(projection.owners);
  for (const auto &classes : ownerQueueClassRefcounts_)
    bytes += classes.retainedBytes();
  return bytes;
}

bool SpatialProgressState::ownerHasCapacityProofDebt(PnrIndex owner) const {
  assert(owner < ownerCount_);
  const std::uint64_t channels = ownerChannelCounts_[owner];
  const bool virtualChannel =
      problem_->progressIndex().ownerQueueDisciplines()[owner] ==
      ::fabric::FifoQueueDiscipline::PerTagVirtualChannel;
  return channels != 0 &&
         (ownerRepeatedChannelNetCounts_[owner] != 0 ||
          (virtualChannel &&
           ownerIndeterminateQueueClassNetCounts_[owner] != 0));
}

std::uint64_t
SpatialProgressState::ownerCapacityShortfall(PnrIndex owner) const {
  assert(owner < ownerCount_);
  if (ownerHasCapacityProofDebt(owner))
    return 0;
  const std::uint64_t required = ownerLogicalNetCounts_[owner];
  const std::uint64_t selected =
      problem_->progressIndex().ownerSharedSlotCapacities()[owner];
  return required > selected ? required - selected : 0;
}

void SpatialProgressState::refreshOwnerCapacityObligation(
    PnrIndex owner, bool oldDebt, std::uint64_t oldShortfall,
    PnrIndex oldRouteAnchorCount) noexcept {
  assert(owner < ownerCount_);
  assert(!(oldDebt && oldShortfall != 0));
  const bool nextDebt = ownerHasCapacityProofDebt(owner);
  const std::uint64_t nextShortfall = ownerCapacityShortfall(owner);
  assert(!(nextDebt && nextShortfall != 0));
  const PnrIndex nextRouteAnchorCount = ownerRouteAnchorCounts_[owner];
  std::uint64_t &word = capacityProofDebtOwnerBits_[owner / 64];
  const std::uint64_t mask = std::uint64_t{1} << (owner % 64);
  if (oldDebt) {
    assert(capacityProofDebtWitnessCount_ != 0);
    --capacityProofDebtWitnessCount_;
    word &= ~mask;
  }
  if (oldShortfall != 0) {
    assert(capacityShortfallOwnerCount_ != 0);
    assert(capacityShortfall_ >= oldShortfall);
    --capacityShortfallOwnerCount_;
    capacityShortfall_ -= oldShortfall;
  }
  if (oldDebt || oldShortfall != 0) {
    assert(capacityObligationRouteAnchorCount_ >= oldRouteAnchorCount);
    capacityObligationRouteAnchorCount_ -= oldRouteAnchorCount;
  }
  if (nextDebt) {
    assert(capacityProofDebtWitnessCount_ !=
           std::numeric_limits<std::uint64_t>::max());
    ++capacityProofDebtWitnessCount_;
    word |= mask;
  }
  if (nextShortfall != 0) {
    assert(capacityShortfallOwnerCount_ !=
           std::numeric_limits<std::uint64_t>::max());
    assert(nextShortfall <= std::numeric_limits<std::uint64_t>::max() -
                                capacityShortfall_);
    ++capacityShortfallOwnerCount_;
    capacityShortfall_ += nextShortfall;
  }
  if (nextDebt || nextShortfall != 0) {
    assert(nextRouteAnchorCount <=
           std::numeric_limits<std::uint64_t>::max() -
               capacityObligationRouteAnchorCount_);
    capacityObligationRouteAnchorCount_ += nextRouteAnchorCount;
  }
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
  ProgressTimer timer(
      statisticsEnabled_ ? &statistics_.incrementalUpdateCount : nullptr,
      statisticsEnabled_ ? &statistics_.incrementalUpdateWallTimeNanoseconds
                         : nullptr);

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

  if (traversal >= traversalSelectionCounts_.size())
    return invalid("capacity proof traversal selection is out of range");
  const PnrIndex traversalRefcount = traversalSelectionCounts_[traversal];
  if (removed > traversalRefcount)
    return invalid("capacity proof traversal removal underflows");
  const PnrIndex traversalRemaining = traversalRefcount - removed;
  if (added > std::numeric_limits<PnrIndex>::max() - traversalRemaining)
    return invalid("capacity proof traversal addition overflows PnrIndex");
  const PnrIndex nextTraversalRefcount = traversalRemaining + added;
  const bool activateAnchor =
      traversalRefcount == 0 && nextTraversalRefcount != 0;
  const bool deactivateAnchor =
      traversalRefcount != 0 && nextTraversalRefcount == 0;
  const PnrIndex oldRouteAnchorCount = ownerRouteAnchorCounts_[owner];
  if (activateAnchor &&
      oldRouteAnchorCount == std::numeric_limits<PnrIndex>::max())
    return invalid("capacity proof route-anchor count exceeds PnrIndex");
  if (deactivateAnchor && oldRouteAnchorCount == 0)
    return invalid("capacity proof route-anchor count underflows");
  const bool oldDebt = ownerHasCapacityProofDebt(owner);
  const std::uint64_t oldShortfall = ownerCapacityShortfall(owner);

  const PnrIndex oldOwnerCount = ownerLogicalNetCounts_[owner];
  if (activate && oldOwnerCount == std::numeric_limits<PnrIndex>::max())
    return invalid("finite-buffer logical-net count overflows PnrIndex");
  if (deactivate && oldOwnerCount == 0)
    return invalid("finite-buffer logical-net count underflows");
  const PnrIndex nextOwnerCount =
      activate ? oldOwnerCount + 1
      : deactivate ? oldOwnerCount - 1
                   : oldOwnerCount;
  if (oldOwnerCount == 1 && nextOwnerCount == 2 &&
      sharedFiniteBufferConflictCount_ ==
          std::numeric_limits<std::uint64_t>::max())
    return invalid("shared finite-buffer conflict count exceeds u64");
  if (oldOwnerCount == 2 && nextOwnerCount == 1 &&
      sharedFiniteBufferConflictCount_ == 0)
    return invalid("shared finite-buffer conflict count underflows");

  traversalSelectionCounts_[traversal] = nextTraversalRefcount;
  ownerRouteAnchorCounts_[owner] =
      activateAnchor    ? oldRouteAnchorCount + 1
      : deactivateAnchor ? oldRouteAnchorCount - 1
                         : oldRouteAnchorCount;
  if (!activate && !deactivate) {
    if (next != 0)
      found->second = next;
    refreshOwnerCapacityObligation(owner, oldDebt, oldShortfall,
                                   oldRouteAnchorCount);
    return llvm::Error::success();
  }

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
  refreshOwnerCapacityObligation(owner, oldDebt, oldShortfall,
                                 oldRouteAnchorCount);
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
  ProgressTimer timer(
      statisticsEnabled_ ? &statistics_.incrementalUpdateCount : nullptr,
      statisticsEnabled_ ? &statistics_.incrementalUpdateWallTimeNanoseconds
                         : nullptr);
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
  ProgressTimer timer(
      statisticsEnabled_ ? &statistics_.incrementalUpdateCount : nullptr,
      statisticsEnabled_ ? &statistics_.incrementalUpdateWallTimeNanoseconds
                         : nullptr);
  const std::uint64_t current =
      netRouteDependencyViolationCounts_[logicalNet];
  assert(current <= routeDependencyViolationCount_);
  const std::uint64_t base = routeDependencyViolationCount_ - current;
  assert(count <= std::numeric_limits<std::uint64_t>::max() - base);
  netRouteDependencyViolationCounts_[logicalNet] = count;
  routeDependencyViolationCount_ = base + count;
}

llvm::Error SpatialProgressState::applyNetCapacityProjection(
    const SpatialProgressNetCapacityProjection &projection, bool add) {
  PnrIndex previous = getInvalidPnrIndex();
  for (const SpatialProgressOwnerCapacityUse &use : projection.owners) {
    if (use.owner >= ownerCount_ || use.channelCount == 0 ||
        use.initializedFeedbackChannelCount > use.channelCount)
      return invalid("capacity proof net projection is malformed");
    if (previous != getInvalidPnrIndex() && use.owner <= previous)
      return invalid("capacity proof net owners are not canonical");
    previous = use.owner;
    const bool virtualChannel =
        problem_->progressIndex().ownerQueueDisciplines()[use.owner] ==
        ::fabric::FifoQueueDiscipline::PerTagVirtualChannel;
    if ((!virtualChannel &&
         (use.queueClassIndeterminate || !use.queueClasses.empty())) ||
        (virtualChannel && !use.queueClassIndeterminate &&
         use.queueClasses.empty()))
      return invalid("capacity proof queue-class projection is malformed");
    for (std::size_t index = 0; index != use.queueClasses.size(); ++index) {
      if (index != 0 &&
          ::fabric::comparePhysicalTagValues(use.queueClasses[index - 1],
                                             use.queueClasses[index]) >= 0)
        return invalid("capacity proof queue classes are not canonical");
      const auto found =
          ownerQueueClassRefcounts_[use.owner].find(use.queueClasses[index]);
      const PnrIndex refcount =
          found == ownerQueueClassRefcounts_[use.owner].end() ? 0
                                                              : found->second;
      if ((add && refcount == std::numeric_limits<PnrIndex>::max()) ||
          (!add && refcount == 0))
        return invalid(add ? "capacity proof queue-class refcount overflows"
                           : "capacity proof queue-class refcount underflows");
    }
    if (add) {
      if (use.channelCount >
              std::numeric_limits<std::uint64_t>::max() -
                  ownerChannelCounts_[use.owner] ||
          use.initializedFeedbackChannelCount >
              std::numeric_limits<std::uint64_t>::max() -
                  ownerInitializedFeedbackChannelCounts_[use.owner] ||
          (use.repeatedWithinChannel &&
           ownerRepeatedChannelNetCounts_[use.owner] ==
               std::numeric_limits<std::uint64_t>::max()) ||
          (use.queueClassIndeterminate &&
           ownerIndeterminateQueueClassNetCounts_[use.owner] ==
               std::numeric_limits<std::uint64_t>::max()))
        return invalid("capacity proof owner aggregate exceeds u64");
    } else if (use.channelCount > ownerChannelCounts_[use.owner] ||
               use.initializedFeedbackChannelCount >
                   ownerInitializedFeedbackChannelCounts_[use.owner] ||
               (use.repeatedWithinChannel &&
                ownerRepeatedChannelNetCounts_[use.owner] == 0) ||
               (use.queueClassIndeterminate &&
                ownerIndeterminateQueueClassNetCounts_[use.owner] == 0)) {
      return invalid("capacity proof owner aggregate underflows");
    }
  }

  for (const SpatialProgressOwnerCapacityUse &use : projection.owners) {
    const bool oldDebt = ownerHasCapacityProofDebt(use.owner);
    const std::uint64_t oldShortfall = ownerCapacityShortfall(use.owner);
    const PnrIndex oldRouteAnchorCount = ownerRouteAnchorCounts_[use.owner];
    if (add) {
      ownerChannelCounts_[use.owner] += use.channelCount;
      ownerInitializedFeedbackChannelCounts_[use.owner] +=
          use.initializedFeedbackChannelCount;
      ownerRepeatedChannelNetCounts_[use.owner] +=
          use.repeatedWithinChannel;
      ownerIndeterminateQueueClassNetCounts_[use.owner] +=
          use.queueClassIndeterminate;
      for (const llvm::APInt &queueClass : use.queueClasses)
        ++ownerQueueClassRefcounts_[use.owner][queueClass];
    } else {
      ownerChannelCounts_[use.owner] -= use.channelCount;
      ownerInitializedFeedbackChannelCounts_[use.owner] -=
          use.initializedFeedbackChannelCount;
      ownerRepeatedChannelNetCounts_[use.owner] -=
          use.repeatedWithinChannel;
      ownerIndeterminateQueueClassNetCounts_[use.owner] -=
          use.queueClassIndeterminate;
      for (const llvm::APInt &queueClass : use.queueClasses) {
        auto position = ownerQueueClassRefcounts_[use.owner].find(queueClass);
        assert(position != ownerQueueClassRefcounts_[use.owner].end() &&
               position->second != 0);
        if (--position->second == 0)
          ownerQueueClassRefcounts_[use.owner].erase(position);
      }
    }
    refreshOwnerCapacityObligation(use.owner, oldDebt, oldShortfall,
                                   oldRouteAnchorCount);
  }
  return llvm::Error::success();
}

llvm::Expected<SpatialProgressNetCapacityProjection>
SpatialProgressState::replaceLogicalNetCapacityProjection(
    const SpatialCandidateState &candidate, PnrIndex logicalNet,
    const RouteTreeState *route) {
  if (&candidate.problem() != problem_ ||
      logicalNet >= netCapacityProjections_.size())
    return invalid("capacity proof projection names a foreign logical net");
  auto projected =
      projectSpatialNetCapacityProofInputs(candidate, logicalNet, route);
  if (!projected)
    return projected.takeError();
  SpatialProgressNetCapacityProjection previous =
      std::move(netCapacityProjections_[logicalNet]);
  if (llvm::Error error = applyNetCapacityProjection(previous, false)) {
    netCapacityProjections_[logicalNet] = std::move(previous);
    return std::move(error);
  }
  if (llvm::Error error = applyNetCapacityProjection(*projected, true)) {
    llvm::cantFail(applyNetCapacityProjection(previous, true));
    netCapacityProjections_[logicalNet] = std::move(previous);
    return std::move(error);
  }
  netCapacityProjections_[logicalNet] = std::move(*projected);
  return previous;
}

void SpatialProgressState::restoreLogicalNetCapacityProjection(
    PnrIndex logicalNet,
    SpatialProgressNetCapacityProjection projection) noexcept {
  assert(logicalNet < netCapacityProjections_.size());
  llvm::cantFail(
      applyNetCapacityProjection(netCapacityProjections_[logicalNet], false));
  llvm::cantFail(applyNetCapacityProjection(projection, true));
  netCapacityProjections_[logicalNet] = std::move(projection);
}

llvm::Expected<std::vector<SpatialFiniteBufferConflictWitness>>
SpatialProgressState::finiteBufferConflictWitnesses(
    const SpatialCandidateState &candidate) const {
  if (&candidate.problem() != problem_)
    return invalid("finite-buffer witness candidate is foreign");
  std::vector<PnrIndex> owners;
  if (llvm::Error error = enumerateFiniteBufferConflictOwners(owners))
    return std::move(error);
  std::vector<SpatialFiniteBufferConflictWitness> witnesses;
  witnesses.reserve(owners.size());
  for (PnrIndex owner : owners) {
    witnesses.emplace_back();
    if (llvm::Error error = rebuildFiniteBufferConflictWitness(
            candidate, owner, witnesses.back()))
      return std::move(error);
  }
  if (witnesses.size() != sharedFiniteBufferConflictCount_)
    return invalid("finite-buffer witness count diverges from conflict state");
  return witnesses;
}

llvm::Error SpatialProgressState::rebuildFiniteBufferConflictWitness(
    const SpatialCandidateState &candidate, PnrIndex owner,
    SpatialFiniteBufferConflictWitness &witness) const {
  witness.ownerOrdinal = getInvalidPnrIndex();
  witness.competingLogicalNets.clear();
  witness.routeAnchors.clear();
  if (&candidate.problem() != problem_)
    return invalid("finite-buffer witness candidate is foreign");
  const auto owners = problem_->progressIndex().finiteBufferOwners();
  if (owner >= ownerCount_ || owner >= owners.size())
    return invalid("finite-buffer witness owner is out of range");
  if (!finiteBufferOwnerConflicts(owner))
    return invalid("finite-buffer witness owner has no live conflict");

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
      witness.competingLogicalNets.push_back(static_cast<PnrIndex>(logicalNet));
      bits &= bits - 1;
    }
  }
  if (witness.competingLogicalNets.size() != ownerLogicalNetCounts_[owner])
    return invalid("finite-buffer owner bitset disagrees with its net count");

  for (PnrIndex logicalNet : witness.competingLogicalNets) {
    const std::size_t anchorBegin = witness.routeAnchors.size();
    if (llvm::Error error = forEachSelectedTraversal(
            candidate, logicalNet,
            [&](PnrIndex traversal,
                const SpatialProgressRouteAnchor &anchor) -> llvm::Error {
              if (problem_->progressIndex().traversalOwner(traversal) == owner)
                witness.routeAnchors.push_back(anchor);
              return llvm::Error::success();
            }))
      return error;
    if (witness.routeAnchors.size() == anchorBegin)
      return invalid("finite-buffer conflict net has no owner route anchor");
  }
  llvm::sort(witness.routeAnchors, [](const SpatialProgressRouteAnchor &lhs,
                                      const SpatialProgressRouteAnchor &rhs) {
    return std::tie(lhs.logicalNet, lhs.kind, lhs.sinkObligation, lhs.endpoint,
                    lhs.traversal) < std::tie(rhs.logicalNet, rhs.kind,
                                              rhs.sinkObligation, rhs.endpoint,
                                              rhs.traversal);
  });
  return llvm::Error::success();
}

llvm::Error SpatialProgressState::rebuildCapacityProofDebtWitness(
    const SpatialCandidateState &candidate, PnrIndex owner,
    SpatialFiniteBufferConflictWitness &witness) const {
  if (&candidate.problem() != problem_ || owner >= ownerCount_)
    return invalid("capacity-debt witness owner is out of range");
  if (!ownerHasCapacityProofDebt(owner))
    return invalid("capacity-debt witness owner has no live debt");
  return rebuildCapacityOwnerWitness(candidate, owner, witness);
}

llvm::Error SpatialProgressState::rebuildCapacityShortfallWitness(
    const SpatialCandidateState &candidate, PnrIndex owner,
    SpatialFiniteBufferConflictWitness &witness) const {
  if (&candidate.problem() != problem_ || owner >= ownerCount_)
    return invalid("capacity-shortfall witness owner is out of range");
  if (ownerCapacityShortfall(owner) == 0)
    return invalid("capacity-shortfall witness owner has no live shortfall");
  return rebuildCapacityOwnerWitness(candidate, owner, witness);
}

llvm::Error SpatialProgressState::rebuildCapacityOwnerWitness(
    const SpatialCandidateState &candidate, PnrIndex owner,
    SpatialFiniteBufferConflictWitness &witness) const {
  witness.ownerOrdinal = getInvalidPnrIndex();
  witness.competingLogicalNets.clear();
  witness.routeAnchors.clear();
  if (&candidate.problem() != problem_ || owner >= ownerCount_)
    return invalid("capacity witness owner is out of range");
  witness.ownerOrdinal = owner;
  witness.owner = problem_->progressIndex().finiteBufferOwners()[owner];
  for (PnrIndex logicalNet = 0; logicalNet < logicalNetCount_; ++logicalNet) {
    const auto &uses = netCapacityProjections_[logicalNet].owners;
    const auto found = llvm::lower_bound(
        uses, owner,
        [](const SpatialProgressOwnerCapacityUse &use, PnrIndex target) {
          return use.owner < target;
        });
    if (found == uses.end() || found->owner != owner)
      continue;
    witness.competingLogicalNets.push_back(logicalNet);
    if (llvm::Error error = forEachSelectedTraversal(
            candidate, logicalNet,
            [&](PnrIndex traversal,
                const SpatialProgressRouteAnchor &anchor) -> llvm::Error {
              if (problem_->progressIndex().traversalOwner(traversal) == owner)
                witness.routeAnchors.push_back(anchor);
              return llvm::Error::success();
            }))
      return error;
  }
  if (witness.competingLogicalNets.empty() || witness.routeAnchors.empty())
    return invalid("capacity witness has no selected route anchor");
  llvm::sort(witness.routeAnchors, [](const SpatialProgressRouteAnchor &lhs,
                                      const SpatialProgressRouteAnchor &rhs) {
    return std::tie(lhs.logicalNet, lhs.kind, lhs.sinkObligation, lhs.endpoint,
                    lhs.traversal) < std::tie(rhs.logicalNet, rhs.kind,
                                              rhs.sinkObligation, rhs.endpoint,
                                              rhs.traversal);
  });
  witness.routeAnchors.erase(
      std::unique(witness.routeAnchors.begin(), witness.routeAnchors.end(),
                  [](const SpatialProgressRouteAnchor &lhs,
                     const SpatialProgressRouteAnchor &rhs) {
                    return std::tie(lhs.logicalNet, lhs.kind,
                                    lhs.sinkObligation, lhs.endpoint,
                                    lhs.traversal) ==
                           std::tie(rhs.logicalNet, rhs.kind,
                                    rhs.sinkObligation, rhs.endpoint,
                                    rhs.traversal);
                  }),
      witness.routeAnchors.end());
  return llvm::Error::success();
}

llvm::Error
SpatialProgressState::verifyCachedState(
    const SpatialCandidateState &candidate) const {
  saturatingAdd(statistics_.cachedVerificationCount, 1);
  if (!problem_ || &candidate.problem() != problem_)
    return invalid("progress state is bound to a foreign candidate");
  const std::size_t expectedLogicalNetCount =
      problem_->transfers().logicalNets().size();
  const std::size_t expectedOwnerCount =
      problem_->progressIndex().finiteBufferOwners().size();
  const std::size_t expectedLogicalNetWordCount =
      expectedLogicalNetCount / 64 + (expectedLogicalNetCount % 64 != 0);
  if (expectedOwnerCount != 0 &&
      expectedLogicalNetWordCount >
          std::numeric_limits<std::size_t>::max() / expectedOwnerCount)
    return invalid("finite-buffer selected-net bitset exceeds native size_t");
  const std::size_t expectedOwnerBitCount =
      expectedOwnerCount * expectedLogicalNetWordCount;
  const std::size_t expectedConflictWordCount =
      expectedOwnerCount / 64 + (expectedOwnerCount % 64 != 0);
  if (logicalNetCount_ != expectedLogicalNetCount ||
      ownerCount_ != expectedOwnerCount ||
      logicalNetWordCount_ != expectedLogicalNetWordCount ||
      netOwnerRefcounts_.size() != logicalNetCount_ ||
      ownerLogicalNetCounts_.size() != ownerCount_ ||
      ownerLogicalNetBits_.size() != expectedOwnerBitCount ||
      conflictingOwnerBits_.size() != expectedConflictWordCount ||
      netRouteDependencyViolationCounts_.size() != logicalNetCount_ ||
      netCapacityProjections_.size() != logicalNetCount_ ||
      ownerChannelCounts_.size() != ownerCount_ ||
      ownerInitializedFeedbackChannelCounts_.size() != ownerCount_ ||
      ownerRepeatedChannelNetCounts_.size() != ownerCount_ ||
      ownerIndeterminateQueueClassNetCounts_.size() != ownerCount_ ||
      ownerQueueClassRefcounts_.size() != ownerCount_ ||
      traversalSelectionCounts_.size() !=
          problem_->routing().traversals().size() ||
      ownerRouteAnchorCounts_.size() != ownerCount_ ||
      capacityProofDebtOwnerBits_.size() != expectedConflictWordCount)
    return invalid("progress state dimensions disagree with the freeze");

  std::uint64_t refcountPairCount = 0;
  for (PnrIndex logicalNet = 0; logicalNet < logicalNetCount_; ++logicalNet) {
    for (const auto &[owner, refcount] : netOwnerRefcounts_[logicalNet]) {
      if (owner >= ownerCount_ || refcount == 0)
        return invalid("finite-buffer owner refcount is invalid");
      const std::uint64_t bits =
          ownerLogicalNetBits_[static_cast<std::size_t>(owner) *
                                   logicalNetWordCount_ +
                               logicalNet / 64];
      if ((bits & (std::uint64_t{1} << (logicalNet % 64))) == 0)
        return invalid("finite-buffer owner refcount has no selected-net bit");
      if (refcountPairCount == std::numeric_limits<std::uint64_t>::max())
        return invalid("finite-buffer owner pair count exceeds u64");
      ++refcountPairCount;
    }
  }

  std::uint64_t bitPairCount = 0;
  std::uint64_t conflictCount = 0;
  for (PnrIndex owner = 0; owner < ownerCount_; ++owner) {
    std::uint64_t ownerNetCount = 0;
    for (std::size_t word = 0; word < logicalNetWordCount_; ++word) {
      const std::uint64_t bits =
          ownerLogicalNetBits_[static_cast<std::size_t>(owner) *
                                   logicalNetWordCount_ +
                               word];
      if (word + 1 == logicalNetWordCount_ && logicalNetCount_ % 64 != 0) {
        const std::uint64_t validMask =
            (std::uint64_t{1} << (logicalNetCount_ % 64)) - 1;
        if ((bits & ~validMask) != 0)
          return invalid("finite-buffer selected-net bitset has excess bits");
      }
      const std::uint64_t selected = llvm::popcount(bits);
      if (selected > std::numeric_limits<std::uint64_t>::max() - ownerNetCount)
        return invalid("finite-buffer owner net count exceeds u64");
      ownerNetCount += selected;
    }
    if (ownerNetCount != ownerLogicalNetCounts_[owner])
      return invalid("finite-buffer owner bitset disagrees with its net count");
    if (ownerNetCount >
        std::numeric_limits<std::uint64_t>::max() - bitPairCount)
      return invalid("finite-buffer owner pair count exceeds u64");
    bitPairCount += ownerNetCount;

    const bool conflictBit =
        (conflictingOwnerBits_[owner / 64] &
         (std::uint64_t{1} << (owner % 64))) != 0;
    const bool expectedConflict = ownerNetCount > 1;
    if (conflictBit != expectedConflict)
      return invalid("finite-buffer conflict bit is stale");
    conflictCount += expectedConflict;
  }
  if (ownerCount_ % 64 != 0 && !conflictingOwnerBits_.empty()) {
    const std::uint64_t validMask =
        (std::uint64_t{1} << (ownerCount_ % 64)) - 1;
    if ((conflictingOwnerBits_.back() & ~validMask) != 0)
      return invalid("finite-buffer conflict bitset has excess bits");
  }
  if (refcountPairCount != bitPairCount)
    return invalid("finite-buffer selected-net bits have no refcount owner");
  if (conflictCount != sharedFiniteBufferConflictCount_)
    return invalid("finite-buffer conflict total is stale");

  std::vector<std::uint64_t> expectedChannels(ownerCount_, 0);
  std::vector<std::uint64_t> expectedFeedbackChannels(ownerCount_, 0);
  std::vector<std::uint64_t> expectedRepeatedNets(ownerCount_, 0);
  std::vector<std::uint64_t> expectedIndeterminateClasses(ownerCount_, 0);
  std::vector<PhysicalTagKeyedMap<PnrIndex>> expectedQueueClasses(
      ownerCount_);
  for (const SpatialProgressNetCapacityProjection &projection :
       netCapacityProjections_) {
    PnrIndex previous = getInvalidPnrIndex();
    for (const SpatialProgressOwnerCapacityUse &use : projection.owners) {
      if (use.owner >= ownerCount_ || use.channelCount == 0 ||
          use.initializedFeedbackChannelCount > use.channelCount ||
          (previous != getInvalidPnrIndex() && use.owner <= previous))
        return invalid("capacity proof net projection is not canonical");
      previous = use.owner;
      if (use.channelCount >
              std::numeric_limits<std::uint64_t>::max() -
                  expectedChannels[use.owner] ||
          use.initializedFeedbackChannelCount >
              std::numeric_limits<std::uint64_t>::max() -
                  expectedFeedbackChannels[use.owner])
        return invalid("capacity proof cold aggregate exceeds u64");
      expectedChannels[use.owner] += use.channelCount;
      expectedFeedbackChannels[use.owner] +=
          use.initializedFeedbackChannelCount;
      expectedRepeatedNets[use.owner] += use.repeatedWithinChannel;
      expectedIndeterminateClasses[use.owner] +=
          use.queueClassIndeterminate;
      for (std::size_t index = 0; index != use.queueClasses.size(); ++index) {
        if (index != 0 &&
            ::fabric::comparePhysicalTagValues(use.queueClasses[index - 1],
                                               use.queueClasses[index]) >= 0)
          return invalid("capacity proof queue classes are not canonical");
        PnrIndex &refcount =
            expectedQueueClasses[use.owner][use.queueClasses[index]];
        if (refcount == std::numeric_limits<PnrIndex>::max())
          return invalid("capacity proof queue-class refcount exceeds "
                         "PnrIndex");
        ++refcount;
      }
    }
  }
  if (ownerChannelCounts_ != expectedChannels ||
      ownerInitializedFeedbackChannelCounts_ != expectedFeedbackChannels ||
      ownerRepeatedChannelNetCounts_ != expectedRepeatedNets ||
      ownerIndeterminateQueueClassNetCounts_ !=
          expectedIndeterminateClasses ||
      ownerQueueClassRefcounts_ != expectedQueueClasses)
    return invalid("capacity proof owner aggregate is stale");

  std::vector<PnrIndex> expectedRouteAnchors(ownerCount_, 0);
  for (PnrIndex traversal = 0;
       traversal < traversalSelectionCounts_.size(); ++traversal) {
    const PnrIndex owner = problem_->progressIndex().traversalOwner(traversal);
    if (owner == getInvalidPnrIndex()) {
      if (traversalSelectionCounts_[traversal] != 0)
        return invalid("non-FIFO traversal has a capacity proof refcount");
      continue;
    }
    if (owner >= ownerCount_)
      return invalid("capacity proof traversal owner is out of range");
    if (traversalSelectionCounts_[traversal] == 0)
      continue;
    if (expectedRouteAnchors[owner] ==
        std::numeric_limits<PnrIndex>::max())
      return invalid("capacity proof route-anchor count exceeds PnrIndex");
    ++expectedRouteAnchors[owner];
  }
  if (ownerRouteAnchorCounts_ != expectedRouteAnchors)
    return invalid("capacity proof route-anchor aggregate is stale");

  std::uint64_t expectedDebtCount = 0;
  std::uint64_t expectedShortfallOwnerCount = 0;
  std::uint64_t expectedShortfall = 0;
  std::uint64_t expectedObligationAnchors = 0;
  for (PnrIndex owner = 0; owner < ownerCount_; ++owner) {
    const bool expectedDebt = expectedChannels[owner] != 0 &&
                              (expectedRepeatedNets[owner] != 0 ||
                               (problem_->progressIndex()
                                        .ownerQueueDisciplines()[owner] ==
                                    ::fabric::FifoQueueDiscipline::
                                        PerTagVirtualChannel &&
                                expectedIndeterminateClasses[owner] != 0));
    const std::uint64_t selectedCapacity =
        problem_->progressIndex().ownerSharedSlotCapacities()[owner];
    const std::uint64_t expectedOwnerShortfall =
        !expectedDebt && ownerLogicalNetCounts_[owner] > selectedCapacity
            ? ownerLogicalNetCounts_[owner] - selectedCapacity
            : 0;
    const bool cachedDebt =
        (capacityProofDebtOwnerBits_[owner / 64] &
         (std::uint64_t{1} << (owner % 64))) != 0;
    if (cachedDebt != expectedDebt)
      return invalid("capacity proof debt bit is stale");
    expectedDebtCount += expectedDebt;
    expectedShortfallOwnerCount += expectedOwnerShortfall != 0;
    if (expectedOwnerShortfall >
        std::numeric_limits<std::uint64_t>::max() - expectedShortfall)
      return invalid("capacity proof shortfall exceeds u64");
    expectedShortfall += expectedOwnerShortfall;
    if (expectedDebt || expectedOwnerShortfall != 0) {
      if (expectedRouteAnchors[owner] >
          std::numeric_limits<std::uint64_t>::max() -
              expectedObligationAnchors)
        return invalid("capacity proof obligation anchor count exceeds u64");
      expectedObligationAnchors += expectedRouteAnchors[owner];
    }
  }
  if (ownerCount_ % 64 != 0 && !capacityProofDebtOwnerBits_.empty()) {
    const std::uint64_t validMask =
        (std::uint64_t{1} << (ownerCount_ % 64)) - 1;
    if ((capacityProofDebtOwnerBits_.back() & ~validMask) != 0)
      return invalid("capacity proof debt bitset has excess bits");
  }
  if (capacityProofDebtWitnessCount_ != expectedDebtCount ||
      capacityShortfallOwnerCount_ != expectedShortfallOwnerCount ||
      capacityShortfall_ != expectedShortfall ||
      capacityObligationRouteAnchorCount_ != expectedObligationAnchors)
    return invalid("capacity proof objective aggregate is stale");

  std::uint64_t dependencyCount = 0;
  for (std::uint64_t count : netRouteDependencyViolationCounts_) {
    if (count > std::numeric_limits<std::uint64_t>::max() - dependencyCount)
      return invalid("route dependency violation total exceeds u64");
    dependencyCount += count;
  }
  if (dependencyCount != routeDependencyViolationCount_)
    return invalid("route dependency violation total is stale");
  if (capacityProofDebtWitnessCount_ >
          std::numeric_limits<std::uint64_t>::max() -
              capacityShortfallOwnerCount_ ||
      capacityProofDebtWitnessCount_ + capacityShortfallOwnerCount_ >
          std::numeric_limits<std::uint64_t>::max() -
              routeDependencyViolationCount_)
    return invalid("activity witness total exceeds u64");
  return llvm::Error::success();
}

llvm::Error
SpatialProgressState::verify(const SpatialCandidateState &candidate) const {
  if (llvm::Error error = verifyCachedState(candidate))
    return error;
  ProgressTimer verificationTimer(
      &statistics_.coldVerificationCount,
      &statistics_.coldVerificationWallTimeNanoseconds);
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
    return invalid("incremental base projection diverges from cold "
                   "reconstruction");
  if (netCapacityProjections_ != expected->netCapacityProjections_) {
    for (const auto [logicalNet, projected] :
         llvm::enumerate(netCapacityProjections_))
      if (!(projected == expected->netCapacityProjections_[logicalNet]))
        return invalid("incremental capacity inputs for logical net " +
                       llvm::Twine(logicalNet) +
                       " diverge from cold reconstruction: " +
                       capacityProjectionDifference(
                           projected,
                           expected->netCapacityProjections_[logicalNet]));
    llvm_unreachable("unequal capacity projection has no differing net");
  }
  if (ownerChannelCounts_ != expected->ownerChannelCounts_ ||
      ownerInitializedFeedbackChannelCounts_ !=
          expected->ownerInitializedFeedbackChannelCounts_ ||
      ownerRepeatedChannelNetCounts_ !=
          expected->ownerRepeatedChannelNetCounts_ ||
      ownerIndeterminateQueueClassNetCounts_ !=
          expected->ownerIndeterminateQueueClassNetCounts_ ||
      ownerQueueClassRefcounts_ != expected->ownerQueueClassRefcounts_)
    return invalid("incremental capacity owner inputs diverge from cold "
                   "reconstruction");
  if (traversalSelectionCounts_ != expected->traversalSelectionCounts_ ||
      ownerRouteAnchorCounts_ != expected->ownerRouteAnchorCounts_)
    return invalid("incremental capacity anchors diverge from cold "
                   "reconstruction");
  if (capacityProofDebtOwnerBits_ !=
          expected->capacityProofDebtOwnerBits_ ||
      capacityProofDebtWitnessCount_ !=
          expected->capacityProofDebtWitnessCount_ ||
      capacityShortfallOwnerCount_ !=
          expected->capacityShortfallOwnerCount_ ||
      capacityShortfall_ != expected->capacityShortfall_ ||
      capacityObligationRouteAnchorCount_ !=
          expected->capacityObligationRouteAnchorCount_)
    return invalid("incremental capacity objective diverges from cold "
                   "reconstruction");

  llvm::Expected<std::uint64_t> cold = [&] {
    ProgressTimer scanTimer(&statistics_.coldProgressScanCount,
                            &statistics_.coldProgressScanWallTimeNanoseconds);
    return spatialCandidateProgressWitnessCount(candidate);
  }();
  if (!cold)
    return cold.takeError();
  if (*cold != capacityProofDebtWitnessCount_ +
                   capacityShortfallOwnerCount_ +
                   routeDependencyViolationCount_)
    return invalid("incremental activity witness total diverges from cold "
                   "verifier");
  verificationTimer.finish();
  if (statisticsEnabled_)
    emitProgressStatistics(statistics_);
  return llvm::Error::success();
}
