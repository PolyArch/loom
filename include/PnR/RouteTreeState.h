#ifndef LOOM_PNR_ROUTETREESTATE_H
#define LOOM_PNR_ROUTETREESTATE_H

#include "PnR/FrozenRoutingGraph.h"
#include "PnR/PnrIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace loom::pnr {

inline constexpr PnrIndex getInvalidPnrIndex() {
  return static_cast<PnrIndex>(getPnrIndexMax());
}

struct RouteTreeNode {
  PnrIndex endpoint = getInvalidPnrIndex();
  PnrIndex parentArc = getInvalidPnrIndex();
  PnrIndex firstChild = getInvalidPnrIndex();
  PnrIndex nextSibling = getInvalidPnrIndex();
  PnrIndex previousSibling = getInvalidPnrIndex();
  PnrIndex firstSinkObligation = getInvalidPnrIndex();
  PnrIndex sinkObligationCount = 0;

  bool isActive() const { return endpoint != getInvalidPnrIndex(); }

  friend bool operator==(const RouteTreeNode &lhs, const RouteTreeNode &rhs) {
    return lhs.endpoint == rhs.endpoint && lhs.parentArc == rhs.parentArc &&
           lhs.firstChild == rhs.firstChild &&
           lhs.nextSibling == rhs.nextSibling &&
           lhs.previousSibling == rhs.previousSibling &&
           lhs.firstSinkObligation == rhs.firstSinkObligation &&
           lhs.sinkObligationCount == rhs.sinkObligationCount;
  }
};

class RouteTreeState;
class RouteTreeTransaction;

// A scratch object must normally outlive its active transaction. Early
// destruction rolls the transaction back and invalidates its handle.
class RouteTreeTransactionScratch {
public:
  RouteTreeTransactionScratch() = default;
  RouteTreeTransactionScratch(const RouteTreeTransactionScratch &) = delete;
  RouteTreeTransactionScratch &
  operator=(const RouteTreeTransactionScratch &) = delete;
  RouteTreeTransactionScratch(RouteTreeTransactionScratch &&) = delete;
  RouteTreeTransactionScratch &
  operator=(RouteTreeTransactionScratch &&) = delete;
  ~RouteTreeTransactionScratch();

private:
  enum class DeltaKind {
    ModifiedNode,
    RemovedNode,
    AddedNode,
    SourceBinding,
    SinkBinding,
  };

  struct Delta {
    DeltaKind kind;
    PnrIndex key = getInvalidPnrIndex();
    RouteTreeNode node;
    bool appended = false;
    PnrIndex value0 = getInvalidPnrIndex();
    PnrIndex value1 = getInvalidPnrIndex();
    PnrIndex value2 = getInvalidPnrIndex();
    PnrIndex value3 = getInvalidPnrIndex();
  };

  void resetTransaction();

  std::vector<Delta> deltas_;
  std::vector<PnrIndex> worklist_;
  std::vector<std::uint64_t> pathMarks_;
  std::uint64_t pathGeneration_ = 0;
  RouteTreeTransaction *activeTransaction_ = nullptr;

  friend class RouteTreeState;
  friend class RouteTreeTransaction;
};

// FrozenRoutingGraph is borrowed and must outlive this state. An active
// transaction normally has a shorter lifetime; moving the state migrates it,
// while early state destruction rolls it back and invalidates its handle.
class RouteTreeState {
public:
  static llvm::Expected<RouteTreeState>
  create(const FrozenRoutingGraph &graph LLVM_LIFETIME_BOUND,
         PnrIndex sinkObligationCount);
  static llvm::Expected<RouteTreeState>
  create(FrozenRoutingGraph &&graph, PnrIndex sinkObligationCount) = delete;
  static llvm::Expected<RouteTreeState>
  create(const FrozenRoutingGraph &&graph,
         PnrIndex sinkObligationCount) = delete;

  RouteTreeState(RouteTreeState &&other) noexcept;
  RouteTreeState(const RouteTreeState &) = delete;
  RouteTreeState &operator=(const RouteTreeState &) = delete;
  RouteTreeState &operator=(RouteTreeState &&) = delete;
  ~RouteTreeState();

  bool isRouted() const { return activeNodeCount_ != 0; }
  bool isUnrouted() const { return activeNodeCount_ == 0; }
  PnrIndex activeNodeCount() const { return activeNodeCount_; }
  PnrIndex sinkObligationCount() const {
    return static_cast<PnrIndex>(sinkBindings_.size());
  }

  std::optional<PnrIndex> sourceEndpoint() const;
  std::optional<PnrIndex> sinkEndpoint(PnrIndex obligation) const;
  llvm::ArrayRef<RouteTreeNode> nodeStorage() const { return nodes_; }
  std::optional<PnrIndex> findNode(PnrIndex endpoint) const;
  const RouteTreeNode &node(PnrIndex slot) const;

  llvm::Error verify() const;
  llvm::Expected<RouteTreeTransaction> beginTransaction(
      RouteTreeTransactionScratch &scratch LLVM_LIFETIME_BOUND) &
      LLVM_LIFETIME_BOUND;
  llvm::Expected<RouteTreeTransaction>
  beginTransaction(RouteTreeTransactionScratch &scratch) && = delete;

private:
  struct SinkBinding {
    PnrIndex endpoint = getInvalidPnrIndex();
    PnrIndex nodeSlot = getInvalidPnrIndex();
    PnrIndex previousAtNode = getInvalidPnrIndex();
    PnrIndex nextAtNode = getInvalidPnrIndex();
  };

  struct LookupEntry {
    PnrIndex endpoint = getInvalidPnrIndex();
    PnrIndex slot = getInvalidPnrIndex();
    bool tombstone = false;

    bool isOccupied() const { return endpoint != getInvalidPnrIndex(); }
    bool isEmpty() const { return !isOccupied() && !tombstone; }
  };

  RouteTreeState(const FrozenRoutingGraph &graph,
                 std::vector<SinkBinding> sinkBindings);

  static std::size_t hashEndpoint(PnrIndex endpoint);
  std::optional<PnrIndex> lookupSlot(PnrIndex endpoint) const;
  llvm::Error ensureLookupCapacity(PnrIndex requiredCount);
  void insertLookup(PnrIndex endpoint, PnrIndex slot);
  void eraseLookup(PnrIndex endpoint);
  void rehashLookup(std::size_t capacity);

  PnrIndex arcSourceEndpoint(PnrIndex arc) const;
  llvm::Error verifyState() const;

  const FrozenRoutingGraph &graph_;
  PnrIndex sourceEndpoint_ = getInvalidPnrIndex();
  std::vector<SinkBinding> sinkBindings_;
  std::vector<RouteTreeNode> nodes_;
  std::vector<PnrIndex> freeSlots_;
  std::vector<LookupEntry> endpointSlots_;
  PnrIndex activeNodeCount_ = 0;
  PnrIndex lookupTombstoneCount_ = 0;
  PnrIndex boundSinkObligationCount_ = 0;
  PnrIndex attachedSinkObligationCount_ = 0;
  RouteTreeTransaction *activeTransaction_ = nullptr;

  friend class RouteTreeTransaction;
};

class RouteTreeTransaction {
public:
  RouteTreeTransaction(RouteTreeTransaction &&other) noexcept;
  RouteTreeTransaction(const RouteTreeTransaction &) = delete;
  RouteTreeTransaction &operator=(const RouteTreeTransaction &) = delete;
  RouteTreeTransaction &operator=(RouteTreeTransaction &&) = delete;
  ~RouteTreeTransaction();

  llvm::Error bindSource(PnrIndex endpoint);
  llvm::Error bindSink(PnrIndex obligation, PnrIndex endpoint);
  llvm::Error attachPath(PnrIndex attachmentEndpoint,
                         llvm::ArrayRef<PnrIndex> forwardArcs,
                         PnrIndex sinkObligation);
  llvm::Error ripUpSink(PnrIndex sinkObligation);
  llvm::Error ripUpSubtree(PnrIndex subtreeRootEndpoint);
  llvm::Error ripUpWholeNet();

  llvm::Error commit();
  void rollback() noexcept;

private:
  RouteTreeTransaction(RouteTreeState &state,
                       RouteTreeTransactionScratch &scratch);

  void recordModifiedNode(PnrIndex slot);
  void setSourceBinding(PnrIndex endpoint);
  void setSinkBinding(PnrIndex obligation, PnrIndex endpoint, PnrIndex nodeSlot,
                      PnrIndex previousAtNode, PnrIndex nextAtNode);
  void attachSinkBinding(PnrIndex obligation, PnrIndex nodeSlot,
                         PnrIndex finalSinkCount);
  void unlinkSinkBinding(PnrIndex obligation);
  llvm::Expected<PnrIndex> addNode(PnrIndex endpoint, PnrIndex parentArc);
  void linkChild(PnrIndex parentSlot, PnrIndex childSlot);
  PnrIndex parentSlot(PnrIndex childSlot) const;
  void detachNode(PnrIndex slot, PnrIndex parentSlot);
  void removeNode(PnrIndex slot);
  void finish();

  RouteTreeState *state_;
  RouteTreeTransactionScratch *scratch_;
  std::size_t initialNodeStorageSize_;
  PnrIndex initialActiveNodeCount_;
  PnrIndex initialBoundSinkObligationCount_;
  PnrIndex initialAttachedSinkObligationCount_;

  friend class RouteTreeState;
  friend class RouteTreeTransactionScratch;
};

namespace detail {

llvm::Error preflightRouteTreeStateCapacity(std::uint64_t reachedEndpointCount,
                                            std::uint64_t sinkObligationCount);

} // namespace detail

} // namespace loom::pnr

#endif // LOOM_PNR_ROUTETREESTATE_H
