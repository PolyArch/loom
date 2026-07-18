#ifndef LOOM_PNR_ROUTETREESTATE_H
#define LOOM_PNR_ROUTETREESTATE_H

#include "PnR/FrozenRoutingGraph.h"
#include "PnR/PnrIndex.h"

#include "llvm/ADT/ArrayRef.h"
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
  PnrIndex sinkObligationCount = 0;

  bool isActive() const { return endpoint != getInvalidPnrIndex(); }

  friend bool operator==(const RouteTreeNode &lhs, const RouteTreeNode &rhs) {
    return lhs.endpoint == rhs.endpoint && lhs.parentArc == rhs.parentArc &&
           lhs.firstChild == rhs.firstChild &&
           lhs.nextSibling == rhs.nextSibling &&
           lhs.previousSibling == rhs.previousSibling &&
           lhs.sinkObligationCount == rhs.sinkObligationCount;
  }
};

class RouteTreeState;
class RouteTreeTransaction;

class RouteTreeTransactionScratch {
public:
  RouteTreeTransactionScratch() = default;
  RouteTreeTransactionScratch(const RouteTreeTransactionScratch &) = delete;
  RouteTreeTransactionScratch &
  operator=(const RouteTreeTransactionScratch &) = delete;
  RouteTreeTransactionScratch(RouteTreeTransactionScratch &&) = delete;
  RouteTreeTransactionScratch &
  operator=(RouteTreeTransactionScratch &&) = delete;

  std::size_t deltaCapacity() const { return deltas_.capacity(); }
  std::size_t workspaceCapacity() const {
    return worklist_.capacity() + childReferences_.capacity() +
           visited_.capacity() + sinkCounts_.capacity() + pathMarks_.capacity();
  }

private:
  enum class DeltaKind {
    ModifiedNode,
    RemovedNode,
    AddedNode,
    SinkMetadata,
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
  };

  void clearRetainingCapacity();

  std::vector<Delta> deltas_;
  std::vector<PnrIndex> worklist_;
  std::vector<std::uint8_t> childReferences_;
  std::vector<std::uint8_t> visited_;
  std::vector<PnrIndex> sinkCounts_;
  std::vector<std::uint64_t> pathMarks_;
  std::uint64_t pathGeneration_ = 0;
  bool inUse_ = false;

  friend class RouteTreeState;
  friend class RouteTreeTransaction;
};

class RouteTreeState {
public:
  static llvm::Expected<RouteTreeState> create(const FrozenRoutingGraph &graph,
                                               PnrIndex sinkObligationCount);

  RouteTreeState(RouteTreeState &&other) noexcept;
  RouteTreeState(const RouteTreeState &) = delete;
  RouteTreeState &operator=(const RouteTreeState &) = delete;
  RouteTreeState &operator=(RouteTreeState &&) = delete;

  bool isRouted() const { return routed_; }
  bool isUnrouted() const { return !routed_; }
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
  llvm::Expected<RouteTreeTransaction>
  beginTransaction(RouteTreeTransactionScratch &scratch);

private:
  struct SinkBinding {
    PnrIndex endpoint = getInvalidPnrIndex();
    PnrIndex nodeSlot = getInvalidPnrIndex();
  };

  struct LookupEntry {
    PnrIndex endpoint = getInvalidPnrIndex();
    PnrIndex slot = getInvalidPnrIndex();

    bool isOccupied() const { return endpoint != getInvalidPnrIndex(); }
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
  llvm::Error verifyCandidate(bool routedCandidate,
                              RouteTreeTransactionScratch &scratch) const;

  const FrozenRoutingGraph &graph_;
  PnrIndex sourceEndpoint_ = getInvalidPnrIndex();
  std::vector<SinkBinding> sinkBindings_;
  std::vector<RouteTreeNode> nodes_;
  std::vector<PnrIndex> freeSlots_;
  std::vector<LookupEntry> endpointSlots_;
  PnrIndex activeNodeCount_ = 0;
  bool routed_ = false;
  bool transactionActive_ = false;

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
  void setSinkMetadata(PnrIndex slot, PnrIndex count);
  void setSourceBinding(PnrIndex endpoint);
  void setSinkBinding(PnrIndex obligation, PnrIndex endpoint,
                      PnrIndex nodeSlot);
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
  bool initialRouted_;

  friend class RouteTreeState;
};

namespace detail {

llvm::Error preflightRouteTreeStateCapacity(std::uint64_t reachedEndpointCount,
                                            std::uint64_t sinkObligationCount);

} // namespace detail

} // namespace loom::pnr

#endif // LOOM_PNR_ROUTETREESTATE_H
