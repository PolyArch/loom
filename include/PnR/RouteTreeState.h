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

class RouteTreeTransaction;

class RouteTreeState {
public:
  static llvm::Expected<RouteTreeState>
  create(const FrozenRoutingGraph &graph, PnrIndex producerEndpoint,
         llvm::ArrayRef<PnrIndex> sinkEndpoints);

  RouteTreeState(RouteTreeState &&other) noexcept;
  RouteTreeState(const RouteTreeState &) = delete;
  RouteTreeState &operator=(const RouteTreeState &) = delete;
  RouteTreeState &operator=(RouteTreeState &&) = delete;

  bool isRouted() const { return routed_; }
  bool isUnrouted() const { return !routed_; }
  PnrIndex producerEndpoint() const { return producerEndpoint_; }
  PnrIndex activeNodeCount() const { return activeNodeCount_; }

  llvm::ArrayRef<RouteTreeNode> nodeStorage() const { return nodes_; }
  std::optional<PnrIndex> findNode(PnrIndex endpoint) const;
  const RouteTreeNode &node(PnrIndex slot) const;

  llvm::Error verify() const;
  llvm::Expected<RouteTreeTransaction> beginTransaction();

private:
  struct SinkObligation {
    PnrIndex endpoint;
    PnrIndex count;
  };

  struct LookupEntry {
    PnrIndex endpoint = getInvalidPnrIndex();
    PnrIndex slot = getInvalidPnrIndex();

    bool isOccupied() const { return endpoint != getInvalidPnrIndex(); }
  };

  RouteTreeState(const FrozenRoutingGraph &graph, PnrIndex producerEndpoint,
                 std::vector<SinkObligation> sinkObligations);

  static std::size_t hashEndpoint(PnrIndex endpoint);
  std::optional<PnrIndex> lookupSlot(PnrIndex endpoint) const;
  llvm::Error ensureLookupCapacity(PnrIndex requiredCount);
  void insertLookup(PnrIndex endpoint, PnrIndex slot);
  void eraseLookup(PnrIndex endpoint);
  void rebuildLookup(std::size_t capacity);

  PnrIndex requiredSinkCount(PnrIndex endpoint) const;
  PnrIndex sourceEndpoint(PnrIndex arc) const;
  llvm::Error verifyCandidate(bool routedCandidate) const;

  const FrozenRoutingGraph &graph_;
  PnrIndex producerEndpoint_;
  std::vector<SinkObligation> sinkObligations_;
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

  llvm::Error attachPath(PnrIndex attachmentEndpoint,
                         llvm::ArrayRef<PnrIndex> forwardArcs,
                         PnrIndex sinkEndpoint);
  llvm::Error ripUpSink(PnrIndex sinkEndpoint);
  llvm::Error ripUpSubtree(PnrIndex subtreeRootEndpoint);
  llvm::Error ripUpWholeNet();

  llvm::Error commit();
  void rollback() noexcept;

private:
  enum class DeltaKind {
    ModifiedNode,
    RemovedNode,
    AddedNode,
    SinkMetadata,
  };

  struct Delta {
    DeltaKind kind;
    PnrIndex slot;
    RouteTreeNode node;
    bool appended = false;
    PnrIndex sinkObligationCount = 0;
  };

  explicit RouteTreeTransaction(RouteTreeState &state);

  void recordModifiedNode(PnrIndex slot);
  void setSinkMetadata(PnrIndex slot, PnrIndex count);
  llvm::Expected<PnrIndex> addNode(PnrIndex endpoint, PnrIndex parentArc);
  void linkChild(PnrIndex parentSlot, PnrIndex childSlot);
  PnrIndex parentSlot(PnrIndex childSlot) const;
  void detachNode(PnrIndex slot, PnrIndex parentSlot);
  void removeNode(PnrIndex slot);

  RouteTreeState *state_;
  std::vector<Delta> deltas_;
  std::size_t initialLookupCapacity_;
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
