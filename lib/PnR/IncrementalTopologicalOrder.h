#ifndef LOOM_LIB_PNR_INCREMENTALTOPOLOGICALORDER_H
#define LOOM_LIB_PNR_INCREMENTALTOPOLOGICALORDER_H

#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace loom::pnr::detail {

struct IncrementalTopologicalGraphView final {
  PnrIndex nodeCount = 0;
  llvm::ArrayRef<FrozenSpatialHandshakeArc> arcs;
  llvm::ArrayRef<PnrIndex> adjacencyOffsets;
  llvm::ArrayRef<PnrIndex> reverseAdjacencyOffsets;
  llvm::ArrayRef<PnrIndex> reverseArcOrdinals;
};

class IncrementalTopologicalOrder;
class IncrementalTopologicalTransaction;

using IncrementalTopologicalOrderHandle =
    std::shared_ptr<IncrementalTopologicalOrder>;

class IncrementalTopologicalScratch final {
public:
  IncrementalTopologicalScratch() = default;
  IncrementalTopologicalScratch(const IncrementalTopologicalScratch &) = delete;
  IncrementalTopologicalScratch &
  operator=(const IncrementalTopologicalScratch &) = delete;
  IncrementalTopologicalScratch(IncrementalTopologicalScratch &&) = delete;
  IncrementalTopologicalScratch &
  operator=(IncrementalTopologicalScratch &&) = delete;
  ~IncrementalTopologicalScratch();

  llvm::Error prepare(IncrementalTopologicalGraphView graph);
  std::size_t retainedStorageBytes() const;

private:
  void beginTransaction();
  void resetTransaction();
  void beginSearch();

  std::vector<std::uint64_t> forwardMarks_;
  std::vector<std::uint64_t> backwardMarks_;
  std::vector<PnrIndex> forwardParentArcs_;
  std::vector<std::uint64_t> rankJournalMarks_;
  std::vector<std::uint64_t> arcJournalMarks_;
  std::vector<PnrIndex> forwardWorklist_;
  std::vector<PnrIndex> backwardWorklist_;
  std::vector<PnrIndex> reorderBuffer_;
  std::vector<PnrIndex> touchedRanks_;
  std::vector<PnrIndex> oldRankNodes_;
  std::vector<PnrIndex> touchedArcs_;
  std::vector<std::uint8_t> oldArcActive_;
  std::vector<PnrIndex> cycleWitness_;
  std::uint64_t transactionEpoch_ = 0;
  std::uint64_t searchEpoch_ = 0;
  IncrementalTopologicalTransaction *activeTransaction_ = nullptr;

  friend class IncrementalTopologicalOrder;
  friend class IncrementalTopologicalTransaction;
};

class IncrementalTopologicalOrder final
    : public std::enable_shared_from_this<IncrementalTopologicalOrder> {
public:
  static llvm::Expected<IncrementalTopologicalOrderHandle>
  create(IncrementalTopologicalGraphView graph,
         llvm::ArrayRef<PnrIndex> initiallyActiveArcs);

  IncrementalTopologicalOrder(const IncrementalTopologicalOrder &) = delete;
  IncrementalTopologicalOrder(IncrementalTopologicalOrder &&) = delete;
  IncrementalTopologicalOrder &
  operator=(const IncrementalTopologicalOrder &) = delete;
  IncrementalTopologicalOrder &
  operator=(IncrementalTopologicalOrder &&) = delete;
  ~IncrementalTopologicalOrder() = default;

  IncrementalTopologicalGraphView graph() const { return graph_; }
  llvm::ArrayRef<PnrIndex> order() const { return order_; }
  llvm::ArrayRef<PnrIndex> ranks() const { return ranks_; }
  PnrIndex rank(PnrIndex node) const;
  bool isArcActive(PnrIndex arc) const;

  llvm::Error rebuild();
  llvm::Error verify() const;
  llvm::Expected<IncrementalTopologicalTransaction>
  beginTransaction(IncrementalTopologicalScratch &scratch) &;
  llvm::Expected<IncrementalTopologicalTransaction>
  beginTransaction(IncrementalTopologicalScratch &scratch) && = delete;

private:
  IncrementalTopologicalOrder(IncrementalTopologicalGraphView graph,
                              std::vector<std::uint64_t> activeArcBits,
                              std::vector<PnrIndex> order,
                              std::vector<PnrIndex> ranks)
      : graph_(graph), activeArcBits_(std::move(activeArcBits)),
        order_(std::move(order)), ranks_(std::move(ranks)) {}

  void setArcActive(PnrIndex arc, bool active);

  IncrementalTopologicalGraphView graph_;
  std::vector<std::uint64_t> activeArcBits_;
  std::vector<PnrIndex> order_;
  std::vector<PnrIndex> ranks_;
  IncrementalTopologicalTransaction *activeTransaction_ = nullptr;

  friend class IncrementalTopologicalTransaction;
};

class IncrementalTopologicalTransaction final {
public:
  IncrementalTopologicalTransaction(
      IncrementalTopologicalTransaction &&other) noexcept;
  IncrementalTopologicalTransaction(const IncrementalTopologicalTransaction &) =
      delete;
  IncrementalTopologicalTransaction &
  operator=(const IncrementalTopologicalTransaction &) = delete;
  IncrementalTopologicalTransaction &
  operator=(IncrementalTopologicalTransaction &&) = delete;
  ~IncrementalTopologicalTransaction();

  llvm::Expected<bool> insertArc(PnrIndex arc);
  llvm::Error removeArc(PnrIndex arc);
  llvm::ArrayRef<PnrIndex> cycleWitness() const;
  llvm::Error commit();
  void rollback() noexcept;

private:
  IncrementalTopologicalTransaction(IncrementalTopologicalOrderHandle order,
                                    IncrementalTopologicalScratch &scratch);

  void recordArc(PnrIndex arc);
  void recordRank(PnrIndex rank);
  llvm::Expected<bool> repairAfterInsertion(PnrIndex arc);
  void finish();

  IncrementalTopologicalOrderHandle order_;
  IncrementalTopologicalScratch *scratch_ = nullptr;
  bool hasCycle_ = false;

  friend class IncrementalTopologicalOrder;
  friend class IncrementalTopologicalScratch;
};

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_INCREMENTALTOPOLOGICALORDER_H
