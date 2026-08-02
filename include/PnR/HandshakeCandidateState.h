#ifndef LOOM_PNR_HANDSHAKECANDIDATESTATE_H
#define LOOM_PNR_HANDSHAKECANDIDATESTATE_H

#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace loom::pnr {

namespace detail {
class IncrementalTopologicalOrder;
struct HandshakeCandidateScratchStorage;
} // namespace detail

class HandshakeCandidateState;
class HandshakeCandidateTransaction;

using FrozenSpatialHandshakeIndexHandle =
    std::shared_ptr<const FrozenSpatialHandshakeIndex>;
using HandshakeCandidateStateHandle = std::shared_ptr<HandshakeCandidateState>;

class HandshakeCandidateScratch final {
public:
  HandshakeCandidateScratch();
  HandshakeCandidateScratch(const HandshakeCandidateScratch &) = delete;
  HandshakeCandidateScratch &
  operator=(const HandshakeCandidateScratch &) = delete;
  HandshakeCandidateScratch(HandshakeCandidateScratch &&) = delete;
  HandshakeCandidateScratch &operator=(HandshakeCandidateScratch &&) = delete;
  ~HandshakeCandidateScratch();

  llvm::Error prepare(const FrozenSpatialHandshakeIndex &index);
  std::size_t retainedStorageBytes() const;

private:
  struct IndexDelta final {
    PnrIndex index = 0;
    PnrIndex oldValue = 0;
  };

  struct BitDelta final {
    PnrIndex index = 0;
    bool oldValue = false;
  };

  void beginTransaction();
  void resetTransaction();

  std::unique_ptr<detail::HandshakeCandidateScratchStorage> storage_;
  std::vector<std::uint64_t> fragmentJournalMarks_;
  std::vector<std::uint64_t> arcJournalMarks_;
  std::vector<std::uint64_t> traversalJournalMarks_;
  std::vector<std::uint64_t> groupJournalMarks_;
  std::vector<IndexDelta> fragmentDeltas_;
  std::vector<IndexDelta> arcDeltas_;
  std::vector<BitDelta> traversalDeltas_;
  std::vector<IndexDelta> groupDeltas_;
  std::uint64_t transactionEpoch_ = 0;
  HandshakeCandidateTransaction *activeTransaction_ = nullptr;

  friend class HandshakeCandidateState;
  friend class HandshakeCandidateTransaction;
};

class HandshakeCandidateState final
    : public std::enable_shared_from_this<HandshakeCandidateState> {
public:
  static llvm::Expected<HandshakeCandidateStateHandle>
  create(FrozenSpatialHandshakeIndexHandle index);
  static llvm::Expected<HandshakeCandidateStateHandle>
  create(const FrozenSpatialHandshakeIndex &index) = delete;
  static llvm::Expected<HandshakeCandidateStateHandle>
  create(FrozenSpatialHandshakeIndex &&index) = delete;

  HandshakeCandidateState(const HandshakeCandidateState &) = delete;
  HandshakeCandidateState(HandshakeCandidateState &&) = delete;
  HandshakeCandidateState &operator=(const HandshakeCandidateState &) = delete;
  HandshakeCandidateState &operator=(HandshakeCandidateState &&) = delete;
  ~HandshakeCandidateState() = default;

  const FrozenSpatialHandshakeIndex &index() const { return *index_; }
  PnrIndex fragmentRefcount(PnrIndex fragment) const;
  PnrIndex arcRefcount(PnrIndex arc) const;
  bool isArcActive(PnrIndex arc) const;
  bool isTraversalSelected(PnrIndex traversal) const;
  llvm::ArrayRef<PnrIndex> topologicalOrder() const;
  llvm::ArrayRef<PnrIndex> topologicalRanks() const;

  llvm::Error verify() const;
  llvm::Expected<HandshakeCandidateTransaction>
  beginTransaction(HandshakeCandidateScratch &scratch) &;
  llvm::Expected<HandshakeCandidateTransaction>
  beginTransaction(HandshakeCandidateScratch &scratch) && = delete;

private:
  HandshakeCandidateState(
      FrozenSpatialHandshakeIndexHandle index,
      std::shared_ptr<detail::IncrementalTopologicalOrder> topology,
      std::vector<PnrIndex> fragmentRefcounts,
      std::vector<PnrIndex> arcRefcounts,
      std::vector<std::uint64_t> traversalSelectedBits,
      std::vector<PnrIndex> allGroupSelectedWitnessCounts)
      : index_(std::move(index)), topology_(std::move(topology)),
        fragmentRefcounts_(std::move(fragmentRefcounts)),
        arcRefcounts_(std::move(arcRefcounts)),
        traversalSelectedBits_(std::move(traversalSelectedBits)),
        allGroupSelectedWitnessCounts_(
            std::move(allGroupSelectedWitnessCounts)) {}

  void setTraversalSelected(PnrIndex traversal, bool selected);

  FrozenSpatialHandshakeIndexHandle index_;
  std::shared_ptr<detail::IncrementalTopologicalOrder> topology_;
  std::vector<PnrIndex> fragmentRefcounts_;
  std::vector<PnrIndex> arcRefcounts_;
  std::vector<std::uint64_t> traversalSelectedBits_;
  std::vector<PnrIndex> allGroupSelectedWitnessCounts_;
  HandshakeCandidateTransaction *activeTransaction_ = nullptr;

  friend class HandshakeCandidateTransaction;
};

class HandshakeCandidateTransaction final {
public:
  HandshakeCandidateTransaction(HandshakeCandidateTransaction &&other) noexcept;
  HandshakeCandidateTransaction(const HandshakeCandidateTransaction &) = delete;
  HandshakeCandidateTransaction &
  operator=(const HandshakeCandidateTransaction &) = delete;
  HandshakeCandidateTransaction &
  operator=(HandshakeCandidateTransaction &&) = delete;
  ~HandshakeCandidateTransaction();

  llvm::Error addFragments(llvm::ArrayRef<PnrIndex> fragments);
  llvm::Error removeFragments(llvm::ArrayRef<PnrIndex> fragments);
  llvm::Error selectTraversal(PnrIndex traversal);
  llvm::Error deselectTraversal(PnrIndex traversal);

  llvm::Expected<bool> close();
  llvm::ArrayRef<PnrIndex> cycleWitness() const;
  llvm::Error commit();
  void rollback() noexcept;

private:
  HandshakeCandidateTransaction(HandshakeCandidateStateHandle state,
                                HandshakeCandidateScratch &scratch);

  llvm::Error validateFragmentSlice(llvm::ArrayRef<PnrIndex> fragments) const;
  llvm::Error changeFragment(PnrIndex fragment, bool add);
  void recordFragment(PnrIndex fragment);
  void recordArc(PnrIndex arc);
  void recordTraversal(PnrIndex traversal);
  void recordGroup(PnrIndex group);
  void finish();

  HandshakeCandidateStateHandle state_;
  HandshakeCandidateScratch *scratch_ = nullptr;
  bool closed_ = false;
  bool cycle_ = false;

  friend class HandshakeCandidateState;
  friend class HandshakeCandidateScratch;
};

} // namespace loom::pnr

#endif // LOOM_PNR_HANDSHAKECANDIDATESTATE_H
