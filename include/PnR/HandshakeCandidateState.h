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
struct MaterializedHandshakeGraph;
struct HandshakeCandidateScratchStorage;
struct HandshakeProjectionScratchStorage;
} // namespace detail

class HandshakeCandidateState;
class HandshakeCandidateTransaction;

using FrozenSpatialHandshakeIndexHandle =
    std::shared_ptr<const FrozenSpatialHandshakeIndex>;
using HandshakeCandidateStateHandle = std::shared_ptr<HandshakeCandidateState>;

struct HandshakeActiveDemandStatistics final {
  std::uint64_t constructionCount = 0;
  std::uint64_t constructionNanoseconds = 0;
  std::uint64_t retainedBytes = 0;
  std::uint64_t deterministicWork = 0;
  std::uint64_t activeFragmentCount = 0;
  std::uint64_t materializedNodeCount = 0;
  std::uint64_t materializedArcCount = 0;
  std::uint64_t fabricUnconditionalArcCount = 0;
  std::uint64_t materializedContributionCount = 0;
  std::uint64_t transactionClosureCount = 0;
  std::uint64_t transactionInsertedArcCount = 0;
  std::uint64_t transactionRemovedArcCount = 0;
  std::uint64_t transactionAffectedNodeCount = 0;
  std::uint64_t transactionAffectedRankSpan = 0;
  std::uint64_t cachedVerificationCount = 0;
  std::uint64_t coldVerificationConstructionCount = 0;
  std::uint64_t coldVerificationConstructionNanoseconds = 0;
};

struct HandshakeProjectionStatistics final {
  std::uint64_t projectionCount = 0;
  std::uint64_t constructionNanoseconds = 0;
  std::uint64_t deterministicWork = 0;
  std::uint64_t retainedBytes = 0;
  std::uint64_t peakActiveNodeCount = 0;
  std::uint64_t peakActiveArcCount = 0;
  std::uint64_t coldVerificationCount = 0;
  std::uint64_t coldVerificationNanoseconds = 0;
};

void emitProvisionalHandshakeProjectionStatistics(
    const HandshakeProjectionStatistics &statistics,
    std::uint64_t seedAttemptOrdinal);
void emitFinalClosureHandshakeProjectionStatistics(
    const HandshakeProjectionStatistics &statistics,
    std::uint64_t seedAttemptOrdinal, std::uint64_t finalClosureAttemptOrdinal);

/// Rebuilds the selected handshake graph from immutable projection inputs and
/// checks its closure with one deterministic whole-graph pass. This path does
/// not read or mutate candidate state.
llvm::Expected<bool> independentlyVerifyHandshakeProjectionAcyclic(
    const FrozenSpatialHandshakeIndex &index,
    llvm::ArrayRef<PnrIndex> selectedFragments,
    llvm::ArrayRef<PnrIndex> traversalUses);

/// Reusable worker-local projection of only the currently selected handshake
/// fragments. It retains storage capacity but never retains a potential union
/// graph or candidate selection between calls.
class HandshakeProjectionScratch final {
public:
  HandshakeProjectionScratch();
  HandshakeProjectionScratch(const HandshakeProjectionScratch &) = delete;
  HandshakeProjectionScratch &
  operator=(const HandshakeProjectionScratch &) = delete;
  HandshakeProjectionScratch(HandshakeProjectionScratch &&) = delete;
  HandshakeProjectionScratch &
  operator=(HandshakeProjectionScratch &&) = delete;
  ~HandshakeProjectionScratch();

  llvm::Error prepare(const FrozenSpatialHandshakeIndex &index);
  llvm::Expected<bool> projectAcyclic(
      const FrozenSpatialHandshakeIndex &index,
      llvm::ArrayRef<PnrIndex> selectedFragments,
      llvm::ArrayRef<PnrIndex> traversalUses);
  HandshakeProjectionStatistics statistics() const;
  std::size_t retainedStorageBytes() const;

private:
  std::unique_ptr<detail::HandshakeProjectionScratchStorage> storage_;
  const FrozenSpatialHandshakeIndex *preparedIndex_ = nullptr;
  std::uint64_t projectionCount_ = 0;
  std::uint64_t constructionNanoseconds_ = 0;
  std::uint64_t deterministicWork_ = 0;
  std::uint64_t peakActiveNodeCount_ = 0;
  std::uint64_t peakActiveArcCount_ = 0;
  std::uint64_t coldVerificationCount_ = 0;
  std::uint64_t coldVerificationNanoseconds_ = 0;
};

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

  void beginTransaction();
  void resetTransaction();

  std::unique_ptr<detail::HandshakeCandidateScratchStorage> storage_;
  std::vector<std::uint64_t> fragmentJournalMarks_;
  std::vector<std::uint64_t> traversalJournalMarks_;
  std::vector<std::uint64_t> groupJournalMarks_;
  std::vector<IndexDelta> fragmentDeltas_;
  std::vector<IndexDelta> traversalDeltas_;
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
  create(FrozenSpatialHandshakeIndexHandle index,
         llvm::ArrayRef<PnrIndex> selectedFragments,
         llvm::ArrayRef<PnrIndex> traversalUses);
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
  PnrIndex traversalRefcount(PnrIndex traversal) const;
  bool isTraversalSelected(PnrIndex traversal) const;
  llvm::ArrayRef<std::optional<::loom::fabric::HandshakeSignalRef>>
  activeNodeSignals() const;
  llvm::ArrayRef<FrozenSpatialHandshakeArc> activeArcs() const;
  llvm::ArrayRef<PnrIndex> activeArcContributors(PnrIndex arc) const;
  std::size_t activeArcContributionCount() const;
  HandshakeActiveDemandStatistics materializationStatistics() const;
  llvm::ArrayRef<PnrIndex> topologicalOrder() const;
  llvm::ArrayRef<PnrIndex> topologicalRanks() const;

  /// Checks only the committed incremental representation. This does not
  /// reconstruct a graph from the frozen fragment selection.
  llvm::Error verifyCachedState() const;
  /// Independently reconstructs the selected graph after checking the cached
  /// representation. Publication boundaries must use this verifier.
  llvm::Error verify() const;
  llvm::Expected<HandshakeCandidateTransaction>
  beginTransaction(HandshakeCandidateScratch &scratch) &;
  llvm::Expected<HandshakeCandidateTransaction>
  beginTransaction(HandshakeCandidateScratch &scratch) && = delete;

private:
  HandshakeCandidateState(
      FrozenSpatialHandshakeIndexHandle index,
      std::shared_ptr<detail::MaterializedHandshakeGraph> graph,
      std::vector<PnrIndex> fragmentRefcounts,
      std::vector<PnrIndex> activeFragments,
      std::vector<PnrIndex> traversalRefcounts,
      std::vector<PnrIndex> allGroupSelectedWitnessCounts)
      : index_(std::move(index)), graph_(std::move(graph)),
        fragmentRefcounts_(std::move(fragmentRefcounts)),
        activeFragments_(std::move(activeFragments)),
        traversalRefcounts_(std::move(traversalRefcounts)),
        allGroupSelectedWitnessCounts_(
            std::move(allGroupSelectedWitnessCounts)) {}

  FrozenSpatialHandshakeIndexHandle index_;
  std::shared_ptr<detail::MaterializedHandshakeGraph> graph_;
  std::vector<PnrIndex> fragmentRefcounts_;
  std::vector<PnrIndex> activeFragments_;
  std::vector<PnrIndex> traversalRefcounts_;
  std::vector<PnrIndex> allGroupSelectedWitnessCounts_;
  std::uint64_t materializationConstructionCount_ = 0;
  std::uint64_t materializationConstructionNanoseconds_ = 0;
  std::uint64_t materializationDeterministicWork_ = 0;
  std::uint64_t transactionClosureCount_ = 0;
  std::uint64_t transactionInsertedArcCount_ = 0;
  std::uint64_t transactionRemovedArcCount_ = 0;
  std::uint64_t transactionAffectedNodeCount_ = 0;
  std::uint64_t transactionAffectedRankSpan_ = 0;
  mutable std::uint64_t cachedVerificationCount_ = 0;
  mutable std::uint64_t coldVerificationConstructionCount_ = 0;
  mutable std::uint64_t coldVerificationConstructionNanoseconds_ = 0;
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
  llvm::Error addTraversalUses(PnrIndex traversal, PnrIndex count);
  llvm::Error removeTraversalUses(PnrIndex traversal, PnrIndex count);

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
  void recordTraversal(PnrIndex traversal);
  void recordGroup(PnrIndex group);
  void finish();

  HandshakeCandidateStateHandle state_;
  HandshakeCandidateScratch *scratch_ = nullptr;
  bool closed_ = false;
  bool cycle_ = false;
  bool rebuildOnCommit_ = false;
  std::shared_ptr<detail::MaterializedHandshakeGraph> pendingGraph_;

  friend class HandshakeCandidateState;
  friend class HandshakeCandidateScratch;
};

} // namespace loom::pnr

#endif // LOOM_PNR_HANDSHAKECANDIDATESTATE_H
