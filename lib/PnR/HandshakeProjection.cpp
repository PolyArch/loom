#include "PnR/HandshakeCandidateState.h"

#include "Common/MappingDebugLog.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "HandshakeProjectionInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <chrono>
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

namespace loom::pnr::detail {

struct HandshakeProjectionScratchStorage final {
  RebuiltHandshakeSelection selection;
  std::vector<std::uint64_t> arcEpochs;
  std::vector<std::uint64_t> arcRefcounts;
  std::vector<PnrIndex> activeArcs;
  std::vector<std::uint64_t> nodeEpochs;
  std::vector<PnrIndex> indegree;
  std::vector<PnrIndex> activeNodes;
  std::vector<PnrIndex> ready;
  std::uint64_t projectionEpoch = 0;
};

} // namespace loom::pnr::detail

namespace {

void emitHandshakeProjectionStatistics(
    const HandshakeProjectionStatistics &statistics,
    llvm::StringRef contextKind, std::uint64_t seedAttemptOrdinal,
    std::optional<std::uint64_t> finalClosureAttemptOrdinal) {
  loom::mapping_debug::emit(
      loom::mapping_debug::Level::Summary,
      loom::mapping_debug::Stage::SpatialPnr,
      loom::mapping_debug::Event::DerivedContext,
      [&](llvm::json::Object &fields) {
        fields["context_kind"] = contextKind;
        fields["seed_attempt"] = seedAttemptOrdinal;
        if (finalClosureAttemptOrdinal)
          fields["final_closure_attempt"] = *finalClosureAttemptOrdinal;
        fields["projection_count"] = statistics.projectionCount;
        fields["construction_time_ns"] = statistics.constructionNanoseconds;
        fields["deterministic_work"] = statistics.deterministicWork;
        fields["retained_bytes"] = statistics.retainedBytes;
        fields["peak_active_node_count"] = statistics.peakActiveNodeCount;
        fields["peak_active_arc_count"] = statistics.peakActiveArcCount;
        fields["cold_verification_count"] = statistics.coldVerificationCount;
        fields["cold_verification_time_ns"] =
            statistics.coldVerificationNanoseconds;
      });
}

llvm::Error projectionError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid handshake candidate state: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

llvm::Error increment(PnrIndex &value, llvm::StringRef subject) {
  if (value == std::numeric_limits<PnrIndex>::max())
    return projectionError(subject + " refcount overflows PnrIndex");
  ++value;
  return llvm::Error::success();
}

void addWork(std::uint64_t &work, std::uint64_t amount = 1) {
  work = amount > std::numeric_limits<std::uint64_t>::max() - work
             ? std::numeric_limits<std::uint64_t>::max()
             : work + amount;
}

std::uint64_t elapsedNanoseconds(std::chrono::steady_clock::time_point begin) {
  const auto count = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now() - begin)
                         .count();
  return count <= 0 ? 0 : static_cast<std::uint64_t>(count);
}

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}


} // namespace

void loom::pnr::detail::assignNodeKey(const HandshakeNodeIdentity &identity,
                                      std::string &key) {
  key.clear();
  if (!identity.boundarySignal) {
    key.resize(13, '\0');
    key[0] = '\2';
    const auto append64 = [&](std::size_t offset, std::uint64_t value) {
      for (std::size_t byte = 0; byte != 8; ++byte)
        key[offset + byte] = static_cast<char>(value >> (56 - byte * 8));
    };
    const auto append32 = [&](std::size_t offset, std::uint32_t value) {
      for (std::size_t byte = 0; byte != 4; ++byte)
        key[offset + byte] = static_cast<char>(value >> (24 - byte * 8));
    };
    append64(1, static_cast<std::uint64_t>(identity.owner));
    append32(9, identity.localNode);
    return;
  }

  std::vector<std::uint8_t> bytes =
      ::loom::fabric::canonicalFabricBytes(identity.boundarySignal->endpoint);
  key.reserve(bytes.size() + 2);
  key.push_back('\1');
  key.append(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  key.push_back(static_cast<char>(identity.boundarySignal->signal));
}

void loom::pnr::emitProvisionalHandshakeProjectionStatistics(
    const HandshakeProjectionStatistics &statistics,
    std::uint64_t seedAttemptOrdinal) {
  emitHandshakeProjectionStatistics(statistics, "spatial_provisional_handshake",
                                    seedAttemptOrdinal, std::nullopt);
}

void loom::pnr::emitFinalClosureHandshakeProjectionStatistics(
    const HandshakeProjectionStatistics &statistics,
    std::uint64_t seedAttemptOrdinal,
    std::uint64_t finalClosureAttemptOrdinal) {
  emitHandshakeProjectionStatistics(
      statistics, "spatial_final_closure_handshake", seedAttemptOrdinal,
      finalClosureAttemptOrdinal);
}

std::string loom::pnr::detail::nodeKey(const HandshakeNodeIdentity &identity) {
  std::string key;
  assignNodeKey(identity, key);
  return key;
}

llvm::Expected<loom::pnr::detail::HandshakeNodeIdentity>
loom::pnr::detail::nodeIdentity(
    PnrIndex owner, const ::loom::fabric::HandshakeOwnerModel &model,
    std::uint32_t localNode) {
  if (localNode >= model.nodeCount())
    return projectionError("active handshake node is out of range");
  const ::loom::fabric::HandshakeOwnerNode node = model.node(localNode);
  HandshakeNodeIdentity identity;
  identity.boundarySignal = node.boundarySignal;
  if (!node.boundarySignal) {
    identity.owner = owner;
    identity.localNode = localNode;
  }
  return identity;
}

llvm::Expected<loom::pnr::detail::HandshakeArcIdentity>
loom::pnr::detail::arcIdentity(PnrIndex owner,
                               const ::loom::fabric::HandshakeOwnerModel &model,
                               const ::loom::fabric::HandshakeOwnerArc &arc) {
  auto source = nodeIdentity(owner, model, arc.source);
  if (!source)
    return source.takeError();
  auto destination = nodeIdentity(owner, model, arc.destination);
  if (!destination)
    return destination.takeError();
  return HandshakeArcIdentity{std::move(*source), std::move(*destination)};
}

llvm::Error loom::pnr::detail::rebuildHandshakeSelectionInto(
    const FrozenSpatialHandshakeIndex &index,
    llvm::ArrayRef<PnrIndex> selectedFragments,
    llvm::ArrayRef<PnrIndex> traversalUses, RebuiltHandshakeSelection &result) {
  const auto traversalFragmentOffsets = index.traversalFragmentOffsets();
  const auto traversalGroupOffsets = index.traversalAllGroupOffsets();
  if (traversalFragmentOffsets.empty() || traversalGroupOffsets.empty())
    return projectionError("projection traversal offsets are empty");
  const std::size_t traversalCount = traversalFragmentOffsets.size() - 1;
  if (traversalUses.size() != traversalCount ||
      traversalGroupOffsets.size() != traversalCount + 1)
    return projectionError(
        "projection traversal dimension does not match its index");

  result.fragmentRefcounts.assign(index.fragments().size(), 0);
  result.activeFragments.clear();
  result.traversalRefcounts.assign(traversalUses.begin(), traversalUses.end());
  result.allGroupSelectedWitnessCounts.assign(index.allTraversalGroups().size(),
                                              0);
  const auto activateFragment = [&](PnrIndex fragment) -> llvm::Error {
    if (fragment >= result.fragmentRefcounts.size())
      return projectionError("projection fragment is out of range");
    return increment(result.fragmentRefcounts[fragment], "fragment");
  };
  for (PnrIndex fragment : index.fixedFragments())
    if (llvm::Error error = activateFragment(fragment))
      return error;
  for (PnrIndex fragment : selectedFragments)
    if (llvm::Error error = activateFragment(fragment))
      return error;

  for (PnrIndex traversal = 0; traversal < traversalCount; ++traversal) {
    if (traversalUses[traversal] == 0)
      continue;
    for (PnrIndex fragment : index.traversalFragments().slice(
             traversalFragmentOffsets[traversal],
             traversalFragmentOffsets[traversal + 1] -
                 traversalFragmentOffsets[traversal]))
      if (llvm::Error error = activateFragment(fragment))
        return error;
    for (PnrIndex group : index.traversalAllGroups().slice(
             traversalGroupOffsets[traversal],
             traversalGroupOffsets[traversal + 1] -
                 traversalGroupOffsets[traversal])) {
      if (group >= result.allGroupSelectedWitnessCounts.size())
        return projectionError("projection traversal group is out of range");
      if (llvm::Error error =
              increment(result.allGroupSelectedWitnessCounts[group],
                        "all-traversal witness"))
        return error;
    }
  }
  for (auto [groupOrdinal, group] :
       llvm::enumerate(index.allTraversalGroups())) {
    const PnrIndex selected =
        result.allGroupSelectedWitnessCounts[groupOrdinal];
    if (selected > group.witnessCount)
      return projectionError(
          "projection selects excess all-traversal witnesses");
    if (selected == group.witnessCount)
      if (llvm::Error error = activateFragment(group.fragment))
        return error;
  }
  result.activeFragments.reserve(result.fragmentRefcounts.size());
  for (auto [fragment, refcount] : llvm::enumerate(result.fragmentRefcounts))
    if (refcount != 0)
      result.activeFragments.push_back(static_cast<PnrIndex>(fragment));
  return llvm::Error::success();
}

llvm::Expected<loom::pnr::detail::RebuiltHandshakeSelection>
loom::pnr::detail::rebuildHandshakeSelection(
    const FrozenSpatialHandshakeIndex &index,
    llvm::ArrayRef<PnrIndex> selectedFragments,
    llvm::ArrayRef<PnrIndex> traversalUses) {
  RebuiltHandshakeSelection result;
  if (llvm::Error error = rebuildHandshakeSelectionInto(
          index, selectedFragments, traversalUses, result))
    return std::move(error);
  return result;
}

HandshakeProjectionScratch::HandshakeProjectionScratch()
    : storage_(std::make_unique<detail::HandshakeProjectionScratchStorage>()) {}

HandshakeProjectionScratch::~HandshakeProjectionScratch() = default;

llvm::Error
HandshakeProjectionScratch::prepare(const FrozenSpatialHandshakeIndex &index) {
  preparedIndex_ = nullptr;
  if (index.traversalFragmentOffsets().empty() ||
      index.traversalAllGroupOffsets().empty())
    return projectionError("handshake projection traversal offsets are empty");
  if (index.projectionFragmentArcOffsets().size() !=
          index.fragments().size() + 1 ||
      index.projectionOutgoingArcOffsets().size() !=
          static_cast<std::size_t>(index.projectionNodeCount()) + 1)
    return projectionError("handshake dense projection index is incomplete");

  projectionCount_ = 0;
  constructionNanoseconds_ = 0;
  deterministicWork_ = 0;
  peakActiveNodeCount_ = 0;
  peakActiveArcCount_ = 0;
  coldVerificationCount_ = 0;
  coldVerificationNanoseconds_ = 0;

  detail::HandshakeProjectionScratchStorage &storage = *storage_;
  storage.selection.fragmentRefcounts.assign(index.fragments().size(), 0);
  storage.selection.activeFragments.clear();
  storage.selection.activeFragments.reserve(index.fragments().size());
  storage.selection.traversalRefcounts.assign(
      index.traversalFragmentOffsets().size() - 1, 0);
  storage.selection.allGroupSelectedWitnessCounts.assign(
      index.allTraversalGroups().size(), 0);
  storage.arcEpochs.assign(index.projectionArcs().size(), 0);
  storage.arcRefcounts.assign(index.projectionArcs().size(), 0);
  storage.activeArcs.clear();
  storage.activeArcs.reserve(index.projectionArcs().size());
  storage.nodeEpochs.assign(index.projectionNodeCount(), 0);
  storage.indegree.assign(index.projectionNodeCount(), 0);
  storage.activeNodes.clear();
  storage.activeNodes.reserve(index.projectionNodeCount());
  storage.ready.clear();
  storage.ready.reserve(index.projectionNodeCount());
  storage.projectionEpoch = 0;
  preparedIndex_ = &index;
  return llvm::Error::success();
}

llvm::Expected<bool> HandshakeProjectionScratch::projectAcyclic(
    const FrozenSpatialHandshakeIndex &index,
    llvm::ArrayRef<PnrIndex> selectedFragments,
    llvm::ArrayRef<PnrIndex> traversalUses) {
  if (preparedIndex_ != &index)
    return projectionError(
        "handshake projection scratch belongs to another frozen index");

  const auto begin = std::chrono::steady_clock::now();
  addWork(projectionCount_);
  std::uint64_t deterministicWork = 0;
  const llvm::scope_exit finishAccounting([&] {
    addWork(constructionNanoseconds_, elapsedNanoseconds(begin));
    addWork(deterministicWork_, deterministicWork);
  });

  detail::HandshakeProjectionScratchStorage &storage = *storage_;
  if (llvm::Error error = detail::rebuildHandshakeSelectionInto(
          index, selectedFragments, traversalUses, storage.selection))
    return std::move(error);

  if (++storage.projectionEpoch == 0) {
    std::fill(storage.arcEpochs.begin(), storage.arcEpochs.end(), 0);
    std::fill(storage.nodeEpochs.begin(), storage.nodeEpochs.end(), 0);
    storage.projectionEpoch = 1;
  }
  const std::uint64_t epoch = storage.projectionEpoch;
  storage.activeArcs.clear();
  storage.activeNodes.clear();
  storage.ready.clear();

  const auto arcs = index.projectionArcs();
  const auto activateNode = [&](PnrIndex node) {
    if (storage.nodeEpochs[node] == epoch)
      return;
    storage.nodeEpochs[node] = epoch;
    storage.indegree[node] = 0;
    storage.activeNodes.push_back(node);
  };
  const auto activateArc = [&](PnrIndex arc) -> llvm::Error {
    if (arc >= arcs.size())
      return projectionError("active projection arc is out of range");
    addWork(deterministicWork);
    if (storage.arcEpochs[arc] != epoch) {
      storage.arcEpochs[arc] = epoch;
      storage.arcRefcounts[arc] = 0;
      storage.activeArcs.push_back(arc);
      activateNode(arcs[arc].source);
      activateNode(arcs[arc].destination);
      if (llvm::Error error = increment(storage.indegree[arcs[arc].destination],
                                        "projection node indegree"))
        return error;
      addWork(deterministicWork);
    }
    if (storage.arcRefcounts[arc] == std::numeric_limits<std::uint64_t>::max())
      return projectionError("active projection arc refcount exceeds u64");
    ++storage.arcRefcounts[arc];
    return llvm::Error::success();
  };

  for (PnrIndex arc : index.projectionFixedArcs())
    if (llvm::Error error = activateArc(arc))
      return std::move(error);
  const auto fragmentOffsets = index.projectionFragmentArcOffsets();
  const auto fragmentArcs = index.projectionFragmentArcs();
  for (PnrIndex fragment : storage.selection.activeFragments) {
    if (fragment >= index.fragments().size())
      return projectionError("active projection fragment is out of range");
    for (PnrIndex arc : fragmentArcs.slice(fragmentOffsets[fragment],
                                           fragmentOffsets[fragment + 1] -
                                               fragmentOffsets[fragment]))
      if (llvm::Error error = activateArc(arc))
        return std::move(error);
  }

  peakActiveNodeCount_ =
      std::max<std::uint64_t>(peakActiveNodeCount_, storage.activeNodes.size());
  peakActiveArcCount_ =
      std::max<std::uint64_t>(peakActiveArcCount_, storage.activeArcs.size());
  for (PnrIndex node : storage.activeNodes) {
    if (storage.indegree[node] == 0)
      storage.ready.push_back(node);
    addWork(deterministicWork);
  }

  const auto outgoingOffsets = index.projectionOutgoingArcOffsets();
  std::size_t cursor = 0;
  while (cursor < storage.ready.size()) {
    const PnrIndex node = storage.ready[cursor++];
    for (PnrIndex arc = outgoingOffsets[node]; arc < outgoingOffsets[node + 1];
         ++arc) {
      addWork(deterministicWork);
      if (storage.arcEpochs[arc] != epoch)
        continue;
      PnrIndex &destinationIndegree = storage.indegree[arcs[arc].destination];
      if (destinationIndegree == 0)
        return projectionError("projection node indegree underflows");
      if (--destinationIndegree == 0)
        storage.ready.push_back(arcs[arc].destination);
    }
  }
  const bool acyclic = storage.ready.size() == storage.activeNodes.size();

  if (loom::mapping_debug::enabled(loom::mapping_debug::Level::Detail)) {
    const auto coldBegin = std::chrono::steady_clock::now();
    addWork(coldVerificationCount_);
    auto cold = independentlyVerifyHandshakeProjectionAcyclic(
        index, selectedFragments, traversalUses);
    addWork(coldVerificationNanoseconds_, elapsedNanoseconds(coldBegin));
    if (!cold)
      return cold.takeError();
    if (*cold != acyclic)
      return projectionError(
          "dense handshake projection disagrees with cold materialization");
  }
  return acyclic;
}

std::size_t HandshakeProjectionScratch::retainedStorageBytes() const {
  const detail::HandshakeProjectionScratchStorage &storage = *storage_;
  return retainedBytes(storage.selection.fragmentRefcounts) +
         retainedBytes(storage.selection.activeFragments) +
         retainedBytes(storage.selection.traversalRefcounts) +
         retainedBytes(storage.selection.allGroupSelectedWitnessCounts) +
         retainedBytes(storage.arcEpochs) +
         retainedBytes(storage.arcRefcounts) +
         retainedBytes(storage.activeArcs) + retainedBytes(storage.nodeEpochs) +
         retainedBytes(storage.indegree) + retainedBytes(storage.activeNodes) +
         retainedBytes(storage.ready);
}

HandshakeProjectionStatistics HandshakeProjectionScratch::statistics() const {
  HandshakeProjectionStatistics result;
  result.projectionCount = projectionCount_;
  result.constructionNanoseconds = constructionNanoseconds_;
  result.deterministicWork = deterministicWork_;
  result.retainedBytes = retainedStorageBytes();
  result.peakActiveNodeCount = peakActiveNodeCount_;
  result.peakActiveArcCount = peakActiveArcCount_;
  result.coldVerificationCount = coldVerificationCount_;
  result.coldVerificationNanoseconds = coldVerificationNanoseconds_;
  return result;
}
