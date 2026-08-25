#include "SpatialProgressIndex.h"

#include "PnR/SpatialPnrProblem.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

using namespace loom::fabric;
using namespace loom::pnr;

namespace {

constexpr llvm::StringLiteral frozenArtifact = "FrozenSpatialProgressIndex";
constexpr PnrCapacityContext ownerCountContext{
    frozenArtifact, "finite_buffer_owners", "finite_buffer_owners",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext traversalCountContext{
    frozenArtifact, "traversal_owner_ordinals", "physical_traversals",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext ownerOffsetContext{
    frozenArtifact, "owner_traversal_offsets", "physical_traversals",
    PnrCapacityMeasure::Offset};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "frozen_spatial_progress_index_invalid: " + message);
}

std::string ownerKey(const FabricFifoOccurrenceRef &owner) {
  const std::vector<std::uint8_t> bytes = canonicalFabricBytes(owner);
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

const FabricFifoTraversalPayload *
bufferedPayload(const FrozenSpatialTraversal &traversal) {
  const auto *fifo =
      std::get_if<FabricFifoTraversalPayload>(&traversal.reference.payload);
  return fifo && fifo->mode == FabricFifoTraversalMode::Buffered ? fifo
                                                                  : nullptr;
}

} // namespace

llvm::Expected<std::shared_ptr<
    const loom::pnr::detail::FrozenSpatialProgressIndex>>
loom::pnr::detail::buildFrozenSpatialProgressIndex(
    const FrozenSpatialRoutingGraph &routing) {
  auto traversalCount = checkedPnrIndex(traversalCountContext,
                                        routing.traversals().size());
  if (!traversalCount)
    return traversalCount.takeError();

  auto result = std::make_shared<FrozenSpatialProgressIndex>();
  result->traversalOwnerOrdinals_.assign(*traversalCount,
                                         getInvalidPnrIndex());
  llvm::StringMap<PnrIndex> ownerOrdinals;
  for (auto [traversalOrdinal, traversal] :
       llvm::enumerate(routing.traversals())) {
    const FabricFifoTraversalPayload *fifo = bufferedPayload(traversal);
    if (!fifo)
      continue;
    const std::string key = ownerKey(fifo->owner);
    auto inserted = ownerOrdinals.try_emplace(key, getInvalidPnrIndex());
    if (inserted.second) {
      auto owner = checkedPnrIndex(ownerCountContext,
                                   result->finiteBufferOwners_.size());
      if (!owner)
        return owner.takeError();
      inserted.first->second = *owner;
      result->finiteBufferOwners_.push_back(fifo->owner);
    }
    result->traversalOwnerOrdinals_[traversalOrdinal] =
        inserted.first->second;
  }

  result->ownerTraversalOffsets_.assign(
      result->finiteBufferOwners_.size() + 1, 0);
  for (PnrIndex owner : result->traversalOwnerOrdinals_) {
    if (owner == getInvalidPnrIndex())
      continue;
    if (owner + 1 >= result->ownerTraversalOffsets_.size())
      return invalid("traversal names an out-of-range finite-buffer owner");
    PnrIndex &offset = result->ownerTraversalOffsets_[owner + 1];
    if (offset == std::numeric_limits<PnrIndex>::max())
      return invalid("finite-buffer traversal count exceeds PnrIndex");
    ++offset;
  }
  for (std::size_t owner = 0;
       owner + 1 < result->ownerTraversalOffsets_.size(); ++owner) {
    auto offset = checkedPnrIndexAdd(
        ownerOffsetContext, result->ownerTraversalOffsets_[owner],
        result->ownerTraversalOffsets_[owner + 1]);
    if (!offset)
      return offset.takeError();
    result->ownerTraversalOffsets_[owner + 1] = *offset;
  }
  result->ownerTraversals_.assign(
      result->ownerTraversalOffsets_.empty()
          ? 0
          : result->ownerTraversalOffsets_.back(),
      getInvalidPnrIndex());
  std::vector<PnrIndex> cursors(result->ownerTraversalOffsets_.begin(),
                                result->ownerTraversalOffsets_.end());
  for (PnrIndex traversal = 0;
       traversal < result->traversalOwnerOrdinals_.size(); ++traversal) {
    const PnrIndex owner = result->traversalOwnerOrdinals_[traversal];
    if (owner == getInvalidPnrIndex())
      continue;
    result->ownerTraversals_[cursors[owner]++] = traversal;
  }
  if (llvm::Error error = result->verify(routing))
    return std::move(error);
  return std::shared_ptr<const FrozenSpatialProgressIndex>(std::move(result));
}

llvm::Error loom::pnr::detail::FrozenSpatialProgressIndex::verify(
    const FrozenSpatialRoutingGraph &routing) const {
  if (traversalOwnerOrdinals_.size() != routing.traversals().size())
    return invalid("traversal owner table does not cover the routing graph");
  if (ownerTraversalOffsets_.size() != finiteBufferOwners_.size() + 1 ||
      ownerTraversalOffsets_.empty() || ownerTraversalOffsets_.front() != 0 ||
      ownerTraversalOffsets_.back() != ownerTraversals_.size())
    return invalid("finite-buffer owner traversal CSR has invalid bounds");

  std::vector<std::uint8_t> seen(routing.traversals().size(), 0);
  for (PnrIndex owner = 0; owner < finiteBufferOwners_.size(); ++owner) {
    PnrIndex previous = getInvalidPnrIndex();
    for (PnrIndex traversal : traversalsForOwner(owner)) {
      if (traversal >= routing.traversals().size() || seen[traversal])
        return invalid("finite-buffer owner traversal CSR is not unique");
      if (previous != getInvalidPnrIndex() && traversal <= previous)
        return invalid("finite-buffer owner traversals are not canonical");
      previous = traversal;
      seen[traversal] = 1;
      const FabricFifoTraversalPayload *fifo =
          bufferedPayload(routing.traversals()[traversal]);
      if (!fifo || fifo->owner != finiteBufferOwners_[owner] ||
          traversalOwnerOrdinals_[traversal] != owner)
        return invalid("finite-buffer owner CSR disagrees with its traversal");
    }
  }
  for (PnrIndex traversal = 0; traversal < routing.traversals().size();
       ++traversal) {
    const FabricFifoTraversalPayload *fifo =
        bufferedPayload(routing.traversals()[traversal]);
    const PnrIndex owner = traversalOwnerOrdinals_[traversal];
    if (!fifo) {
      if (owner != getInvalidPnrIndex() || seen[traversal])
        return invalid("non-buffered traversal has a finite-buffer owner");
      continue;
    }
    if (owner >= finiteBufferOwners_.size() || !seen[traversal] ||
        finiteBufferOwners_[owner] != fifo->owner)
      return invalid("buffered traversal has no exact finite-buffer owner");
  }
  return llvm::Error::success();
}
