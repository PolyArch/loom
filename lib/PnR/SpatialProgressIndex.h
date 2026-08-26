#ifndef LOOM_LIB_PNR_SPATIALPROGRESSINDEX_H
#define LOOM_LIB_PNR_SPATIALPROGRESSINDEX_H

#include "PnR/PnrIndex.h"

#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <vector>

namespace loom::pnr {

class FrozenSpatialRoutingGraph;

namespace detail {

class FrozenSpatialProgressIndex final {
public:
  llvm::ArrayRef<::loom::fabric::FabricFifoOccurrenceRef>
  finiteBufferOwners() const {
    return finiteBufferOwners_;
  }
  llvm::ArrayRef<PnrIndex> traversalOwnerOrdinals() const {
    return traversalOwnerOrdinals_;
  }
  llvm::ArrayRef<PnrIndex> ownerTraversalOffsets() const {
    return ownerTraversalOffsets_;
  }
  llvm::ArrayRef<PnrIndex> ownerTraversals() const {
    return ownerTraversals_;
  }
  PnrIndex traversalOwner(PnrIndex traversal) const {
    return traversal < traversalOwnerOrdinals_.size()
               ? traversalOwnerOrdinals_[traversal]
               : getInvalidPnrIndex();
  }
  llvm::ArrayRef<PnrIndex> traversalsForOwner(PnrIndex owner) const {
    if (owner >= finiteBufferOwners_.size())
      return {};
    return llvm::ArrayRef(ownerTraversals_)
        .slice(ownerTraversalOffsets_[owner],
               ownerTraversalOffsets_[owner + 1] -
                   ownerTraversalOffsets_[owner]);
  }

  llvm::Error verify(const FrozenSpatialRoutingGraph &routing) const;

private:
  std::vector<::loom::fabric::FabricFifoOccurrenceRef> finiteBufferOwners_;
  std::vector<PnrIndex> traversalOwnerOrdinals_;
  std::vector<PnrIndex> ownerTraversalOffsets_;
  std::vector<PnrIndex> ownerTraversals_;

  friend llvm::Expected<std::shared_ptr<const FrozenSpatialProgressIndex>>
  buildFrozenSpatialProgressIndex(const FrozenSpatialRoutingGraph &routing);
};

llvm::Expected<std::shared_ptr<const FrozenSpatialProgressIndex>>
buildFrozenSpatialProgressIndex(const FrozenSpatialRoutingGraph &routing);

} // namespace detail
} // namespace loom::pnr

#endif // LOOM_LIB_PNR_SPATIALPROGRESSINDEX_H
