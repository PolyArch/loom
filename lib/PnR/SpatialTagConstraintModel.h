#ifndef LOOM_LIB_PNR_SPATIALTAGCONSTRAINTMODEL_H
#define LOOM_LIB_PNR_SPATIALTAGCONSTRAINTMODEL_H

#include "Common/Artifact.h"
#include "PnR/FrozenConstraintIndex.h"
#include "PnR/PnrIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <vector>

namespace loom::pnr {

class FrozenSpatialTransferIndex;

namespace detail {

/// Dense removable relation index for the net-assigned Physical Tag set.
/// SpatialTagAssignmentState remains the sole selected-value authority.
class SpatialTagConstraintModel final {
public:
  static llvm::Expected<std::shared_ptr<const SpatialTagConstraintModel>>
  create(const ArtifactIdentity &dataflowIdentity,
         const FrozenSpatialTransferIndex &transfers,
         const FrozenConstraintIndex &constraints);

  bool hasRelations() const { return hasRelations_; }
  PnrIndex classCount() const {
    return static_cast<PnrIndex>(classMemberOffsets_.size() - 1);
  }
  PnrIndex classOfNet(PnrIndex logicalNet) const {
    return netClasses_[logicalNet];
  }
  llvm::ArrayRef<PnrIndex> classMembers(PnrIndex equalityClass) const;
  llvm::ArrayRef<PnrIndex> classDisjointGroups(PnrIndex equalityClass) const;
  llvm::ArrayRef<PnrIndex> disjointGroupMembers(PnrIndex group) const;
  bool netHasRelations(PnrIndex logicalNet) const {
    const PnrIndex equalityClass = classOfNet(logicalNet);
    return classMembers(equalityClass).size() > 1 ||
           !classDisjointGroups(equalityClass).empty();
  }

private:
  std::vector<PnrIndex> netClasses_;
  std::vector<PnrIndex> classMemberOffsets_;
  std::vector<PnrIndex> classMembers_;
  std::vector<PnrIndex> classGroupOffsets_;
  std::vector<PnrIndex> classGroups_;
  std::vector<PnrIndex> groupMemberOffsets_;
  std::vector<PnrIndex> groupMembers_;
  bool hasRelations_ = false;
};

} // namespace detail
} // namespace loom::pnr

#endif // LOOM_LIB_PNR_SPATIALTAGCONSTRAINTMODEL_H
