#ifndef LOOM_LIB_PNR_SPATIALROUTECONSTRAINTMODEL_H
#define LOOM_LIB_PNR_SPATIALROUTECONSTRAINTMODEL_H

#include "PnR/FrozenConstraintIndex.h"
#include "PnR/PnrIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace loom::pnr {

class FrozenSpatialPnrProblem;
class FrozenSpatialResourceIndex;
class FrozenSpatialRoutingGraph;
class FrozenSpatialTransferIndex;
class SpatialCandidateState;

namespace detail {

enum class SpatialRouteConstraintProjection : std::uint8_t {
  Traversal,
  ResourceState,
};

enum class SpatialRouteConstraintRelationKind : std::uint8_t {
  Equal,
  Disjoint,
};

struct SpatialRouteConstraintDomain final {
  PnrIndex valueOffset = 0;
  PnrIndex valueCount = 0;
  bool restricted = false;
};

struct SpatialRouteConstraintRelation final {
  SpatialRouteConstraintProjection projection =
      SpatialRouteConstraintProjection::Traversal;
  SpatialRouteConstraintRelationKind kind =
      SpatialRouteConstraintRelationKind::Equal;
  PnrIndex memberOffset = 0;
  PnrIndex memberCount = 0;
};

/// Removable dense projection of route-set constraints. RouteTree remains the
/// only selected traversal authority; this model contains only exact K-to-F
/// domains and relation incidence needed by hot search.
class SpatialRouteConstraintModel final {
public:
  static llvm::Expected<std::shared_ptr<const SpatialRouteConstraintModel>>
  create(const ArtifactIdentity &dataflowIdentity,
         const FrozenConstraintIndex &constraints,
         const FrozenSpatialTransferIndex &transfers,
         const FrozenSpatialResourceIndex &resources,
         const FrozenSpatialRoutingGraph &routing);

  llvm::ArrayRef<PnrIndex> equalityClosure(PnrIndex logicalNet) const;
  llvm::ArrayRef<PnrIndex> netRelations(PnrIndex logicalNet) const;
  llvm::ArrayRef<SpatialRouteConstraintRelation> relations() const {
    return relations_;
  }
  llvm::ArrayRef<PnrIndex>
  relationMembers(const SpatialRouteConstraintRelation &relation) const;
  bool netHasConstraints(PnrIndex logicalNet) const;

private:
  std::vector<SpatialRouteConstraintDomain> traversalDomains_;
  std::vector<PnrIndex> traversalDomainValues_;
  std::vector<SpatialRouteConstraintDomain> resourceStateDomains_;
  std::vector<PnrIndex> resourceStateDomainValues_;
  std::vector<SpatialRouteConstraintRelation> relations_;
  std::vector<PnrIndex> relationMembers_;
  std::vector<PnrIndex> netRelationOffsets_;
  std::vector<PnrIndex> netRelations_;
  std::vector<PnrIndex> netEqualityComponents_;
  std::vector<PnrIndex> equalityComponentOffsets_;
  std::vector<PnrIndex> equalityComponentMembers_;
  std::vector<std::uint8_t> netConstraintFlags_;

  friend class SpatialRouteConstraintScratch;
};

/// Allocation-free worker-local route constraint projection and verifier.
class SpatialRouteConstraintScratch final {
public:
  llvm::Error prepare(const FrozenSpatialPnrProblem &problem);
  llvm::Error beginSweep(llvm::ArrayRef<PnrIndex> logicalNets);
  llvm::Expected<llvm::ArrayRef<std::uint64_t>>
  eligibleTraversals(const SpatialCandidateState &candidate,
                     PnrIndex logicalNet);
  llvm::Error finishNet(PnrIndex logicalNet);
  llvm::Error verifyAll(const SpatialCandidateState &candidate);
  llvm::Error verifyAffected(const SpatialCandidateState &candidate,
                             llvm::ArrayRef<PnrIndex> logicalNets);
  std::size_t retainedStorageBytes() const;

private:
  void clearBits(std::vector<std::uint64_t> &bits);
  llvm::Error collectSelected(const SpatialCandidateState &candidate,
                              PnrIndex logicalNet,
                              SpatialRouteConstraintProjection projection,
                              std::vector<std::uint64_t> &bits);
  bool traversalAllowedByResourceBits(PnrIndex traversal,
                                      llvm::ArrayRef<std::uint64_t> bits,
                                      bool requireSubset) const;
  llvm::Error verifyNetDomains(const SpatialCandidateState &candidate,
                               PnrIndex logicalNet);
  llvm::Error verifyRelation(const SpatialCandidateState &candidate,
                             PnrIndex relation);

  const FrozenSpatialPnrProblem *problem_ = nullptr;
  const SpatialRouteConstraintModel *model_ = nullptr;
  std::vector<std::uint64_t> eligibleTraversalBits_;
  std::vector<std::uint64_t> selectedTraversalBits_;
  std::vector<std::uint64_t> selectedResourceStateBits_;
  std::vector<std::uint64_t> referenceBits_;
  std::vector<std::uint64_t> seenBits_;
  std::vector<std::uint8_t> pendingNets_;
  std::vector<std::uint64_t> relationMarks_;
  std::vector<PnrIndex> affectedRelations_;
  std::uint64_t relationEpoch_ = 0;
  bool sweepActive_ = false;
};

} // namespace detail
} // namespace loom::pnr

#endif // LOOM_LIB_PNR_SPATIALROUTECONSTRAINTMODEL_H
