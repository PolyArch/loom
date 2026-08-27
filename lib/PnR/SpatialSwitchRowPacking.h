#ifndef LOOM_LIB_PNR_SPATIALSWITCHROWPACKING_H
#define LOOM_LIB_PNR_SPATIALSWITCHROWPACKING_H

#include "PnR/RouteTreeState.h"
#include "PnR/SpatialTagContinuity.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::pnr::detail {

struct SpatialTemporalSwitchInputSignature final {
  ::loom::fabric::FabricSwitchOccurrenceRef occurrence;
  ::loom::fabric::FabricOrdinal input = 0;
  std::vector<::loom::fabric::FabricOrdinal> outputs;
  std::vector<PnrIndex> traversals;

  friend bool operator==(const SpatialTemporalSwitchInputSignature &lhs,
                         const SpatialTemporalSwitchInputSignature &rhs) {
    return lhs.occurrence == rhs.occurrence && lhs.input == rhs.input &&
           lhs.outputs == rhs.outputs && lhs.traversals == rhs.traversals;
  }
};

/// One continuity segment's complete row demand in one Temporal switch table.
/// Multiple input signatures are retained so re-entry cannot be mistaken for
/// one widened crossbar selection.
struct SpatialTemporalSwitchSegmentDemand final {
  PnrIndex domain = 0;
  PnrIndex logicalNet = 0;
  PnrIndex segment = 0;
  std::vector<SpatialTemporalSwitchInputSignature> signatures;

  friend bool operator==(const SpatialTemporalSwitchSegmentDemand &lhs,
                         const SpatialTemporalSwitchSegmentDemand &rhs) {
    return lhs.domain == rhs.domain && lhs.logicalNet == rhs.logicalNet &&
           lhs.segment == rhs.segment && lhs.signatures == rhs.signatures;
  }
};

struct SpatialTagVertexRef final {
  PnrIndex logicalNet = 0;
  SpatialTagContinuityOriginKind originKind =
      SpatialTagContinuityOriginKind::RouteSource;
  PnrIndex origin = 0;

  friend bool operator==(SpatialTagVertexRef lhs, SpatialTagVertexRef rhs) {
    return lhs.logicalNet == rhs.logicalNet &&
           lhs.originKind == rhs.originKind && lhs.origin == rhs.origin;
  }
  friend bool operator!=(SpatialTagVertexRef lhs, SpatialTagVertexRef rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(SpatialTagVertexRef lhs, SpatialTagVertexRef rhs) {
    return std::tie(lhs.logicalNet, lhs.originKind, lhs.origin) <
           std::tie(rhs.logicalNet, rhs.originKind, rhs.origin);
  }
};

struct SpatialTagConflictPair final {
  SpatialTagVertexRef lhs;
  SpatialTagVertexRef rhs;

  friend bool operator==(const SpatialTagConflictPair &left,
                         const SpatialTagConflictPair &right) {
    return left.lhs == right.lhs && left.rhs == right.rhs;
  }
  friend bool operator<(const SpatialTagConflictPair &left,
                        const SpatialTagConflictPair &right) {
    return std::tie(left.lhs, left.rhs) < std::tie(right.lhs, right.rhs);
  }
};

class SpatialTagInterferenceUpdateScratch;
struct SpatialTagInterferenceBuilder;

/// Exact segment interference after applying Temporal switch row compatibility.
/// Non-switch match domains remain cliques. The CSR is symmetric and excludes
/// self edges.
class SpatialTagInterferenceProjection final {
public:
  llvm::ArrayRef<PnrIndex> netSegmentOffsets() const {
    return netSegmentOffsets_;
  }
  llvm::ArrayRef<PnrIndex> conflictOffsets() const { return conflictOffsets_; }
  llvm::ArrayRef<PnrIndex> conflicts() const { return conflicts_; }
  llvm::ArrayRef<PnrIndex> conflicts(PnrIndex vertex) const;
  llvm::ArrayRef<SpatialTagVertexRef> domainVertices(PnrIndex domain) const {
    return domainVertices_[domain];
  }
  llvm::ArrayRef<SpatialTemporalSwitchSegmentDemand>
  switchDemands(PnrIndex logicalNet) const {
    return netSwitchDemands_[logicalNet];
  }
  bool interferes(PnrIndex lhs, PnrIndex rhs) const;
  bool interferes(PnrIndex domain, PnrIndex lhs, PnrIndex rhs) const;
  bool interferes(SpatialTagVertexRef lhs, SpatialTagVertexRef rhs) const;
  bool interferes(PnrIndex domain, SpatialTagVertexRef lhs,
                  SpatialTagVertexRef rhs) const;
  SpatialTagVertexRef vertexRef(PnrIndex vertex) const;
  PnrIndex vertexOrdinal(SpatialTagVertexRef vertex) const;
  bool
  equivalentDerivedState(const SpatialTagInterferenceProjection &other) const;
  std::size_t retainedStorageBytes() const;

private:
  std::vector<PnrIndex> netSegmentOffsets_;
  std::vector<SpatialTagVertexRef> vertexRefs_;
  std::vector<PnrIndex> conflictOffsets_;
  std::vector<PnrIndex> conflicts_;
  std::vector<SpatialTagConflictPair> globalConflicts_;
  std::vector<std::vector<SpatialTagVertexRef>> domainVertices_;
  std::vector<std::vector<SpatialTagConflictPair>> domainConflicts_;
  std::vector<std::vector<PnrIndex>> netDomains_;
  std::vector<std::vector<SpatialTemporalSwitchSegmentDemand>>
      netSwitchDemands_;

  friend llvm::Expected<SpatialTagInterferenceProjection>
  deriveSpatialTagInterference(
      const FrozenSpatialPnrProblem &problem,
      llvm::ArrayRef<const RouteTreeState *> routes,
      llvm::ArrayRef<const SpatialTagContinuityProjection *> continuity);
  friend llvm::Error stageSpatialTagInterferenceUpdate(
      const FrozenSpatialPnrProblem &problem,
      llvm::ArrayRef<const RouteTreeState *> routes,
      llvm::ArrayRef<const SpatialTagContinuityProjection *> continuity,
      llvm::ArrayRef<PnrIndex> touchedLogicalNets,
      SpatialTagInterferenceProjection &projection,
      SpatialTagInterferenceUpdateScratch &scratch);
  friend void rollbackSpatialTagInterferenceUpdate(
      SpatialTagInterferenceProjection &projection,
      SpatialTagInterferenceUpdateScratch &scratch) noexcept;
  friend struct SpatialTagInterferenceBuilder;
};

class SpatialTagInterferenceUpdateScratch final {
public:
  SpatialTagInterferenceUpdateScratch() = default;
  SpatialTagInterferenceUpdateScratch(
      const SpatialTagInterferenceUpdateScratch &) = delete;
  SpatialTagInterferenceUpdateScratch &
  operator=(const SpatialTagInterferenceUpdateScratch &) = delete;

  llvm::ArrayRef<PnrIndex> previousNetSegmentOffsets() const {
    return previousNetSegmentOffsets_;
  }
  llvm::ArrayRef<SpatialTagVertexRef> previousVertexRefs() const {
    return previousVertexRefs_;
  }
  llvm::ArrayRef<PnrIndex> affectedDomains() const { return affectedDomains_; }
  bool active() const { return active_; }
  std::size_t retainedStorageBytes() const;

private:
  struct DomainDelta final {
    PnrIndex domain = 0;
    std::vector<SpatialTagVertexRef> vertices;
    std::vector<SpatialTagConflictPair> conflicts;
  };
  struct NetDemandDelta final {
    PnrIndex logicalNet = 0;
    std::vector<PnrIndex> domains;
    std::vector<SpatialTemporalSwitchSegmentDemand> demands;
  };

  std::vector<PnrIndex> previousNetSegmentOffsets_;
  std::vector<SpatialTagVertexRef> previousVertexRefs_;
  std::vector<PnrIndex> previousConflictOffsets_;
  std::vector<PnrIndex> previousConflicts_;
  std::vector<SpatialTagConflictPair> previousGlobalConflicts_;
  std::vector<PnrIndex> affectedDomains_;
  std::vector<DomainDelta> domainDeltas_;
  std::vector<NetDemandDelta> netDemandDeltas_;
  bool active_ = false;

  friend llvm::Error stageSpatialTagInterferenceUpdate(
      const FrozenSpatialPnrProblem &problem,
      llvm::ArrayRef<const RouteTreeState *> routes,
      llvm::ArrayRef<const SpatialTagContinuityProjection *> continuity,
      llvm::ArrayRef<PnrIndex> touchedLogicalNets,
      SpatialTagInterferenceProjection &projection,
      SpatialTagInterferenceUpdateScratch &scratch);
  friend void commitSpatialTagInterferenceUpdate(
      SpatialTagInterferenceUpdateScratch &scratch) noexcept;
  friend void rollbackSpatialTagInterferenceUpdate(
      SpatialTagInterferenceProjection &projection,
      SpatialTagInterferenceUpdateScratch &scratch) noexcept;
};

llvm::Expected<std::vector<SpatialTemporalSwitchSegmentDemand>>
deriveSpatialTemporalSwitchSegmentDemands(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<const RouteTreeState *> routes,
    llvm::ArrayRef<const SpatialTagContinuityProjection *> continuity);

llvm::Expected<std::vector<SpatialTemporalSwitchSegmentDemand>>
deriveSpatialTemporalSwitchSegmentDemands(
    const FrozenSpatialPnrProblem &problem, PnrIndex logicalNet,
    const RouteTreeState &route,
    const SpatialTagContinuityProjection &continuity);

bool compatibleSpatialTemporalSwitchDemands(
    const SpatialTemporalSwitchSegmentDemand &lhs,
    const SpatialTemporalSwitchSegmentDemand &rhs);

llvm::Expected<SpatialTagInterferenceProjection> deriveSpatialTagInterference(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<const RouteTreeState *> routes,
    llvm::ArrayRef<const SpatialTagContinuityProjection *> continuity);

llvm::Error stageSpatialTagInterferenceUpdate(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<const RouteTreeState *> routes,
    llvm::ArrayRef<const SpatialTagContinuityProjection *> continuity,
    llvm::ArrayRef<PnrIndex> touchedLogicalNets,
    SpatialTagInterferenceProjection &projection,
    SpatialTagInterferenceUpdateScratch &scratch);
void commitSpatialTagInterferenceUpdate(
    SpatialTagInterferenceUpdateScratch &scratch) noexcept;
void rollbackSpatialTagInterferenceUpdate(
    SpatialTagInterferenceProjection &projection,
    SpatialTagInterferenceUpdateScratch &scratch) noexcept;

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALSWITCHROWPACKING_H
