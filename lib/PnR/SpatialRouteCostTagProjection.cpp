#include "SpatialRouteCostStateInternal.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

using namespace loom::pnr;
using namespace loom::pnr::detail;

std::size_t SpatialRouteCostSwitchRowState::retainedStorageBytes() const {
  std::size_t bytes =
      netDemands.capacity() *
          sizeof(std::vector<SpatialTemporalSwitchSegmentDemand>) +
      netDemandsSettled.capacity() * sizeof(std::uint8_t) +
      demandJournal.capacity() * sizeof(DemandJournal) +
      retainedSpatialTemporalSwitchDemandStorageBytes(selectedNetDemands) +
      updateDomainDemands.capacity() * sizeof(std::vector<SelectedDemandRef>) +
      updateTouchedDomains.capacity() * sizeof(PnrIndex) +
      updateDomainMarks.capacity() * sizeof(std::uint64_t) +
      updateMarginalRows.capacity() * sizeof(std::uint64_t) +
      updateSignatureViews.capacity() *
          sizeof(::loom::fabric::FabricTemporalSwitchRouteSignatureView) +
      updateDemandViews.capacity() *
          sizeof(::loom::fabric::FabricTemporalSwitchCandidateRouteDemandView) +
      updateProjectionScratch.retainedStorageBytes() +
      updateUses.capacity() * sizeof(SpatialTagDomainUse) +
      demandScratch.retainedStorageBytes();
  for (const auto &demands : netDemands)
    bytes += retainedSpatialTemporalSwitchDemandStorageBytes(demands);
  for (const DemandJournal &journal : demandJournal)
    bytes += retainedSpatialTemporalSwitchDemandStorageBytes(journal.demands);
  for (const auto &demands : updateDomainDemands)
    bytes += demands.capacity() * sizeof(SelectedDemandRef);
  return bytes;
}

llvm::Error SpatialRouteCostState::synchronizeTagProjection(
    const SpatialTagAssignmentDelta &delta,
    llvm::ArrayRef<PnrIndex> routeChangedLogicalNets) {
  if (selectedLogicalNet_)
    return routeCostStateError(
        "cannot synchronize tags while a logical net is selected");
  if (inverseTagDelta_)
    return routeCostStateError("a tag projection delta is already active");
  const std::size_t domainCount = workingTagDomainUsage_.size();
  if (!llvm::is_sorted(delta.domains) ||
      std::adjacent_find(delta.domains.begin(), delta.domains.end()) !=
          delta.domains.end() ||
      delta.domainResidentCounts.size() != delta.domains.size() ||
      delta.domainConflictCounts.size() != delta.domains.size() ||
      !llvm::is_sorted(delta.logicalNets) ||
      std::adjacent_find(delta.logicalNets.begin(), delta.logicalNets.end()) !=
          delta.logicalNets.end() ||
      delta.netDomainUseOffsets.size() != delta.logicalNets.size() + 1 ||
      delta.netDomainUseDomains.size() !=
          delta.netDomainMarginalResidentCounts.size() ||
      delta.netDomainUseOffsets.empty() ||
      delta.netDomainUseOffsets.front() != 0 ||
      delta.netDomainUseOffsets.back() != delta.netDomainUseDomains.size() ||
      delta.netUnassignedCounts.size() != delta.logicalNets.size() ||
      delta.netTagValueOffsets.size() != delta.logicalNets.size() + 1 ||
      delta.netTagValueOffsets.empty() ||
      delta.netTagValueOffsets.front() != 0 ||
      delta.netTagValueOffsets.back() != delta.netTagValues.size())
    return routeCostStateError("tag projection delta dimensions disagree");

  SpatialTagAssignmentDelta inverse;
  inverse.domains = delta.domains;
  inverse.logicalNets = delta.logicalNets;
  inverse.domainResidentCounts.reserve(delta.domains.size());
  inverse.domainConflictCounts.reserve(delta.domains.size());
  for (PnrIndex domain : delta.domains) {
    if (domain >= domainCount)
      return routeCostStateError("tag projection delta domain is out of range");
    inverse.domainResidentCounts.push_back(workingTagDomainUsage_[domain]);
    inverse.domainConflictCounts.push_back(tagDomainConflictCounts_[domain]);
  }
  inverse.netDomainUseOffsets.reserve(delta.logicalNets.size() + 1);
  inverse.netTagValueOffsets.reserve(delta.logicalNets.size() + 1);
  inverse.netDomainUseOffsets.push_back(0);
  inverse.netTagValueOffsets.push_back(0);
  const std::uint64_t oldUnassigned = tagUnassignedCount_;
  std::uint64_t oldChangedUnassigned = 0;
  std::uint64_t newChangedUnassigned = 0;
  for (auto [local, logicalNet] : llvm::enumerate(delta.logicalNets)) {
    if (logicalNet >= logicalNetCount_ ||
        logicalNet >= logicalNetTagValues_.size())
      return routeCostStateError(
          "tag projection delta logical net is out of range");
    oldChangedUnassigned = saturatedAdd(
        oldChangedUnassigned, logicalNetTagUnassignedCounts_[logicalNet]);
    const std::size_t valueBegin = delta.netTagValueOffsets[local];
    const std::size_t valueEnd = delta.netTagValueOffsets[local + 1];
    if (valueBegin > valueEnd || valueEnd > delta.netTagValues.size())
      return routeCostStateError("tag projection delta value range is invalid");
    const std::uint64_t newUnassigned = llvm::count_if(
        llvm::ArrayRef<std::optional<llvm::APInt>>(
            delta.netTagValues.data() + valueBegin, valueEnd - valueBegin),
        [](const auto &value) { return !value.has_value(); });
    if (newUnassigned != delta.netUnassignedCounts[local])
      return routeCostStateError(
          "tag projection delta unassigned count disagrees with its values");
    newChangedUnassigned = saturatedAdd(newChangedUnassigned, newUnassigned);
    inverse.netUnassignedCounts.push_back(
        logicalNetTagUnassignedCounts_[logicalNet]);
    const auto &oldValues = logicalNetTagValues_[logicalNet];
    inverse.netTagValues.insert(inverse.netTagValues.end(), oldValues.begin(),
                                oldValues.end());
    inverse.netTagValueOffsets.push_back(inverse.netTagValues.size());
    for (const SpatialTagDomainUse &use : logicalNetTagUses_[logicalNet]) {
      inverse.netDomainUseDomains.push_back(use.domain);
      inverse.netDomainMarginalResidentCounts.push_back(
          use.marginalResidentCount);
    }
    inverse.netDomainUseOffsets.push_back(inverse.netDomainUseDomains.size());
  }
  if (oldChangedUnassigned > oldUnassigned ||
      newChangedUnassigned > std::numeric_limits<std::uint64_t>::max() -
                                 (oldUnassigned - oldChangedUnassigned) ||
      oldUnassigned - oldChangedUnassigned + newChangedUnassigned !=
          delta.unassignedCount)
    return routeCostStateError(
        "tag projection delta total unassigned count is inconsistent");
  inverse.unassignedCount = oldUnassigned;
  inverseTagDelta_ = std::move(inverse);

  if (switchRows_ && switchRows_->enabled) {
    llvm::SmallVector<PnrIndex, 8> rebuildNets;
    for (PnrIndex logicalNet : delta.logicalNets)
      if (llvm::is_contained(routeChangedLogicalNets, logicalNet))
        rebuildNets.push_back(logicalNet);
    switchRows_->demandJournal.clear();
    switchRows_->demandJournal.reserve(rebuildNets.size());
    for (PnrIndex logicalNet : rebuildNets) {
      switchRows_->demandJournal.push_back(
          {logicalNet, {}, switchRows_->netDemandsSettled[logicalNet]});
      std::swap(switchRows_->demandJournal.back().demands,
                switchRows_->netDemands[logicalNet]);
      switchRows_->netDemandsSettled[logicalNet] = 0;
    }
    if (llvm::Error error = synchronizeCandidateSwitchRows(rebuildNets)) {
      llvm::Error rollback = rollbackTagProjectionDelta();
      return rollback ? llvm::joinErrors(std::move(error), std::move(rollback))
                      : std::move(error);
    }
    for (auto [local, logicalNet] : llvm::enumerate(delta.logicalNets)) {
      const std::size_t valueCount =
          delta.netTagValueOffsets[local + 1] - delta.netTagValueOffsets[local];
      for (const auto &demand : switchRows_->netDemands[logicalNet])
        if (demand.segment >= valueCount) {
          llvm::Error error = routeCostStateError(
              "switch row demand is outside the tag projection delta");
          llvm::Error rollback = rollbackTagProjectionDelta();
          return rollback
                     ? llvm::joinErrors(std::move(error), std::move(rollback))
                     : std::move(error);
        }
    }
  }

  for (auto [local, logicalNet] : llvm::enumerate(delta.logicalNets)) {
    const std::size_t valueBegin = delta.netTagValueOffsets[local];
    const std::size_t valueEnd = delta.netTagValueOffsets[local + 1];
    logicalNetTagValues_[logicalNet].assign(
        delta.netTagValues.begin() + valueBegin,
        delta.netTagValues.begin() + valueEnd);
    logicalNetTagUnassignedCounts_[logicalNet] =
        delta.netUnassignedCounts[local];
    auto &uses = logicalNetTagUses_[logicalNet];
    uses.clear();
    const std::size_t useBegin = delta.netDomainUseOffsets[local];
    const std::size_t useEnd = delta.netDomainUseOffsets[local + 1];
    if (useBegin > useEnd || useEnd > delta.netDomainUseDomains.size()) {
      llvm::Error error =
          routeCostStateError("tag projection delta use range is invalid");
      llvm::Error rollback = rollbackTagProjectionDelta();
      return rollback ? llvm::joinErrors(std::move(error), std::move(rollback))
                      : std::move(error);
    }
    uses.reserve(useEnd - useBegin);
    for (std::size_t incidence = useBegin; incidence < useEnd; ++incidence) {
      const PnrIndex domain = delta.netDomainUseDomains[incidence];
      const std::uint64_t count =
          delta.netDomainMarginalResidentCounts[incidence];
      if (domain >= domainCount || count == 0) {
        llvm::Error error =
            routeCostStateError("tag projection delta use is out of range");
        llvm::Error rollback = rollbackTagProjectionDelta();
        return rollback
                   ? llvm::joinErrors(std::move(error), std::move(rollback))
                   : std::move(error);
      }
      uses.push_back({domain, count});
    }
  }
  for (auto [local, domain] : llvm::enumerate(delta.domains))
    tagDomainConflictCounts_[domain] = delta.domainConflictCounts[local];
  tagUnassignedCount_ = delta.unassignedCount;

  beginUpdate();
  for (auto [local, domain] : llvm::enumerate(delta.domains)) {
    if (workingTagDomainUsage_[domain] == delta.domainResidentCounts[local])
      continue;
    tagDomainUpdateEpochs_[domain] = updateEpoch_;
    stagedTagDomainUsage_[domain] = delta.domainResidentCounts[local];
    affectedTagDomains_.push_back(domain);
  }
  if (llvm::Error error = finishUpdate()) {
    llvm::Error rollback = rollbackTagProjectionDelta();
    return rollback ? llvm::joinErrors(std::move(error), std::move(rollback))
                    : std::move(error);
  }
  return llvm::Error::success();
}

llvm::Error SpatialRouteCostState::commitTagProjectionDelta() {
  if (!inverseTagDelta_)
    return llvm::Error::success();
  inverseTagDelta_.reset();
  if (switchRows_)
    switchRows_->demandJournal.clear();
  return llvm::Error::success();
}

llvm::Error SpatialRouteCostState::rollbackTagProjectionDelta() {
  if (!inverseTagDelta_)
    return llvm::Error::success();
  if (selectedLogicalNet_)
    return routeCostStateError(
        "cannot roll back tags while a logical net is selected");
  SpatialTagAssignmentDelta inverse = std::move(*inverseTagDelta_);
  if (switchRows_ && switchRows_->enabled) {
    for (auto &journal : switchRows_->demandJournal) {
      std::swap(journal.demands, switchRows_->netDemands[journal.logicalNet]);
      switchRows_->netDemandsSettled[journal.logicalNet] = journal.settled;
    }
    switchRows_->demandJournal.clear();
  }
  for (auto [local, logicalNet] : llvm::enumerate(inverse.logicalNets)) {
    const std::size_t valueBegin = inverse.netTagValueOffsets[local];
    const std::size_t valueEnd = inverse.netTagValueOffsets[local + 1];
    logicalNetTagValues_[logicalNet].assign(
        inverse.netTagValues.begin() + valueBegin,
        inverse.netTagValues.begin() + valueEnd);
    logicalNetTagUnassignedCounts_[logicalNet] =
        inverse.netUnassignedCounts[local];
    auto &uses = logicalNetTagUses_[logicalNet];
    uses.clear();
    for (std::size_t incidence = inverse.netDomainUseOffsets[local];
         incidence < inverse.netDomainUseOffsets[local + 1]; ++incidence)
      uses.push_back({inverse.netDomainUseDomains[incidence],
                      inverse.netDomainMarginalResidentCounts[incidence]});
  }
  for (auto [local, domain] : llvm::enumerate(inverse.domains))
    tagDomainConflictCounts_[domain] = inverse.domainConflictCounts[local];
  tagUnassignedCount_ = inverse.unassignedCount;
  beginUpdate();
  for (auto [local, domain] : llvm::enumerate(inverse.domains)) {
    if (workingTagDomainUsage_[domain] == inverse.domainResidentCounts[local])
      continue;
    tagDomainUpdateEpochs_[domain] = updateEpoch_;
    stagedTagDomainUsage_[domain] = inverse.domainResidentCounts[local];
    affectedTagDomains_.push_back(domain);
  }
  llvm::Error result = finishUpdate();
  if (!result)
    inverseTagDelta_.reset();
  else
    inverseTagDelta_ = std::move(inverse);
  return result;
}
