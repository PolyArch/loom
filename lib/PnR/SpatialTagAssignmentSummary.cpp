#include "SpatialTagAssignmentState.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

using namespace loom::pnr;
using namespace loom::pnr::detail;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 "invalid Spatial tag assignment: %s",
                                 message.str().c_str());
}

// Domain membership is independent of the marginal rows removed with one
// net. A shared row can cost zero while still coupling every participating net.
void completeTagDomainIncidence(
    const SpatialTagNetState &net,
    llvm::ArrayRef<::loom::fabric::FabricPhysicalTagMatchDomainView> domains,
    std::map<PnrIndex, std::uint64_t> &marginalRows) {
  for (PnrIndex segment = 0; segment < net.values.size(); ++segment)
    for (PnrIndex domain : tagSegmentDomains(net, segment)) {
      std::uint64_t &count = marginalRows[domain];
      if (!net.values[segment] ||
          domains[domain].kind !=
              ::loom::fabric::FabricPhysicalTagMatchDomainKind::
                  TemporalSwitchTable)
        ++count;
    }
}

} // namespace

std::uint64_t loom::pnr::detail::tagDomainConflictCount(
    llvm::ArrayRef<SpatialTagDomainOccupancy> occupancy,
    const SpatialTagInterferenceProjection &interference, PnrIndex domain) {
  assert(domain < occupancy.size());
  std::uint64_t conflicts = 0;
  for (const auto &entry : occupancy[domain])
    for (std::size_t lhs = 0; lhs != entry.second.size(); ++lhs)
      for (std::size_t rhs = lhs + 1; rhs != entry.second.size(); ++rhs)
        conflicts += interference.interferes(domain, entry.second[lhs],
                                             entry.second[rhs]);
  return conflicts;
}

llvm::Expected<SpatialTagAssignmentSummary>
loom::pnr::detail::summarizeTagAssignmentState(
    const SpatialTagAssignmentStateStorage &storage,
    bool includeDomainDetails) {
  SpatialTagAssignmentSummary summary;
  summary.unassignedCount = storage.unassignedCount;
  summary.conflictCount = storage.conflictCount;
  summary.residentCapacityOveruse = storage.residentCapacityOveruse;
  if (!includeDomainDetails)
    return summary;
  summary.domainResidentCounts.reserve(storage.residentCounts.size());
  summary.domainConflictCounts.reserve(storage.occupancy.size());
  for (PnrIndex domain = 0; domain < storage.residentCounts.size(); ++domain) {
    summary.domainResidentCounts.push_back(storage.residentCounts[domain]);
    summary.domainConflictCounts.push_back(tagDomainConflictCount(
        storage.occupancy, storage.interference, domain));
  }

  summary.netDomainUseOffsets.reserve(storage.nets.size() + 1);
  summary.netUnassignedCounts.reserve(storage.nets.size());
  summary.netTagValueOffsets.reserve(storage.nets.size() + 1);
  summary.netDomainUseOffsets.push_back(0);
  summary.netTagValueOffsets.push_back(0);
  std::vector<std::map<PnrIndex, std::uint64_t>> marginalRows(
      storage.nets.size());
  const auto matchDomains =
      storage.problem->routing().tagContinuity().matchDomains();
  for (PnrIndex domain = 0; domain < storage.occupancy.size(); ++domain) {
    if (matchDomains[domain].kind !=
        ::loom::fabric::FabricPhysicalTagMatchDomainKind::TemporalSwitchTable)
      continue;
    for (const auto &entry : storage.occupancy[domain]) {
      std::optional<PnrIndex> soleNet;
      bool shared = false;
      for (const auto &vertex : entry.second) {
        const PnrIndex logicalNet = vertex.logicalNet;
        if (logicalNet >= storage.nets.size())
          return invalid("tag occupancy vertex is outside its logical net");
        if (!soleNet)
          soleNet = logicalNet;
        else if (*soleNet != logicalNet)
          shared = true;
      }
      if (soleNet && !shared)
        ++marginalRows[*soleNet][domain];
    }
  }
  for (PnrIndex logicalNet = 0; logicalNet < storage.nets.size();
       ++logicalNet) {
    const SpatialTagNetState &net = storage.nets[logicalNet];
    summary.netUnassignedCounts.push_back(llvm::count_if(
        net.values, [](const auto &value) { return !value.has_value(); }));
    summary.netTagValues.insert(summary.netTagValues.end(), net.values.begin(),
                                net.values.end());
    completeTagDomainIncidence(net, matchDomains, marginalRows[logicalNet]);
    for (const auto &[domain, count] : marginalRows[logicalNet]) {
      summary.netDomainUseDomains.push_back(domain);
      summary.netDomainMarginalResidentCounts.push_back(count);
    }
    summary.netDomainUseOffsets.push_back(summary.netDomainUseDomains.size());
    summary.netTagValueOffsets.push_back(summary.netTagValues.size());
  }
  return summary;
}

llvm::Expected<SpatialTagAssignmentDelta>
loom::pnr::detail::summarizeTagAssignmentDelta(
    const SpatialTagAssignmentStateStorage &storage,
    llvm::ArrayRef<PnrIndex> logicalNets,
    llvm::ArrayRef<PnrIndex> changedDomains) {
  if (!llvm::is_sorted(logicalNets) ||
      std::adjacent_find(logicalNets.begin(), logicalNets.end()) !=
          logicalNets.end() ||
      !llvm::is_sorted(changedDomains) ||
      std::adjacent_find(changedDomains.begin(), changedDomains.end()) !=
          changedDomains.end())
    return invalid("tag assignment delta inventory is not canonical");

  SpatialTagAssignmentDelta delta;
  delta.unassignedCount = storage.unassignedCount;
  delta.domains.assign(changedDomains.begin(), changedDomains.end());
  delta.logicalNets.assign(logicalNets.begin(), logicalNets.end());
  delta.domainResidentCounts.reserve(changedDomains.size());
  delta.domainConflictCounts.reserve(changedDomains.size());
  for (PnrIndex domain : changedDomains) {
    if (domain >= storage.residentCounts.size() ||
        domain >= storage.occupancy.size())
      return invalid("tag assignment delta domain is out of range");
    delta.domainResidentCounts.push_back(storage.residentCounts[domain]);
    delta.domainConflictCounts.push_back(tagDomainConflictCount(
        storage.occupancy, storage.interference, domain));
  }

  std::vector<std::map<PnrIndex, std::uint64_t>> marginalRows(
      logicalNets.size());
  const auto matchDomains =
      storage.problem->routing().tagContinuity().matchDomains();
  std::vector<PnrIndex> temporalDomains;
  for (PnrIndex logicalNet : logicalNets) {
    if (logicalNet >= storage.nets.size())
      return invalid("tag assignment delta logical net is out of range");
    const auto &net = storage.nets[logicalNet];
    for (PnrIndex segment = 0; segment < net.values.size(); ++segment)
      for (PnrIndex domain : tagSegmentDomains(net, segment)) {
        if (domain >= matchDomains.size())
          return invalid("tag assignment delta use is out of range");
        if (matchDomains[domain].kind ==
            ::loom::fabric::FabricPhysicalTagMatchDomainKind::
                TemporalSwitchTable)
          temporalDomains.push_back(domain);
      }
  }
  llvm::sort(temporalDomains);
  temporalDomains.erase(
      std::unique(temporalDomains.begin(), temporalDomains.end()),
      temporalDomains.end());
  for (PnrIndex domain : temporalDomains)
    for (const auto &entry : storage.occupancy[domain]) {
      std::optional<PnrIndex> soleNet;
      bool shared = false;
      for (const auto &vertex : entry.second) {
        if (!soleNet)
          soleNet = vertex.logicalNet;
        else if (*soleNet != vertex.logicalNet)
          shared = true;
      }
      if (!soleNet || shared)
        continue;
      const auto found = llvm::lower_bound(logicalNets, *soleNet);
      if (found != logicalNets.end() && *found == *soleNet)
        ++marginalRows[found - logicalNets.begin()][domain];
    }

  delta.netDomainUseOffsets.reserve(logicalNets.size() + 1);
  delta.netUnassignedCounts.reserve(logicalNets.size());
  delta.netTagValueOffsets.reserve(logicalNets.size() + 1);
  delta.netDomainUseOffsets.push_back(0);
  delta.netTagValueOffsets.push_back(0);
  for (auto [local, logicalNet] : llvm::enumerate(logicalNets)) {
    const auto &net = storage.nets[logicalNet];
    const std::uint64_t unassigned = llvm::count_if(
        net.values, [](const auto &value) { return !value.has_value(); });
    delta.netUnassignedCounts.push_back(unassigned);
    delta.netTagValues.insert(delta.netTagValues.end(), net.values.begin(),
                              net.values.end());
    completeTagDomainIncidence(net, matchDomains, marginalRows[local]);
    for (const auto &[domain, count] : marginalRows[local]) {
      delta.netDomainUseDomains.push_back(domain);
      delta.netDomainMarginalResidentCounts.push_back(count);
    }
    delta.netDomainUseOffsets.push_back(delta.netDomainUseDomains.size());
    delta.netTagValueOffsets.push_back(delta.netTagValues.size());
  }
  return delta;
}
