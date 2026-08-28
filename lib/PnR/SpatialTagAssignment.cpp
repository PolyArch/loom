#include "PnR/SpatialTagAssignment.h"

#include "Fabric/IR/PhysicalTag.h"

#include "SpatialSwitchRowPacking.h"
#include "SpatialTagAssignmentState.h"
#include "SpatialTagColoring.h"
#include "SpatialTagConstraintModel.h"

#include "Fabric/Identity/FabricTemporalSwitchRoute.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Errc.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <system_error>
#include <tuple>
#include <variant>
#include <vector>

using namespace loom;
using namespace loom::mapping;
using namespace loom::pnr;

namespace {

using TagDomainOccupancy = ::loom::pnr::detail::SpatialTagDomainOccupancy;

constexpr llvm::StringLiteral tagAssignment = "SpatialTagAssignmentProjection";
constexpr PnrCapacityContext segmentCountContext{tagAssignment, "segments",
                                                 "tag_continuity_segments",
                                                 PnrCapacityMeasure::Count};
constexpr PnrCapacityContext segmentOffsetContext{
    tagAssignment, "net_segment_offsets", "tag_continuity_segments",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext incidenceCountContext{
    tagAssignment, "segment_domains", "segment_domain_incidence",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext colorIntervalCountContext{
    tagAssignment, "color_intervals", "tag_color_intervals",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext matchDomainCountContext{
    tagAssignment, "match_domains", "tag_match_domains",
    PnrCapacityMeasure::Count};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 "invalid Spatial tag assignment: %s",
                                 message.str().c_str());
}

llvm::Error unsupported(const llvm::Twine &message) {
  return llvm::createStringError(llvm::errc::not_supported,
                                 "Spatial tag assignment unavailable: %s",
                                 message.str().c_str());
}

int compareUnsigned(const llvm::APInt &lhs, const llvm::APInt &rhs) {
  return ::fabric::comparePhysicalTagValues(lhs, rhs);
}

llvm::APInt canonicalUnsigned(const llvm::APInt &value) {
  const unsigned width = std::max(1u, value.getActiveBits());
  return value.zextOrTrunc(width);
}

bool unsignedLess(const llvm::APInt &lhs, const llvm::APInt &rhs) {
  return compareUnsigned(lhs, rhs) < 0;
}

void normalizeValues(std::vector<llvm::APInt> &values) {
  for (llvm::APInt &value : values)
    value = canonicalUnsigned(value);
  llvm::sort(values, unsignedLess);
  values.erase(std::unique(values.begin(), values.end(),
                           [](const llvm::APInt &lhs, const llvm::APInt &rhs) {
                             return compareUnsigned(lhs, rhs) == 0;
                           }),
               values.end());
}

bool equalUnsignedValues(llvm::ArrayRef<llvm::APInt> lhs,
                         llvm::ArrayRef<llvm::APInt> rhs) {
  return lhs.size() == rhs.size() &&
         llvm::equal(lhs, rhs,
                     [](const llvm::APInt &left, const llvm::APInt &right) {
                       return compareUnsigned(left, right) == 0;
                     });
}

bool equalOptionalUnsignedValues(
    llvm::ArrayRef<std::optional<llvm::APInt>> lhs,
    llvm::ArrayRef<std::optional<llvm::APInt>> rhs) {
  return lhs.size() == rhs.size() &&
         llvm::equal(lhs, rhs, [](const auto &left, const auto &right) {
           return left.has_value() == right.has_value() &&
                  (!left || compareUnsigned(*left, *right) == 0);
         });
}

bool containsValue(llvm::ArrayRef<llvm::APInt> values,
                   const llvm::APInt &value) {
  const auto found = llvm::lower_bound(values, value, unsignedLess);
  return found != values.end() && compareUnsigned(*found, value) == 0;
}

llvm::Expected<llvm::APInt> nextUnsigned(const llvm::APInt &value) {
  if (value.getBitWidth() == std::numeric_limits<unsigned>::max())
    return invalid("Physical Tag candidate width overflows");
  llvm::APInt next =
      value.isAllOnes() ? value.zext(value.getBitWidth() + 1) : value;
  ++next;
  return canonicalUnsigned(next);
}

bool isFree(llvm::ArrayRef<PnrIndex> domains, const llvm::APInt &value,
            llvm::ArrayRef<TagDomainOccupancy> occupancy) {
  return llvm::all_of(domains, [&](PnrIndex domain) {
    const auto found = occupancy[domain].find(value);
    return found == occupancy[domain].end() || found->second.empty();
  });
}

std::uint64_t conflictCost(llvm::ArrayRef<PnrIndex> domains,
                           const llvm::APInt &value,
                           llvm::ArrayRef<TagDomainOccupancy> occupancy) {
  return llvm::count_if(domains, [&](PnrIndex domain) {
    const auto found = occupancy[domain].find(value);
    return found != occupancy[domain].end() && !found->second.empty();
  });
}

llvm::Expected<std::optional<llvm::APInt>>
chooseValue(std::uint32_t tagWidthBits, llvm::ArrayRef<PnrIndex> domains,
            const std::optional<llvm::ArrayRef<SpatialConstraintDomainValue>>
                &restriction,
            llvm::ArrayRef<llvm::APInt> forbidden,
            llvm::ArrayRef<TagDomainOccupancy> occupancy) {
  std::optional<llvm::APInt> bestAllowed;
  std::uint64_t bestCost = std::numeric_limits<std::uint64_t>::max();
  const auto considerRange = [&](llvm::APInt candidate,
                                 const llvm::APInt *upper)
      -> llvm::Expected<std::optional<llvm::APInt>> {
    candidate = canonicalUnsigned(candidate);
    while ((!upper || compareUnsigned(candidate, *upper) < 0) &&
           ::fabric::isRepresentablePhysicalTagValue(tagWidthBits, candidate)) {
      if (containsValue(forbidden, candidate)) {
        auto next = nextUnsigned(candidate);
        if (!next)
          return next.takeError();
        candidate = std::move(*next);
        continue;
      }
      const std::uint64_t cost = conflictCost(domains, candidate, occupancy);
      if (cost < bestCost) {
        bestAllowed = candidate;
        bestCost = cost;
      }
      if (cost == 0)
        return std::optional<llvm::APInt>(std::move(candidate));
      auto next = nextUnsigned(candidate);
      if (!next)
        return next.takeError();
      candidate = std::move(*next);
    }
    return std::optional<llvm::APInt>();
  };

  if (!restriction) {
    auto selected = considerRange(llvm::APInt(1, 0), nullptr);
    if (!selected)
      return selected.takeError();
    if (*selected)
      return std::move(*selected);
    return std::move(bestAllowed);
  }

  for (const SpatialConstraintDomainValue &domainValue : *restriction) {
    const auto *interval =
        std::get_if<SpatialConstraintUnsignedInterval>(&domainValue);
    if (!interval)
      return invalid("tag restriction contains a non-interval value");
    auto selected = considerRange(interval->lower, &interval->upper);
    if (!selected)
      return selected.takeError();
    if (*selected)
      return std::move(*selected);
    if (!::fabric::isRepresentablePhysicalTagValue(tagWidthBits,
                                                   interval->lower))
      break;
  }
  return std::move(bestAllowed);
}

llvm::Expected<PnrIndex> checkedIndex(PnrCapacityContext context,
                                      std::size_t value) {
  return checkedPnrIndex(context, static_cast<std::uint64_t>(value));
}

bool valueAllowed(
    const llvm::APInt &value,
    const std::optional<llvm::ArrayRef<SpatialConstraintDomainValue>>
        &restriction) {
  if (!restriction)
    return true;
  return llvm::any_of(*restriction, [&](const auto &domainValue) {
    const auto *interval =
        std::get_if<SpatialConstraintUnsignedInterval>(&domainValue);
    return interval && compareUnsigned(interval->lower, value) <= 0 &&
           compareUnsigned(value, interval->upper) < 0;
  });
}

llvm::Error addAssignment(
    llvm::MutableArrayRef<TagDomainOccupancy> occupancy,
    const ::loom::pnr::detail::SpatialTagInterferenceProjection &interference,
    ::loom::pnr::detail::SpatialTagVertexRef vertex,
    std::uint64_t &unassignedCount, std::uint64_t &conflictCount,
    llvm::ArrayRef<PnrIndex> domains, const std::optional<llvm::APInt> &value) {
  if (!value) {
    if (unassignedCount == std::numeric_limits<std::uint64_t>::max())
      return invalid("unassigned count overflows u64");
    ++unassignedCount;
    return llvm::Error::success();
  }

  for (PnrIndex domain : domains) {
    if (domain >= occupancy.size())
      return invalid("assignment names an out-of-range tag match domain");
    const auto found = occupancy[domain].find(*value);
    if (found == occupancy[domain].end())
      continue;
    if (found->second.size() >= getInvalidPnrIndex() - PnrIndex{1})
      return invalid("tag match-domain occupancy overflows PnrIndex");
    const auto member = llvm::lower_bound(found->second, vertex);
    if (member != found->second.end() && *member == vertex)
      return invalid("tag match-domain occupancy repeats a vertex");
  }
  std::vector<::loom::pnr::detail::SpatialTagVertexRef> peers;
  for (PnrIndex domain : domains) {
    const auto found = occupancy[domain].find(*value);
    if (found != occupancy[domain].end())
      peers.insert(peers.end(), found->second.begin(), found->second.end());
  }
  llvm::sort(peers);
  peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
  const std::uint64_t addedConflicts = llvm::count_if(
      peers, [&](auto peer) { return interference.interferes(vertex, peer); });
  if (addedConflicts >
      std::numeric_limits<std::uint64_t>::max() - conflictCount)
    return invalid("tag conflict count overflows u64");
  conflictCount += addedConflicts;
  for (PnrIndex domain : domains) {
    auto &members = occupancy[domain][*value];
    members.insert(llvm::lower_bound(members, vertex), vertex);
  }
  return llvm::Error::success();
}

void removeAssignment(
    llvm::MutableArrayRef<TagDomainOccupancy> occupancy,
    const ::loom::pnr::detail::SpatialTagInterferenceProjection &interference,
    ::loom::pnr::detail::SpatialTagVertexRef vertex,
    std::uint64_t &unassignedCount, std::uint64_t &conflictCount,
    llvm::ArrayRef<PnrIndex> domains,
    const std::optional<llvm::APInt> &value) noexcept {
  if (!value) {
    assert(unassignedCount != 0);
    --unassignedCount;
    return;
  }
  std::vector<::loom::pnr::detail::SpatialTagVertexRef> peers;
  for (PnrIndex domain : domains) {
    const auto found = occupancy[domain].find(*value);
    if (found != occupancy[domain].end())
      peers.insert(peers.end(), found->second.begin(), found->second.end());
  }
  llvm::sort(peers);
  peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
  const std::uint64_t removedConflicts = llvm::count_if(peers, [&](auto peer) {
    return peer != vertex && interference.interferes(vertex, peer);
  });
  for (PnrIndex domain : domains) {
    assert(domain < occupancy.size());
    auto found = occupancy[domain].find(*value);
    assert(found != occupancy[domain].end());
    auto member = llvm::lower_bound(found->second, vertex);
    assert(member != found->second.end());
    assert(*member == vertex);
    found->second.erase(member);
    if (found->second.empty())
      occupancy[domain].erase(found);
  }
  assert(removedConflicts <= conflictCount);
  conflictCount -= removedConflicts;
}

} // namespace

namespace {

using TagNetState = loom::pnr::detail::SpatialTagNetState;
using TagStateStorage = loom::pnr::detail::SpatialTagAssignmentStateStorage;

::loom::pnr::detail::SpatialTagVertexRef
segmentVertex(const TagStateStorage &storage, PnrIndex logicalNet,
              PnrIndex segment) {
  assert(logicalNet < storage.nets.size() &&
         segment < storage.nets[logicalNet].continuity.segments().size());
  const auto &descriptor =
      storage.nets[logicalNet].continuity.segments()[segment];
  return {logicalNet, descriptor.originKind, descriptor.origin};
}

llvm::ArrayRef<PnrIndex> segmentDomains(const TagNetState &net,
                                        PnrIndex segment) {
  return ::loom::pnr::detail::tagSegmentDomains(net, segment);
}

llvm::Error verifyFabricTemporalSwitchRows(
    const TagStateStorage &storage,
    llvm::ArrayRef<const RouteTreeState *> routes,
    llvm::ArrayRef<const SpatialTagContinuityProjection *> continuity,
    llvm::ArrayRef<TagDomainOccupancy> occupancy,
    llvm::ArrayRef<PnrIndex> residentCounts) {
  using Demand = ::loom::pnr::detail::SpatialTemporalSwitchSegmentDemand;
  const auto domains =
      storage.problem->routing().tagContinuity().matchDomains();
  if (llvm::none_of(domains, [](const auto &domain) {
        return domain.kind == ::loom::fabric::FabricPhysicalTagMatchDomainKind::
                                  TemporalSwitchTable;
      }))
    return llvm::Error::success();
  auto demands = ::loom::pnr::detail::deriveSpatialTemporalSwitchSegmentDemands(
      *storage.problem, routes, continuity);
  if (!demands)
    return demands.takeError();
  std::vector<std::vector<const Demand *>> demandsByDomain(domains.size());
  for (const Demand &demand : *demands) {
    if (demand.domain >= domains.size())
      return invalid("Temporal switch row demand is out of range");
    demandsByDomain[demand.domain].push_back(&demand);
  }

  for (PnrIndex domain = 0; domain < domains.size(); ++domain) {
    if (domains[domain].kind !=
        ::loom::fabric::FabricPhysicalTagMatchDomainKind::TemporalSwitchTable)
      continue;
    const auto &domainDemands = demandsByDomain[domain];
    std::vector<
        std::vector<::loom::fabric::FabricTemporalSwitchRouteSignatureView>>
        signatureStorage;
    signatureStorage.reserve(domainDemands.size());
    for (const Demand *demand : domainDemands) {
      signatureStorage.emplace_back();
      auto &signatures = signatureStorage.back();
      signatures.reserve(demand->signatures.size());
      for (const ::loom::pnr::detail::SpatialTemporalSwitchInputSignature
               &signature : demand->signatures)
        signatures.push_back(
            {signature.occurrence, signature.input, signature.outputs});
    }
    std::vector<::loom::fabric::FabricTemporalSwitchTaggedRouteDemandView>
        ownerDemands;
    ownerDemands.reserve(domainDemands.size());
    for (auto [ordinal, signatures] : llvm::enumerate(signatureStorage)) {
      const Demand &demand = *domainDemands[ordinal];
      if (demand.logicalNet >= storage.nets.size() ||
          demand.segment >= storage.nets[demand.logicalNet].values.size() ||
          !storage.nets[demand.logicalNet].values[demand.segment])
        return invalid(
            "exact Temporal switch row verification found TagUnassigned");
      ownerDemands.push_back(
          {{signatures},
           storage.nets[demand.logicalNet].values[demand.segment]->zextOrTrunc(
               domains[domain].tagWidthBits)});
    }
    auto ownerRows =
        ::loom::fabric::projectFabricTemporalSwitchRouteRows(ownerDemands);
    if (!ownerRows)
      return ownerRows.takeError();
    if (ownerRows->size() != residentCounts[domain] ||
        ownerRows->size() != occupancy[domain].size())
      return invalid(
          "cached Temporal switch resident count diverges from Fabric rows");

    for (const auto &row : *ownerRows) {
      std::vector<::loom::pnr::detail::SpatialTagVertexRef> rowVertices;
      rowVertices.reserve(row.demandOrdinals.size());
      for (std::uint64_t demandOrdinal : row.demandOrdinals) {
        if (demandOrdinal >= domainDemands.size())
          return invalid("Fabric Temporal switch row has an absent demand");
        const Demand &demand = *domainDemands[demandOrdinal];
        rowVertices.push_back(
            segmentVertex(storage, demand.logicalNet, demand.segment));
      }
      llvm::sort(rowVertices);
      rowVertices.erase(std::unique(rowVertices.begin(), rowVertices.end()),
                        rowVertices.end());
      bool cacheCompatible = true;
      for (std::size_t lhs = 0; lhs != rowVertices.size(); ++lhs)
        for (std::size_t rhs = lhs + 1; rhs != rowVertices.size(); ++rhs)
          cacheCompatible &= !storage.interference.interferes(
              domain, rowVertices[lhs], rowVertices[rhs]);
      if (cacheCompatible != row.compatible)
        return invalid(
            "cached Temporal switch compatibility diverges from Fabric row");

      const auto occupied =
          llvm::find_if(occupancy[domain], [&](const auto &entry) {
            return compareUnsigned(entry.first, row.tag) == 0;
          });
      if (occupied == occupancy[domain].end())
        return invalid("Fabric Temporal switch row has no cached tag");
      std::vector<::loom::pnr::detail::SpatialTagVertexRef> occupiedVertices =
          occupied->second;
      llvm::sort(occupiedVertices);
      occupiedVertices.erase(
          std::unique(occupiedVertices.begin(), occupiedVertices.end()),
          occupiedVertices.end());
      if (occupiedVertices != rowVertices)
        return invalid(
            "cached Temporal switch row membership diverges from Fabric row");
    }
  }
  return llvm::Error::success();
}

std::uint64_t residentOveruse(PnrIndex count,
                              std::optional<std::uint64_t> capacity) {
  return capacity && count > *capacity ? count - *capacity : 0;
}

llvm::Error addDomainResidency(TagStateStorage &storage,
                               llvm::ArrayRef<PnrIndex> domains,
                               const std::optional<llvm::APInt> &value) {
  const auto matchDomains =
      storage.problem->routing().tagContinuity().matchDomains();
  std::uint64_t addedOveruse = 0;
  for (PnrIndex domain : domains) {
    if (domain >= storage.residentCounts.size() ||
        domain >= matchDomains.size())
      return invalid("segment names an out-of-range tag match domain");
    const PnrIndex count = storage.residentCounts[domain];
    const auto found = value ? storage.occupancy[domain].find(*value)
                             : storage.occupancy[domain].end();
    const bool packedSwitch =
        matchDomains[domain].kind ==
        ::loom::fabric::FabricPhysicalTagMatchDomainKind::TemporalSwitchTable;
    if (packedSwitch && value && found != storage.occupancy[domain].end() &&
        !found->second.empty())
      continue;
    if (count >= getInvalidPnrIndex() - PnrIndex{1})
      return invalid("tag match-domain residency overflows PnrIndex");
    const auto capacity = matchDomains[domain].residentEntryCapacity;
    addedOveruse +=
        residentOveruse(count + 1, capacity) - residentOveruse(count, capacity);
  }
  if (addedOveruse > std::numeric_limits<std::uint64_t>::max() -
                         storage.residentCapacityOveruse)
    return invalid("tag match-domain capacity overuse exceeds u64");
  for (PnrIndex domain : domains) {
    const auto found = value ? storage.occupancy[domain].find(*value)
                             : storage.occupancy[domain].end();
    const bool packedSwitch =
        matchDomains[domain].kind ==
        ::loom::fabric::FabricPhysicalTagMatchDomainKind::TemporalSwitchTable;
    if (!packedSwitch || !value || found == storage.occupancy[domain].end() ||
        found->second.empty())
      ++storage.residentCounts[domain];
  }
  storage.residentCapacityOveruse += addedOveruse;
  return llvm::Error::success();
}

void removeDomainResidency(TagStateStorage &storage,
                           llvm::ArrayRef<PnrIndex> domains,
                           const std::optional<llvm::APInt> &value) noexcept {
  const auto matchDomains =
      storage.problem->routing().tagContinuity().matchDomains();
  for (PnrIndex domain : domains) {
    assert(domain < storage.residentCounts.size() &&
           domain < matchDomains.size());
    const PnrIndex count = storage.residentCounts[domain];
    const auto found = value ? storage.occupancy[domain].find(*value)
                             : storage.occupancy[domain].end();
    const bool packedSwitch =
        matchDomains[domain].kind ==
        ::loom::fabric::FabricPhysicalTagMatchDomainKind::TemporalSwitchTable;
    if (packedSwitch && value) {
      assert(found != storage.occupancy[domain].end() &&
             !found->second.empty());
      if (found->second.size() != 1)
        continue;
    }
    assert(count != 0);
    const auto capacity = matchDomains[domain].residentEntryCapacity;
    const std::uint64_t removedOveruse =
        residentOveruse(count, capacity) - residentOveruse(count - 1, capacity);
    assert(removedOveruse <= storage.residentCapacityOveruse);
    storage.residentCapacityOveruse -= removedOveruse;
    --storage.residentCounts[domain];
  }
}

llvm::Error addSegmentState(TagStateStorage &storage, PnrIndex logicalNet,
                            PnrIndex segment, llvm::ArrayRef<PnrIndex> domains,
                            const std::optional<llvm::APInt> &value) {
  if (llvm::Error error = addDomainResidency(storage, domains, value))
    return error;
  if (llvm::Error error = addAssignment(
          storage.occupancy, storage.interference,
          segmentVertex(storage, logicalNet, segment), storage.unassignedCount,
          storage.conflictCount, domains, value)) {
    removeDomainResidency(storage, domains, value);
    return error;
  }
  return llvm::Error::success();
}

void removeSegmentState(TagStateStorage &storage, PnrIndex logicalNet,
                        PnrIndex segment, llvm::ArrayRef<PnrIndex> domains,
                        const std::optional<llvm::APInt> &value) noexcept {
  removeDomainResidency(storage, domains, value);
  removeAssignment(storage.occupancy, storage.interference,
                   segmentVertex(storage, logicalNet, segment),
                   storage.unassignedCount, storage.conflictCount, domains,
                   value);
}

void removeNet(TagStateStorage &storage, PnrIndex logicalNet,
               const TagNetState &net) noexcept {
  for (PnrIndex segment = 0; segment < net.values.size(); ++segment)
    removeSegmentState(storage, logicalNet, segment,
                       segmentDomains(net, segment), net.values[segment]);
}

llvm::Error
installNetValues(TagStateStorage &storage, PnrIndex logicalNet,
                 llvm::ArrayRef<std::optional<llvm::APInt>> values) {
  TagNetState &net = storage.nets[logicalNet];
  if (values.size() != net.continuity.segments().size())
    return invalid("installed tag value inventory has the wrong width");
  net.values.assign(values.begin(), values.end());
  PnrIndex added = 0;
  bool committed = false;
  const llvm::scope_exit rollback([&] {
    if (committed)
      return;
    while (added != 0) {
      --added;
      removeSegmentState(storage, logicalNet, added, segmentDomains(net, added),
                         net.values[added]);
    }
  });
  for (; added < net.values.size(); ++added)
    if (llvm::Error error =
            addSegmentState(storage, logicalNet, added,
                            segmentDomains(net, added), net.values[added]))
      return error;
  committed = true;
  return llvm::Error::success();
}

struct TagAssignmentPreservationView final {
  llvm::ArrayRef<SpatialTagContinuitySegment> segments;
  llvm::ArrayRef<std::optional<llvm::APInt>> values;
};

std::optional<llvm::APInt>
preservedValue(std::optional<TagAssignmentPreservationView> oldNet,
               const SpatialTagContinuitySegment &segment) {
  if (!oldNet)
    return std::nullopt;
  const auto oldSegments = oldNet->segments;
  const auto found =
      llvm::lower_bound(oldSegments, segment,
                        [](const SpatialTagContinuitySegment &lhs,
                           const SpatialTagContinuitySegment &rhs) {
                          return std::tie(lhs.originKind, lhs.origin) <
                                 std::tie(rhs.originKind, rhs.origin);
                        });
  if (found == oldSegments.end() || found->originKind != segment.originKind ||
      found->origin != segment.origin)
    return std::nullopt;
  const auto index = static_cast<std::size_t>(found - oldSegments.begin());
  if (index >= oldNet->values.size())
    return std::nullopt;
  return oldNet->values[index];
}

llvm::Error buildNet(TagStateStorage &storage, PnrIndex logicalNet,
                     std::optional<TagAssignmentPreservationView> oldNet,
                     TagNetState &result,
                     std::optional<llvm::ArrayRef<llvm::APInt>> requiredValues,
                     llvm::ArrayRef<llvm::APInt> forbiddenValues) {
  result.values.assign(result.continuity.segments().size(), std::nullopt);
  std::vector<std::uint8_t> added(result.values.size(), 0);
  bool committed = false;
  const llvm::scope_exit rollback([&] {
    if (committed)
      return;
    for (PnrIndex segment = 0; segment < result.values.size(); ++segment)
      if (added[segment])
        removeSegmentState(storage, logicalNet, segment,
                           segmentDomains(result, segment),
                           result.values[segment]);
  });

  const auto logicalNets = storage.problem->transfers().logicalNets();
  if (logicalNet >= logicalNets.size())
    return invalid("logical net ordinal is out of range");
  const FrozenConstraintShard &tagConstraints =
      storage.problem->constraints().shard(
          ::mapping::SpatialConstraintProjection::NetAssignedTagValues);
  const auto restriction = tagConstraints.restrictedDomain(
      SpatialConstraintSubject{logicalNets[logicalNet].producer});
  const auto matchDomains =
      storage.problem->routing().tagContinuity().matchDomains();

  for (PnrIndex ordinal = 0; ordinal < result.continuity.segments().size();
       ++ordinal) {
    const SpatialTagContinuitySegment &segment =
        result.continuity.segments()[ordinal];
    const auto domains = segmentDomains(result, ordinal);
    for (PnrIndex domain : domains)
      if (domain >= matchDomains.size() ||
          matchDomains[domain].tagWidthBits != segment.tagWidthBits)
        return invalid("segment and local match-domain widths disagree");
  }

  if (requiredValues) {
    if (requiredValues->size() > result.values.size())
      return unsupported(
          "tag equality requires more values than one member has segments");
    if (requiredValues->empty()) {
      if (!result.values.empty())
        return unsupported(
            "nonempty tagged route cannot realize an empty equality set");
      committed = true;
      return llvm::Error::success();
    }

    std::vector<PnrIndex> valueForSegment(result.values.size(),
                                          getInvalidPnrIndex());
    const auto compatible = [&](PnrIndex value, PnrIndex segment) {
      const llvm::APInt &candidate = (*requiredValues)[value];
      const auto &descriptor = result.continuity.segments()[segment];
      return ::fabric::isRepresentablePhysicalTagValue(descriptor.tagWidthBits,
                                                       candidate) &&
             valueAllowed(candidate, restriction) &&
             isFree(segmentDomains(result, segment), candidate,
                    storage.occupancy);
    };
    std::function<bool(PnrIndex, std::vector<std::uint8_t> &)> matchValue =
        [&](PnrIndex value, std::vector<std::uint8_t> &visited) {
          for (PnrIndex segment = 0; segment < result.values.size();
               ++segment) {
            if (visited[segment] || !compatible(value, segment))
              continue;
            visited[segment] = 1;
            if (valueForSegment[segment] == getInvalidPnrIndex() ||
                matchValue(valueForSegment[segment], visited)) {
              valueForSegment[segment] = value;
              return true;
            }
          }
          return false;
        };
    for (PnrIndex value = 0; value < requiredValues->size(); ++value) {
      std::vector<std::uint8_t> visited(result.values.size(), 0);
      if (!matchValue(value, visited))
        return unsupported(
            "tag equality set has no representative segment matching");
    }
    for (PnrIndex segment = 0; segment < valueForSegment.size(); ++segment) {
      const PnrIndex value = valueForSegment[segment];
      if (value == getInvalidPnrIndex())
        continue;
      result.values[segment] = (*requiredValues)[value];
      if (llvm::Error error = addSegmentState(storage, logicalNet, segment,
                                              segmentDomains(result, segment),
                                              result.values[segment]))
        return error;
      added[segment] = 1;
    }
    std::vector<PnrIndex> remaining;
    remaining.reserve(result.values.size() - requiredValues->size());
    for (PnrIndex segment = 0; segment < result.values.size(); ++segment)
      if (!result.values[segment])
        remaining.push_back(segment);
    llvm::sort(remaining, [&](PnrIndex lhs, PnrIndex rhs) {
      const auto left = segmentDomains(result, lhs).size();
      const auto right = segmentDomains(result, rhs).size();
      if (left != right)
        return left > right;
      return lhs < rhs;
    });
    for (PnrIndex segment : remaining) {
      const auto &descriptor = result.continuity.segments()[segment];
      const auto domains = segmentDomains(result, segment);
      for (const llvm::APInt &candidate : *requiredValues)
        if (::fabric::isRepresentablePhysicalTagValue(descriptor.tagWidthBits,
                                                      candidate) &&
            valueAllowed(candidate, restriction) &&
            isFree(domains, candidate, storage.occupancy)) {
          result.values[segment] = candidate;
          break;
        }
      if (!result.values[segment])
        return unsupported(
            "tag equality set cannot color one continuity segment");
      if (llvm::Error error = addSegmentState(storage, logicalNet, segment,
                                              domains, result.values[segment]))
        return error;
      added[segment] = 1;
    }
    committed = true;
    return llvm::Error::success();
  }

  for (PnrIndex ordinal = 0; ordinal < result.continuity.segments().size();
       ++ordinal) {
    const SpatialTagContinuitySegment &segment =
        result.continuity.segments()[ordinal];
    const auto domains = segmentDomains(result, ordinal);
    std::optional<llvm::APInt> selected = preservedValue(oldNet, segment);
    if (selected && (!::fabric::isRepresentablePhysicalTagValue(
                         segment.tagWidthBits, *selected) ||
                     !valueAllowed(*selected, restriction) ||
                     containsValue(forbiddenValues, *selected) ||
                     !isFree(domains, *selected, storage.occupancy)))
      selected.reset();
    if (!selected) {
      auto chosen = chooseValue(segment.tagWidthBits, domains, restriction,
                                forbiddenValues, storage.occupancy);
      if (!chosen)
        return chosen.takeError();
      selected = std::move(*chosen);
    }
    result.values[ordinal] = std::move(selected);
    if (llvm::Error error = addSegmentState(storage, logicalNet, ordinal,
                                            domains, result.values[ordinal]))
      return error;
    added[ordinal] = 1;
  }
  committed = true;
  return llvm::Error::success();
}

std::vector<llvm::APInt> projectedValues(const TagNetState &net) {
  std::vector<llvm::APInt> result;
  result.reserve(net.values.size());
  for (const auto &value : net.values)
    if (value)
      result.push_back(*value);
  normalizeValues(result);
  return result;
}

std::vector<llvm::APInt> disjointForbiddenValues(const TagStateStorage &storage,
                                                 PnrIndex equalityClass) {
  std::vector<llvm::APInt> result;
  for (PnrIndex group : storage.constraints->classDisjointGroups(equalityClass))
    for (PnrIndex peer : storage.constraints->disjointGroupMembers(group)) {
      if (peer == equalityClass || !storage.classBuilt[peer])
        continue;
      for (PnrIndex net : storage.constraints->classMembers(peer))
        for (const auto &value : storage.nets[net].values)
          if (value)
            result.push_back(*value);
    }
  normalizeValues(result);
  return result;
}

llvm::Error buildClass(
    TagStateStorage &storage, PnrIndex equalityClass,
    const std::function<std::optional<TagAssignmentPreservationView>(PnrIndex)>
        &oldNet,
    std::vector<PnrIndex> *builtNets = nullptr) {
  const auto members = storage.constraints->classMembers(equalityClass);
  if (members.empty())
    return invalid("Physical Tag equality class has no member");
  PnrIndex leader = members.front();
  for (PnrIndex net : members)
    if (storage.nets[net].continuity.segments().size() <
            storage.nets[leader].continuity.segments().size() ||
        (storage.nets[net].continuity.segments().size() ==
             storage.nets[leader].continuity.segments().size() &&
         net < leader))
      leader = net;
  std::vector<PnrIndex> order;
  order.reserve(members.size());
  order.push_back(leader);
  for (PnrIndex net : members)
    if (net != leader)
      order.push_back(net);

  const std::vector<llvm::APInt> forbidden =
      disjointForbiddenValues(storage, equalityClass);
  const auto oldLeader = oldNet(leader);
  if (llvm::Error error =
          buildNet(storage, leader, oldLeader, storage.nets[leader],
                   std::nullopt, forbidden))
    return error;
  if (builtNets)
    builtNets->push_back(leader);
  const std::vector<llvm::APInt> required =
      projectedValues(storage.nets[leader]);
  for (PnrIndex net : llvm::drop_begin(order)) {
    if (llvm::Error error = buildNet(storage, net, std::nullopt,
                                     storage.nets[net], required, forbidden))
      return error;
    if (builtNets)
      builtNets->push_back(net);
  }
  storage.classBuilt[equalityClass] = 1;
  return llvm::Error::success();
}

llvm::Expected<::loom::pnr::detail::SpatialTagColoringResult>
deriveIndependentColoring(
    const TagStateStorage &storage,
    const ::loom::pnr::detail::SpatialTagColoringCache *previous = nullptr) {
  if (storage.constraints->hasRelations())
    return invalid("independent coloring received related logical nets");

  std::vector<::loom::pnr::detail::SpatialTagColoringVertex> vertices;
  std::vector<::loom::pnr::detail::SpatialTagColoringVertexIdentity> identities;
  std::vector<PnrIndex> domainOffsets;
  std::vector<PnrIndex> domains;
  std::vector<PnrIndex> intervalOffsets;
  std::vector<::loom::pnr::detail::SpatialTagColoringInterval> intervals;
  domainOffsets.push_back(0);
  intervalOffsets.push_back(0);
  const auto logicalNets = storage.problem->transfers().logicalNets();
  const FrozenConstraintShard &tagConstraints =
      storage.problem->constraints().shard(
          ::mapping::SpatialConstraintProjection::NetAssignedTagValues);
  for (PnrIndex logicalNet = 0; logicalNet < storage.nets.size();
       ++logicalNet) {
    const auto restriction = tagConstraints.restrictedDomain(
        SpatialConstraintSubject{logicalNets[logicalNet].producer});
    std::vector<::loom::pnr::detail::SpatialTagColoringInterval> localIntervals;
    if (restriction)
      for (const SpatialConstraintDomainValue &value : *restriction) {
        const auto *interval =
            std::get_if<SpatialConstraintUnsignedInterval>(&value);
        if (!interval)
          return invalid("tag restriction contains a non-interval value");
        localIntervals.push_back({interval->lower, interval->upper});
      }
    const TagNetState &net = storage.nets[logicalNet];
    for (PnrIndex segment = 0; segment < net.continuity.segments().size();
         ++segment) {
      const auto &descriptor = net.continuity.segments()[segment];
      vertices.push_back({descriptor.tagWidthBits, restriction.has_value()});
      identities.push_back({logicalNet,
                            static_cast<std::uint64_t>(descriptor.originKind),
                            descriptor.origin});
      const auto localDomains = segmentDomains(net, segment);
      if (llvm::Error error = preflightPnrIndexCapacity(
              incidenceCountContext, domains.size() + localDomains.size()))
        return error;
      domains.insert(domains.end(), localDomains.begin(), localDomains.end());
      auto domainEnd = checkedIndex(incidenceCountContext, domains.size());
      if (!domainEnd)
        return domainEnd.takeError();
      domainOffsets.push_back(*domainEnd);

      if (llvm::Error error = preflightPnrIndexCapacity(
              colorIntervalCountContext,
              intervals.size() + localIntervals.size()))
        return error;
      intervals.insert(intervals.end(), localIntervals.begin(),
                       localIntervals.end());
      auto intervalEnd =
          checkedIndex(colorIntervalCountContext, intervals.size());
      if (!intervalEnd)
        return intervalEnd.takeError();
      intervalOffsets.push_back(*intervalEnd);
    }
  }
  if (llvm::Error error =
          preflightPnrIndexCapacity(segmentCountContext, vertices.size()))
    return error;
  auto domainCount = checkedIndex(
      matchDomainCountContext,
      storage.problem->routing().tagContinuity().matchDomains().size());
  if (!domainCount)
    return domainCount.takeError();
  auto coloring = ::loom::pnr::detail::colorSpatialTagInterference(
      {vertices, domainOffsets, domains, intervalOffsets, intervals,
       *domainCount, storage.interference.conflictOffsets(),
       storage.interference.conflicts()},
      identities, previous);
  if (!coloring)
    return coloring.takeError();
  if (coloring->values.size() != vertices.size())
    return invalid("independent tag coloring returned the wrong width");

  return std::move(*coloring);
}

llvm::Error colorIndependentNets(TagStateStorage &storage,
                                 std::vector<PnrIndex> *builtNets = nullptr) {
  if (storage.unassignedCount != 0 || storage.conflictCount != 0 ||
      storage.residentCapacityOveruse != 0 ||
      llvm::any_of(storage.residentCounts,
                   [](PnrIndex count) { return count != 0; }) ||
      llvm::any_of(storage.occupancy,
                   [](const auto &domain) { return !domain.empty(); }))
    return invalid("independent coloring requires empty assignment state");
  auto coloring = deriveIndependentColoring(storage);
  if (!coloring)
    return coloring.takeError();

  std::vector<std::pair<PnrIndex, PnrIndex>> added;
  added.reserve(coloring->values.size());
  bool committed = false;
  const llvm::scope_exit rollback([&] {
    if (committed)
      return;
    for (const auto &[logicalNet, segment] : llvm::reverse(added))
      removeSegmentState(storage, logicalNet, segment,
                         segmentDomains(storage.nets[logicalNet], segment),
                         storage.nets[logicalNet].values[segment]);
  });
  std::size_t vertex = 0;
  for (PnrIndex logicalNet = 0; logicalNet < storage.nets.size();
       ++logicalNet) {
    TagNetState &net = storage.nets[logicalNet];
    net.values.assign(net.continuity.segments().size(), std::nullopt);
    for (PnrIndex segment = 0; segment < net.values.size(); ++segment) {
      net.values[segment] = coloring->values[vertex++];
      if (llvm::Error error = addSegmentState(storage, logicalNet, segment,
                                              segmentDomains(net, segment),
                                              net.values[segment]))
        return error;
      added.emplace_back(logicalNet, segment);
    }
  }
  if (storage.unassignedCount != coloring->unassignedCount ||
      storage.conflictCount != coloring->conflictCount)
    return invalid("independent tag coloring summary is inconsistent");
  storage.coloringCache = std::move(coloring->cache);
  std::fill(storage.classBuilt.begin(), storage.classBuilt.end(), 1);
  if (builtNets)
    for (PnrIndex logicalNet = 0; logicalNet < storage.nets.size();
         ++logicalNet)
      builtNets->push_back(logicalNet);
  committed = true;
  return llvm::Error::success();
}

llvm::Error buildAssignments(TagStateStorage &storage,
                             const std::vector<TagNetState> *oldNets,
                             std::vector<PnrIndex> *builtNets = nullptr) {
  if (!storage.constraints->hasRelations())
    return colorIndependentNets(storage, builtNets);
  for (PnrIndex equalityClass = 0;
       equalityClass < storage.constraints->classCount(); ++equalityClass)
    if (llvm::Error error = buildClass(
            storage, equalityClass,
            [&](PnrIndex logicalNet)
                -> std::optional<TagAssignmentPreservationView> {
              if (!oldNets)
                return std::nullopt;
              const TagNetState &old = (*oldNets)[logicalNet];
              return TagAssignmentPreservationView{old.continuity.segments(),
                                                   old.values};
            },
            builtNets))
      return error;
  return llvm::Error::success();
}

llvm::Error verifyRelations(const TagStateStorage &storage) {
  if (!storage.constraints->hasRelations())
    return llvm::Error::success();
  std::vector<std::vector<llvm::APInt>> classValues(
      storage.constraints->classCount());
  for (PnrIndex equalityClass = 0;
       equalityClass < storage.constraints->classCount(); ++equalityClass) {
    const auto members = storage.constraints->classMembers(equalityClass);
    if (members.empty())
      return invalid("Physical Tag equality class has no member");
    classValues[equalityClass] = projectedValues(storage.nets[members.front()]);
    for (PnrIndex net : members.drop_front())
      if (!equalUnsignedValues(projectedValues(storage.nets[net]),
                               classValues[equalityClass]))
        return invalid("Physical Tag value-set equality is violated");
  }
  for (PnrIndex equalityClass = 0;
       equalityClass < storage.constraints->classCount(); ++equalityClass)
    for (PnrIndex group :
         storage.constraints->classDisjointGroups(equalityClass)) {
      const auto members = storage.constraints->disjointGroupMembers(group);
      if (members.empty() || members.front() != equalityClass)
        continue;
      std::vector<llvm::APInt> observed;
      for (PnrIndex member : members) {
        for (const llvm::APInt &value : classValues[member])
          if (containsValue(observed, value))
            return invalid("Physical Tag value-set disjointness is violated");
        observed.insert(observed.end(), classValues[member].begin(),
                        classValues[member].end());
        normalizeValues(observed);
      }
    }
  return llvm::Error::success();
}

llvm::Expected<SpatialTagContinuityProjection>
deriveVerifiedSpatialTagContinuity(const RouteTreeState &route) {
  SpatialTagContinuityProjection result;
  SpatialTagContinuityScratch scratch;
  if (llvm::Error error =
          ::loom::pnr::detail::rebuildSpatialTagContinuityUnchecked(
              route, result, scratch))
    return std::move(error);
  return result;
}

enum class RouteReadMode : std::uint8_t {
  Stable,
  AlreadyVerified,
};

llvm::Expected<std::unique_ptr<TagStateStorage>>
buildStorage(const FrozenSpatialPnrProblem &problem,
             llvm::ArrayRef<const RouteTreeState *> routes,
             const std::vector<TagNetState> *oldNets = nullptr,
             RouteReadMode routeReadMode = RouteReadMode::Stable) {
  if (routes.size() != problem.transfers().logicalNets().size())
    return invalid("route count does not match the frozen logical nets");
  if (oldNets && oldNets->size() != routes.size())
    return invalid("preserved assignment count does not match the routes");
  auto storage = std::make_unique<TagStateStorage>();
  storage->problem = &problem;
  storage->constraints = &problem.tagConstraints();
  storage->hasTaggedTransport = llvm::any_of(
      problem.routing().routingEndpoints(), [](const auto &endpoint) {
        return endpoint.dataPath.kind == ::fabric::DataPathKind::BitsTag;
      });
  storage->nets.resize(routes.size());
  storage->occupancy.resize(
      problem.routing().tagContinuity().matchDomains().size());
  storage->residentCounts.assign(
      problem.routing().tagContinuity().matchDomains().size(), 0);
  storage->classBuilt.assign(storage->constraints->classCount(), 0);
  for (PnrIndex logicalNet = 0; logicalNet < routes.size(); ++logicalNet) {
    if (!routes[logicalNet] ||
        &routes[logicalNet]->routingGraph() != &problem.routing())
      return invalid("route does not belong to the frozen routing graph");
    auto continuity =
        routeReadMode == RouteReadMode::Stable
            ? deriveSpatialTagContinuity(*routes[logicalNet])
            : deriveVerifiedSpatialTagContinuity(*routes[logicalNet]);
    if (!continuity)
      return continuity.takeError();
    storage->nets[logicalNet].continuity = std::move(*continuity);
  }
  std::vector<const SpatialTagContinuityProjection *> continuity;
  continuity.reserve(storage->nets.size());
  for (const TagNetState &net : storage->nets)
    continuity.push_back(&net.continuity);
  auto interference = ::loom::pnr::detail::deriveSpatialTagInterference(
      problem, routes, continuity);
  if (!interference)
    return interference.takeError();
  storage->interference = std::move(*interference);
  if (llvm::Error error = buildAssignments(*storage, oldNets))
    return std::move(error);
  if (llvm::Error error = verifyRelations(*storage))
    return std::move(error);
  return storage;
}

std::size_t retainedBytes(const TagNetState &net) {
  return net.continuity.retainedStorageBytes() +
         net.values.capacity() * sizeof(std::optional<llvm::APInt>);
}

std::size_t
retainedBytes(const ::loom::pnr::detail::SpatialTagColoringCache &cache) {
  std::size_t bytes =
      cache.components.capacity() *
      sizeof(::loom::pnr::detail::SpatialTagColoringComponentCache);
  for (const auto &component : cache.components) {
    bytes += component.identities.capacity() *
                 sizeof(::loom::pnr::detail::SpatialTagColoringVertexIdentity) +
             component.vertices.capacity() *
                 sizeof(::loom::pnr::detail::SpatialTagColoringVertex) +
             component.domainOffsets.capacity() * sizeof(PnrIndex) +
             component.domains.capacity() * sizeof(PnrIndex) +
             component.intervalOffsets.capacity() * sizeof(PnrIndex) +
             component.intervals.capacity() *
                 sizeof(::loom::pnr::detail::SpatialTagColoringInterval) +
             component.conflictOffsets.capacity() * sizeof(PnrIndex) +
             component.conflicts.capacity() * sizeof(PnrIndex) +
             component.values.capacity() * sizeof(std::optional<llvm::APInt>);
  }
  return bytes;
}

} // namespace

llvm::ArrayRef<PnrIndex>
SpatialTagAssignmentProjection::domainSegments(PnrIndex domain) const {
  assert(domain + 1 < domainSegmentOffsets_.size());
  return llvm::ArrayRef<PnrIndex>(domainSegments_)
      .slice(domainSegmentOffsets_[domain],
             domainSegmentOffsets_[domain + 1] - domainSegmentOffsets_[domain]);
}

llvm::Expected<SpatialTagAssignmentProjection>
loom::pnr::deriveCanonicalSpatialTagAssignments(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<const RouteTreeState *> routes) {
  auto storage = buildStorage(problem, routes);
  if (!storage)
    return storage.takeError();

  SpatialTagAssignmentProjection result;
  result.netSegmentOffsets_.reserve(routes.size() + 1);
  result.netSegmentOffsets_.push_back(0);
  result.segmentDomainOffsets_.push_back(0);
  for (const TagNetState &net : (*storage)->nets) {
    const auto &continuity = net.continuity;
    if (llvm::Error error = preflightPnrIndexCapacity(
            segmentCountContext,
            result.segments_.size() + continuity.segments().size()))
      return std::move(error);
    result.segments_.insert(result.segments_.end(),
                            continuity.segments().begin(),
                            continuity.segments().end());
    result.values_.insert(result.values_.end(), net.values.begin(),
                          net.values.end());
    const auto localOffsets = continuity.segmentDomainOffsets();
    const auto localDomains = continuity.segmentDomains();
    for (PnrIndex segment = 0; segment < continuity.segments().size();
         ++segment) {
      const PnrIndex begin = localOffsets[segment];
      const PnrIndex end = localOffsets[segment + 1];
      if (llvm::Error error = preflightPnrIndexCapacity(
              incidenceCountContext,
              result.segmentDomains_.size() + (end - begin)))
        return std::move(error);
      result.segmentDomains_.insert(result.segmentDomains_.end(),
                                    localDomains.begin() + begin,
                                    localDomains.begin() + end);
      auto offset =
          checkedIndex(incidenceCountContext, result.segmentDomains_.size());
      if (!offset)
        return offset.takeError();
      result.segmentDomainOffsets_.push_back(*offset);
    }
    auto netEnd = checkedIndex(segmentOffsetContext, result.segments_.size());
    if (!netEnd)
      return netEnd.takeError();
    result.netSegmentOffsets_.push_back(*netEnd);
  }

  const std::size_t domainCount =
      problem.routing().tagContinuity().matchDomains().size();
  std::vector<PnrIndex> domainCounts(domainCount, 0);
  for (PnrIndex domain : result.segmentDomains_) {
    if (domain >= domainCount)
      return invalid("segment names an out-of-range tag match domain");
    if (domainCounts[domain] >= getInvalidPnrIndex() - PnrIndex{1})
      return invalid("tag match-domain incidence count overflows PnrIndex");
    ++domainCounts[domain];
  }
  result.domainSegmentOffsets_.reserve(domainCount + 1);
  result.domainSegmentOffsets_.push_back(0);
  for (PnrIndex count : domainCounts) {
    auto end = checkedPnrIndexAdd(incidenceCountContext,
                                  result.domainSegmentOffsets_.back(), count);
    if (!end)
      return end.takeError();
    result.domainSegmentOffsets_.push_back(*end);
  }
  result.domainSegments_.resize(result.segmentDomains_.size());
  std::vector<PnrIndex> cursors(result.domainSegmentOffsets_.begin(),
                                result.domainSegmentOffsets_.end() - 1);
  for (PnrIndex segment = 0; segment < result.segments_.size(); ++segment)
    for (PnrIndex incidence = result.segmentDomainOffsets_[segment];
         incidence < result.segmentDomainOffsets_[segment + 1]; ++incidence) {
      const PnrIndex domain = result.segmentDomains_[incidence];
      result.domainSegments_[cursors[domain]++] = segment;
    }

  result.unassignedCount_ = (*storage)->unassignedCount;
  result.conflictCount_ = (*storage)->conflictCount;
  result.residentCapacityOveruse_ = (*storage)->residentCapacityOveruse;
  return result;
}

SpatialTagAssignmentScratch::SpatialTagAssignmentScratch()
    : storage_(std::make_unique<detail::SpatialTagAssignmentScratchStorage>()) {
}

SpatialTagAssignmentScratch::~SpatialTagAssignmentScratch() {
  assert(!storage_->active &&
         "destroying Physical Tag scratch during an active transaction");
}

llvm::Error
SpatialTagAssignmentScratch::prepare(const FrozenSpatialPnrProblem &problem) {
  if (storage_->active)
    return invalid("cannot prepare scratch during an active transaction");
  storage_->problem = &problem;
  storage_->stagedNets.resize(problem.transfers().logicalNets().size());
  storage_->stagedValues.resize(problem.transfers().logicalNets().size());
  storage_->touchedRoutes.clear();
  storage_->touchedRoutes.reserve(problem.transfers().logicalNets().size());
  storage_->routedNets.clear();
  storage_->routedNets.reserve(problem.transfers().logicalNets().size());
  storage_->valueOnlyNets.clear();
  storage_->valueOnlyNets.reserve(problem.transfers().logicalNets().size());
  storage_->rebuiltNets.clear();
  storage_->rebuiltNets.reserve(problem.transfers().logicalNets().size());
  storage_->synchronizedNets.clear();
  storage_->synchronizedNets.reserve(problem.transfers().logicalNets().size());
  storage_->changedDomains.clear();
  storage_->changedDomains.reserve(
      problem.routing().tagContinuity().matchDomains().size());
  storage_->stagedColoringCache.components.clear();
  storage_->coloringCacheActive = false;
  return llvm::Error::success();
}

std::size_t SpatialTagAssignmentScratch::retainedStorageBytes() const {
  std::size_t bytes =
      storage_->stagedNets.capacity() * sizeof(detail::SpatialTagNetState) +
      storage_->stagedValues.capacity() *
          sizeof(std::vector<std::optional<llvm::APInt>>) +
      storage_->touchedRoutes.capacity() * sizeof(PnrIndex) +
      storage_->routedNets.capacity() * sizeof(PnrIndex) +
      storage_->valueOnlyNets.capacity() * sizeof(PnrIndex) +
      storage_->rebuiltNets.capacity() * sizeof(PnrIndex) +
      storage_->synchronizedNets.capacity() * sizeof(PnrIndex) +
      storage_->changedDomains.capacity() * sizeof(PnrIndex) +
      storage_->continuityScratch.retainedStorageBytes() +
      storage_->interferenceScratch.retainedStorageBytes() +
      retainedBytes(storage_->stagedColoringCache);
  for (const auto &net : storage_->stagedNets)
    bytes += retainedBytes(net);
  for (const auto &values : storage_->stagedValues)
    bytes += values.capacity() * sizeof(std::optional<llvm::APInt>);
  return bytes;
}

SpatialTagAssignmentState::SpatialTagAssignmentState(
    std::unique_ptr<detail::SpatialTagAssignmentStateStorage> storage)
    : storage_(std::move(storage)) {}

SpatialTagAssignmentState::SpatialTagAssignmentState(
    SpatialTagAssignmentState &&) noexcept = default;

SpatialTagAssignmentState::~SpatialTagAssignmentState() = default;

llvm::Expected<SpatialTagAssignmentState>
SpatialTagAssignmentState::create(const FrozenSpatialPnrProblem &problem,
                                  llvm::ArrayRef<RouteTreeStateHandle> routes) {
  std::vector<const RouteTreeState *> rawRoutes;
  rawRoutes.reserve(routes.size());
  for (const RouteTreeStateHandle &route : routes)
    rawRoutes.push_back(route.get());
  auto storage = buildStorage(problem, rawRoutes);
  if (!storage)
    return storage.takeError();
  return SpatialTagAssignmentState(std::move(*storage));
}

llvm::ArrayRef<SpatialTagContinuitySegment>
SpatialTagAssignmentState::segments(PnrIndex logicalNet) const {
  assert(logicalNet < storage_->nets.size());
  return storage_->nets[logicalNet].continuity.segments();
}

llvm::ArrayRef<std::optional<llvm::APInt>>
SpatialTagAssignmentState::values(PnrIndex logicalNet) const {
  assert(logicalNet < storage_->nets.size());
  return storage_->nets[logicalNet].values;
}

llvm::ArrayRef<PnrIndex>
SpatialTagAssignmentState::segmentDomains(PnrIndex logicalNet,
                                          PnrIndex segment) const {
  assert(logicalNet < storage_->nets.size());
  return ::segmentDomains(storage_->nets[logicalNet], segment);
}

std::uint64_t SpatialTagAssignmentState::unassignedCount() const {
  return storage_->unassignedCount;
}

std::uint64_t SpatialTagAssignmentState::conflictCount() const {
  return storage_->conflictCount;
}

std::uint64_t SpatialTagAssignmentState::residentCapacityOveruse() const {
  return storage_->residentCapacityOveruse;
}

std::uint64_t
SpatialTagAssignmentState::domainResidentCount(PnrIndex domain) const {
  assert(domain < storage_->residentCounts.size());
  return storage_->residentCounts[domain];
}

std::uint64_t SpatialTagAssignmentState::domainResidentCapacityOveruse(
    PnrIndex domain) const {
  assert(domain < storage_->residentCounts.size());
  const auto matchDomains =
      storage_->problem->routing().tagContinuity().matchDomains();
  assert(domain < matchDomains.size());
  return residentOveruse(storage_->residentCounts[domain],
                         matchDomains[domain].residentEntryCapacity);
}

std::uint64_t
SpatialTagAssignmentState::domainConflictCount(PnrIndex domain) const {
  const std::uint64_t conflicts = ::loom::pnr::detail::tagDomainConflictCount(
      storage_->occupancy, storage_->interference, domain);
  assert(conflicts <= storage_->conflictCount);
  return conflicts;
}

bool SpatialTagAssignmentState::domainValueConflicts(
    PnrIndex domain, const llvm::APInt &value) const {
  assert(domain < storage_->occupancy.size());
  const auto found = storage_->occupancy[domain].find(value);
  if (found == storage_->occupancy[domain].end())
    return false;
  for (std::size_t lhs = 0; lhs != found->second.size(); ++lhs)
    for (std::size_t rhs = lhs + 1; rhs != found->second.size(); ++rhs)
      if (storage_->interference.interferes(domain, found->second[lhs],
                                            found->second[rhs]))
        return true;
  return false;
}

llvm::Expected<SpatialTagAssignmentSummary>
SpatialTagAssignmentState::projectVerifiedRoutes(
    llvm::ArrayRef<const RouteTreeState *> routes,
    bool includeDomainDetails) const {
  auto projected = buildStorage(*storage_->problem, routes, &storage_->nets,
                                RouteReadMode::AlreadyVerified);
  if (!projected)
    return projected.takeError();
  return detail::summarizeTagAssignmentState(**projected, includeDomainDetails);
}

llvm::Expected<SpatialTagAssignmentSummary>
SpatialTagAssignmentState::summarizeCurrentState(
    bool includeDomainDetails) const {
  return detail::summarizeTagAssignmentState(*storage_, includeDomainDetails);
}

llvm::Expected<SpatialTagAssignmentDelta>
SpatialTagAssignmentState::summarizeCurrentDelta(
    const SpatialTagAssignmentScratch &scratch) const {
  const auto &transaction = *scratch.storage_;
  if (!transaction.active || transaction.problem != storage_->problem)
    return invalid("tag assignment delta has no active transaction");
  return detail::summarizeTagAssignmentDelta(
      *storage_, transaction.synchronizedNets, transaction.changedDomains);
}

llvm::Expected<SpatialTagAssignmentDelta>
SpatialTagAssignmentState::summarizeCurrentDelta(
    llvm::ArrayRef<PnrIndex> logicalNets,
    llvm::ArrayRef<PnrIndex> changedDomains) const {
  return detail::summarizeTagAssignmentDelta(*storage_, logicalNets,
                                             changedDomains);
}

llvm::ArrayRef<PnrIndex> SpatialTagAssignmentState::changedDomains(
    const SpatialTagAssignmentScratch &scratch) const {
  const auto &transaction = *scratch.storage_;
  assert(transaction.active && transaction.problem == storage_->problem);
  return transaction.changedDomains;
}

llvm::Error SpatialTagAssignmentState::stageRouteUpdates(
    llvm::ArrayRef<RouteTreeStateHandle> routes,
    llvm::ArrayRef<std::optional<RouteTreeTransaction>> routeTransactions,
    llvm::ArrayRef<PnrIndex> touchedRoutes,
    SpatialTagAssignmentScratch &scratch) {
  auto &transaction = *scratch.storage_;
  if (transaction.active)
    return invalid("Physical Tag transaction is already active");
  if (transaction.coloringCacheActive)
    return invalid("Physical Tag scratch retained a coloring transaction");
  if (transaction.problem != storage_->problem ||
      transaction.stagedNets.size() != storage_->nets.size() ||
      routes.size() != storage_->nets.size() ||
      routeTransactions.size() != storage_->nets.size())
    return invalid("Physical Tag scratch belongs to another frozen problem");

  transaction.routedNets.assign(touchedRoutes.begin(), touchedRoutes.end());
  llvm::sort(transaction.routedNets);
  transaction.routedNets.erase(
      std::unique(transaction.routedNets.begin(), transaction.routedNets.end()),
      transaction.routedNets.end());
  if (transaction.routedNets.empty())
    return llvm::Error::success();
  for (PnrIndex logicalNet : transaction.routedNets)
    if (logicalNet >= routes.size() || !routes[logicalNet] ||
        &routes[logicalNet]->routingGraph() != &storage_->problem->routing() ||
        !routeTransactions[logicalNet])
      return invalid("touched route does not belong to the frozen problem");

  if (!storage_->hasTaggedTransport) {
    transaction.touchedRoutes = transaction.routedNets;
    for (PnrIndex logicalNet : transaction.touchedRoutes)
      if (!storage_->nets[logicalNet].continuity.segments().empty() ||
          !storage_->nets[logicalNet].values.empty())
        return invalid("untagged Fabric retained a Physical Tag assignment");
    transaction.active = true;
    transaction.rebuiltNets.clear();
    for (PnrIndex logicalNet : transaction.touchedRoutes)
      std::swap(storage_->nets[logicalNet], transaction.stagedNets[logicalNet]);
    for (PnrIndex logicalNet : transaction.touchedRoutes) {
      if (llvm::Error error =
              rebuildSpatialTagContinuity(*routeTransactions[logicalNet],
                                          storage_->nets[logicalNet].continuity,
                                          transaction.continuityScratch)) {
        rollback(scratch);
        return error;
      }
      if (!storage_->nets[logicalNet].continuity.segments().empty()) {
        rollback(scratch);
        return invalid("untagged Fabric produced a Physical Tag segment");
      }
      storage_->nets[logicalNet].values.clear();
    }
    return llvm::Error::success();
  }

  transaction.active = true;
  transaction.touchedRoutes.clear();
  transaction.valueOnlyNets.clear();
  transaction.rebuiltNets.clear();
  transaction.synchronizedNets = transaction.routedNets;
  transaction.changedDomains.clear();
  for (PnrIndex logicalNet : transaction.routedNets)
    transaction.changedDomains.insert(
        transaction.changedDomains.end(),
        storage_->nets[logicalNet].continuity.segmentDomains().begin(),
        storage_->nets[logicalNet].continuity.segmentDomains().end());
  llvm::sort(transaction.changedDomains);
  transaction.changedDomains.erase(
      std::unique(transaction.changedDomains.begin(),
                  transaction.changedDomains.end()),
      transaction.changedDomains.end());
  for (PnrIndex domain : transaction.changedDomains) {
    if (domain >= storage_->occupancy.size()) {
      rollback(scratch);
      return invalid("changed tag match domain is out of range");
    }
    for (const auto &vertex : storage_->interference.domainVertices(domain))
      transaction.synchronizedNets.push_back(vertex.logicalNet);
  }

  for (PnrIndex logicalNet : transaction.routedNets) {
    removeNet(*storage_, logicalNet, storage_->nets[logicalNet]);
    std::swap(storage_->nets[logicalNet], transaction.stagedNets[logicalNet]);
    transaction.touchedRoutes.push_back(logicalNet);
    if (llvm::Error error =
            rebuildSpatialTagContinuity(*routeTransactions[logicalNet],
                                        storage_->nets[logicalNet].continuity,
                                        transaction.continuityScratch)) {
      rollback(scratch);
      return error;
    }
    storage_->nets[logicalNet].values.clear();
  }
  std::vector<const RouteTreeState *> projectedRoutes;
  std::vector<const SpatialTagContinuityProjection *> continuity;
  projectedRoutes.reserve(routes.size());
  continuity.reserve(storage_->nets.size());
  for (PnrIndex logicalNet = 0; logicalNet < routes.size(); ++logicalNet) {
    if (routeTransactions[logicalNet]) {
      auto prepared = routeTransactions[logicalNet]->preparedState();
      if (!prepared) {
        rollback(scratch);
        return prepared.takeError();
      }
      projectedRoutes.push_back(*prepared);
    } else {
      projectedRoutes.push_back(routes[logicalNet].get());
    }
    continuity.push_back(&storage_->nets[logicalNet].continuity);
  }
  if (llvm::Error error =
          ::loom::pnr::detail::stageSpatialTagInterferenceUpdate(
              *storage_->problem, projectedRoutes, continuity,
              transaction.routedNets, storage_->interference,
              transaction.interferenceScratch)) {
    rollback(scratch);
    return error;
  }
  const auto journalValueOnly = [&](PnrIndex logicalNet) {
    removeNet(*storage_, logicalNet, storage_->nets[logicalNet]);
    std::swap(storage_->nets[logicalNet].values,
              transaction.stagedValues[logicalNet]);
    transaction.valueOnlyNets.push_back(logicalNet);
    transaction.touchedRoutes.push_back(logicalNet);
    transaction.synchronizedNets.push_back(logicalNet);
  };

  if (!storage_->constraints->hasRelations()) {
    auto coloring =
        deriveIndependentColoring(*storage_, &storage_->coloringCache);
    if (!coloring) {
      rollback(scratch);
      return coloring.takeError();
    }
    std::swap(storage_->coloringCache, transaction.stagedColoringCache);
    storage_->coloringCache = std::move(coloring->cache);
    transaction.coloringCacheActive = true;
    std::vector<PnrIndex> recoloredNets = transaction.routedNets;
    recoloredNets.reserve(recoloredNets.size() +
                          coloring->recomputedIdentities.size());
    for (const auto identity : coloring->recomputedIdentities) {
      if (identity.owner >= storage_->nets.size()) {
        rollback(scratch);
        return invalid("canonical tag coloring names an unknown logical net");
      }
      recoloredNets.push_back(static_cast<PnrIndex>(identity.owner));
    }
    llvm::sort(recoloredNets);
    recoloredNets.erase(std::unique(recoloredNets.begin(), recoloredNets.end()),
                        recoloredNets.end());
    const auto netOffsets = storage_->interference.netSegmentOffsets();
    if (netOffsets.size() != storage_->nets.size() + 1 || netOffsets.empty() ||
        netOffsets.front() != 0 ||
        netOffsets.back() != coloring->values.size()) {
      rollback(scratch);
      return invalid("canonical tag coloring is not grouped by logical net");
    }
    for (PnrIndex logicalNet : recoloredNets) {
      const PnrIndex begin = netOffsets[logicalNet];
      const PnrIndex end = netOffsets[logicalNet + 1];
      if (begin > end || end > coloring->values.size()) {
        rollback(scratch);
        return invalid("canonical tag coloring has invalid net offsets");
      }
      const auto values =
          llvm::ArrayRef(coloring->values)
              .slice(begin, static_cast<std::size_t>(end - begin));
      const bool routed =
          std::binary_search(transaction.routedNets.begin(),
                             transaction.routedNets.end(), logicalNet);
      if (!routed && !equalOptionalUnsignedValues(
                         storage_->nets[logicalNet].values, values))
        journalValueOnly(logicalNet);
      if (routed ||
          std::binary_search(transaction.valueOnlyNets.begin(),
                             transaction.valueOnlyNets.end(), logicalNet)) {
        if (llvm::Error error =
                installNetValues(*storage_, logicalNet, values)) {
          rollback(scratch);
          return error;
        }
        transaction.rebuiltNets.push_back(logicalNet);
      }
    }
    if (storage_->unassignedCount != coloring->unassignedCount ||
        storage_->conflictCount != coloring->conflictCount) {
      rollback(scratch);
      return invalid(
          "incremental tag coloring disagrees with canonical output");
    }
  } else {
    PnrIndex firstClass = storage_->constraints->classCount();
    for (PnrIndex logicalNet : transaction.routedNets)
      firstClass =
          std::min(firstClass, storage_->constraints->classOfNet(logicalNet));
    std::vector<PnrIndex> affectedClasses;
    affectedClasses.reserve(storage_->constraints->classCount() - firstClass);
    for (PnrIndex equalityClass = firstClass;
         equalityClass < storage_->constraints->classCount(); ++equalityClass) {
      affectedClasses.push_back(equalityClass);
      storage_->classBuilt[equalityClass] = 0;
      for (PnrIndex logicalNet :
           storage_->constraints->classMembers(equalityClass))
        if (!std::binary_search(transaction.routedNets.begin(),
                                transaction.routedNets.end(), logicalNet))
          journalValueOnly(logicalNet);
    }
    llvm::sort(transaction.valueOnlyNets);
    for (PnrIndex equalityClass : affectedClasses)
      if (llvm::Error error = buildClass(
              *storage_, equalityClass,
              [&](PnrIndex logicalNet)
                  -> std::optional<TagAssignmentPreservationView> {
                if (std::binary_search(transaction.routedNets.begin(),
                                       transaction.routedNets.end(),
                                       logicalNet)) {
                  const TagNetState &old = transaction.stagedNets[logicalNet];
                  return TagAssignmentPreservationView{
                      old.continuity.segments(), old.values};
                }
                if (std::binary_search(transaction.valueOnlyNets.begin(),
                                       transaction.valueOnlyNets.end(),
                                       logicalNet))
                  return TagAssignmentPreservationView{
                      storage_->nets[logicalNet].continuity.segments(),
                      transaction.stagedValues[logicalNet]};
                return std::nullopt;
              },
              &transaction.rebuiltNets)) {
        rollback(scratch);
        return error;
      }
  }
  if (llvm::Error error = verifyRelations(*storage_)) {
    rollback(scratch);
    return error;
  }
  llvm::sort(transaction.touchedRoutes);
  transaction.touchedRoutes.erase(std::unique(transaction.touchedRoutes.begin(),
                                              transaction.touchedRoutes.end()),
                                  transaction.touchedRoutes.end());
  for (PnrIndex logicalNet : transaction.touchedRoutes)
    transaction.changedDomains.insert(
        transaction.changedDomains.end(),
        storage_->nets[logicalNet].continuity.segmentDomains().begin(),
        storage_->nets[logicalNet].continuity.segmentDomains().end());
  llvm::sort(transaction.changedDomains);
  transaction.changedDomains.erase(
      std::unique(transaction.changedDomains.begin(),
                  transaction.changedDomains.end()),
      transaction.changedDomains.end());
  for (PnrIndex domain : transaction.changedDomains)
    for (const auto &vertex : storage_->interference.domainVertices(domain))
      transaction.synchronizedNets.push_back(vertex.logicalNet);
  llvm::sort(transaction.synchronizedNets);
  transaction.synchronizedNets.erase(
      std::unique(transaction.synchronizedNets.begin(),
                  transaction.synchronizedNets.end()),
      transaction.synchronizedNets.end());
  return llvm::Error::success();
}

void SpatialTagAssignmentState::commit(
    SpatialTagAssignmentScratch &scratch) noexcept {
  auto &transaction = *scratch.storage_;
  if (!transaction.active)
    return;
  transaction.active = false;
  for (PnrIndex logicalNet : transaction.valueOnlyNets)
    transaction.stagedValues[logicalNet].clear();
  transaction.touchedRoutes.clear();
  transaction.routedNets.clear();
  transaction.valueOnlyNets.clear();
  transaction.rebuiltNets.clear();
  transaction.synchronizedNets.clear();
  transaction.changedDomains.clear();
  if (transaction.coloringCacheActive) {
    transaction.stagedColoringCache.components.clear();
    transaction.coloringCacheActive = false;
  }
  ::loom::pnr::detail::commitSpatialTagInterferenceUpdate(
      transaction.interferenceScratch);
}

void SpatialTagAssignmentState::rollback(
    SpatialTagAssignmentScratch &scratch) noexcept {
  auto &transaction = *scratch.storage_;
  if (!transaction.active)
    return;
  for (PnrIndex logicalNet : transaction.rebuiltNets)
    removeNet(*storage_, logicalNet, storage_->nets[logicalNet]);
  ::loom::pnr::detail::rollbackSpatialTagInterferenceUpdate(
      storage_->interference, transaction.interferenceScratch);
  if (transaction.coloringCacheActive) {
    std::swap(storage_->coloringCache, transaction.stagedColoringCache);
    transaction.stagedColoringCache.components.clear();
    transaction.coloringCacheActive = false;
  }
  for (PnrIndex logicalNet : transaction.touchedRoutes) {
    if (!std::binary_search(transaction.routedNets.begin(),
                            transaction.routedNets.end(), logicalNet))
      continue;
    std::swap(storage_->nets[logicalNet], transaction.stagedNets[logicalNet]);
    const TagNetState &restored = storage_->nets[logicalNet];
    for (PnrIndex segment = 0; segment < restored.values.size(); ++segment)
      llvm::cantFail(addSegmentState(*storage_, logicalNet, segment,
                                     ::segmentDomains(restored, segment),
                                     restored.values[segment]));
  }
  for (PnrIndex logicalNet : transaction.valueOnlyNets) {
    std::swap(storage_->nets[logicalNet].values,
              transaction.stagedValues[logicalNet]);
    const TagNetState &restored = storage_->nets[logicalNet];
    for (PnrIndex segment = 0; segment < restored.values.size(); ++segment)
      llvm::cantFail(addSegmentState(*storage_, logicalNet, segment,
                                     ::segmentDomains(restored, segment),
                                     restored.values[segment]));
  }
  for (PnrIndex logicalNet : transaction.touchedRoutes)
    storage_->classBuilt[storage_->constraints->classOfNet(logicalNet)] = 1;
  transaction.active = false;
  transaction.touchedRoutes.clear();
  transaction.routedNets.clear();
  transaction.valueOnlyNets.clear();
  transaction.rebuiltNets.clear();
  transaction.synchronizedNets.clear();
  transaction.changedDomains.clear();
}

llvm::Error SpatialTagAssignmentState::verify(
    llvm::ArrayRef<RouteTreeStateHandle> routes) const {
  if (routes.size() != storage_->nets.size())
    return invalid("candidate route count changed after Tag initialization");
  std::vector<TagDomainOccupancy> expectedOccupancy(storage_->occupancy.size());
  std::vector<PnrIndex> expectedResidentCounts(storage_->occupancy.size(), 0);
  std::uint64_t expectedUnassigned = 0;
  std::uint64_t expectedConflicts = 0;
  std::uint64_t expectedResidentOveruse = 0;
  const auto logicalNets = storage_->problem->transfers().logicalNets();
  const FrozenConstraintShard &tagConstraints =
      storage_->problem->constraints().shard(
          ::mapping::SpatialConstraintProjection::NetAssignedTagValues);
  const auto matchDomains =
      storage_->problem->routing().tagContinuity().matchDomains();

  for (PnrIndex logicalNet = 0; logicalNet < routes.size(); ++logicalNet) {
    if (!routes[logicalNet] ||
        &routes[logicalNet]->routingGraph() != &storage_->problem->routing())
      return invalid("candidate route belongs to another routing graph");
    auto continuity = deriveSpatialTagContinuity(*routes[logicalNet]);
    if (!continuity)
      return continuity.takeError();
    const TagNetState &net = storage_->nets[logicalNet];
    const auto diverged = [&](llvm::StringRef field) {
      return invalid("cached Tag continuity " + field +
                     " diverges from RouteTree state for logical net " +
                     llvm::Twine(logicalNet));
    };
    if (!llvm::equal(net.continuity.segments(), continuity->segments()))
      return diverged("segments");
    if (!llvm::equal(net.continuity.nodeSegments(),
                     continuity->nodeSegments())) {
      const auto cached = net.continuity.nodeSegments();
      const auto derived = continuity->nodeSegments();
      const std::size_t common = std::min(cached.size(), derived.size());
      std::size_t first = 0;
      while (first != common && cached[first] == derived[first])
        ++first;
      return invalid("cached Tag continuity node segments diverge from "
                     "RouteTree state for logical net " +
                     llvm::Twine(logicalNet) + " at node slot " +
                     llvm::Twine(first) + " (cached size " +
                     llvm::Twine(cached.size()) + ", derived size " +
                     llvm::Twine(derived.size()) + ", cached value " +
                     (first == cached.size() ? llvm::Twine("absent")
                                             : llvm::Twine(cached[first])) +
                     ", derived value " +
                     (first == derived.size() ? llvm::Twine("absent")
                                              : llvm::Twine(derived[first])) +
                     ")");
    }
    if (!llvm::equal(net.continuity.segmentDomainOffsets(),
                     continuity->segmentDomainOffsets()))
      return diverged("segment-domain offsets");
    if (!llvm::equal(net.continuity.segmentDomains(),
                     continuity->segmentDomains()))
      return diverged("segment domains");
    if (!llvm::equal(net.continuity.domainSegmentOffsets(),
                     continuity->domainSegmentOffsets()))
      return diverged("domain-segment offsets");
    if (!llvm::equal(net.continuity.domainSegments(),
                     continuity->domainSegments()))
      return diverged("domain segments");
    if (net.values.size() != continuity->segments().size())
      return diverged("value cardinality");
    const auto restriction = tagConstraints.restrictedDomain(
        SpatialConstraintSubject{logicalNets[logicalNet].producer});
    for (PnrIndex segment = 0; segment < net.values.size(); ++segment) {
      const auto domains = ::segmentDomains(net, segment);
      for (PnrIndex domain : domains)
        if (domain >= matchDomains.size() ||
            matchDomains[domain].tagWidthBits !=
                net.continuity.segments()[segment].tagWidthBits)
          return invalid("cached Tag domain has an incompatible width");
      if (net.values[segment] &&
          (!::fabric::isRepresentablePhysicalTagValue(
               net.continuity.segments()[segment].tagWidthBits,
               *net.values[segment]) ||
           !valueAllowed(*net.values[segment], restriction)))
        return invalid("candidate Physical Tag value is outside its domain");
    }
  }
  std::vector<const RouteTreeState *> rawRoutes;
  std::vector<const SpatialTagContinuityProjection *> continuity;
  rawRoutes.reserve(routes.size());
  continuity.reserve(storage_->nets.size());
  for (PnrIndex logicalNet = 0; logicalNet < routes.size(); ++logicalNet) {
    rawRoutes.push_back(routes[logicalNet].get());
    continuity.push_back(&storage_->nets[logicalNet].continuity);
  }
  auto expectedInterference = ::loom::pnr::detail::deriveSpatialTagInterference(
      *storage_->problem, rawRoutes, continuity);
  if (!expectedInterference)
    return expectedInterference.takeError();
  if (!llvm::equal(expectedInterference->netSegmentOffsets(),
                   storage_->interference.netSegmentOffsets()) ||
      !llvm::equal(expectedInterference->conflictOffsets(),
                   storage_->interference.conflictOffsets()) ||
      !llvm::equal(expectedInterference->conflicts(),
                   storage_->interference.conflicts()) ||
      !expectedInterference->equivalentDerivedState(storage_->interference))
    return invalid("cached Physical Tag interference has drifted");
  for (PnrIndex logicalNet = 0; logicalNet < storage_->nets.size();
       ++logicalNet) {
    const TagNetState &net = storage_->nets[logicalNet];
    for (PnrIndex segment = 0; segment < net.values.size(); ++segment) {
      const auto domains = ::segmentDomains(net, segment);
      for (PnrIndex domain : domains) {
        const auto found =
            net.values[segment]
                ? expectedOccupancy[domain].find(*net.values[segment])
                : expectedOccupancy[domain].end();
        const bool packedSwitch =
            matchDomains[domain].kind ==
            ::loom::fabric::FabricPhysicalTagMatchDomainKind::
                TemporalSwitchTable;
        if (!packedSwitch || !net.values[segment] ||
            found == expectedOccupancy[domain].end() || found->second.empty()) {
          if (expectedResidentCounts[domain] >=
              getInvalidPnrIndex() - PnrIndex{1})
            return invalid("tag match-domain residency overflows PnrIndex");
          ++expectedResidentCounts[domain];
        }
      }
      if (llvm::Error error = addAssignment(
              expectedOccupancy, *expectedInterference,
              segmentVertex(*storage_, logicalNet, segment), expectedUnassigned,
              expectedConflicts, domains, net.values[segment]))
        return error;
    }
  }
  for (PnrIndex domain = 0; domain < expectedResidentCounts.size(); ++domain) {
    const std::uint64_t overuse =
        residentOveruse(expectedResidentCounts[domain],
                        matchDomains[domain].residentEntryCapacity);
    if (overuse >
        std::numeric_limits<std::uint64_t>::max() - expectedResidentOveruse)
      return invalid("tag match-domain capacity overuse exceeds u64");
    expectedResidentOveruse += overuse;
  }
  if (expectedUnassigned == 0)
    if (llvm::Error error = verifyFabricTemporalSwitchRows(
            *storage_, rawRoutes, continuity, expectedOccupancy,
            expectedResidentCounts))
      return error;
  if (expectedUnassigned != storage_->unassignedCount ||
      expectedConflicts != storage_->conflictCount ||
      expectedResidentCounts != storage_->residentCounts ||
      expectedResidentOveruse != storage_->residentCapacityOveruse)
    return invalid("cached Physical Tag violation counts have drifted");
  for (PnrIndex domain = 0; domain < storage_->occupancy.size(); ++domain) {
    if (storage_->occupancy[domain].size() != expectedOccupancy[domain].size())
      return invalid("cached Physical Tag occupancy has drifted");
    for (const auto &entry : storage_->occupancy[domain]) {
      const auto found = expectedOccupancy[domain].find(entry.first);
      if (found == expectedOccupancy[domain].end() ||
          found->second != entry.second)
        return invalid("cached Physical Tag occupancy has drifted");
    }
  }
  if (llvm::Error error = verifyRelations(*storage_))
    return error;
  return llvm::Error::success();
}
