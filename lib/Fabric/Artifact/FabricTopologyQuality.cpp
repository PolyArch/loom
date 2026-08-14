#include "Fabric/Artifact/FabricTopologyQuality.h"

#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefText.h"

#include "Common/ArtifactText.h"
#include "Fabric/IR/FabricEnums.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace loom::fabric {
namespace {

using CanonicalKey = std::vector<std::uint8_t>;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_topology_quality_invalid: " + message);
}

CanonicalKey key(const FabricTransportEndpointOwnerRef &owner) {
  return canonicalFabricBytes(owner);
}

CanonicalKey key(const FabricTransportEndpointRef &endpoint) {
  return canonicalFabricBytes(endpoint);
}

bool isTransparent(FabricTransportEndpointOwnerKind kind) {
  return kind == FabricTransportEndpointOwnerKind::FabricFifoOccurrence ||
         kind == FabricTransportEndpointOwnerKind::FabricBoundaryOccurrence;
}

bool isRoutingResource(FabricRootKind rootKind,
                       FabricTransportEndpointOwnerKind ownerKind) {
  if (rootKind == FabricRootKind::Module)
    return ownerKind ==
           FabricTransportEndpointOwnerKind::FabricSwitchOccurrence;
  return rootKind == FabricRootKind::System &&
         ownerKind == FabricTransportEndpointOwnerKind::SystemTransportResource;
}

template <typename Ref>
void appendSubject(
    std::vector<std::pair<FabricTopologyTerminalKind,
                          FabricTransportEndpointOwnerRef>> &subjects,
    FabricTopologyTerminalKind kind, const Ref &reference) {
  subjects.emplace_back(kind, FabricTransportEndpointOwnerRef::of(reference));
}

void sortUniqueOwners(std::vector<FabricTransportEndpointOwnerRef> &owners) {
  llvm::sort(owners, [](const auto &lhs, const auto &rhs) {
    return key(lhs) < key(rhs);
  });
  owners.erase(std::unique(owners.begin(), owners.end()), owners.end());
}

int compareRatios(std::uint64_t lhsNumerator, std::uint64_t lhsDenominator,
                  std::uint64_t rhsNumerator, std::uint64_t rhsDenominator) {
  const unsigned __int128 lhs =
      static_cast<unsigned __int128>(lhsNumerator) * rhsDenominator;
  const unsigned __int128 rhs =
      static_cast<unsigned __int128>(rhsNumerator) * lhsDenominator;
  return lhs < rhs ? -1 : lhs > rhs ? 1 : 0;
}

FabricTopologyRatioExtreme
canonicalRatio(std::uint64_t numerator, std::uint64_t denominator,
               const FabricTransportEndpointOwnerRef &owner) {
  const std::uint64_t divisor = std::gcd(numerator, denominator);
  return FabricTopologyRatioExtreme{
      numerator / divisor, denominator / divisor, {owner}};
}

void includeCountExtreme(FabricTopologyCountExtreme &minimum,
                         FabricTopologyCountExtreme &maximum,
                         std::uint64_t value,
                         const FabricTransportEndpointOwnerRef &owner,
                         bool first) {
  if (first || value < minimum.value)
    minimum = FabricTopologyCountExtreme{value, {owner}};
  else if (value == minimum.value)
    minimum.owners.push_back(owner);

  if (first || value > maximum.value)
    maximum = FabricTopologyCountExtreme{value, {owner}};
  else if (value == maximum.value)
    maximum.owners.push_back(owner);
}

void includeRatioExtreme(std::optional<FabricTopologyRatioExtreme> &minimum,
                         std::optional<FabricTopologyRatioExtreme> &maximum,
                         std::uint64_t numerator, std::uint64_t denominator,
                         const FabricTransportEndpointOwnerRef &owner) {
  if (!minimum || compareRatios(numerator, denominator, minimum->numerator,
                                minimum->denominator) < 0)
    minimum = canonicalRatio(numerator, denominator, owner);
  else if (compareRatios(numerator, denominator, minimum->numerator,
                         minimum->denominator) == 0)
    minimum->owners.push_back(owner);

  if (!maximum || compareRatios(numerator, denominator, maximum->numerator,
                                maximum->denominator) > 0)
    maximum = canonicalRatio(numerator, denominator, owner);
  else if (compareRatios(numerator, denominator, maximum->numerator,
                         maximum->denominator) == 0)
    maximum->owners.push_back(owner);
}

void writeOwnerArray(llvm::json::OStream &json,
                     llvm::ArrayRef<FabricTransportEndpointOwnerRef> owners) {
  for (const FabricTransportEndpointOwnerRef &owner : owners)
    json.value(printFabricRef(owner));
}

void writeCountExtreme(llvm::json::OStream &json, llvm::StringRef name,
                       const FabricTopologyCountExtreme &extreme) {
  json.attributeObject(name, [&] {
    json.attribute("value", extreme.value);
    json.attributeArray("owners",
                        [&] { writeOwnerArray(json, extreme.owners); });
  });
}

void writeRatioExtreme(
    llvm::json::OStream &json, llvm::StringRef name,
    const std::optional<FabricTopologyRatioExtreme> &extreme) {
  if (!extreme) {
    json.attribute(name, nullptr);
    return;
  }
  json.attributeObject(name, [&] {
    json.attribute("numerator", extreme->numerator);
    json.attribute("denominator", extreme->denominator);
    json.attributeArray("owners",
                        [&] { writeOwnerArray(json, extreme->owners); });
  });
}

void writePeArray(llvm::json::OStream &json,
                  llvm::ArrayRef<FabricPeOccurrenceRef> pes) {
  for (FabricPeOccurrenceRef pe : pes)
    json.value(printFabricRef(pe));
}

void writeHopExtreme(llvm::json::OStream &json, llvm::StringRef name,
                     const std::optional<FabricTopologyHopExtreme> &extreme) {
  if (!extreme) {
    json.attribute(name, nullptr);
    return;
  }
  json.attributeObject(name, [&] {
    json.attribute("hops", extreme->hops);
    json.attributeArray("subjects",
                        [&] { writePeArray(json, extreme->subjects); });
  });
}

void writeHopDistribution(llvm::json::OStream &json, llvm::StringRef name,
                          const FabricTopologyHopDistribution &distribution) {
  json.attributeObject(name, [&] {
    json.attribute("subject_count", distribution.subjectCount);
    json.attribute("reachable_subject_count",
                   distribution.reachableSubjectCount);
    json.attribute("total_reachable_hops",
                   distribution.totalReachableHops);
    json.attributeArray("unreachable_subjects", [&] {
      writePeArray(json, distribution.unreachableSubjects);
    });
    writeHopExtreme(json, "minimum", distribution.minimum);
    writeHopExtreme(json, "maximum", distribution.maximum);
  });
}

struct EndpointGraph final {
  std::vector<FabricTransportEndpointRef> endpoints;
  std::vector<FabricPortDirection> directions;
  std::vector<std::vector<std::size_t>> outgoing;
  std::vector<std::vector<std::size_t>> incoming;
  std::set<CanonicalKey> moduleBoundaryAttachments;
  std::map<CanonicalKey, std::size_t> ordinals;
};

using EndpointDistance = std::optional<std::uint64_t>;

struct ModuleQualityInventory final {
  std::vector<FabricPeOccurrenceRef> pes;
  std::vector<FabricMemoryOccurrenceRef> memories;
  std::map<CanonicalKey, std::vector<std::size_t>> endpointsByOwner;
  std::map<CanonicalKey, std::size_t> peOrdinals;
};

llvm::Expected<EndpointGraph>
buildEndpointGraph(const FabricArtifactView &fabric) {
  EndpointGraph graph;
  graph.endpoints.assign(fabric.transportEndpoints().begin(),
                         fabric.transportEndpoints().end());
  graph.directions.reserve(graph.endpoints.size());
  graph.outgoing.resize(graph.endpoints.size());
  graph.incoming.resize(graph.endpoints.size());

  for (auto [ordinal, endpoint] : llvm::enumerate(graph.endpoints)) {
    const auto direction = fabric.transportEndpointDirection(endpoint);
    if (!direction)
      return invalid("a canonical endpoint has no direction");
    graph.directions.push_back(*direction);
    if (!graph.ordinals.emplace(key(endpoint), ordinal).second)
      return invalid("the canonical endpoint inventory has a duplicate");
  }

  for (const FabricPhysicalTraversalView &traversal :
       fabric.physicalTraversals()) {
    for (const FabricTransportEndpointRef &source : traversal.sources) {
      const auto sourcePosition = graph.ordinals.find(key(source));
      if (sourcePosition == graph.ordinals.end())
        return invalid("a traversal source is absent from the endpoint "
                       "inventory");
      for (const FabricTransportEndpointRef &destination :
           traversal.destinations) {
        const auto destinationPosition = graph.ordinals.find(key(destination));
        if (destinationPosition == graph.ordinals.end())
          return invalid("a traversal destination is absent from the endpoint "
                         "inventory");
        graph.outgoing[sourcePosition->second].push_back(
            destinationPosition->second);
        graph.incoming[destinationPosition->second].push_back(
            sourcePosition->second);
      }
    }
  }

  for (auto &adjacency : graph.outgoing) {
    llvm::sort(adjacency);
    adjacency.erase(std::unique(adjacency.begin(), adjacency.end()),
                    adjacency.end());
  }
  for (auto &adjacency : graph.incoming) {
    llvm::sort(adjacency);
    adjacency.erase(std::unique(adjacency.begin(), adjacency.end()),
                    adjacency.end());
  }
  for (const FabricModuleBoundaryTransportAttachmentView &attachment :
       fabric.moduleBoundaryTransportAttachments())
    graph.moduleBoundaryAttachments.insert(key(attachment.endpoint));
  return graph;
}

ModuleQualityInventory buildModuleQualityInventory(
    const FabricArtifactView &fabric, const EndpointGraph &graph) {
  ModuleQualityInventory inventory;
  inventory.pes.assign(fabric.peOccurrences().begin(),
                       fabric.peOccurrences().end());
  inventory.memories.assign(fabric.memoryOccurrences().begin(),
                            fabric.memoryOccurrences().end());
  llvm::sort(inventory.pes, [](const auto &lhs, const auto &rhs) {
    return canonicalFabricBytes(lhs) < canonicalFabricBytes(rhs);
  });
  llvm::sort(inventory.memories, [](const auto &lhs, const auto &rhs) {
    return canonicalFabricBytes(lhs) < canonicalFabricBytes(rhs);
  });
  for (auto [ordinal, pe] : llvm::enumerate(inventory.pes))
    inventory.peOrdinals.emplace(canonicalFabricBytes(pe), ordinal);
  for (auto [ordinal, endpoint] : llvm::enumerate(graph.endpoints))
    inventory.endpointsByOwner[key(endpoint.owner)].push_back(ordinal);
  return inventory;
}

std::vector<std::uint64_t>
undirectedEndpointDistances(const EndpointGraph &graph,
                            llvm::ArrayRef<std::size_t> starts,
                            const CanonicalKey &sourceOwner) {
  constexpr std::uint64_t unreachable =
      std::numeric_limits<std::uint64_t>::max();
  std::vector<std::uint64_t> distance(graph.endpoints.size(), unreachable);
  std::deque<std::size_t> pending;
  for (std::size_t start : starts) {
    distance[start] = 0;
    pending.push_back(start);
  }
  while (!pending.empty()) {
    const std::size_t current = pending.front();
    pending.pop_front();
    const FabricTransportEndpointOwnerRef &owner =
        graph.endpoints[current].owner;
    if (key(owner) != sourceOwner && !isTransparent(owner.kind()) &&
        owner.kind() !=
            FabricTransportEndpointOwnerKind::FabricSwitchOccurrence)
      continue;
    const auto include = [&](std::size_t next) {
      if (distance[next] != unreachable)
        return;
      distance[next] = distance[current] + 1;
      pending.push_back(next);
    };
    for (std::size_t next : graph.outgoing[current])
      include(next);
    for (std::size_t next : graph.incoming[current])
      include(next);
  }
  return distance;
}

llvm::Expected<std::vector<std::vector<std::uint64_t>>>
buildPeDistanceRows(const EndpointGraph &graph,
                    const ModuleQualityInventory &inventory) {
  std::vector<std::vector<std::uint64_t>> rows;
  rows.reserve(inventory.pes.size());
  for (FabricPeOccurrenceRef pe : inventory.pes) {
    const CanonicalKey peKey =
        key(FabricTransportEndpointOwnerRef::of(pe));
    const auto starts = inventory.endpointsByOwner.find(peKey);
    if (starts == inventory.endpointsByOwner.end())
      return invalid("a Module PE has no canonical transport endpoints");
    rows.push_back(
        undirectedEndpointDistances(graph, starts->second, peKey));
  }
  return rows;
}

llvm::Expected<llvm::ArrayRef<std::uint64_t>>
peDistanceRow(const ModuleQualityInventory &inventory,
              llvm::ArrayRef<std::vector<std::uint64_t>> rows,
              FabricPeOccurrenceRef pe) {
  const auto ordinal = inventory.peOrdinals.find(canonicalFabricBytes(pe));
  if (ordinal == inventory.peOrdinals.end() || ordinal->second >= rows.size())
    return invalid("a Module PE is absent from the locality inventory");
  return llvm::ArrayRef<std::uint64_t>(rows[ordinal->second]);
}

template <typename Ref>
EndpointDistance nearestOwnerDistance(
    const ModuleQualityInventory &inventory,
    llvm::ArrayRef<std::uint64_t> distances, llvm::ArrayRef<Ref> targets,
    const std::optional<CanonicalKey> &excluded = std::nullopt) {
  constexpr std::uint64_t unreachable =
      std::numeric_limits<std::uint64_t>::max();
  std::uint64_t best = unreachable;
  for (const Ref &target : targets) {
    const CanonicalKey targetKey =
        key(FabricTransportEndpointOwnerRef::of(target));
    if (excluded && targetKey == *excluded)
      continue;
    const auto endpoints = inventory.endpointsByOwner.find(targetKey);
    if (endpoints == inventory.endpointsByOwner.end())
      continue;
    for (std::size_t endpoint : endpoints->second)
      best = std::min(best, distances[endpoint]);
  }
  if (best == unreachable)
    return std::nullopt;
  return best;
}

void includeHopDistance(FabricTopologyHopDistribution &distribution,
                        FabricPeOccurrenceRef subject,
                        EndpointDistance distance) {
  ++distribution.subjectCount;
  if (!distance) {
    distribution.unreachableSubjects.push_back(subject);
    return;
  }
  ++distribution.reachableSubjectCount;
  distribution.totalReachableHops += *distance;
  if (!distribution.minimum || *distance < distribution.minimum->hops)
    distribution.minimum = FabricTopologyHopExtreme{*distance, {subject}};
  else if (*distance == distribution.minimum->hops)
    distribution.minimum->subjects.push_back(subject);
  if (!distribution.maximum || *distance > distribution.maximum->hops)
    distribution.maximum = FabricTopologyHopExtreme{*distance, {subject}};
  else if (*distance == distribution.maximum->hops)
    distribution.maximum->subjects.push_back(subject);
}

llvm::Expected<std::vector<FabricTopologyScheduleQuality>>
analyzeScheduleQuality(const FabricArtifactView &fabric,
                       const ModuleQualityInventory &inventory,
                       llvm::ArrayRef<std::vector<std::uint64_t>>
                           distanceRows) {
  constexpr ::fabric::Schedule schedules[] = {::fabric::Schedule::Spatial,
                                               ::fabric::Schedule::Temporal};
  std::vector<FabricTopologyScheduleQuality> result;
  result.reserve(std::size(schedules));
  for (::fabric::Schedule schedule : schedules) {
    FabricTopologyScheduleQuality quality{
        schedule, 0, 0, 0, {}, {}, {}, {}};
    std::vector<FabricPeOccurrenceRef> samePes;
    std::vector<FabricPeOccurrenceRef> otherPes;
    std::vector<FabricMemoryOccurrenceRef> sameMemories;
    std::vector<FabricMemoryOccurrenceRef> otherMemories;
    for (FabricPeOccurrenceRef pe : inventory.pes) {
      const auto ownerSchedule = fabric.peSchedule(pe);
      if (!ownerSchedule)
        return invalid("a Module PE has no scheduling domain");
      (*ownerSchedule == schedule ? samePes : otherPes).push_back(pe);
    }
    for (FabricMemoryOccurrenceRef memory : inventory.memories) {
      const auto ownerSchedule = fabric.memorySchedule(memory);
      if (!ownerSchedule)
        continue;
      (*ownerSchedule == schedule ? sameMemories : otherMemories)
          .push_back(memory);
    }
    quality.peCount = samePes.size();
    quality.memoryCount = sameMemories.size();
    for (FabricSwitchOccurrenceRef sw : fabric.switchOccurrences()) {
      const auto ownerSchedule = fabric.switchSchedule(sw);
      if (!ownerSchedule)
        return invalid("a Module switch has no scheduling domain");
      quality.switchCount += *ownerSchedule == schedule;
    }
    for (FabricPeOccurrenceRef pe : samePes) {
      const CanonicalKey peKey =
          key(FabricTransportEndpointOwnerRef::of(pe));
      auto distances = peDistanceRow(inventory, distanceRows, pe);
      if (!distances)
        return distances.takeError();
      includeHopDistance(
          quality.nearestSameSchedulePe, pe,
          nearestOwnerDistance(inventory, *distances,
                               llvm::ArrayRef(samePes), peKey));
      includeHopDistance(
          quality.nearestOtherSchedulePe, pe,
          nearestOwnerDistance(inventory, *distances,
                               llvm::ArrayRef(otherPes)));
      includeHopDistance(
          quality.nearestMatchingMemory, pe,
          nearestOwnerDistance(inventory, *distances,
                               llvm::ArrayRef(sameMemories)));
      includeHopDistance(
          quality.nearestOtherScheduleMemory, pe,
          nearestOwnerDistance(inventory, *distances,
                               llvm::ArrayRef(otherMemories)));
    }
    result.push_back(std::move(quality));
  }
  return result;
}

llvm::Expected<std::vector<FabricTopologyCapabilityQuality>>
analyzeCapabilityQuality(const FabricArtifactView &fabric,
                         const ModuleQualityInventory &inventory,
                         llvm::ArrayRef<std::vector<std::uint64_t>>
                             distanceRows) {
  std::vector<std::set<CanonicalKey>> supportingKeys(
      ::dataflow::operationSchemaCount());
  for (FabricFuOccurrenceRef fu : fabric.fuOccurrences()) {
    const auto parent = fabric.parentPeOf(fu);
    const auto definition = fabric.fuTemplateOf(fu);
    if (!parent || !definition)
      return invalid("a Module FU lost its PE or template owner");
    std::set<::dataflow::OperationSchemaId> schemas;
    for (const FabricFuCapabilityTemplateRecord &row :
         fabric.fuCapabilityTemplates(*definition)) {
      for (const FabricFuTemplateNodeRef &node : row.activeNodes) {
        if (node.node != FabricFuNodeKind::Op)
          continue;
        const ResolvedFabricOpCapabilityView *capability =
            fabric.resolvedFabricOpCapability(node);
        if (!capability)
          return invalid("an active FU operation has no resolved capability");
        schemas.insert(capability->enabledOperationSchemas.begin(),
                       capability->enabledOperationSchemas.end());
      }
    }
    for (::dataflow::OperationSchemaId schema : schemas)
      supportingKeys[static_cast<std::size_t>(schema)].insert(
          canonicalFabricBytes(*parent));
  }

  std::vector<FabricTopologyCapabilityQuality> result;
  for (std::uint32_t ordinal = 0;
       ordinal != ::dataflow::operationSchemaCount(); ++ordinal) {
    if (supportingKeys[ordinal].empty())
      continue;
    FabricTopologyCapabilityQuality quality{
        static_cast<::dataflow::OperationSchemaId>(ordinal), {}, 0, 0, {}, {}};
    for (FabricPeOccurrenceRef pe : inventory.pes) {
      if (supportingKeys[ordinal].count(canonicalFabricBytes(pe)))
        quality.supportingPes.push_back(pe);
    }
    for (FabricPeOccurrenceRef pe : quality.supportingPes) {
      const auto schedule = fabric.peSchedule(pe);
      if (!schedule)
        return invalid("a capability-supporting PE has no scheduling domain");
      if (*schedule == ::fabric::Schedule::Spatial)
        ++quality.spatialPeCount;
      else
        ++quality.temporalPeCount;
    }
    for (FabricPeOccurrenceRef pe : inventory.pes) {
      const CanonicalKey peKey =
          key(FabricTransportEndpointOwnerRef::of(pe));
      auto distances = peDistanceRow(inventory, distanceRows, pe);
      if (!distances)
        return distances.takeError();
      includeHopDistance(
          quality.coverage, pe,
          nearestOwnerDistance(inventory, *distances,
                               llvm::ArrayRef(quality.supportingPes)));
      if (supportingKeys[ordinal].count(canonicalFabricBytes(pe)))
        includeHopDistance(
            quality.supportingPeer, pe,
            nearestOwnerDistance(inventory, *distances,
                                 llvm::ArrayRef(quality.supportingPes), peKey));
    }
    result.push_back(std::move(quality));
  }
  llvm::sort(result, [](const auto &lhs, const auto &rhs) {
    return ::dataflow::operationSchemaSpelling(lhs.schema) <
           ::dataflow::operationSchemaSpelling(rhs.schema);
  });
  return result;
}

struct EndpointTraversalScratch final {
  explicit EndpointTraversalScratch(std::size_t endpointCount)
      : visitEpoch(endpointCount, 0) {}

  void begin(std::size_t start) {
    if (epoch == std::numeric_limits<std::uint64_t>::max()) {
      std::fill(visitEpoch.begin(), visitEpoch.end(), 0);
      epoch = 0;
    }
    ++epoch;
    pending.clear();
    pending.push_back(start);
    visitEpoch[start] = epoch;
  }

  bool mark(std::size_t endpoint) {
    if (visitEpoch[endpoint] == epoch)
      return false;
    visitEpoch[endpoint] = epoch;
    return true;
  }

  std::vector<std::uint64_t> visitEpoch;
  std::uint64_t epoch = 0;
  std::vector<std::size_t> pending;
};

FabricTopologyPortQuality
analyzePort(const EndpointGraph &graph, std::size_t start,
            const FabricTransportEndpointOwnerRef &subject,
            FabricRootKind rootKind, EndpointTraversalScratch &scratch) {
  FabricTopologyPortQuality result{graph.endpoints[start], {}, {}, false};
  scratch.begin(start);

  while (!scratch.pending.empty()) {
    const std::size_t current = scratch.pending.back();
    scratch.pending.pop_back();
    const FabricTransportEndpointRef &endpoint = graph.endpoints[current];

    if (graph.moduleBoundaryAttachments.count(key(endpoint)))
      result.reachesModuleBoundary = true;

    if (current != start && endpoint.owner != subject) {
      const FabricTransportEndpointOwnerKind ownerKind = endpoint.owner.kind();
      if (isRoutingResource(rootKind, ownerKind)) {
        result.routingResources.push_back(endpoint.owner);
        continue;
      }
      if (!isTransparent(ownerKind)) {
        result.directResources.push_back(endpoint.owner);
        continue;
      }
    }

    const auto &adjacency =
        graph.directions[start] == FabricPortDirection::Output
            ? graph.outgoing[current]
            : graph.incoming[current];
    for (std::size_t next : adjacency) {
      if (!scratch.mark(next))
        continue;
      scratch.pending.push_back(next);
    }
  }

  sortUniqueOwners(result.routingResources);
  sortUniqueOwners(result.directResources);
  return result;
}

} // namespace

llvm::StringRef
fabricTopologyTerminalKindSpelling(FabricTopologyTerminalKind kind) {
  switch (kind) {
  case FabricTopologyTerminalKind::ProcessingElement:
    return "processing_element";
  case FabricTopologyTerminalKind::Memory:
    return "memory";
  case FabricTopologyTerminalKind::SpatialCore:
    return "spatial_core";
  case FabricTopologyTerminalKind::ServiceEndpoint:
    return "service_endpoint";
  }
  llvm_unreachable("unknown Fabric topology terminal kind");
}

llvm::Expected<FabricTopologyQualityReport>
analyzeFabricTopologyQuality(const FabricArtifactView &fabric) {
  const FabricRootKind rootKind = fabric.rootKind();
  if (rootKind != FabricRootKind::Module && rootKind != FabricRootKind::System)
    return invalid("only Module and System roots have terminal quality "
                   "metrics");

  auto graph = buildEndpointGraph(fabric);
  if (!graph)
    return graph.takeError();

  std::vector<
      std::pair<FabricTopologyTerminalKind, FabricTransportEndpointOwnerRef>>
      subjects;
  if (rootKind == FabricRootKind::Module) {
    subjects.reserve(fabric.peOccurrences().size() +
                     fabric.memoryOccurrences().size());
    for (FabricPeOccurrenceRef occurrence : fabric.peOccurrences())
      appendSubject(subjects, FabricTopologyTerminalKind::ProcessingElement,
                    occurrence);
    for (FabricMemoryOccurrenceRef occurrence : fabric.memoryOccurrences())
      appendSubject(subjects, FabricTopologyTerminalKind::Memory, occurrence);
  } else {
    subjects.reserve(fabric.accCoreOccurrences().size() +
                     fabric.systemServiceEndpoints().size());
    for (AccCoreOccurrenceRef occurrence : fabric.accCoreOccurrences())
      appendSubject(subjects, FabricTopologyTerminalKind::SpatialCore,
                    SpatialCoreOccurrenceRef{occurrence});
    for (SystemServiceEndpointRef endpoint : fabric.systemServiceEndpoints())
      appendSubject(subjects, FabricTopologyTerminalKind::ServiceEndpoint,
                    endpoint);
  }
  llvm::sort(subjects, [](const auto &lhs, const auto &rhs) {
    return key(lhs.second) < key(rhs.second);
  });

  FabricTopologyQualityReport report{fabric.identity(), rootKind, {}, 0, {}, {}};
  report.owners.reserve(subjects.size());
  EndpointTraversalScratch scratch(graph->endpoints.size());
  for (const auto &[kind, subject] : subjects) {
    FabricTopologyOwnerQuality owner{kind, subject, {}, {}, {}, 0, 0};
    const std::uint64_t endpointCount = fabric.transportEndpointCount(subject);
    owner.ports.reserve(endpointCount);
    for (FabricOrdinal ordinal = 0; ordinal != endpointCount; ++ordinal) {
      const FabricTransportEndpointRef endpoint{subject, ordinal};
      const auto position = graph->ordinals.find(key(endpoint));
      if (position == graph->ordinals.end())
        return invalid("a terminal endpoint is absent from the canonical "
                       "endpoint inventory");
      owner.ports.push_back(
          analyzePort(*graph, position->second, subject, rootKind, scratch));
      FabricTopologyPortQuality &port = owner.ports.back();
      owner.distinctRoutingResources.insert(
          owner.distinctRoutingResources.end(), port.routingResources.begin(),
          port.routingResources.end());
      owner.distinctDirectResources.insert(owner.distinctDirectResources.end(),
                                           port.directResources.begin(),
                                           port.directResources.end());
      owner.boundaryPortCount += port.reachesModuleBoundary;
      owner.unreachablePortCount += port.unreachable();
    }
    sortUniqueOwners(owner.distinctRoutingResources);
    sortUniqueOwners(owner.distinctDirectResources);
    report.owners.push_back(std::move(owner));
  }
  if (rootKind == FabricRootKind::Module) {
    const ModuleQualityInventory inventory =
        buildModuleQualityInventory(fabric, *graph);
    auto distanceRows = buildPeDistanceRows(*graph, inventory);
    if (!distanceRows)
      return distanceRows.takeError();
    for (FabricMemoryOccurrenceRef memory : inventory.memories)
      report.unscheduledMemoryCount += !fabric.memorySchedule(memory);
    auto schedules = analyzeScheduleQuality(fabric, inventory, *distanceRows);
    if (!schedules)
      return schedules.takeError();
    report.schedules = std::move(*schedules);
    auto capabilities =
        analyzeCapabilityQuality(fabric, inventory, *distanceRows);
    if (!capabilities)
      return capabilities.takeError();
    report.capabilities = std::move(*capabilities);
  }
  return report;
}

llvm::Expected<std::vector<FabricTopologyQualityReport>>
analyzeFabricTopologyQualityClosure(const FabricArtifactView &fabric) {
  std::vector<FabricTopologyQualityReport> reports;
  auto root = analyzeFabricTopologyQuality(fabric);
  if (!root)
    return root.takeError();
  reports.push_back(std::move(*root));
  reports.reserve(1 + fabric.importedModules().size());
  for (const FabricArtifactView &module : fabric.importedModules()) {
    auto report = analyzeFabricTopologyQuality(module);
    if (!report)
      return report.takeError();
    reports.push_back(std::move(*report));
  }
  return reports;
}

std::vector<FabricTopologyKindDistribution>
summarizeFabricTopologyQuality(const FabricTopologyQualityReport &report) {
  constexpr FabricTopologyTerminalKind kinds[] = {
      FabricTopologyTerminalKind::ProcessingElement,
      FabricTopologyTerminalKind::Memory,
      FabricTopologyTerminalKind::SpatialCore,
      FabricTopologyTerminalKind::ServiceEndpoint,
  };
  std::vector<FabricTopologyKindDistribution> result;
  for (FabricTopologyTerminalKind kind : kinds) {
    FabricTopologyKindDistribution distribution{
        kind,         0,           0, {}, {}, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt};
    for (const FabricTopologyOwnerQuality &owner : report.owners) {
      if (owner.kind != kind)
        continue;
      const bool first = distribution.ownerCount == 0;
      ++distribution.ownerCount;
      includeCountExtreme(distribution.minimumPortCount,
                          distribution.maximumPortCount, owner.portCount(),
                          owner.owner, first);
      if (owner.portCount() == 0) {
        ++distribution.zeroPortOwnerCount;
        continue;
      }
      includeRatioExtreme(
          distribution.minimumRoutingRatio, distribution.maximumRoutingRatio,
          owner.routingResourceCount(), owner.portCount(), owner.owner);
      includeRatioExtreme(
          distribution.minimumDirectRatio, distribution.maximumDirectRatio,
          owner.directResourceCount(), owner.portCount(), owner.owner);
    }
    if (distribution.ownerCount != 0)
      result.push_back(std::move(distribution));
  }
  return result;
}

FabricTopologyDseQuality projectFabricTopologyDseQuality(
    const FabricTopologyQualityReport &report) {
  FabricTopologyDseQuality result;
  result.unscheduledMemoryCount = report.unscheduledMemoryCount;
  for (const FabricTopologyScheduleQuality &schedule : report.schedules) {
    if ((schedule.peCount == 0) != (schedule.memoryCount == 0))
      result.scheduleSupplyGap +=
          schedule.peCount == 0 ? schedule.memoryCount : schedule.peCount;
    result.matchingMemoryUnreachablePeCount +=
        schedule.nearestMatchingMemory.unreachableSubjects.size();
    result.matchingMemoryTotalReachableHops +=
        schedule.nearestMatchingMemory.totalReachableHops;
  }
  for (const FabricTopologyCapabilityQuality &capability :
       report.capabilities) {
    result.capabilityCoverageUnreachablePeCount +=
        capability.coverage.unreachableSubjects.size();
    result.capabilityCoverageTotalReachableHops +=
        capability.coverage.totalReachableHops;
    result.isolatedCapabilitySupportingPeCount +=
        capability.supportingPeer.unreachableSubjects.size();
  }
  return result;
}

llvm::Error writeFabricTopologyQualityJson(
    llvm::ArrayRef<FabricTopologyQualityReport> reports,
    llvm::raw_ostream &output) {
  if (reports.empty())
    return invalid("a JSON report closure must be nonempty");

  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", "loom.fabric.topology_quality.1");
    json.attribute("schema_version", "1.1");
    json.attributeArray("roots", [&] {
      for (const FabricTopologyQualityReport &report : reports) {
        json.object([&] {
          json.attribute("artifact",
                         formatArtifactIdentityHex(report.artifact));
          json.attribute("root_kind", fabricRefKeyword(report.rootKind));
          json.attribute("unscheduled_memory_count",
                         report.unscheduledMemoryCount);
          json.attributeArray("schedule_distributions", [&] {
            for (const FabricTopologyScheduleQuality &schedule :
                 report.schedules) {
              json.object([&] {
                json.attribute("schedule",
                               ::fabric::stringifySchedule(schedule.schedule));
                json.attribute("pe_count", schedule.peCount);
                json.attribute("memory_count", schedule.memoryCount);
                json.attribute("switch_count", schedule.switchCount);
                writeHopDistribution(json, "nearest_same_schedule_pe",
                                     schedule.nearestSameSchedulePe);
                writeHopDistribution(json, "nearest_other_schedule_pe",
                                     schedule.nearestOtherSchedulePe);
                writeHopDistribution(json, "nearest_matching_memory",
                                     schedule.nearestMatchingMemory);
                writeHopDistribution(json, "nearest_other_schedule_memory",
                                     schedule.nearestOtherScheduleMemory);
              });
            }
          });
          json.attributeArray("capability_distributions", [&] {
            for (const FabricTopologyCapabilityQuality &capability :
                 report.capabilities) {
              json.object([&] {
                json.attribute(
                    "schema",
                    ::dataflow::operationSchemaSpelling(capability.schema));
                json.attributeArray("supporting_pes", [&] {
                  writePeArray(json, capability.supportingPes);
                });
                json.attribute("spatial_pe_count",
                               capability.spatialPeCount);
                json.attribute("temporal_pe_count",
                               capability.temporalPeCount);
                writeHopDistribution(json, "coverage", capability.coverage);
                writeHopDistribution(json, "supporting_peer",
                                     capability.supportingPeer);
              });
            }
          });
          json.attributeArray("terminal_distributions", [&] {
            for (const FabricTopologyKindDistribution &distribution :
                 summarizeFabricTopologyQuality(report)) {
              json.object([&] {
                json.attribute("kind", fabricTopologyTerminalKindSpelling(
                                           distribution.kind));
                json.attribute("owner_count", distribution.ownerCount);
                json.attribute("zero_port_owner_count",
                               distribution.zeroPortOwnerCount);
                writeCountExtreme(json, "minimum_port_count",
                                  distribution.minimumPortCount);
                writeCountExtreme(json, "maximum_port_count",
                                  distribution.maximumPortCount);
                writeRatioExtreme(json, "minimum_routing_ratio",
                                  distribution.minimumRoutingRatio);
                writeRatioExtreme(json, "maximum_routing_ratio",
                                  distribution.maximumRoutingRatio);
                writeRatioExtreme(json, "minimum_direct_ratio",
                                  distribution.minimumDirectRatio);
                writeRatioExtreme(json, "maximum_direct_ratio",
                                  distribution.maximumDirectRatio);
              });
            }
          });
          json.attributeArray("owners", [&] {
            for (const FabricTopologyOwnerQuality &owner : report.owners) {
              json.object([&] {
                json.attribute("kind",
                               fabricTopologyTerminalKindSpelling(owner.kind));
                json.attribute("owner", printFabricRef(owner.owner));
                json.attribute("port_count", owner.portCount());
                json.attribute("routing_resource_count",
                               owner.routingResourceCount());
                json.attribute("direct_resource_count",
                               owner.directResourceCount());
                json.attribute("boundary_port_count", owner.boundaryPortCount);
                json.attribute("unreachable_port_count",
                               owner.unreachablePortCount);
                json.attributeArray("ports", [&] {
                  for (const FabricTopologyPortQuality &port : owner.ports) {
                    json.object([&] {
                      json.attribute("endpoint", printFabricRef(port.endpoint));
                      json.attributeArray("routing_resources", [&] {
                        for (const auto &resource : port.routingResources)
                          json.value(printFabricRef(resource));
                      });
                      json.attributeArray("direct_resources", [&] {
                        for (const auto &resource : port.directResources)
                          json.value(printFabricRef(resource));
                      });
                      json.attribute("reaches_module_boundary",
                                     port.reachesModuleBoundary);
                      json.attribute("unreachable", port.unreachable());
                    });
                  }
                });
              });
            }
          });
        });
      }
    });
  });
  output << '\n';
  return llvm::Error::success();
}

} // namespace loom::fabric
