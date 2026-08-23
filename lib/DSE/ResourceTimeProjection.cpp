#include "DSE/ResourceTimeFrontier.h"

#include "ResourceTimeFrontierInternal.h"

#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Deployment/Deployment.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingProgressAnalysis.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <vector>

namespace loom::dse {
using namespace detail;
namespace {

constexpr llvm::StringLiteral resourceTimeTransitionCacheDescriptor{
    "loom.dse.resource_time_transition_cache.1"};
constexpr llvm::StringLiteral resourceTimeAnalyticModelDescriptor{
    "loom.dse.resource_time_analytic_model.1"};
constexpr llvm::StringLiteral resourceTimeProjectionMemoDescriptor{
    "loom.dse.resource_time_projection_memo.1"};

llvm::Error invalid(const llvm::Twine &message) {
  return invalidResourceTimeFrontier(message);
}

std::vector<std::uint8_t>
canonicalAllocationBytes(const pnr::ResourceTimeRegionAllocation &allocation) {
  std::vector<std::vector<std::uint8_t>> resources;
  resources.reserve(allocation.resources.size());
  for (const auto &resource : allocation.resources)
    resources.push_back(fabric::canonicalFabricBytes(resource));
  llvm::sort(resources);
  std::vector<std::uint8_t> bytes;
  appendDataflowRoot(bytes, allocation.region);
  appendU64(bytes, resources.size());
  for (const auto &resource : resources)
    appendBlob(bytes, resource);
  return bytes;
}

void appendAllocations(
    std::vector<std::uint8_t> &bytes,
    llvm::ArrayRef<pnr::ResourceTimeRegionAllocation> allocations) {
  std::vector<std::vector<std::uint8_t>> encoded;
  encoded.reserve(allocations.size());
  for (const auto &allocation : allocations)
    encoded.push_back(canonicalAllocationBytes(allocation));
  llvm::sort(encoded);
  appendU64(bytes, encoded.size());
  for (const auto &allocation : encoded)
    appendBlob(bytes, allocation);
}

void appendRoots(std::vector<std::uint8_t> &bytes,
                 llvm::ArrayRef<ArtifactRootReference> roots) {
  std::vector<ArtifactRootReference> canonical(roots.begin(), roots.end());
  llvm::sort(canonical, artifactRootReferenceLess);
  appendU64(bytes, canonical.size());
  for (const auto &root : canonical)
    appendRoot(bytes, root);
}

} // namespace

llvm::Expected<ComponentViewDigest> resourceTimeAnalyticModelSnapshotDigest() {
  return computeComponentViewDigest(
      {reinterpret_cast<const std::uint8_t *>(
           resourceTimeAnalyticModelDescriptor.data()),
       resourceTimeAnalyticModelDescriptor.size()},
      {});
}

llvm::Expected<ComponentViewDigest> deriveResourceTimeProjectionCacheKey(
    const ResourceTimeInvocationKey &invocation) {
  if (invocation.sourceLineage.schemaIdentity.empty() ||
      invocation.dataflow.schemaIdentity.empty() ||
      invocation.fabric.schemaIdentity.empty() ||
      invocation.workload.schemaIdentity.empty() ||
      invocation.runtimeInput.schemaIdentity.empty() ||
      invocation.entrySymbol.empty())
    return invalid("projection cache key contains an empty semantic input");
  std::vector<std::uint8_t> bytes;
  appendString(bytes, resourceTimeProjectionMemoDescriptor);
  appendResourceTimeInvocationKey(bytes, invocation);
  return computeComponentViewDigest(
      {reinterpret_cast<const std::uint8_t *>(
           resourceTimeProjectionMemoDescriptor.data()),
       resourceTimeProjectionMemoDescriptor.size()},
      bytes);
}

llvm::Expected<ResourceTimeDataflowProjection> projectResourceTimeDataflow(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &system,
    llvm::StringRef entrySymbol,
    std::optional<std::uint64_t> estimatedRuntimePicoseconds) {
  if (entrySymbol.empty())
    return invalid("resource-time projection requires an ABI entry symbol");
  auto reachable =
      dataflow.projectRootThreadLaunchesReachableFromAbiEntry(entrySymbol);
  if (!reachable)
    return reachable.takeError();
  if (reachable->empty())
    return invalid("resource-time projection has no reachable root thread");
  llvm::sort(*reachable, rootLess);
  if (std::adjacent_find(reachable->begin(), reachable->end()) !=
      reachable->end())
    return invalid("resource-time projection has duplicate roots");
  const std::uint64_t availableAccCores =
      system.artifact().accCoreOccurrences().size();
  if (availableAccCores == 0)
    return invalid("resource-time projection has no AccCore capacity");

  std::vector<std::vector<::dataflow::RootedGraphLaunchRef>> launches(
      reachable->size());
  dataflow.forEachRootedGraphLaunch(
      [&](::dataflow::RootedGraphLaunchRef launch) {
        const auto found =
            llvm::lower_bound(*reachable, launch.rootThreadLaunch, rootLess);
        if (found != reachable->end() && *found == launch.rootThreadLaunch)
          launches[static_cast<std::size_t>(found - reachable->begin())]
              .push_back(launch);
      });

  std::vector<std::uint64_t> weights(reachable->size(), 1);
  std::vector<std::uint64_t> maximumUseful(reachable->size(),
                                           availableAccCores);
  std::vector<std::uint64_t> logicalEpochCounts(reachable->size(), 0);
  std::vector<ResourceTimeRegionAnalyticFeatures> analyticFeatures(
      reachable->size());
  std::vector<ResourceTimeEstimateSupport> boundSupport(
      reachable->size(), ResourceTimeEstimateSupport::Unsupported);
  std::uint64_t totalWeight = 0;
  for (std::size_t ordinal = 0; ordinal != reachable->size(); ++ordinal) {
    std::uint64_t weight = 0;
    std::optional<std::uint64_t> pointCount;
    bool exactPoints = !launches[ordinal].empty();
    if (launches[ordinal].empty()) {
      auto logical =
          dataflow.projectRootThreadLogicalDomain((*reachable)[ordinal]);
      if (!logical)
        return logical.takeError();
      if (logical->coordinateRank == 0) {
        pointCount = 1;
        exactPoints = true;
      }
    }
    for (const ::dataflow::RootedGraphLaunchRef launch : launches[ordinal]) {
      auto graph = dataflow.resolve(launch);
      if (!graph)
        return graph.takeError();
      ++analyticFeatures[ordinal].graphCount;
      const std::uint64_t actors =
          llvm::count_if(dataflow.actors(), [&](const auto &actor) {
            return actor.graph == *graph;
          });
      analyticFeatures[ordinal].actorCount =
          llvm::checkedAddUnsigned(analyticFeatures[ordinal].actorCount, actors)
              .value_or(std::numeric_limits<std::uint64_t>::max());
      for (const auto &actor : dataflow.actors()) {
        if (actor.graph != *graph)
          continue;
        switch (actor.kind) {
        case ::dataflow::CanonicalDataflowActorKind::Compute:
          ++analyticFeatures[ordinal].computeActorCount;
          break;
        case ::dataflow::CanonicalDataflowActorKind::Control:
          ++analyticFeatures[ordinal].controlActorCount;
          break;
        case ::dataflow::CanonicalDataflowActorKind::Memory:
          ++analyticFeatures[ordinal].memoryActorCount;
          break;
        }
      }
      const auto addedWeight =
          llvm::checkedAddUnsigned(weight, std::max<std::uint64_t>(1, actors));
      if (!addedWeight)
        return invalid("resource-time region weight overflows");
      weight = *addedWeight;
      auto extents = dataflow.projectStaticDenseExtents(launch, entrySymbol);
      if (!extents)
        return extents.takeError();
      if (!*extents) {
        exactPoints = false;
        continue;
      }
      std::uint64_t points = 1;
      for (std::uint64_t extent : **extents) {
        auto product = llvm::checkedMulUnsigned(points, extent);
        if (!product)
          return invalid("resource-time logical-domain size overflows");
        points = *product;
      }
      if (pointCount && *pointCount != points)
        return invalid("one root has inconsistent static logical domains");
      pointCount = points;
    }
    weights[ordinal] = std::max<std::uint64_t>(1, weight);
    auto added = llvm::checkedAddUnsigned(totalWeight, weights[ordinal]);
    if (!added)
      return invalid("resource-time total region weight overflows");
    totalWeight = *added;
    if (exactPoints && pointCount && *pointCount != 0) {
      logicalEpochCounts[ordinal] = *pointCount;
      maximumUseful[ordinal] = std::min(*pointCount, availableAccCores);
      boundSupport[ordinal] = ResourceTimeEstimateSupport::Exact;
    }
  }

  std::vector<::dataflow::EventFamilyKey> boundaryEvents;
  boundaryEvents.reserve(reachable->size() * 2);
  for (const auto root : *reachable) {
    boundaryEvents.push_back(dataflow::rootThreadStartEventFamily(root));
    boundaryEvents.push_back(dataflow::rootThreadCompletionEventFamily(root));
  }
  auto causality =
      mapping::freezeMappingProgressModel(dataflow, boundaryEvents);
  if (!causality)
    return causality.takeError();

  ResourceTimeDataflowProjection result;
  result.acceleratedGraphCount = dataflow.graphs().size();
  result.acceleratedActorCount = dataflow.actors().size();
  result.resourceClasses.push_back({fabric::fabricArtifactSchema.identity.str(),
                                    fabric::fabricArtifactSchema.version,
                                    system.artifact().identity()});
  result.availableResourceUnits.push_back(availableAccCores);
  result.regions.reserve(reachable->size());
  result.regionBounds.reserve(reachable->size());
  for (std::size_t ordinal = 0; ordinal != reachable->size(); ++ordinal) {
    ResourceTimeRegionFeature feature{(*reachable)[ordinal],       {},    {},
                                      logicalEpochCounts[ordinal], false, {}};
    feature.allocationDomainExhaustive = true;
    feature.analyticFeatures = analyticFeatures[ordinal];
    feature.analyticFeatures.launchSynchronizationCost =
        feature.dependencies.size();
    feature.analyticFeatures.parallelismLowerBound =
        std::max<std::uint64_t>(1, logicalEpochCounts[ordinal]);
    feature.analyticFeatures.topologyCongestionProxy =
        feature.analyticFeatures.actorCount + feature.dependencies.size();
    for (std::size_t producer = 0; producer != reachable->size(); ++producer) {
      if (producer == ordinal)
        continue;
      auto completionPrecedes = mapping::mappingEventPrecedes(
          *causality,
          dataflow::rootThreadCompletionEventFamily((*reachable)[producer]),
          dataflow::rootThreadStartEventFamily((*reachable)[ordinal]));
      if (!completionPrecedes)
        return completionPrecedes.takeError();
      if (*completionPrecedes) {
        feature.dependencies.push_back(
            {(*reachable)[producer],
             pnr::ResourceTimeReadinessKind::Completion});
        continue;
      }
      auto startPrecedes = mapping::mappingEventPrecedes(
          *causality,
          dataflow::rootThreadStartEventFamily((*reachable)[producer]),
          dataflow::rootThreadStartEventFamily((*reachable)[ordinal]));
      if (!startPrecedes)
        return startPrecedes.takeError();
      if (*startPrecedes)
        feature.dependencies.push_back(
            {(*reachable)[producer],
             pnr::ResourceTimeReadinessKind::FifoToken});
    }
    llvm::sort(feature.dependencies, [](const auto &lhs, const auto &rhs) {
      return rootLess(lhs.producer, rhs.producer);
    });
    const unsigned __int128 scaled =
        static_cast<unsigned __int128>(
            estimatedRuntimePicoseconds.value_or(totalWeight)) *
        weights[ordinal];
    const std::uint64_t baseDuration = std::max<std::uint64_t>(
        1, static_cast<std::uint64_t>(std::min<unsigned __int128>(
               std::numeric_limits<std::uint64_t>::max(),
               (scaled + totalWeight - 1) / totalWeight)));
    const ResourceTimeEstimateSupport estimateSupport =
        estimatedRuntimePicoseconds ? ResourceTimeEstimateSupport::Analytic
                                    : ResourceTimeEstimateSupport::Unsupported;
    for (std::uint64_t units = 1; units <= maximumUseful[ordinal]; ++units)
      feature.speedupCurve.push_back(
          {{units},
           baseDuration / units + (baseDuration % units != 0),
           std::nullopt,
           std::nullopt,
           0,
           0,
           0,
           estimateSupport});
    result.regions.push_back(std::move(feature));
    result.regionBounds.push_back(
        {(*reachable)[ordinal], maximumUseful[ordinal], boundSupport[ordinal]});
  }
  return result;
}

std::uint64_t resourceTimeProjectionRetainedBytes(
    const ResourceTimeDataflowProjection &projection) {
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  const auto add = [](std::uint64_t lhs, std::uint64_t rhs) {
    return rhs > maximum - lhs ? maximum : lhs + rhs;
  };
  const auto product = [](std::size_t lhs, std::size_t rhs) {
    if (rhs != 0 && lhs > maximum / rhs)
      return maximum;
    return static_cast<std::uint64_t>(lhs * rhs);
  };
  std::uint64_t bytes = sizeof(ResourceTimeDataflowProjection);
  bytes = add(bytes, product(projection.resourceClasses.size(),
                             sizeof(ArtifactRootReference)));
  bytes = add(bytes, product(projection.availableResourceUnits.size(),
                             sizeof(std::uint64_t)));
  bytes = add(bytes, product(projection.regions.size(),
                             sizeof(ResourceTimeRegionFeature)));
  bytes = add(bytes, product(projection.regionBounds.size(),
                             sizeof(ResourceTimeRegionResourceBound)));
  for (const ResourceTimeRegionFeature &region : projection.regions) {
    bytes = add(bytes, product(region.dependencies.size(),
                               sizeof(ResourceTimeDependencyFeature)));
    bytes = add(bytes, product(region.speedupCurve.size(),
                               sizeof(ResourceTimeSpeedupPoint)));
    for (const ResourceTimeSpeedupPoint &point : region.speedupCurve)
      bytes = add(bytes,
                  product(point.resourceUnits.size(), sizeof(std::uint64_t)));
  }
  return bytes;
}

llvm::Expected<ComponentViewDigest> deriveResourceTimeTransitionCacheKey(
    const pnr::ResourceTimeTransition &transition,
    const ResourceTimeTransitionCacheKeyInput &input) {
  if (llvm::Error error = pnr::validateResourceTimeTransition(transition))
    return std::move(error);
  const auto hasSchema = [](const ArtifactRootReference &reference,
                            const ArtifactSchemaDescriptor &schema) {
    return reference.schemaIdentity == schema.identity &&
           reference.schemaVersion == schema.version;
  };
  if (!hasSchema(transition.beforeMapping, mapping::mappingArtifactSchema) ||
      !hasSchema(transition.afterMapping, mapping::mappingArtifactSchema))
    return invalid("transition cache key has a non-Mapping endpoint");
  if (!hasSchema(input.parentDeployment, deployment::deploymentSchema) ||
      !hasSchema(input.childDeployment, deployment::deploymentSchema))
    return invalid("transition cache key has a non-Deployment endpoint");
  if (!hasSchema(input.constraints, mapping::mappingConstraintSetSchema))
    return invalid("transition cache key has a non-constraint root");
  if (!hasSchema(input.childTarget, fabric::fabricArtifactSchema))
    return invalid("transition cache key has a non-Fabric child target");
  if (!transition.resourceDeltaDigest || !transition.configurationDeltaDigest ||
      !transition.routeDeltaDigest)
    return invalid("transition cache key requires every derived delta");
  auto trigger = dataflow::encodeDataflowReference(transition.trigger);
  if (!trigger)
    return trigger.takeError();

  std::vector<std::uint8_t> bytes;
  appendBlob(bytes, *trigger);
  appendRoot(bytes, transition.safePoint);
  appendRoot(bytes, transition.beforeMapping);
  appendRoot(bytes, transition.afterMapping);
  appendRoot(bytes, input.parentDeployment);
  appendRoot(bytes, input.childDeployment);
  appendAllocations(bytes, transition.beforeActive);
  appendAllocations(bytes, transition.afterActive);
  appendRoots(bytes, transition.beforeLiveWork);
  appendRoots(bytes, transition.afterLiveWork);
  appendOptionalRoot(bytes, transition.tokenLiveStateCorrespondence);
  appendBlob(bytes, transition.resourceDeltaDigest->bytes());
  appendBlob(bytes, transition.configurationDeltaDigest->bytes());
  appendBlob(bytes, transition.routeDeltaDigest->bytes());
  appendRoot(bytes, input.constraints);
  appendBlob(bytes, input.algorithmIdentity.bytes());
  appendRoot(bytes, input.childTarget);
  appendBlob(bytes, input.scheduleDeltaDigest.bytes());
  appendBlob(bytes, input.hardwareDeltaDigest.bytes());
  return computeComponentViewDigest(
      {reinterpret_cast<const std::uint8_t *>(
           resourceTimeTransitionCacheDescriptor.data()),
       resourceTimeTransitionCacheDescriptor.size()},
      bytes);
}

} // namespace loom::dse
