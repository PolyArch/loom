#ifndef LOOM_DSE_RESOURCETIMEFRONTIERINTERNAL_H
#define LOOM_DSE_RESOURCETIMEFRONTIERINTERNAL_H

#include "Common/ArtifactLocalReference.h"
#include "DSE/ResourceTimeFrontier.h"

#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <system_error>
#include <tuple>
#include <vector>

namespace loom::dse::detail {

inline llvm::Error invalidResourceTimeFrontier(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "resource_time_frontier_invalid: " + message);
}

inline bool rootLess(::dataflow::RootThreadLaunchRef lhs,
                     ::dataflow::RootThreadLaunchRef rhs) {
  if (lhs.artifact != rhs.artifact)
    return lhs.artifact.bytes() < rhs.artifact.bytes();
  return lhs.entity.value() < rhs.entity.value();
}

inline void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

inline void appendBlob(std::vector<std::uint8_t> &bytes,
                       llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

inline void appendRoot(std::vector<std::uint8_t> &bytes,
                       const ArtifactRootReference &reference) {
  appendBlob(bytes, encodeArtifactRootReference(reference));
}

inline void appendDataflowRoot(std::vector<std::uint8_t> &bytes,
                               ::dataflow::RootThreadLaunchRef reference) {
  appendBlob(bytes, reference.artifact.bytes());
  appendU64(bytes, reference.entity.value());
}

inline void
appendOptionalRoot(std::vector<std::uint8_t> &bytes,
                   const std::optional<ArtifactRootReference> &reference) {
  bytes.push_back(reference ? 1 : 0);
  if (reference)
    appendRoot(bytes, *reference);
}

inline void appendOptionalU64(std::vector<std::uint8_t> &bytes,
                              std::optional<std::uint64_t> value) {
  bytes.push_back(value ? 1 : 0);
  if (value)
    appendU64(bytes, *value);
}

inline void appendString(std::vector<std::uint8_t> &bytes,
                         llvm::StringRef value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.bytes_begin(), value.bytes_end());
}

inline void appendDigest(std::vector<std::uint8_t> &bytes,
                         const ComponentViewDigest &digest) {
  appendBlob(bytes, digest.bytes());
}

inline void
appendResourceTimeInvocationKey(std::vector<std::uint8_t> &bytes,
                                const ResourceTimeInvocationKey &invocation) {
  appendRoot(bytes, invocation.sourceLineage);
  appendRoot(bytes, invocation.dataflow);
  appendRoot(bytes, invocation.fabric);
  appendRoot(bytes, invocation.workload);
  appendRoot(bytes, invocation.runtimeInput);
  appendDigest(bytes, invocation.resolvedConfigDigest);
  appendDigest(bytes, invocation.modelSnapshotDigest);
  appendString(bytes, invocation.entrySymbol);
  appendOptionalU64(bytes, invocation.estimatedRuntimePicoseconds);
}

inline void appendResourceTimeFeatures(
    std::vector<std::uint8_t> &bytes,
    llvm::ArrayRef<ArtifactRootReference> resourceClasses,
    llvm::ArrayRef<ResourceTimeRegionFeature> regions) {
  appendU64(bytes, resourceClasses.size());
  for (const ArtifactRootReference &resource : resourceClasses)
    appendRoot(bytes, resource);
  appendU64(bytes, regions.size());
  for (const ResourceTimeRegionFeature &region : regions) {
    appendDataflowRoot(bytes, region.region);
    appendU64(bytes, region.logicalEpochCount);
    bytes.push_back(region.allocationDomainExhaustive ? 1 : 0);
    appendU64(bytes, region.analyticFeatures.actorCount);
    appendU64(bytes, region.analyticFeatures.computeActorCount);
    appendU64(bytes, region.analyticFeatures.controlActorCount);
    appendU64(bytes, region.analyticFeatures.memoryActorCount);
    appendU64(bytes, region.analyticFeatures.graphCount);
    appendU64(bytes, region.analyticFeatures.launchSynchronizationCost);
    appendU64(bytes, region.analyticFeatures.parallelismLowerBound);
    appendU64(bytes, region.analyticFeatures.topologyCongestionProxy);
    appendU64(bytes, region.dependencies.size());
    for (const ResourceTimeDependencyFeature &dependency :
         region.dependencies) {
      appendDataflowRoot(bytes, dependency.producer);
      appendU64(bytes, static_cast<std::uint64_t>(dependency.readiness));
    }
    appendU64(bytes, region.speedupCurve.size());
    for (const ResourceTimeSpeedupPoint &point : region.speedupCurve) {
      appendU64(bytes, point.resourceUnits.size());
      for (std::uint64_t units : point.resourceUnits)
        appendU64(bytes, units);
      appendU64(bytes, point.executionTimePicoseconds);
      appendOptionalU64(bytes, point.firstTokenLatencyPicoseconds);
      appendOptionalU64(bytes, point.initiationIntervalPicoseconds);
      appendU64(bytes, point.hostTransferTimePicoseconds);
      appendU64(bytes, point.configurationTimePicoseconds);
      appendU64(bytes, point.liveStateMigrationTimePicoseconds);
      appendU64(bytes, static_cast<std::uint64_t>(point.support));
    }
  }
}

inline void appendResourceTimePolicy(std::vector<std::uint8_t> &bytes,
                                     const ResourceTimeFrontierPolicy &policy) {
  appendU64(bytes, policy.availableResourceUnits.size());
  for (std::uint64_t units : policy.availableResourceUnits)
    appendU64(bytes, units);
  appendU64(bytes, policy.maximumStatesGenerated);
  appendU64(bytes, policy.maximumActionsGenerated);
  appendU64(bytes, policy.maximumStateCacheEntries);
  appendU64(bytes, policy.maximumRetainedBytes);
  appendU64(bytes, policy.beamWidth);
  appendU64(bytes, policy.maximumFinalists);
}

inline std::uint64_t
allocationMagnitude(llvm::ArrayRef<std::uint64_t> allocation) {
  return std::accumulate(allocation.begin(), allocation.end(), 0ULL);
}

inline bool fits(llvm::ArrayRef<std::uint64_t> used,
                 llvm::ArrayRef<std::uint64_t> requested,
                 llvm::ArrayRef<std::uint64_t> available) {
  if (used.size() != requested.size() || used.size() != available.size())
    return false;
  for (std::size_t index = 0; index != used.size(); ++index)
    if (requested[index] > available[index] ||
        used[index] > available[index] - requested[index])
      return false;
  return true;
}

inline std::uint64_t pointDuration(const ResourceTimeSpeedupPoint &point) {
  const auto withHost = llvm::checkedAddUnsigned(
      point.executionTimePicoseconds, point.hostTransferTimePicoseconds);
  if (!withHost)
    return std::numeric_limits<std::uint64_t>::max();
  const auto withConfiguration =
      llvm::checkedAddUnsigned(*withHost, point.configurationTimePicoseconds);
  if (!withConfiguration)
    return std::numeric_limits<std::uint64_t>::max();
  return llvm::checkedAddUnsigned(*withConfiguration,
                                  point.liveStateMigrationTimePicoseconds)
      .value_or(std::numeric_limits<std::uint64_t>::max());
}

inline std::uint8_t estimateSupportRank(ResourceTimeEstimateSupport support) {
  switch (support) {
  case ResourceTimeEstimateSupport::Exact:
    return 0;
  case ResourceTimeEstimateSupport::Calibrated:
    return 1;
  case ResourceTimeEstimateSupport::Analytic:
    return 2;
  case ResourceTimeEstimateSupport::OutOfDomain:
    return 3;
  case ResourceTimeEstimateSupport::Unsupported:
    return 4;
  }
  llvm_unreachable("unknown resource-time estimate support");
}

inline ResourceTimeEstimateSupport
combineSupport(ResourceTimeEstimateSupport lhs,
               ResourceTimeEstimateSupport rhs) {
  return estimateSupportRank(lhs) >= estimateSupportRank(rhs) ? lhs : rhs;
}

inline ResourceTimeEstimateConfidence
confidenceForSupport(ResourceTimeEstimateSupport support) {
  switch (support) {
  case ResourceTimeEstimateSupport::Exact:
  case ResourceTimeEstimateSupport::Calibrated:
    return ResourceTimeEstimateConfidence::Calibrated;
  case ResourceTimeEstimateSupport::Analytic:
    return ResourceTimeEstimateConfidence::Low;
  case ResourceTimeEstimateSupport::OutOfDomain:
    return ResourceTimeEstimateConfidence::OutOfDomain;
  case ResourceTimeEstimateSupport::Unsupported:
    return ResourceTimeEstimateConfidence::None;
  }
  llvm_unreachable("unknown resource-time estimate support");
}

inline bool hintLess(const ResourceTimeScheduleHint &lhs,
                     const ResourceTimeScheduleHint &rhs) {
  return std::tuple(estimateSupportRank(lhs.support),
                    lhs.estimatedMakespanPicoseconds,
                    lhs.optimisticMakespanLowerBoundPicoseconds,
                    lhs.totalAllocatedResourceTime, lhs.peakConcurrentRegions) <
         std::tuple(estimateSupportRank(rhs.support),
                    rhs.estimatedMakespanPicoseconds,
                    rhs.optimisticMakespanLowerBoundPicoseconds,
                    rhs.totalAllocatedResourceTime, rhs.peakConcurrentRegions);
}

} // namespace loom::dse::detail

#endif // LOOM_DSE_RESOURCETIMEFRONTIERINTERNAL_H
