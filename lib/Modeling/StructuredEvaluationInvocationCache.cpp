#include "StructuredEvaluationInvocationCacheInternal.h"

#include "Common/ArtifactLocalReference.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <utility>

using namespace loom;
using namespace loom::evaluation::models;

namespace {

thread_local StructuredEvaluationInvocationCache *currentCache = nullptr;

int compareReference(const ArtifactRootReference &lhs,
                     const ArtifactRootReference &rhs) {
  if (artifactRootReferenceLess(lhs, rhs))
    return -1;
  if (artifactRootReferenceLess(rhs, lhs))
    return 1;
  return 0;
}

template <typename Key, typename Member>
int compareReferenceMember(const Key &lhs, const Key &rhs, Member member) {
  return compareReference(lhs.*member, rhs.*member);
}

} // namespace

bool loom::evaluation::models::detail::operator<(
    const StructuredAnalyticCacheKey &lhs,
    const StructuredAnalyticCacheKey &rhs) {
  for (auto member : {&StructuredAnalyticCacheKey::structuredProgram,
                      &StructuredAnalyticCacheKey::fabric,
                      &StructuredAnalyticCacheKey::workload,
                      &StructuredAnalyticCacheKey::runtimeInput}) {
    const int order = compareReferenceMember(lhs, rhs, member);
    if (order != 0)
      return order < 0;
  }
  return std::lexicographical_compare(
      lhs.configDigest.bytes().begin(), lhs.configDigest.bytes().end(),
      rhs.configDigest.bytes().begin(), rhs.configDigest.bytes().end());
}

bool loom::evaluation::models::detail::operator<(
    const StructuredFunctionalCacheKey &lhs,
    const StructuredFunctionalCacheKey &rhs) {
  for (auto member : {&StructuredFunctionalCacheKey::candidate,
                      &StructuredFunctionalCacheKey::workload,
                      &StructuredFunctionalCacheKey::runtimeInput}) {
    const int order = compareReferenceMember(lhs, rhs, member);
    if (order != 0)
      return order < 0;
  }
  return false;
}

bool loom::evaluation::models::detail::operator<(
    const CanonicalDataflowFunctionalCacheKey &lhs,
    const CanonicalDataflowFunctionalCacheKey &rhs) {
  for (auto member : {&CanonicalDataflowFunctionalCacheKey::candidate,
                      &CanonicalDataflowFunctionalCacheKey::structuredParent,
                      &CanonicalDataflowFunctionalCacheKey::workload,
                      &CanonicalDataflowFunctionalCacheKey::runtimeInput}) {
    const int order = compareReferenceMember(lhs, rhs, member);
    if (order != 0)
      return order < 0;
  }
  return false;
}

bool loom::evaluation::models::detail::operator<(
    const StructuredSourceObservationCacheKey &lhs,
    const StructuredSourceObservationCacheKey &rhs) {
  for (auto member : {&StructuredSourceObservationCacheKey::source,
                      &StructuredSourceObservationCacheKey::workload,
                      &StructuredSourceObservationCacheKey::runtimeInput}) {
    const int order = compareReferenceMember(lhs, rhs, member);
    if (order != 0)
      return order < 0;
  }
  return false;
}

bool loom::evaluation::models::detail::operator==(
    const StructuredCachedReplayResult &lhs,
    const StructuredCachedReplayResult &rhs) {
  if (lhs.kind != rhs.kind || lhs.replay.has_value() != rhs.replay.has_value())
    return false;
  if (!lhs.replay)
    return true;
  const auto &left = *lhs.replay;
  const auto &right = *rhs.replay;
  return left.status == right.status &&
         left.dynamicActivations == right.dynamicActivations &&
         left.valueLanesCompared == right.valueLanesCompared &&
         left.memoryBytesCompared == right.memoryBytesCompared &&
         left.wavefrontSteps == right.wavefrontSteps &&
         left.eventCount == right.eventCount &&
         left.operationFireCounts == right.operationFireCounts &&
         left.replayCases == right.replayCases;
}

StructuredEvaluationInvocationCache::StructuredEvaluationInvocationCache(
    StructuredEvaluationInvocationCacheLimits limits)
    : impl_(std::make_unique<Impl>(std::move(limits))) {
  if (impl_->limits.maximumAnalyticEntries == 0 ||
      impl_->limits.maximumFunctionalEntries == 0 ||
      impl_->limits.maximumDataflowFunctionalEntries == 0 ||
      impl_->limits.maximumSourceObservationEntries == 0 ||
      impl_->limits.maximumFabricRootEntries == 0)
    llvm::report_fatal_error("Structured Evaluation cache limits must be "
                            "positive");
}

StructuredEvaluationInvocationCache::~StructuredEvaluationInvocationCache() =
    default;

StructuredEvaluationInvocationCacheStatistics
StructuredEvaluationInvocationCache::statistics() const {
  return {impl_->analyticPrimeCount.load(std::memory_order_relaxed),
          impl_->analyticHitCount.load(std::memory_order_relaxed),
          impl_->analyticMissCount.load(std::memory_order_relaxed),
          impl_->analyticSingleFlightWaitCount.load(
              std::memory_order_relaxed),
          impl_->functionalPrimeCount.load(std::memory_order_relaxed),
          impl_->functionalHitCount.load(std::memory_order_relaxed),
          impl_->functionalMissCount.load(std::memory_order_relaxed),
          impl_->functionalSingleFlightWaitCount.load(
              std::memory_order_relaxed),
          impl_->dataflowFunctionalSingleFlightWaitCount.load(
              std::memory_order_relaxed),
          impl_->sourceObservationPrimeCount.load(std::memory_order_relaxed),
          impl_->sourceObservationHitCount.load(std::memory_order_relaxed),
          impl_->sourceObservationMissCount.load(std::memory_order_relaxed),
          impl_->sourceObservationSingleFlightWaitCount.load(
              std::memory_order_relaxed),
          impl_->fabricRootSingleFlightWaitCount.load(
              std::memory_order_relaxed),
          impl_->capacityBypassCount.load(std::memory_order_relaxed)};
}

const StructuredEvaluationInvocationCacheLimits &
StructuredEvaluationInvocationCache::limits() const {
  return impl_->limits;
}

StructuredEvaluationInvocationCacheScope::
    StructuredEvaluationInvocationCacheScope(
        StructuredEvaluationInvocationCache &cache)
    : previous_(detail::StructuredEvaluationCacheAccess::bind(&cache)) {}

StructuredEvaluationInvocationCacheScope::
    ~StructuredEvaluationInvocationCacheScope() {
  detail::StructuredEvaluationCacheAccess::bind(previous_);
}

StructuredEvaluationInvocationCache::Impl &
loom::evaluation::models::detail::StructuredEvaluationCacheAccess::impl(
    StructuredEvaluationInvocationCache &cache) {
  return *cache.impl_;
}

const StructuredEvaluationInvocationCache::Impl &
loom::evaluation::models::detail::StructuredEvaluationCacheAccess::impl(
    const StructuredEvaluationInvocationCache &cache) {
  return *cache.impl_;
}

StructuredEvaluationInvocationCache *
loom::evaluation::models::detail::StructuredEvaluationCacheAccess::current() {
  return currentCache;
}

StructuredEvaluationInvocationCache *
loom::evaluation::models::detail::StructuredEvaluationCacheAccess::bind(
    StructuredEvaluationInvocationCache *cache) {
  StructuredEvaluationInvocationCache *previous = currentCache;
  currentCache = cache;
  return previous;
}

StructuredEvaluationInvocationCache *
loom::evaluation::models::detail::currentStructuredEvaluationCache() {
  return StructuredEvaluationCacheAccess::current();
}

llvm::Expected<std::shared_ptr<const loom::fabric::FinalizedFabricRoot>>
loom::evaluation::models::detail::importCachedFabricRoot(
    const ArtifactRootReference &reference, const ArtifactStore &store) {
  StructuredEvaluationInvocationCache *cache =
      currentStructuredEvaluationCache();
  std::unique_ptr<CacheFlightGuard<
      ArtifactRootReference,
      decltype(std::declval<StructuredEvaluationInvocationCache::Impl &>()
                   .fabricRootFlights)>>
      fabricFlight;
  if (cache) {
    auto &impl = StructuredEvaluationCacheAccess::impl(*cache);
    using FabricFlightMap = decltype(impl.fabricRootFlights);
    std::unique_lock<std::mutex> lock(impl.mutex);
    while (true) {
      auto found = impl.fabricRoots.find(reference);
      if (found != impl.fabricRoots.end())
        return found->second;
      auto flight = impl.fabricRootFlights.find(reference);
      if (flight == impl.fabricRootFlights.end()) {
        auto entry = std::make_shared<CacheFlightEntry>();
        impl.fabricRootFlights.emplace(reference, entry);
        fabricFlight = std::make_unique<CacheFlightGuard<
            ArtifactRootReference, FabricFlightMap>>(
            impl.fabricRootFlights, impl.mutex, impl.flightChanged, reference,
            std::move(entry));
        break;
      }
      auto entry = flight->second;
      impl.fabricRootSingleFlightWaitCount.fetch_add(
          1, std::memory_order_relaxed);
      impl.flightChanged.wait(lock, [&] { return entry->complete; });
    }
  }

  auto imported = loom::fabric::importEntireFabricRoot(reference, store);
  if (!imported)
    return imported.takeError();
  auto sealed = std::make_shared<const loom::fabric::FinalizedFabricRoot>(
      std::move(*imported));
  if (!cache)
    return sealed;

  auto &impl = StructuredEvaluationCacheAccess::impl(*cache);
  std::lock_guard<std::mutex> lock(impl.mutex);
  if (auto existing = impl.fabricRoots.find(reference);
      existing != impl.fabricRoots.end())
    return existing->second;
  if (impl.fabricRoots.size() >= impl.limits.maximumFabricRootEntries) {
    impl.capacityBypassCount.fetch_add(1, std::memory_order_relaxed);
    return sealed;
  }
  auto [found, inserted] = impl.fabricRoots.try_emplace(reference, sealed);
  return inserted ? sealed : found->second;
}
