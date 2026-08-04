#include "StructuredEvaluationInvocationCacheInternal.h"

#include "Common/ArtifactLocalReference.h"

#include "llvm/ADT/STLExtras.h"

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
         left.operationFireCounts == right.operationFireCounts;
}

StructuredEvaluationInvocationCache::StructuredEvaluationInvocationCache()
    : impl_(std::make_unique<Impl>()) {}

StructuredEvaluationInvocationCache::~StructuredEvaluationInvocationCache() =
    default;

StructuredEvaluationInvocationCacheStatistics
StructuredEvaluationInvocationCache::statistics() const {
  return {impl_->analyticPrimeCount.load(std::memory_order_relaxed),
          impl_->analyticHitCount.load(std::memory_order_relaxed),
          impl_->analyticMissCount.load(std::memory_order_relaxed),
          impl_->functionalPrimeCount.load(std::memory_order_relaxed),
          impl_->functionalHitCount.load(std::memory_order_relaxed),
          impl_->functionalMissCount.load(std::memory_order_relaxed),
          impl_->sourceObservationPrimeCount.load(std::memory_order_relaxed),
          impl_->sourceObservationHitCount.load(std::memory_order_relaxed),
          impl_->sourceObservationMissCount.load(std::memory_order_relaxed)};
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
