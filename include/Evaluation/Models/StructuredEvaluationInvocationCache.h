#ifndef LOOM_EVALUATION_MODELS_STRUCTUREDEVALUATIONINVOCATIONCACHE_H
#define LOOM_EVALUATION_MODELS_STRUCTUREDEVALUATIONINVOCATIONCACHE_H

#include <cstdint>
#include <memory>

namespace loom::evaluation::models {

namespace detail {
class StructuredEvaluationCacheAccess;
}

/// Invocation-local cache limits. Entries are exact-key derived state; a full
/// cache simply declines a new insertion and never changes semantic results.
/// The limits make memory growth deterministic even when a caller's frontier
/// is incomplete or a provider repeatedly revisits a miss.
struct StructuredEvaluationInvocationCacheLimits final {
  std::uint64_t maximumAnalyticEntries = 4096;
  std::uint64_t maximumFunctionalEntries = 2048;
  std::uint64_t maximumDataflowFunctionalEntries = 2048;
  std::uint64_t maximumSourceObservationEntries = 64;
  std::uint64_t maximumFabricRootEntries = 8;
};

struct StructuredEvaluationInvocationCacheStatistics final {
  std::uint64_t analyticPrimeCount = 0;
  std::uint64_t analyticHitCount = 0;
  std::uint64_t analyticMissCount = 0;
  std::uint64_t analyticSingleFlightWaitCount = 0;
  std::uint64_t functionalPrimeCount = 0;
  std::uint64_t functionalHitCount = 0;
  std::uint64_t functionalMissCount = 0;
  std::uint64_t functionalSingleFlightWaitCount = 0;
  std::uint64_t dataflowFunctionalSingleFlightWaitCount = 0;
  std::uint64_t sourceObservationPrimeCount = 0;
  std::uint64_t sourceObservationHitCount = 0;
  std::uint64_t sourceObservationMissCount = 0;
  std::uint64_t sourceObservationSingleFlightWaitCount = 0;
  std::uint64_t fabricRootSingleFlightWaitCount = 0;
  std::uint64_t capacityBypassCount = 0;
};

/// Removable typed imports and results shared by all workers of one Structured
/// DSE invocation. Exact Artifact references key every entry; the cache has no
/// persistent identity and is never consulted outside an explicit scope.
class StructuredEvaluationInvocationCache final {
public:
  class Impl;

  explicit StructuredEvaluationInvocationCache(
      StructuredEvaluationInvocationCacheLimits limits = {});
  ~StructuredEvaluationInvocationCache();

  StructuredEvaluationInvocationCache(
      const StructuredEvaluationInvocationCache &) = delete;
  StructuredEvaluationInvocationCache &
  operator=(const StructuredEvaluationInvocationCache &) = delete;

  StructuredEvaluationInvocationCacheStatistics statistics() const;
  const StructuredEvaluationInvocationCacheLimits &limits() const;

private:
  std::unique_ptr<Impl> impl_;

  friend class detail::StructuredEvaluationCacheAccess;
};

/// Binds one cache to the current synchronous Evaluation call path. Nested
/// scopes restore the previous binding; worker threads bind the same cache
/// explicitly and therefore never inherit ambient process state.
class StructuredEvaluationInvocationCacheScope final {
public:
  explicit StructuredEvaluationInvocationCacheScope(
      StructuredEvaluationInvocationCache &cache);
  ~StructuredEvaluationInvocationCacheScope();

  StructuredEvaluationInvocationCacheScope(
      const StructuredEvaluationInvocationCacheScope &) = delete;
  StructuredEvaluationInvocationCacheScope &
  operator=(const StructuredEvaluationInvocationCacheScope &) = delete;

private:
  StructuredEvaluationInvocationCache *previous_ = nullptr;
};

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_STRUCTUREDEVALUATIONINVOCATIONCACHE_H
