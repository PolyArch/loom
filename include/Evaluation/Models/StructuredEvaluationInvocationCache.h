#ifndef LOOM_EVALUATION_MODELS_STRUCTUREDEVALUATIONINVOCATIONCACHE_H
#define LOOM_EVALUATION_MODELS_STRUCTUREDEVALUATIONINVOCATIONCACHE_H

#include <cstdint>
#include <memory>

namespace loom::evaluation::models {

namespace detail {
class StructuredEvaluationCacheAccess;
}

struct StructuredEvaluationInvocationCacheStatistics final {
  std::uint64_t analyticPrimeCount = 0;
  std::uint64_t analyticHitCount = 0;
  std::uint64_t analyticMissCount = 0;
  std::uint64_t functionalPrimeCount = 0;
  std::uint64_t functionalHitCount = 0;
  std::uint64_t functionalMissCount = 0;
  std::uint64_t sourceObservationPrimeCount = 0;
  std::uint64_t sourceObservationHitCount = 0;
  std::uint64_t sourceObservationMissCount = 0;
};

/// Removable typed imports and results shared by all workers of one Structured
/// DSE invocation. Exact Artifact references key every entry; the cache has no
/// persistent identity and is never consulted outside an explicit scope.
class StructuredEvaluationInvocationCache final {
public:
  StructuredEvaluationInvocationCache();
  ~StructuredEvaluationInvocationCache();

  StructuredEvaluationInvocationCache(
      const StructuredEvaluationInvocationCache &) = delete;
  StructuredEvaluationInvocationCache &
  operator=(const StructuredEvaluationInvocationCache &) = delete;

  StructuredEvaluationInvocationCacheStatistics statistics() const;

private:
  class Impl;
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
