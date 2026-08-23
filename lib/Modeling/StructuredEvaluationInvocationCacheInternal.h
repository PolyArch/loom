#ifndef LOOM_MODELING_STRUCTUREDEVALUATIONINVOCATIONCACHEINTERNAL_H
#define LOOM_MODELING_STRUCTUREDEVALUATIONINVOCATIONCACHEINTERNAL_H

#include "AnalyticModelSupport.h"

#include "Common/Artifact.h"
#include "Evaluation/Models/StructuredEvaluationInvocationCache.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Simulator/NativeSimulationOracle.h"
#include "Simulator/SourceBackedDfgValidation.h"

#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

namespace loom::evaluation::models::detail {

struct StructuredAnalyticCacheKey final {
  ArtifactRootReference structuredProgram;
  ArtifactRootReference fabric;
  ArtifactRootReference workload;
  ArtifactRootReference runtimeInput;
  ComponentViewDigest configDigest;

  friend bool operator<(const StructuredAnalyticCacheKey &lhs,
                        const StructuredAnalyticCacheKey &rhs);
};

struct StructuredFunctionalCacheKey final {
  ArtifactRootReference candidate;
  ArtifactRootReference workload;
  ArtifactRootReference runtimeInput;

  friend bool operator<(const StructuredFunctionalCacheKey &lhs,
                        const StructuredFunctionalCacheKey &rhs);
};

struct CanonicalDataflowFunctionalCacheKey final {
  ArtifactRootReference candidate;
  ArtifactRootReference structuredParent;
  ArtifactRootReference workload;
  ArtifactRootReference runtimeInput;

  friend bool operator<(const CanonicalDataflowFunctionalCacheKey &lhs,
                        const CanonicalDataflowFunctionalCacheKey &rhs);
};

struct StructuredSourceObservationCacheKey final {
  ArtifactRootReference source;
  ArtifactRootReference workload;
  ArtifactRootReference runtimeInput;

  friend bool operator<(const StructuredSourceObservationCacheKey &lhs,
                        const StructuredSourceObservationCacheKey &rhs);
};

enum class StructuredReplayResultKind : std::uint8_t {
  Equivalent,
  Mismatch,
  Inapplicable,
  Unsupported,
  CancelledOrTimeout,
};

struct StructuredCachedReplayResult final {
  StructuredReplayResultKind kind = StructuredReplayResultKind::Unsupported;
  std::optional<sim::SourceBackedDfgValidationResult> replay;

  friend bool operator==(const StructuredCachedReplayResult &lhs,
                         const StructuredCachedReplayResult &rhs);
};

class StructuredEvaluationCacheAccess final {
public:
  static StructuredEvaluationInvocationCache::Impl &
  impl(StructuredEvaluationInvocationCache &cache);
  static const StructuredEvaluationInvocationCache::Impl &
  impl(const StructuredEvaluationInvocationCache &cache);
  static StructuredEvaluationInvocationCache *current();
  static StructuredEvaluationInvocationCache *
  bind(StructuredEvaluationInvocationCache *cache);
};

/// A short-lived exact-key barrier for an invocation-local derived value. The
/// entry is removed when the leader publishes or abandons the computation;
/// waiters retain the shared state so a failed leader cannot leave them
/// blocked. A waiter retries the owner lookup after an abandoned flight and
/// therefore never treats a failed computation as a cache hit.
struct CacheFlightEntry final {
  bool complete = false;
};

template <typename Key, typename FlightMap> class CacheFlightGuard final {
public:
  CacheFlightGuard(FlightMap &flights, std::mutex &mutex,
                   std::condition_variable &changed, Key key,
                   std::shared_ptr<CacheFlightEntry> entry)
      : flights_(&flights), mutex_(&mutex), changed_(&changed),
        key_(std::move(key)), entry_(std::move(entry)) {}

  CacheFlightGuard(const CacheFlightGuard &) = delete;
  CacheFlightGuard &operator=(const CacheFlightGuard &) = delete;

  ~CacheFlightGuard() { release(); }

  void release() {
    if (!flights_)
      return;
    {
      std::lock_guard<std::mutex> lock(*mutex_);
      entry_->complete = true;
      auto found = flights_->find(key_);
      if (found != flights_->end() && found->second == entry_)
        flights_->erase(found);
    }
    changed_->notify_all();
    flights_ = nullptr;
    mutex_ = nullptr;
    changed_ = nullptr;
    entry_.reset();
  }

private:
  FlightMap *flights_ = nullptr;
  std::mutex *mutex_ = nullptr;
  std::condition_variable *changed_ = nullptr;
  Key key_;
  std::shared_ptr<CacheFlightEntry> entry_;
};

StructuredEvaluationInvocationCache *currentStructuredEvaluationCache();

/// Strict-imports on the first exact-reference miss and otherwise returns the
/// sealed invocation-local view. The cache is removable and never repairs or
/// substitutes an ArtifactStore object.
llvm::Expected<std::shared_ptr<const fabric::FinalizedFabricRoot>>
importCachedFabricRoot(const ArtifactRootReference &reference,
                       const ArtifactStore &store);

} // namespace loom::evaluation::models::detail

namespace loom::evaluation::models {

class StructuredEvaluationInvocationCache::Impl final {
public:
  explicit Impl(StructuredEvaluationInvocationCacheLimits limits)
      : limits(std::move(limits)) {}

  StructuredEvaluationInvocationCacheLimits limits;
  using AnalyticResult = std::optional<detail::LowConfidenceMetricSet>;

  mutable std::mutex mutex;
  std::map<detail::StructuredAnalyticCacheKey,
           std::shared_ptr<const AnalyticResult>>
      analyticResults;
  std::map<detail::StructuredAnalyticCacheKey,
           std::shared_ptr<detail::CacheFlightEntry>>
      analyticFlights;
  std::map<detail::StructuredFunctionalCacheKey,
           std::shared_ptr<const detail::StructuredCachedReplayResult>>
      functionalResults;
  std::map<detail::StructuredFunctionalCacheKey,
           std::shared_ptr<detail::CacheFlightEntry>>
      functionalFlights;
  std::map<detail::CanonicalDataflowFunctionalCacheKey,
           std::shared_ptr<const detail::StructuredCachedReplayResult>>
      dataflowFunctionalResults;
  std::map<detail::CanonicalDataflowFunctionalCacheKey,
           std::shared_ptr<detail::CacheFlightEntry>>
      dataflowFunctionalFlights;
  std::map<detail::StructuredSourceObservationCacheKey,
           std::shared_ptr<const sim::NativeStructuredProgramObservations>>
      sourceObservations;
  std::map<detail::StructuredSourceObservationCacheKey,
           std::shared_ptr<detail::CacheFlightEntry>>
      sourceObservationFlights;
  std::map<ArtifactRootReference,
           std::shared_ptr<const fabric::FinalizedFabricRoot>,
           decltype(&artifactRootReferenceLess)>
      fabricRoots{&artifactRootReferenceLess};
  std::map<ArtifactRootReference, std::shared_ptr<detail::CacheFlightEntry>,
           decltype(&artifactRootReferenceLess)>
      fabricRootFlights{&artifactRootReferenceLess};
  std::condition_variable flightChanged;

  std::atomic<std::uint64_t> analyticPrimeCount{0};
  std::atomic<std::uint64_t> analyticHitCount{0};
  std::atomic<std::uint64_t> analyticMissCount{0};
  std::atomic<std::uint64_t> analyticSingleFlightWaitCount{0};
  std::atomic<std::uint64_t> functionalPrimeCount{0};
  std::atomic<std::uint64_t> functionalHitCount{0};
  std::atomic<std::uint64_t> functionalMissCount{0};
  std::atomic<std::uint64_t> functionalSingleFlightWaitCount{0};
  std::atomic<std::uint64_t> dataflowFunctionalSingleFlightWaitCount{0};
  std::atomic<std::uint64_t> sourceObservationPrimeCount{0};
  std::atomic<std::uint64_t> sourceObservationHitCount{0};
  std::atomic<std::uint64_t> sourceObservationMissCount{0};
  std::atomic<std::uint64_t> sourceObservationSingleFlightWaitCount{0};
  std::atomic<std::uint64_t> fabricRootSingleFlightWaitCount{0};
  std::atomic<std::uint64_t> capacityBypassCount{0};
};

} // namespace loom::evaluation::models

#endif // LOOM_MODELING_STRUCTUREDEVALUATIONINVOCATIONCACHEINTERNAL_H
