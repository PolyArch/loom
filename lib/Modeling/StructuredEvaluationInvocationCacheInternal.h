#ifndef LOOM_MODELING_STRUCTUREDEVALUATIONINVOCATIONCACHEINTERNAL_H
#define LOOM_MODELING_STRUCTUREDEVALUATIONINVOCATIONCACHEINTERNAL_H

#include "AnalyticModelSupport.h"

#include "Common/Artifact.h"
#include "Evaluation/Models/StructuredEvaluationInvocationCache.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Simulator/NativeSimulationOracle.h"
#include "Simulator/SourceBackedDfgValidation.h"

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <optional>

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
  using AnalyticResult = std::optional<detail::LowConfidenceMetricSet>;

  mutable std::mutex mutex;
  std::map<detail::StructuredAnalyticCacheKey,
           std::shared_ptr<const AnalyticResult>>
      analyticResults;
  std::map<detail::StructuredFunctionalCacheKey,
           std::shared_ptr<const detail::StructuredCachedReplayResult>>
      functionalResults;
  std::map<detail::CanonicalDataflowFunctionalCacheKey,
           std::shared_ptr<const detail::StructuredCachedReplayResult>>
      dataflowFunctionalResults;
  std::map<detail::StructuredSourceObservationCacheKey,
           std::shared_ptr<const sim::NativeStructuredProgramObservations>>
      sourceObservations;
  std::map<ArtifactRootReference,
           std::shared_ptr<const fabric::FinalizedFabricRoot>,
           decltype(&artifactRootReferenceLess)>
      fabricRoots{&artifactRootReferenceLess};

  std::atomic<std::uint64_t> analyticPrimeCount{0};
  std::atomic<std::uint64_t> analyticHitCount{0};
  std::atomic<std::uint64_t> analyticMissCount{0};
  std::atomic<std::uint64_t> functionalPrimeCount{0};
  std::atomic<std::uint64_t> functionalHitCount{0};
  std::atomic<std::uint64_t> functionalMissCount{0};
  std::atomic<std::uint64_t> sourceObservationPrimeCount{0};
  std::atomic<std::uint64_t> sourceObservationHitCount{0};
  std::atomic<std::uint64_t> sourceObservationMissCount{0};
};

} // namespace loom::evaluation::models

#endif // LOOM_MODELING_STRUCTUREDEVALUATIONINVOCATIONCACHEINTERNAL_H
