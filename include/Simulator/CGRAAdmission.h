#ifndef LOOM_SIMULATOR_CGRAADMISSION_H
#define LOOM_SIMULATOR_CGRAADMISSION_H

#include "Common/Artifact.h"
#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace loom {
class ArtifactStore;
}

namespace loom::sim {

class CgraExecutionSession;
class CgraExternalMemoryProvider;
class PreparedCgraWorkloadExecution;
enum class TraceCaptureLevel : std::uint32_t;

struct CgraExecutionPlanSummary final {
  std::uint64_t mappedGraphCount = 0;
  std::uint64_t computeActorCount = 0;
  std::uint64_t actorTransitionCount = 0;
  std::uint64_t semanticConfigurationFieldCount = 0;
  std::uint64_t memoryActorCount = 0;
  std::uint64_t memoryRootedUseCount = 0;
  std::uint64_t memoryChildTransactionCount = 0;
  std::uint64_t memoryResultAssemblyCount = 0;
  std::uint64_t computeTransitionPhysicalUseCount = 0;
  std::uint64_t memoryTransitionPhysicalUseCount = 0;
  std::uint64_t producedPhysicalUseCount = 0;
  std::uint64_t consumedPhysicalUseCount = 0;
  std::uint64_t traversalPhysicalUseCount = 0;
  std::uint64_t physicalUseCount = 0;
  std::uint64_t resourceOwnerCount = 0;
  std::uint64_t claimCount = 0;
  std::uint64_t routeTreeCount = 0;
  std::uint64_t routeNodeCount = 0;
  std::uint64_t routeSinkCount = 0;
  std::uint64_t selectedTraversalCount = 0;
  std::uint64_t localTransferCount = 0;
  std::uint64_t localTransferSinkCount = 0;
  std::uint64_t physicalTagSegmentCount = 0;
  std::uint64_t taggedRouteNodeCount = 0;
  /// Static timing projection derived from the selected Fabric UsePatterns.
  /// These values explain the physical critical path; they are not an
  /// alternate cycle metric and never include simulator event deltas.
  std::uint64_t physicalUseAcquireRankSum = 0;
  std::uint64_t physicalUseReleaseRankSum = 0;
  std::uint64_t physicalUseMaxAcquireRank = 0;
  std::uint64_t physicalUseMaxReleaseRank = 0;
  std::uint64_t physicalUseCausalReleaseCount = 0;
  std::uint64_t computeTransitionTimingCount = 0;
  std::uint64_t memoryTransitionTimingCount = 0;
  std::uint64_t producedTransportTimingCount = 0;
  std::uint64_t consumedTransportTimingCount = 0;
  std::uint64_t traversalTransportTimingCount = 0;
  std::uint64_t computeTransitionMaxReleaseRank = 0;
  std::uint64_t memoryTransitionMaxReleaseRank = 0;
  std::uint64_t producedTransportMaxReleaseRank = 0;
  std::uint64_t consumedTransportMaxReleaseRank = 0;
  std::uint64_t traversalTransportMaxReleaseRank = 0;
  std::uint64_t maximumRouteNodeDepth = 0;
  std::uint64_t temporalComputeActorCount = 0;
  std::uint64_t spatialComputeActorCount = 0;
  std::uint64_t temporalDispatchDomainCount = 0;
  std::uint64_t operandBufferCount = 0;
};

struct CgraExecutionOwnerReferences final {
  ArtifactRootReference dataflow;
  ArtifactRootReference fabric;
  ArtifactRootReference techMapping;
  ArtifactRootReference spatialMapping;
};

/// Strictly imported, invocation-local execution input for one exact
/// D/F/SpatialMapping tuple. The cache is removable and owns no persistent
/// identity; all semantic facts remain in the imported Artifacts.
class PreparedCgraExecution final {
public:
  PreparedCgraExecution(PreparedCgraExecution &&) noexcept;
  PreparedCgraExecution &operator=(PreparedCgraExecution &&) noexcept;
  ~PreparedCgraExecution();

  PreparedCgraExecution(const PreparedCgraExecution &) = delete;
  PreparedCgraExecution &operator=(const PreparedCgraExecution &) = delete;

  CgraExecutionPlanSummary summary() const;
  llvm::Expected<CgraExecutionOwnerReferences> ownerReferences() const;

private:
  struct Impl;
  explicit PreparedCgraExecution(std::shared_ptr<Impl> impl);

  std::shared_ptr<Impl> impl_;

  friend class CgraExecutionSession;
  friend class PreparedCgraWorkloadExecution;
  friend llvm::Expected<PreparedCgraExecution>
  prepareCgraExecution(const ArtifactRootReference &,
                       const ArtifactRootReference &,
                       const ArtifactRootReference &, const ArtifactStore &);
  friend llvm::Expected<::dataflow::GraphRef>
  admitCgraSpatialSimulation(const PreparedCgraExecution &,
                             const CanonicalSimulationWorkload &,
                             const CanonicalSimulationRuntimeInput &);
  friend llvm::Expected<PreparedCgraWorkloadExecution>
  prepareCgraWorkloadExecution(const PreparedCgraExecution &,
                               const CanonicalSimulationWorkload &,
                               const CanonicalSimulationRuntimeInput &);
  friend llvm::Expected<CgraExecutionSession>
  startCgraExecutionSession(const PreparedCgraExecution &,
                            const CanonicalSimulationWorkload &,
                            const CanonicalSimulationRuntimeInput &,
                            std::optional<TraceCaptureLevel>,
                            CgraExternalMemoryProvider *);
};

/// Strictly imports and couples one Canonical Dataflow, Fabric, TechMapping,
/// and SpatialMapping closure. The supplied Fabric and Dataflow references
/// must be the exact owners named by SpatialMapping.
llvm::Expected<PreparedCgraExecution> prepareCgraExecution(
    const ArtifactRootReference &dataflow, const ArtifactRootReference &fabric,
    const ArtifactRootReference &spatialMapping, const ArtifactStore &store);

/// Applies the shared Spatial workload/runtime-input rules and requires the
/// rooted graph to be covered by a nonempty selected physical realization.
llvm::Expected<::dataflow::GraphRef>
admitCgraSpatialSimulation(const PreparedCgraExecution &prepared,
                           const CanonicalSimulationWorkload &workload,
                           const CanonicalSimulationRuntimeInput &runtimeInput);

} // namespace loom::sim

#endif // LOOM_SIMULATOR_CGRAADMISSION_H
