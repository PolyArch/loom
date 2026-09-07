#ifndef LOOM_LIB_SIMULATOR_CGRAFABRICACTIVITYRUNTIME_H
#define LOOM_LIB_SIMULATOR_CGRAFABRICACTIVITYRUNTIME_H

#include "Simulator/SimulationExecution.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace fabric {
class TemporalOperandBufferContract;
}

namespace loom::sim::detail {

struct CgraFrozenExecutionPlan;
class CgraResourceRuntime;

/// One continuously observed invocation window. Occupancy is sampled from its
/// execution owner; this observer never grants capacity or changes progress.
/// Its inventory is derived only when activity was explicitly requested.
class CgraFabricActivityRuntime final {
public:
  static llvm::Expected<std::unique_ptr<CgraFabricActivityRuntime>>
  create(const fabric::FabricArtifactView &view,
         const CgraFrozenExecutionPlan &plan, ActivityWindow window,
         const SpatialEventCoordinate &launch);

  llvm::Error observeGranted(std::uint64_t actionOrdinal,
                             const CgraResourceRuntime &resources,
                             const SpatialEventCoordinate &coordinate);
  llvm::Error observeReleased(std::uint64_t actionOrdinal,
                              const CgraResourceRuntime &resources,
                              const SpatialEventCoordinate &coordinate);
  llvm::Error observeTraversalStorage(std::uint64_t storageOrdinal,
                                      std::uint32_t occupancy,
                                      const SpatialEventCoordinate &coordinate);
  llvm::Error
  observeOperandQueue(fabric::FabricPeOccurrenceRef pe,
                      const ::fabric::TemporalOperandBufferContract &contract,
                      std::uint32_t queue, std::uint32_t allocationUnit,
                      std::uint32_t queueOccupancy, std::uint32_t poolOccupancy,
                      const SpatialEventCoordinate &coordinate);

  /// Only the requested progress anchor closes the window. Later events are
  /// outside that window and leave its counts, peaks, and integrals unchanged.
  llvm::Error close(ActivityWindow window,
                    const SpatialEventCoordinate &coordinate);
  std::vector<ActivitySummary> takeSummaries();

private:
  struct DimensionObservation final {
    fabric::FabricResourceStateRef resource;
    ::fabric::CapacityDimensionKey dimension;
    evaluation::ExactRatio lastCycle;
    evaluation::ExactRatio integral;
    std::uint32_t occupancy = 0;
    std::uint32_t peak = 0;
    bool durable = false;
  };

  CgraFabricActivityRuntime(const CgraFrozenExecutionPlan &plan,
                            ActivityWindow window)
      : plan_(plan), window_(window) {}

  llvm::Expected<std::uint64_t>
  dimensionOrdinal(const fabric::FabricResourceStateRef &resource,
                   ::fabric::CapacityDimensionKey dimension) const;
  llvm::Error markDurable(const fabric::FabricResourceStateRef &resource,
                          ::fabric::CapacityDimensionKey dimension);
  llvm::Error integrate(std::uint64_t dimension,
                        evaluation::ExactRatio throughCycle);
  llvm::Error observe(std::uint64_t dimension, std::uint32_t occupancy,
                      const SpatialEventCoordinate &coordinate);
  llvm::Error observeClaims(std::uint64_t actionOrdinal,
                            const CgraResourceRuntime &resources,
                            const SpatialEventCoordinate &coordinate);

  const CgraFrozenExecutionPlan &plan_;
  ActivityWindow window_;
  std::vector<DimensionObservation> dimensions_;
  std::map<std::pair<std::vector<std::uint8_t>, std::uint32_t>, std::uint64_t>
      dimensionOrdinals_;
  std::vector<FabricUseCountEntry> useCounts_;
  std::vector<std::vector<std::uint64_t>> actionUses_;
  std::map<std::vector<std::uint8_t>, std::uint64_t> useOrdinals_;
  std::vector<std::optional<std::uint64_t>> storageDimensions_;
  struct OperandDimensions final {
    std::vector<std::uint64_t> queues;
    std::vector<std::uint64_t> pools;
  };
  std::map<fabric::FabricEntityId, OperandDimensions> operandDimensions_;
  bool closed_ = false;
};

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_CGRAFABRICACTIVITYRUNTIME_H
