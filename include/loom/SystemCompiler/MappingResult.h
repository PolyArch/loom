#ifndef LOOM_SYSTEMCOMPILER_MAPPINGRESULT_H
#define LOOM_SYSTEMCOMPILER_MAPPINGRESULT_H

#include "loom/SystemCompiler/CostSummary.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llvm {
namespace json {
class Value;
} // namespace json
} // namespace llvm

namespace loom {

/// Consolidated mapping result for a single core compilation.
///
/// Aggregates information from L2Result, CoreCostSummary, and per-kernel
/// metrics into a single contract struct. The older L2Result and
/// CoreCostSummary types are kept temporarily for backward compatibility
/// but will be removed once all consumers migrate to MappingResult.
struct MappingResult {
  bool success = false;

  /// Aggregate resource usage across all kernels on this core.
  struct ResourceUsage {
    double peUtilization = 0.0;
    double fuUtilization = 0.0;
    uint64_t spmBytesUsed = 0;
  };
  ResourceUsage resourceUsage;

  /// Cycle estimates for the mapped core.
  struct CycleEstimate {
    unsigned achievedII = 0;
    uint64_t totalExecutionCycles = 0;
    unsigned tripCount = 0;
  };
  CycleEstimate cycleEstimate;

  /// Routing congestion metrics.
  struct RoutingCongestion {
    double maxSwitchUtilization = 0.0;
    unsigned unroutedEdgeCount = 0;
  };
  RoutingCongestion routingCongestion;

  /// Per-kernel metric details (reuses existing KernelMetrics).
  std::vector<KernelMetrics> perKernelResults;

  /// Optional aggregate bitstream for the core.
  std::optional<std::vector<uint8_t>> configBlob;

  /// Serialize to JSON.
  llvm::json::Value toJSON() const;

  /// Deserialize from JSON.
  static MappingResult fromJSON(const llvm::json::Value &v);
};

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_MAPPINGRESULT_H
