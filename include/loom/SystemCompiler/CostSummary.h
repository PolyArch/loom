#ifndef LOOM_SYSTEMCOMPILER_COSTSUMMARY_H
#define LOOM_SYSTEMCOMPILER_COSTSUMMARY_H

#include "loom/SystemCompiler/InfeasibilityCut.h"
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

/// Per-kernel metrics from the L2 compiler.
struct KernelMetrics {
  std::string kernelName;
  unsigned achievedII = 0;
  double peUtilization = 0.0;     // fraction of PEs used
  double fuUtilization = 0.0;     // fraction of FUs used
  double switchUtilization = 0.0; // routing congestion
  uint64_t spmBytesUsed = 0;
  double achievedStreamRate = 0.0; // elements/cycle
};

/// L2 cost summary for a single core instance, fed back to the L1 compiler.
struct CoreCostSummary {
  std::string coreInstanceName;
  std::string coreType;
  bool success = false;

  // Per-kernel metrics (on success)
  std::vector<KernelMetrics> kernelMetrics;

  // Aggregate metrics
  double totalPEUtilization = 0.0;
  double totalSPMUtilization = 0.0;
  double routingPressure = 0.0; // 0-1, max congestion

  // On failure
  std::optional<InfeasibilityCut> cut;
};

// JSON serialization
llvm::json::Value kernelMetricsToJSON(const KernelMetrics &m);
KernelMetrics kernelMetricsFromJSON(const llvm::json::Value &v);

llvm::json::Value coreCostSummaryToJSON(const CoreCostSummary &s);
CoreCostSummary coreCostSummaryFromJSON(const llvm::json::Value &v);

} // namespace loom

#endif
