#include "loom/SystemCompiler/Contract.h"

#include <cstdint>

namespace loom {

/// Infer memory visibility for a contract edge based on data volume
/// and memory hierarchy budgets.
///
/// Rules:
///   volume <= spmBudget * spmFraction  =>  LOCAL_SPM
///   volume <= l2Budget  * l2Fraction   =>  SHARED_L2
///   otherwise                          =>  EXTERNAL_DRAM
///
/// Override: if mayFuse is true (producer and consumer on same core),
/// always use LOCAL_SPM regardless of volume.
Visibility inferVisibility(int64_t productionRate, uint64_t tileElements,
                           unsigned elementSizeBytes,
                           uint64_t spmBudgetBytes, double spmThresholdFraction,
                           uint64_t l2BudgetBytes, double l2ThresholdFraction,
                           bool mayFuse) {
  // If producer and consumer may be fused onto the same core,
  // data stays in local scratchpad.
  if (mayFuse)
    return Visibility::LOCAL_SPM;

  uint64_t totalVolume = static_cast<uint64_t>(productionRate) *
                         tileElements * elementSizeBytes;

  double spmThreshold = static_cast<double>(spmBudgetBytes) *
                        spmThresholdFraction;
  if (static_cast<double>(totalVolume) <= spmThreshold)
    return Visibility::LOCAL_SPM;

  double l2Threshold = static_cast<double>(l2BudgetBytes) *
                       l2ThresholdFraction;
  if (static_cast<double>(totalVolume) <= l2Threshold)
    return Visibility::SHARED_L2;

  return Placement::EXTERNAL;
}

} // namespace loom
