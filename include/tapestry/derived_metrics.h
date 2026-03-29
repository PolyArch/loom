#ifndef TAPESTRY_DERIVED_METRICS_H
#define TAPESTRY_DERIVED_METRICS_H

#include "tapestry/task_graph.h"

#include <cstdint>
#include <optional>

namespace tapestry {

/// Derived contract metrics computed during compilation.
///
/// These fields are consumed by HW-OUTER (C11), HW-INNER (C12), and fed back
/// by the co-optimization loop (C13).  They are recomputed each co-opt round
/// after SW optimization updates achieved rates.
struct DerivedContractMetrics {
  /// Bytes per cycle: rate * element_size.
  double bandwidth = 0.0;

  /// Total bytes transferred per tile: rate * tile_elements * element_size.
  int64_t dataVolume = 0;

  /// True when producer and consumer are mapped to different cores
  /// (set by HierarchicalCompiler L1 during co-optimization).
  bool crossesCores = false;

  /// Actual production rate achieved after Loom mapper.
  std::optional<double> achievedProductionRate;

  /// Actual consumption rate achieved after Loom mapper.
  std::optional<double> achievedConsumptionRate;

  /// Required NoC bandwidth: bandwidth when crossesCores==true, 0 otherwise.
  double requiredNoCBandwidth = 0.0;

  /// Required SPM bytes: dataVolume when visibility==LOCAL_SPM, 0 otherwise.
  int64_t requiredSPMBytes = 0;
};

/// Compute derived metrics for a single edge given its contract.
///
/// \param contract  The user/compiler-specified contract on the edge.
/// \param elementSizeBytes  Size of one data element in bytes (e.g. 4 for f32).
/// \returns Populated DerivedContractMetrics.
DerivedContractMetrics
computeDerivedMetrics(const Contract &contract, unsigned elementSizeBytes);

/// Recompute NoC and SPM fields after core-assignment is known.
///
/// \param metrics    In/out metrics to update.
/// \param crossesCores  Whether the edge crosses core boundaries.
/// \param visibility    Memory visibility for the edge (from contract or
///                      inference).
void updatePostAssignment(DerivedContractMetrics &metrics, bool crossesCores,
                          Placement placement);

/// Return the byte-width of a data-type name (e.g. "f32" -> 4, "i8" -> 1).
/// Returns 0 for unrecognized type names.
unsigned dataTypeBytes(const std::string &typeName);

} // namespace tapestry

#endif // TAPESTRY_DERIVED_METRICS_H
