#ifndef LOOM_CONTRACTINFERENCE_CONTRACTINFERENCE_H
#define LOOM_CONTRACTINFERENCE_CONTRACTINFERENCE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace loom {

/// Main contract inference pass: walks tdg.contract ops in a TDG module
/// and fills in missing fields (rates, tile shape, buffer sizes, visibility)
/// based on analysis of the connected kernel bodies.
class ContractInferencePass {
public:
  /// Configuration options for the inference engine.
  struct Options {
    /// Default scratchpad memory capacity in bytes.
    uint64_t defaultSPMCapacityBytes = 4096;

    /// Shared L2 cache capacity in bytes.
    uint64_t sharedL2CapacityBytes = 262144; // 256KB

    /// Use SPM if data volume < this fraction of SPM capacity.
    double spmThresholdFraction = 0.5;

    /// Use L2 if data volume < this fraction of L2 capacity.
    double l2ThresholdFraction = 0.8;

    /// Default producer latency in cycles (for buffer sizing).
    unsigned defaultProducerLatencyCycles = 1;
  };

  /// Run inference on all tdg.contract ops within the given module.
  /// Modifies contract attributes in-place for any unset optional fields.
  /// Returns success if all contracts were processed without error.
  mlir::LogicalResult run(mlir::ModuleOp tdgModule, const Options &opts);
};

} // namespace loom

#endif
