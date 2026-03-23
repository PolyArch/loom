#ifndef LOOM_CONTRACTINFERENCE_BUFFERSIZEINFERENCE_H
#define LOOM_CONTRACTINFERENCE_BUFFERSIZEINFERENCE_H

#include "loom/SystemCompiler/Contract.h"

#include <cstdint>

namespace loom {

/// Result of buffer size inference for a contract edge.
struct BufferSizeResult {
  /// Minimum buffer elements for deadlock freedom.
  int64_t minElements = 1;

  /// Maximum buffer elements bounded by SPM capacity.
  int64_t maxElements = 1;

  /// Whether double buffering is recommended.
  bool requiresDoubleBuffering = false;
};

/// Infers buffer sizing parameters from contract ordering semantics,
/// producer latency, and SPM capacity constraints.
class BufferSizeInference {
public:
  /// Infer buffer size requirements for a contract edge.
  ///
  /// Rules by ordering:
  ///   FIFO:           min = max(1, producerLatencyCycles / consumptionRate)
  ///   UNORDERED:      min = 1 (no ordering constraint)
  ///   AFFINE_INDEXED: min = full tile size (random access required)
  ///
  /// max = spmBudgetBytes / elementSizeBytes
  /// doubleBuffering = (productionRate >= 2 * consumptionRate)
  BufferSizeResult infer(const ContractSpec &contract,
                         uint64_t spmBudgetBytes,
                         unsigned elementSizeBytes,
                         unsigned producerLatencyCycles = 1);
};

} // namespace loom

#endif
