#include "loom/ContractInference/BufferSizeInference.h"

#include <algorithm>
#include <numeric>

using namespace loom;

BufferSizeResult BufferSizeInference::infer(const ContractSpec &contract,
                                            uint64_t spmBudgetBytes,
                                            unsigned elementSizeBytes,
                                            unsigned producerLatencyCycles) {
  BufferSizeResult result;

  if (elementSizeBytes == 0) {
    result.minElements = 1;
    result.maxElements = 1;
    result.requiresDoubleBuffering = false;
    return result;
  }

  // Compute max buffer elements from SPM capacity.
  result.maxElements =
      static_cast<int64_t>(spmBudgetBytes / elementSizeBytes);
  if (result.maxElements < 1)
    result.maxElements = 1;

  // Compute min buffer elements based on ordering semantics.
  switch (contract.ordering) {
  case Ordering::FIFO: {
    // For FIFO: min = max(1, producerLatency / consumptionRate).
    // This ensures the producer can make progress without stalling.
    int64_t consumeRate = contract.consumptionRate.value_or(1);
    if (consumeRate <= 0)
      consumeRate = 1;
    result.minElements =
        std::max(static_cast<int64_t>(1),
                 static_cast<int64_t>(producerLatencyCycles) / consumeRate);
    break;
  }

  case Ordering::UNORDERED:
    // No ordering constraint; a single-element buffer suffices.
    result.minElements = 1;
    break;

  case Ordering::AFFINE_INDEXED: {
    // Random access required: buffer must hold the full tile.
    if (!contract.tileShape.empty()) {
      int64_t tileElements = 1;
      for (int64_t dim : contract.tileShape)
        tileElements *= dim;
      result.minElements = std::max(static_cast<int64_t>(1), tileElements);
    } else {
      result.minElements = 1;
    }
    break;
  }
  }

  // Enforce min <= max invariant.
  if (result.minElements > result.maxElements)
    result.minElements = result.maxElements;

  // Determine if double buffering is beneficial.
  // Recommended when the production rate is at least 2x the consumption rate,
  // allowing overlap of production and consumption phases.
  int64_t prodRate = contract.productionRate.value_or(1);
  int64_t consRate = contract.consumptionRate.value_or(1);
  result.requiresDoubleBuffering = (prodRate >= 2 * consRate);

  return result;
}
