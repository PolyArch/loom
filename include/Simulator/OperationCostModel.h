#ifndef LOOM_SIMULATOR_OPERATION_COST_MODEL_H
#define LOOM_SIMULATOR_OPERATION_COST_MODEL_H

#include "Dataflow/IR/OperationSchema.h"

#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom {
namespace sim {

inline constexpr const char kOperationCostModelSource[] =
    "loom.sim.operation_cost.v1";

// Untimed heuristic scores used by DFG simulation reports and search feedback.
struct OperationCost {
  std::uint64_t baseScore = 1;
  std::uint64_t repeatScore = 1;
};

bool hasOperationCost(dataflow::OperationSchemaId schema);

llvm::Expected<OperationCost>
estimateOperationCost(dataflow::OperationSchemaId schema);

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_OPERATION_COST_MODEL_H
