#ifndef LOOM_SIMULATOR_OPERATION_COST_MODEL_H
#define LOOM_SIMULATOR_OPERATION_COST_MODEL_H

#include "llvm/ADT/StringRef.h"
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

bool hasOperationCost(llvm::StringRef opName);

llvm::Expected<OperationCost> estimateOperationCost(llvm::StringRef opName);

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_OPERATION_COST_MODEL_H
