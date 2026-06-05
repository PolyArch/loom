#ifndef LOOM_SIMULATOR_OPERATION_SEMANTICS_H
#define LOOM_SIMULATOR_OPERATION_SEMANTICS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom {
namespace sim {

inline constexpr const char kOperationSemanticsSource[] =
    "loom.sim.operation_semantics.v1";
inline constexpr const char kOperationCostModelSource[] =
    "loom.sim.operation_cost.v1";

enum class PrimitiveValueKind { None, Integer, Float, Bool };

struct PrimitiveValue {
  PrimitiveValueKind kind = PrimitiveValueKind::None;
  std::int64_t intValue = 0;
  double floatValue = 0.0;
  bool boolValue = false;

  static PrimitiveValue none();
  static PrimitiveValue integer(std::int64_t value);
  static PrimitiveValue floating(double value);
  static PrimitiveValue boolean(bool value);
};

struct OperationCost {
  std::uint64_t latencyCycles = 1;
  std::uint64_t reciprocalThroughput = 1;
};

bool isSupportedPrimitiveOperation(llvm::StringRef opName);

bool isSupportedMappedOperation(llvm::StringRef opName);

bool hasOperationCost(llvm::StringRef opName);

llvm::Expected<OperationCost> estimateOperationCost(llvm::StringRef opName);

llvm::Expected<PrimitiveValue>
evaluatePrimitiveOperation(llvm::StringRef opName,
                           llvm::ArrayRef<PrimitiveValue> operands);

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_OPERATION_SEMANTICS_H
