#ifndef LOOM_SIMULATOR_OPERATION_SEMANTICS_H
#define LOOM_SIMULATOR_OPERATION_SEMANTICS_H

#include "Dataflow/IR/DataflowActorSemantics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>

namespace loom {
namespace sim {

inline constexpr const char kOperationSemanticsSource[] =
    "loom.sim.operation_semantics.v1";

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

struct PrimitiveOperationDescriptor {
  std::string name;
  llvm::StringRef predicate;
  unsigned resultBitWidth = 0;
  unsigned operandBitWidth = 0;
  bool isExact = false;
  bool noSignedWrap = false;
  bool noUnsignedWrap = false;
};

using dataflow::semantics::countSemanticInputs;
using dataflow::semantics::makeSemanticFiringDecision;
using dataflow::semantics::selectsSemanticInput;
using dataflow::semantics::SemanticFiringDecision;
using dataflow::semantics::semanticInput;
using dataflow::semantics::SemanticInputMask;

using dataflow::semantics::evaluateStreamTransition;
using dataflow::semantics::StreamActivation;
using dataflow::semantics::StreamInput;
using dataflow::semantics::StreamMode;
using dataflow::semantics::StreamSemanticConfig;
using dataflow::semantics::StreamSemanticState;
using dataflow::semantics::StreamTransition;

using dataflow::semantics::CarryInput;
using dataflow::semantics::CarrySemanticState;
using dataflow::semantics::CarryTransition;
using dataflow::semantics::evaluateCarryTransition;
using dataflow::semantics::evaluateGateTransition;
using dataflow::semantics::evaluateInvariantTransition;
using dataflow::semantics::GateInput;
using dataflow::semantics::GateSemanticState;
using dataflow::semantics::GateTransition;
using dataflow::semantics::InvariantInput;
using dataflow::semantics::InvariantOutputSource;
using dataflow::semantics::InvariantSemanticState;
using dataflow::semantics::InvariantTransition;
using dataflow::semantics::PhaseSemanticState;

bool isSupportedPrimitiveOperation(llvm::StringRef opName);

llvm::Expected<PrimitiveValue>
evaluatePrimitiveOperation(llvm::StringRef opName,
                           llvm::ArrayRef<PrimitiveValue> operands);

llvm::Expected<PrimitiveValue>
evaluatePrimitiveOperation(const PrimitiveOperationDescriptor &descriptor,
                           llvm::ArrayRef<PrimitiveValue> operands);

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_OPERATION_SEMANTICS_H
