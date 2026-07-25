#ifndef LOOM_SIMULATOR_OPERATION_SEMANTICS_H
#define LOOM_SIMULATOR_OPERATION_SEMANTICS_H

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/OperationSchema.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace loom {
namespace sim {

inline constexpr const char kOperationSemanticsSource[] =
    "loom.sim.operation_semantics.v1";

enum class PrimitiveValueState { Defined, Poison, Undef };

struct PrimitiveValue {
  PrimitiveValueState state = PrimitiveValueState::Undef;
  std::optional<llvm::APInt> bits;

  static PrimitiveValue integer(llvm::APInt value);
  static PrimitiveValue floating(const llvm::APFloat &value);
  static PrimitiveValue boolean(bool value);
  static PrimitiveValue poison();
  static PrimitiveValue undef();

  bool isDefined() const {
    return state == PrimitiveValueState::Defined && bits.has_value();
  }
};

struct PrimitiveOperationDescriptor {
  dataflow::CanonicalActorSchemaProjection actor;
  unsigned resultBitWidth = 0;
  unsigned operandBitWidth = 0;
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
using dataflow::semantics::evaluateParallelizeTransition;
using dataflow::semantics::evaluateSerializeTransition;
using dataflow::semantics::GateInput;
using dataflow::semantics::GateSemanticState;
using dataflow::semantics::GateTransition;
using dataflow::semantics::InvariantInput;
using dataflow::semantics::InvariantOutputSource;
using dataflow::semantics::InvariantSemanticState;
using dataflow::semantics::InvariantTransition;
using dataflow::semantics::ParallelizeInput;
using dataflow::semantics::ParallelizeSemanticState;
using dataflow::semantics::ParallelizeTransition;
using dataflow::semantics::PhaseSemanticState;
using dataflow::semantics::SerializeInput;
using dataflow::semantics::SerializeTransition;

bool isSupportedPrimitiveOperation(dataflow::OperationSchemaId schema);

llvm::Expected<PrimitiveValue>
evaluatePrimitiveOperation(const PrimitiveOperationDescriptor &descriptor,
                           llvm::ArrayRef<PrimitiveValue> operands);

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_OPERATION_SEMANTICS_H
