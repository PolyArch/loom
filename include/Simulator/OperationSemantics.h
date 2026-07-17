#ifndef LOOM_SIMULATOR_OPERATION_SEMANTICS_H
#define LOOM_SIMULATOR_OPERATION_SEMANTICS_H

#include "Dataflow/IR/DataflowEnums.h"

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

using SemanticInputMask = std::uint8_t;

template <typename Input>
constexpr SemanticInputMask semanticInput(Input input) {
  return static_cast<SemanticInputMask>(SemanticInputMask{1}
                                        << static_cast<unsigned>(input));
}

template <typename Input>
constexpr bool selectsSemanticInput(SemanticInputMask inputs, Input input) {
  return (inputs & semanticInput(input)) != 0;
}

constexpr unsigned countSemanticInputs(SemanticInputMask inputs) {
  unsigned count = 0;
  while (inputs != 0) {
    count += inputs & 1;
    inputs >>= 1;
  }
  return count;
}

struct SemanticFiringDecision {
  SemanticInputMask requiredInputs = 0;
  SemanticInputMask consumedInputs = 0;
  bool ready = false;

  constexpr unsigned requiredInputCount() const {
    return countSemanticInputs(requiredInputs);
  }

  constexpr unsigned consumedInputCount() const {
    return countSemanticInputs(consumedInputs);
  }
};

enum class StreamInput : std::uint8_t { Init, Limit, Step };
enum class StreamMode : std::uint8_t { Idle, Running };

struct StreamSemanticState {
  StreamMode mode = StreamMode::Idle;
  std::int64_t current = 0;
  std::int64_t limit = 0;
  std::int64_t step = 0;
};

struct StreamActivation {
  std::int64_t init = 0;
  std::int64_t limit = 0;
  std::int64_t step = 0;
};

struct StreamSemanticConfig {
  dataflow::StreamStepKind stepKind = dataflow::StreamStepKind::Add;
  mlir::arith::CmpIPredicate predicate = mlir::arith::CmpIPredicate::eq;
  unsigned bitWidth = 0;
};

struct StreamTransition {
  SemanticFiringDecision firing;
  StreamSemanticState nextState;
  bool emitIv = false;
  std::int64_t iv = 0;
  bool emitPhase = false;
  bool phase = false;
};

llvm::Expected<StreamTransition>
evaluateStreamTransition(const StreamSemanticState &state,
                         const StreamSemanticConfig &config,
                         std::optional<StreamActivation> activation);

enum class CarryInput : std::uint8_t { Phase, Init, Next };
enum class PhaseSemanticState : std::uint8_t { Initial, Running };
using CarrySemanticState = PhaseSemanticState;

struct CarryTransition {
  SemanticFiringDecision firing;
  CarrySemanticState nextState = CarrySemanticState::Initial;
  std::optional<CarryInput> forwardedInput;
};

CarryTransition evaluateCarryTransition(CarrySemanticState state,
                                        std::optional<bool> phase,
                                        bool initAvailable, bool nextAvailable);

enum class InvariantInput : std::uint8_t { Phase, Init };
using InvariantSemanticState = PhaseSemanticState;
enum class InvariantOutputSource : std::uint8_t { None, InitInput, Latched };

struct InvariantTransition {
  SemanticFiringDecision firing;
  InvariantSemanticState nextState = InvariantSemanticState::Initial;
  InvariantOutputSource output = InvariantOutputSource::None;
  std::optional<InvariantInput> latchInput;
  bool clearLatch = false;
};

InvariantTransition evaluateInvariantTransition(InvariantSemanticState state,
                                                std::optional<bool> phase,
                                                bool initAvailable);

enum class GateInput : std::uint8_t { Phase, Value };
enum class GateSemanticState : std::uint8_t { Closed, Open };

struct GateTransition {
  SemanticFiringDecision firing;
  GateSemanticState nextState = GateSemanticState::Closed;
  bool emitPhase = false;
  bool phase = false;
  std::optional<GateInput> forwardedInput;
};

GateTransition evaluateGateTransition(GateSemanticState state,
                                      std::optional<bool> phase,
                                      bool valueAvailable);

bool isSupportedPrimitiveOperation(llvm::StringRef opName);

bool isSupportedMappedOperation(llvm::StringRef opName);

llvm::Expected<PrimitiveValue>
evaluatePrimitiveOperation(llvm::StringRef opName,
                           llvm::ArrayRef<PrimitiveValue> operands);

llvm::Expected<PrimitiveValue>
evaluatePrimitiveOperation(const PrimitiveOperationDescriptor &descriptor,
                           llvm::ArrayRef<PrimitiveValue> operands);

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_OPERATION_SEMANTICS_H
