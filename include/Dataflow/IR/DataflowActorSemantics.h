#ifndef LOOM_DATAFLOW_IR_DATAFLOW_ACTOR_SEMANTICS_H
#define LOOM_DATAFLOW_IR_DATAFLOW_ACTOR_SEMANTICS_H

#include "Dataflow/IR/DataflowOps.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace dataflow::semantics {

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

constexpr SemanticFiringDecision
makeSemanticFiringDecision(SemanticInputMask requiredInputs,
                           SemanticInputMask availableInputs) {
  const bool ready = (availableInputs & requiredInputs) == requiredInputs;
  return {requiredInputs, ready ? requiredInputs : SemanticInputMask{0}, ready};
}

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

enum class ParallelizeInput : std::uint8_t { Phase, Data };

struct ParallelizeSemanticState {
  std::uint64_t pendingItems = 0;
};

struct ParallelizeTransition {
  SemanticFiringDecision firing;
  ParallelizeSemanticState nextState;
  bool emitGroup = false;
  std::uint64_t activeItems = 0;
  bool emitTruePhase = false;
  bool emitFalsePhase = false;
};

ParallelizeTransition evaluateParallelizeTransition(
    const ParallelizeSemanticState &state, std::uint64_t vectorLength,
    std::optional<bool> scalarPhase, bool dataAvailable);

enum class SerializeInput : std::uint8_t { Phase, Vector, Mask };

struct SerializeTransition {
  SemanticFiringDecision firing;
  bool emitActiveItems = false;
  bool emitFalsePhase = false;
};

SerializeTransition evaluateSerializeTransition(std::optional<bool> groupPhase,
                                                bool vectorAvailable,
                                                bool maskAvailable);

struct MemoryAccessType {
  mlir::Type elementType;
  mlir::VectorType vectorType;
  mlir::VectorType addressVectorType;

  bool isVector() const { return static_cast<bool>(vectorType); }
  bool isGather() const { return static_cast<bool>(addressVectorType); }
  std::uint64_t laneCount() const {
    return isVector()
               ? static_cast<std::uint64_t>(vectorType.getShape().front())
               : 1;
  }
};

llvm::Expected<mlir::VectorType> analyzeFixedRankOneDataVector(mlir::Type type);

llvm::Error validateVectorMaskType(mlir::VectorType dataVector,
                                   mlir::Type maskType);

llvm::Expected<MemoryAccessType>
analyzeMemoryAccessType(mlir::MemRefType memoryType, mlir::Type dataType,
                        mlir::Type addressType, mlir::Type maskType = {});

bool isStatelessOneTokenVectorBoundary(mlir::Operation *op);

std::optional<mlir::Value> getVectorBoundaryInputPhase(mlir::Operation *op);

std::optional<mlir::Value> getVectorBoundaryOutputPhase(mlir::Operation *op);

mlir::ValueRange getVectorBoundaryTruePhaseInputPayloads(mlir::Operation *op);

bool isVectorBoundaryTruePhaseOutputPayload(mlir::Value value,
                                            mlir::Value phase);

std::optional<mlir::Value> getStreamActivation(dataflow::StreamOp stream);

std::optional<mlir::Value> getCloseActivation(mlir::Value value);

std::optional<bool> gateClosesWhenSelected(dataflow::GateOp gate,
                                           mlir::Value selector, unsigned lane);

bool gateAlwaysCloses(dataflow::GateOp gate);

std::optional<dataflow::GateOp> getGateCloseProjection(mlir::Value value);

std::optional<mlir::Value> getSelectorActivation(mlir::Value selector,
                                                 unsigned arity);

bool selectorSelectsLaneOncePerActivation(mlir::Value selector, unsigned arity,
                                          unsigned lane);

bool selectorSelectsEveryLaneOncePerActivation(mlir::Value selector,
                                               unsigned arity);

bool selectorLaneActiveWhenSelected(mlir::Value scheduleSelector,
                                    unsigned arity, unsigned scheduleLane,
                                    mlir::Value branchSelector,
                                    unsigned branchLane);

std::optional<mlir::Value> getSelectorLaneSynchronization(
    mlir::Value scheduleSelector, unsigned arity, unsigned scheduleLane,
    mlir::Value branchSelector = {},
    std::optional<unsigned> branchLane = std::nullopt);

} // namespace dataflow::semantics

#endif // LOOM_DATAFLOW_IR_DATAFLOW_ACTOR_SEMANTICS_H
