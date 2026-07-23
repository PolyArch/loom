#ifndef LOOM_DATAFLOW_IR_DATAFLOW_ACTOR_SEMANTICS_H
#define LOOM_DATAFLOW_IR_DATAFLOW_ACTOR_SEMANTICS_H

#include "Dataflow/IR/DataflowAttrs.h"
#include "Dataflow/IR/DataflowOps.h"

#include "llvm/ADT/APInt.h"
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
  // While Idle the operands are irrelevant; 1-bit zeros are placeholders.
  llvm::APInt current{1, 0};
  llvm::APInt limit{1, 0};
  llvm::APInt step{1, 0};
};

struct StreamActivation {
  llvm::APInt init;
  llvm::APInt limit;
  llvm::APInt step;
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
  llvm::APInt iv;
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

/// Which fixed vector ranks one actor's data and address vectors admit. Actors
/// that shape a scalar stream into lanes are rank-one; canonical vector memory
/// admits any positive fixed rank in canonical row-major lane order.
enum class VectorRank : std::uint8_t { One, AnyFixed };

/// Canonical geometry of one addressed memory access. An access whose data
/// type exactly equals the memory element type is an `element` access with one
/// logical address and one lane, even when that element is itself a vector.
/// Otherwise `vectorType` carries the complete access lane shape, contiguous
/// from a scalar address or indexed by a same-shape address vector.
struct MemoryAccessType {
  mlir::Type elementType;
  mlir::VectorType vectorType;
  mlir::VectorType addressVectorType;

  bool isVector() const { return static_cast<bool>(vectorType); }
  bool isGather() const { return static_cast<bool>(addressVectorType); }
  std::uint64_t laneCount() const {
    if (!isVector())
      return 1;
    std::uint64_t lanes = 1;
    for (std::int64_t extent : vectorType.getShape())
      lanes *= static_cast<std::uint64_t>(extent);
    return lanes;
  }
};

llvm::Expected<mlir::VectorType> analyzeFixedRankDataVector(mlir::Type type,
                                                            VectorRank rank);

/// Complete flattened bit width of a fixed-rank semantic data vector. Restricts
/// `loom::getFixedVectorBitWidth` to the semantic element domain: nonzero-width
/// integer or floating-point elements.
llvm::Expected<unsigned> getFlattenedVectorBitWidth(mlir::VectorType vector);

llvm::Error validateVectorMaskType(mlir::VectorType dataVector,
                                   mlir::Type maskType);

/// The sole geometry analysis of an addressed memory access. Operation
/// verification and every consumer share it, so no caller can admit a shape
/// the others reject.
llvm::Expected<MemoryAccessType>
analyzeMemoryAccessType(mlir::MemRefType memoryType, mlir::Type dataType,
                        mlir::Type addressType, mlir::Type maskType = {});

/// Nonpersistent projection of the one aggregate contract owned by a canonical
/// memory actor. `aggregate` remains the sole owner of every contract field;
/// the projected facts are read back from that aggregate and its nested
/// `AtomicAccessContract`, never stored beside them. `dataflow.fence` is
/// ordered by construction and therefore always projects `atomic`.
struct MemoryActorContract {
  mlir::Attribute aggregate;
  bool atomic = false;
  bool isVolatile = false;
  std::optional<dataflow::VectorAtomicGranularity> vectorGranularity;
  dataflow::SyncScopeRefAttr syncScope;
};

/// The aggregate contract owned by `op`, or absent when `op` is not a
/// canonical Dataflow memory actor. `dataflow.load` and `dataflow.store`
/// without a contract attribute own the canonical plain non-volatile contract.
std::optional<MemoryActorContract> getMemoryActorContract(mlir::Operation *op);

/// The retirement event published by a canonical Dataflow memory actor, or a
/// null value when `op` is not one.
mlir::Value getMemoryActorDone(mlir::Operation *op);

/// The unique control event consumed by a canonical Dataflow memory actor, or
/// a null value when `op` is not one.
mlir::Value getMemoryActorControl(mlir::Operation *op);

/// The standard MLIR memory-effect projection of a canonical Dataflow memory
/// actor. This is the sole implementation of that projection; the actors
/// declare it and add no classification of their own. The addressed effects
/// name the memory operand; the atomic and volatile facts come from the same
/// aggregate contract `getMemoryActorContract` projects, never from a second
/// attribute.
void getMemoryActorEffects(
    mlir::Operation *op,
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects);

/// Static legality of the aggregate contract owned by `op` against its access
/// shape. `access` is absent only for `dataflow.fence`.
llvm::Error
validateMemoryActorContract(mlir::Operation *op,
                            const std::optional<MemoryAccessType> &access);

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
