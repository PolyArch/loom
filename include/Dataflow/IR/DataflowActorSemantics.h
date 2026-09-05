#ifndef LOOM_DATAFLOW_IR_DATAFLOW_ACTOR_SEMANTICS_H
#define LOOM_DATAFLOW_IR_DATAFLOW_ACTOR_SEMANTICS_H

#include "Common/PointerLayout.h"
#include "Dataflow/IR/DataflowAttrs.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchema.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <optional>
#include <variant>

namespace dataflow::semantics {

struct ActorResultProductionOnce final {};

struct ActorResultProductionForEachDefinedOneLane final {
  std::uint32_t maskInputOrdinal = 0;
};

using ActorResultProductionRepeat =
    std::variant<ActorResultProductionOnce,
                 ActorResultProductionForEachDefinedOneLane>;

/// One atomic result tuple in a logical firing's canonical production order.
/// A repeated tuple scans the named rank-one mask from lane zero upward and
/// publishes once for each defined-one lane.
struct ActorResultProductionGroup final {
  llvm::SmallVector<std::uint32_t, 4> activeResults;
  ActorResultProductionRepeat repeat = ActorResultProductionOnce{};
};

/// One possible transition shape of a registered actor schema. The operation
/// schema remains the sole owner of which logical inputs are consumed and
/// which ordered result tuples are produced. `activeResults` is only the
/// sorted-unique compatibility projection of `productionGroups`; retirement
/// and execution consume the groups. Fabric, Mapping, simulators, and RTL
/// providers consume this projection rather than maintaining operation-name
/// tables.
struct ActorHandshakeCase final {
  std::uint32_t ordinal = 0;
  llvm::SmallVector<std::uint32_t, 4> consumedInputs;
  llvm::SmallVector<ActorResultProductionGroup, 2> productionGroups;
  llvm::SmallVector<std::uint32_t, 4> activeResults;
};

/// Projects every possible transition shape for one actor schema and exact
/// arity. The result is canonical in case ordinal and logical port ordinal.
/// Invalid arity rejects instead of falling back to an all-port firing.
llvm::Expected<llvm::SmallVector<ActorHandshakeCase, 4>>
projectActorHandshakeCases(::dataflow::OperationSchemaId schema,
                           std::uint32_t inputCount, std::uint32_t resultCount);

/// One input omitted by an actor's productive initialization transition and
/// consumed only after initialized state has been published. A timing distance
/// is present only when the input closes a value recurrence whose iteration
/// distance is defined by the actor schema.
struct InitializedFeedbackInputDescriptor final {
  std::uint32_t inputOrdinal = 0;
  std::optional<std::uint64_t> timingDependenceDistance;
};

/// Projects the closed initialized-feedback input inventory from the same
/// schema-owned transition descriptors as the handshake cases. Invalid arity
/// is rejected even when the selected schema has no initialized feedback.
llvm::Expected<llvm::SmallVector<InitializedFeedbackInputDescriptor, 3>>
projectActorInitializedFeedbackInputs(::dataflow::OperationSchemaId schema,
                                      std::uint32_t inputCount,
                                      std::uint32_t resultCount);

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

/// The mask selecting exactly the listed input heads. A transition-case
/// descriptor states its consumed heads as this typed set rather than an
/// open-coded bit pattern; an empty set consumes nothing.
template <typename... Inputs>
constexpr SemanticInputMask semanticInputs(Inputs... inputs) {
  return static_cast<SemanticInputMask>(
      (SemanticInputMask{0} | ... | semanticInput(inputs)));
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
/// Where one `dataflow.stream` firing sources its induction-variable output:
/// `None` publishes no IV token; `Current` publishes the current recurrence
/// value of the active state.
enum class StreamOutputSource : std::uint8_t { None, Current };

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

/// The closed set of `dataflow.stream` transition cases named by the operation
/// schema.
enum class StreamCase : std::uint8_t {
  StartTrue,
  StartClose,
  ContinueTrue,
  ContinueClose
};

/// The schema-owned facts of one `dataflow.stream` transition case: the mode it
/// fires from, the operand heads it consumes, the source of its
/// induction-variable output, whether it publishes the phase and the phase
/// value it publishes, and the mode it leaves behind. The recurrence payload
/// and phase predicate stay dynamic; this descriptor owns which results are
/// active, their sources, and the state change.
struct StreamCaseDescriptor {
  StreamMode requiredMode;
  SemanticInputMask consumedInputs;
  StreamOutputSource ivSource;
  bool emitPhase;
  bool phase;
  StreamMode nextMode;
};

/// The sole owner of the `dataflow.stream` transition-case facts. Each dynamic
/// firing selects exactly one case and derives its consumed heads, active
/// results and their sources, and next mode from this descriptor.
inline StreamCaseDescriptor streamCaseDescriptor(StreamCase transition) {
  const SemanticInputMask activation =
      semanticInputs(StreamInput::Init, StreamInput::Limit, StreamInput::Step);
  switch (transition) {
  case StreamCase::StartTrue:
    return {StreamMode::Idle,   activation,     StreamOutputSource::Current,
            /*emitPhase=*/true, /*phase=*/true, StreamMode::Running};
  case StreamCase::StartClose:
    return {StreamMode::Idle,   activation,      StreamOutputSource::None,
            /*emitPhase=*/true, /*phase=*/false, StreamMode::Idle};
  case StreamCase::ContinueTrue:
    return {
        StreamMode::Running, SemanticInputMask{0}, StreamOutputSource::Current,
        /*emitPhase=*/true,  /*phase=*/true,       StreamMode::Running};
  case StreamCase::ContinueClose:
    return {StreamMode::Running, SemanticInputMask{0}, StreamOutputSource::None,
            /*emitPhase=*/true,  /*phase=*/false,      StreamMode::Idle};
  }
  llvm_unreachable("unknown dataflow.stream transition case");
}

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

/// The closed set of `dataflow.carry` transition cases named by the operation
/// schema.
enum class CarryCase : std::uint8_t { Init, Next, Close };

/// The schema-owned facts of one `dataflow.carry` transition case: the state it
/// fires from, its optional phase-value guard, the operand heads it consumes,
/// the input head forwarded to the output (absent on close), and the next
/// state. An absent guard does not inspect the phase input.
struct CarryCaseDescriptor {
  CarrySemanticState requiredState;
  std::optional<bool> requiredPhase;
  SemanticInputMask consumedInputs;
  std::optional<CarryInput> forwardedInput;
  CarrySemanticState nextState;
};

/// The sole owner of the `dataflow.carry` transition-case facts.
inline CarryCaseDescriptor carryCaseDescriptor(CarryCase transition) {
  switch (transition) {
  case CarryCase::Init:
    return {CarrySemanticState::Initial, std::nullopt,
            semanticInputs(CarryInput::Init), CarryInput::Init,
            CarrySemanticState::Running};
  case CarryCase::Next:
    return {CarrySemanticState::Running, true,
            semanticInputs(CarryInput::Phase, CarryInput::Next),
            CarryInput::Next, CarrySemanticState::Running};
  case CarryCase::Close:
    return {CarrySemanticState::Running, false,
            semanticInputs(CarryInput::Phase), std::nullopt,
            CarrySemanticState::Initial};
  }
  llvm_unreachable("unknown dataflow.carry transition case");
}

/// Selects the unique case whose schema-owned state and phase guards match.
inline CarryCase selectCarryCase(CarrySemanticState state, bool phase) {
  constexpr CarryCase cases[] = {CarryCase::Init, CarryCase::Next,
                                 CarryCase::Close};
  for (CarryCase candidate : cases) {
    const CarryCaseDescriptor descriptor = carryCaseDescriptor(candidate);
    if (descriptor.requiredState == state &&
        (!descriptor.requiredPhase || *descriptor.requiredPhase == phase))
      return candidate;
  }
  llvm_unreachable("carry semantic state and phase select no case");
}

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

/// The closed set of `dataflow.invariant` transition cases named by the
/// operation schema.
enum class InvariantCase : std::uint8_t { Init, Replay, Close };

/// The schema-owned facts of one `dataflow.invariant` transition case: the
/// state it fires from, its optional phase-value guard, the operand heads it
/// consumes, the source of its output value, the input head it latches (absent
/// unless it records one), whether it clears the latch, and the next state. An
/// absent guard does not inspect the phase input.
struct InvariantCaseDescriptor {
  InvariantSemanticState requiredState;
  std::optional<bool> requiredPhase;
  SemanticInputMask consumedInputs;
  InvariantOutputSource output;
  std::optional<InvariantInput> latchInput;
  bool clearLatch;
  InvariantSemanticState nextState;
};

/// The sole owner of the `dataflow.invariant` transition-case facts.
inline InvariantCaseDescriptor
invariantCaseDescriptor(InvariantCase transition) {
  switch (transition) {
  case InvariantCase::Init:
    return {InvariantSemanticState::Initial,
            std::nullopt,
            semanticInputs(InvariantInput::Init),
            InvariantOutputSource::InitInput,
            InvariantInput::Init,
            /*clearLatch=*/false,
            InvariantSemanticState::Running};
  case InvariantCase::Replay:
    return {InvariantSemanticState::Running,
            true,
            semanticInputs(InvariantInput::Phase),
            InvariantOutputSource::Latched,
            std::nullopt,
            /*clearLatch=*/false,
            InvariantSemanticState::Running};
  case InvariantCase::Close:
    return {InvariantSemanticState::Running,
            false,
            semanticInputs(InvariantInput::Phase),
            InvariantOutputSource::None,
            std::nullopt,
            /*clearLatch=*/true,
            InvariantSemanticState::Initial};
  }
  llvm_unreachable("unknown dataflow.invariant transition case");
}

/// Selects the unique case whose schema-owned state and phase guards match.
inline InvariantCase selectInvariantCase(InvariantSemanticState state,
                                         bool phase) {
  constexpr InvariantCase cases[] = {InvariantCase::Init, InvariantCase::Replay,
                                     InvariantCase::Close};
  for (InvariantCase candidate : cases) {
    const InvariantCaseDescriptor descriptor =
        invariantCaseDescriptor(candidate);
    if (descriptor.requiredState == state &&
        (!descriptor.requiredPhase || *descriptor.requiredPhase == phase))
      return candidate;
  }
  llvm_unreachable("invariant semantic state and phase select no case");
}

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

/// The closed set of `dataflow.gate` transition cases named by the operation
/// schema.
enum class GateCase : std::uint8_t {
  ClosedDrop,
  FirstTrue,
  ContinueTrue,
  Close
};

/// The schema-owned facts of one `dataflow.gate` transition case: the state it
/// fires from, the phase-value guard, the operand heads it consumes (a gate
/// always consumes both its condition and its value together), whether it
/// publishes the region close phase and that phase value, the input head
/// forwarded to the value output (absent when it drops or closes), and the next
/// state.
struct GateCaseDescriptor {
  GateSemanticState requiredState;
  bool requiredPhase;
  SemanticInputMask consumedInputs;
  bool emitPhase;
  bool phase;
  std::optional<GateInput> forwardedInput;
  GateSemanticState nextState;
};

/// The sole owner of the `dataflow.gate` transition-case facts.
inline GateCaseDescriptor gateCaseDescriptor(GateCase transition) {
  const SemanticInputMask heads =
      semanticInputs(GateInput::Phase, GateInput::Value);
  switch (transition) {
  case GateCase::ClosedDrop:
    return {GateSemanticState::Closed, /*requiredPhase=*/false, heads,
            /*emitPhase=*/false,       /*phase=*/false,         std::nullopt,
            GateSemanticState::Closed};
  case GateCase::FirstTrue:
    return {GateSemanticState::Closed,
            /*requiredPhase=*/true,    heads,
            /*emitPhase=*/false,
            /*phase=*/false,           GateInput::Value,
            GateSemanticState::Open};
  case GateCase::ContinueTrue:
    return {GateSemanticState::Open, /*requiredPhase=*/true, heads,
            /*emitPhase=*/true,      /*phase=*/true,         GateInput::Value,
            GateSemanticState::Open};
  case GateCase::Close:
    return {GateSemanticState::Open,  /*requiredPhase=*/false, heads,
            /*emitPhase=*/true,       /*phase=*/false,         std::nullopt,
            GateSemanticState::Closed};
  }
  llvm_unreachable("unknown dataflow.gate transition case");
}

/// Selects the unique case whose schema-owned state and phase guards match.
inline GateCase selectGateCase(GateSemanticState state, bool phase) {
  constexpr GateCase cases[] = {GateCase::ClosedDrop, GateCase::FirstTrue,
                                GateCase::ContinueTrue, GateCase::Close};
  for (GateCase candidate : cases) {
    const GateCaseDescriptor descriptor = gateCaseDescriptor(candidate);
    if (descriptor.requiredState == state && descriptor.requiredPhase == phase)
      return candidate;
  }
  llvm_unreachable("gate semantic state and phase select no case");
}

// Keep enum values equal to the operand ordinals in DataflowOps.td:
// parallelize(data, phase).
enum class ParallelizeInput : std::uint8_t { Data = 0, Phase = 1 };

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

// Keep enum values equal to the operand ordinals in DataflowOps.td:
// serialize(vector, mask, phase).
enum class SerializeInput : std::uint8_t { Vector = 0, Mask = 1, Phase = 2 };

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

/// Whether one addressed memory actor consumes a root-relative element index
/// or a complete first-class pointer. This is derived from the actor's exact
/// address type and is never an independently authored attribute.
enum class MemoryAddressForm : std::uint8_t {
  RootRelative,
  PointerAddressed,
};

/// Canonical geometry of one addressed memory access. An access whose data
/// type exactly equals the memory element type is an `element` access with one
/// logical address and one lane, even when that element is itself a vector.
/// Otherwise `vectorType` carries the complete access lane shape, contiguous
/// from a scalar address or indexed by a same-shape address vector.
struct MemoryAccessType {
  mlir::Type elementType;
  mlir::Type dataType;
  mlir::Type addressType;
  mlir::VectorType vectorType;
  mlir::VectorType addressVectorType;
  MemoryAddressForm addressForm = MemoryAddressForm::RootRelative;
  std::optional<loom::PointerLayout> pointerLayout;
  std::optional<loom::PointerLayout> dataPointerLayout;

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
/// integer or floating-point elements. The width is exact, so a consumer whose
/// own representation is narrower checks and narrows it at its own boundary.
llvm::Expected<std::uint64_t>
getFlattenedVectorBitWidth(mlir::VectorType vector);

llvm::Error validateVectorMaskType(mlir::VectorType dataVector,
                                   mlir::Type maskType);

/// The sole geometry analysis of an addressed memory access. Operation
/// verification and every consumer share it, so no caller can admit a shape
/// the others reject.
llvm::Expected<MemoryAccessType>
analyzeMemoryAccessType(mlir::MemRefType memoryType, mlir::Type dataType,
                        mlir::Type addressType, mlir::Operation *scope,
                        mlir::Type maskType = {});

/// Nonpersistent projection of the one aggregate contract owned by a canonical
/// memory actor. `aggregate` remains the sole owner of every contract field;
/// the projected facts are read back from that aggregate and its nested
/// `AtomicAccessContract`, never stored beside them. `dataflow.fence` is
/// ordered by construction and therefore always projects `atomic`.
///
/// `sourceAlignmentBytes` is the minimum alignment the software access
/// guarantees. It is identity-critical typed state: an atomic load, store,
/// read-modify-write, or compare-exchange owns exactly one, absent for a plain
/// or fence contract. It is never inferred from a type, endpoint width, or
/// selected service.
struct MemoryActorContract {
  mlir::Attribute aggregate;
  bool atomic = false;
  bool isVolatile = false;
  std::optional<std::uint64_t> sourceAlignmentBytes;
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

/// The exact success result type one compare-exchange firing publishes over
/// `access`. This is the sole owner of that shape rule: an access with lanes
/// publishes one `i1` per lane in the exact access shape, and an access to one
/// complete memory element publishes one `i1` whatever that element's payload
/// type is.
mlir::Type getCompareExchangeSuccessType(const MemoryAccessType &access);

bool isStatelessOneTokenVectorBoundary(mlir::Operation *op);

std::optional<mlir::Value> getVectorBoundaryInputPhase(mlir::Operation *op);

std::optional<mlir::Value> getVectorBoundaryOutputPhase(mlir::Operation *op);

mlir::ValueRange getVectorBoundaryTruePhaseInputPayloads(mlir::Operation *op);

bool isVectorBoundaryTruePhaseOutputPayload(mlir::Value value,
                                            mlir::Value phase);

/// Whether two distinct vector-boundary output phases have the same ordered
/// cardinality by construction. Parallelizers share this relation when their
/// scalar phase and fixed width agree. Serializers share it when their mask,
/// group phase, and fixed width agree. Payload element types are irrelevant to
/// the ordering relation.
bool haveEquivalentOrderedCardinality(mlir::Value lhsPhase,
                                      mlir::Value rhsPhase);

std::optional<mlir::Value> getStreamActivation(dataflow::StreamOp stream);

std::optional<mlir::Value> getCloseActivation(mlir::Value value);

std::optional<bool> gateClosesWhenSelected(dataflow::GateOp gate,
                                           mlir::Value selector, unsigned lane);

bool gateAlwaysCloses(dataflow::GateOp gate);

std::optional<dataflow::GateOp> getGateCloseProjection(mlir::Value value);

std::optional<mlir::Value> getSelectorActivation(mlir::Value selector,
                                                 unsigned arity);

/// The graph activation of a stream input leaf selected exactly once by a
/// balanced binary router. When a branch is supplied, activity need only be
/// true in that branch; otherwise it must be unconditionally true.
std::optional<mlir::Value> getSelectiveRouterLeafActivation(
    mlir::Value value, mlir::Value branchSelector = {},
    std::optional<unsigned> branchLane = std::nullopt);

std::optional<mlir::Value> getSelectiveRouterLeafSynchronization(
    mlir::Value value, mlir::Value branchSelector = {},
    std::optional<unsigned> branchLane = std::nullopt);

/// The root event of a total stream activity projection. A single endpoint or
/// every leaf of a balanced router must define one Boolean for its event.
std::optional<mlir::Value> getStreamActivityEvent(mlir::Value value);

/// Whether two Boolean selectors differ only by event-preserving rendezvous or
/// an identity selection between false and true.
bool haveEquivalentSynchronizedSelectionCorrespondence(mlir::Value lhs,
                                                       mlir::Value rhs);

/// The active event of a stream publication. A single rendezvous or every leaf
/// of a balanced collector must use the corresponding routed event.
std::optional<mlir::Value> getStreamPublicationEvent(mlir::Value value);

bool selectorSelectsLaneOncePerActivation(mlir::Value selector, unsigned arity,
                                          unsigned lane);

/// The parent activation of a selector lane whose input is the true-phase
/// event of the selector's finite stream and whose lane is visited exactly
/// once per activation.
std::optional<mlir::Value> getSelectorLaneEventActivation(mlir::Value selector,
                                                          unsigned arity,
                                                          unsigned lane,
                                                          mlir::Value event);

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
