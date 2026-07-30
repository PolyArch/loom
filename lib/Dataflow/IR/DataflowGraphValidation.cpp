#include "Dataflow/IR/DataflowGraphValidation.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/OperationSchema.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"

#include <cstdint>
#include <limits>
#include <memory>

namespace {

llvm::Error graphError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::errc::invalid_argument, message.str());
}

bool isMemoryCapabilityType(mlir::Type type) {
  return dataflow::DataflowDialect::isMemoryCapabilityType(type);
}

bool containsChannelType(mlir::Type type) {
  return type
      .walk<mlir::WalkOrder::PreOrder>([](mlir::Type nested) {
        return llvm::isa<dataflow::ChannelType>(nested)
                   ? mlir::WalkResult::interrupt()
                   : mlir::WalkResult::advance();
      })
      .wasInterrupted();
}

bool isGraphMemoryInput(dataflow::GraphOp graph, mlir::Value value) {
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  return argument && argument.getOwner() == &graph.getBody().front() &&
         argument.getArgNumber() > 0 &&
         graph.getInputPortKind(argument.getArgNumber() - 1) ==
             dataflow::GraphPortKind::Memory;
}

bool isLaunchAvailableValueInput(dataflow::GraphOp graph, mlir::Value value) {
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  return argument && argument.getOwner() == &graph.getBody().front() &&
         argument.getArgNumber() > 0 &&
         graph.getInputPortKind(argument.getArgNumber() - 1) ==
             dataflow::GraphPortKind::Value;
}

bool isSupportedMemoryView(mlir::Operation *op) {
  return llvm::isa<mlir::memref::CastOp>(op);
}

bool isProtocolEstablishedMemory(dataflow::GraphOp graph, mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    if (isGraphMemoryInput(graph, value))
      return true;
    mlir::Operation *def = value.getDefiningOp();
    if (!def)
      return false;
    if (llvm::isa<mlir::memref::AllocOp>(def))
      return true;
    if (isSupportedMemoryView(def)) {
      value = mlir::cast<mlir::ViewLikeOpInterface>(def).getViewSource();
      continue;
    }
    return false;
  }
  return false;
}

bool isFreshMemoryRoot(mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    mlir::Operation *def = value.getDefiningOp();
    if (!def)
      return false;
    if (llvm::isa<mlir::memref::AllocOp>(def))
      return true;
    if (isSupportedMemoryView(def)) {
      value = mlir::cast<mlir::ViewLikeOpInterface>(def).getViewSource();
      continue;
    }
    return false;
  }
  return false;
}

bool isResidualLLVMMemoryOperation(mlir::Operation *op) {
  return llvm::isa<mlir::LLVM::LoadOp, mlir::LLVM::StoreOp,
                   mlir::LLVM::MemcpyOp, mlir::LLVM::MemmoveOp,
                   mlir::LLVM::MemsetOp>(op);
}

bool hasRawPointerUse(mlir::Operation *op) {
  return llvm::any_of(op->getOperandTypes(),
                      [](mlir::Type type) {
                        return mlir::isa<mlir::LLVM::LLVMPointerType>(type);
                      }) ||
         llvm::any_of(op->getResultTypes(), [](mlir::Type type) {
           return mlir::isa<mlir::LLVM::LLVMPointerType>(type);
         });
}

using SelectorLanes = llvm::DenseMap<mlir::Value, unsigned>;

/// A canonical memory actor publishes its remaining results together with
/// `done` as one retirement event.
bool isRetirementPublication(mlir::Value witness, mlir::Value prerequisite) {
  mlir::Operation *def = witness.getDefiningOp();
  return def && witness != prerequisite &&
         witness == dataflow::semantics::getMemoryActorDone(def) &&
         prerequisite.getDefiningOp() == def;
}

struct CausalConstraintTransition {
  unsigned parent;
  mlir::Value selector;
  unsigned lane;

  bool operator==(const CausalConstraintTransition &other) const {
    return parent == other.parent && selector == other.selector &&
           lane == other.lane;
  }
};

struct CausalConstraintTransitionInfo {
  static CausalConstraintTransition getEmptyKey() {
    return {std::numeric_limits<unsigned>::max(), {}, 0};
  }
  static CausalConstraintTransition getTombstoneKey() {
    return {std::numeric_limits<unsigned>::max() - 1, {}, 0};
  }
  static unsigned getHashValue(const CausalConstraintTransition &key) {
    return llvm::hash_combine(key.parent, key.selector.getAsOpaquePointer(),
                              key.lane);
  }
  static bool isEqual(const CausalConstraintTransition &lhs,
                      const CausalConstraintTransition &rhs) {
    return lhs == rhs;
  }
};

struct CausalMemoKey {
  mlir::Value event;
  unsigned constraints;

  bool operator==(const CausalMemoKey &other) const {
    return event == other.event && constraints == other.constraints;
  }
};

struct CausalMemoKeyInfo {
  static CausalMemoKey getEmptyKey() {
    return {{}, std::numeric_limits<unsigned>::max()};
  }
  static CausalMemoKey getTombstoneKey() {
    return {{}, std::numeric_limits<unsigned>::max() - 1};
  }
  static unsigned getHashValue(const CausalMemoKey &key) {
    return llvm::hash_combine(key.event.getAsOpaquePointer(), key.constraints);
  }
  static bool isEqual(const CausalMemoKey &lhs, const CausalMemoKey &rhs) {
    return lhs == rhs;
  }
};

class CausalDependencyAnalysis {
public:
  CausalDependencyAnalysis(mlir::Value prerequisite, mlir::Value event)
      : prerequisite(prerequisite), event(event) {
    constraintStates.emplace_back();
    constraintStatesByHash[hashConstraintState(constraintStates.front())]
        .push_back(0);
  }

  bool dependsOn() { return reaches(prerequisite, 0); }

private:
  enum class MemoState : uint8_t { Visiting, False, True };

  mlir::Value prerequisite;
  mlir::Value event;
  llvm::DenseMap<mlir::Value, unsigned> selectorIds;
  llvm::SmallVector<llvm::SmallVector<std::uint64_t, 4>, 8> constraintStates;
  llvm::DenseMap<std::uint64_t, llvm::SmallVector<unsigned, 1>>
      constraintStatesByHash;
  llvm::DenseMap<CausalConstraintTransition, unsigned,
                 CausalConstraintTransitionInfo>
      constraintTransitions;
  llvm::DenseMap<CausalMemoKey, MemoState, CausalMemoKeyInfo> memo;

  static std::uint64_t
  hashConstraintState(llvm::ArrayRef<std::uint64_t> assignments) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::uint64_t assignment : assignments) {
      hash ^= assignment;
      hash *= 1099511628211ULL;
    }
    if (hash >= std::numeric_limits<std::uint64_t>::max() - 1)
      hash -= 2;
    return hash;
  }

  unsigned
  internConstraintState(llvm::SmallVector<std::uint64_t, 4> assignments) {
    std::uint64_t hash = hashConstraintState(assignments);
    auto &bucket = constraintStatesByHash[hash];
    for (unsigned candidate : bucket) {
      if (constraintStates[candidate] == assignments)
        return candidate;
    }
    unsigned next = constraintStates.size();
    constraintStates.push_back(std::move(assignments));
    bucket.push_back(next);
    return next;
  }

  std::optional<unsigned> constrain(unsigned state, mlir::Value selector,
                                    unsigned lane) {
    auto selectorIt =
        selectorIds.try_emplace(selector, selectorIds.size()).first;
    unsigned selectorId = selectorIt->second;
    std::uint64_t assignment =
        (static_cast<std::uint64_t>(selectorId) << 32) | lane;
    const auto &current = constraintStates[state];
    unsigned position = 0;
    while (position != current.size() && (current[position] >> 32) < selectorId)
      ++position;
    if (position != current.size() && (current[position] >> 32) == selectorId) {
      return current[position] == assignment ? std::optional<unsigned>(state)
                                             : std::nullopt;
    }

    CausalConstraintTransition transition{state, selector, lane};
    auto known = constraintTransitions.find(transition);
    if (known != constraintTransitions.end())
      return known->second;
    llvm::SmallVector<std::uint64_t, 4> nextState(current);
    nextState.insert(nextState.begin() + position, assignment);
    unsigned next = internConstraintState(std::move(nextState));
    constraintTransitions.try_emplace(transition, next);
    return next;
  }

  bool reaches(mlir::Value value, unsigned state) {
    if (!value)
      return false;
    if (auto result = llvm::dyn_cast<mlir::OpResult>(value)) {
      if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(result.getOwner())) {
        auto constrained =
            constrain(state, demux.getSel(), result.getResultNumber());
        if (!constrained)
          return false;
        state = *constrained;
      }
    }

    CausalMemoKey key{value, state};
    auto known = memo.find(key);
    if (known != memo.end())
      return known->second == MemoState::True;
    memo.try_emplace(key, MemoState::Visiting);
    bool result = compute(value, state);
    memo[key] = result ? MemoState::True : MemoState::False;
    return result;
  }

  bool reachesAnyResult(mlir::Operation *operation, unsigned state) {
    return llvm::any_of(operation->getResults(), [&](mlir::Value result) {
      return reaches(result, state);
    });
  }

  bool compute(mlir::Value value, unsigned state) {
    if (value == event)
      return true;

    // Sync outputs are one atomic publication. A memory actor's done result
    // likewise follows every sibling result from the same retirement event.
    // These are the only causal edges that are not ordinary operand-to-result
    // SSA edges.
    if (value == prerequisite) {
      if (auto result = llvm::dyn_cast<mlir::OpResult>(value)) {
        mlir::Operation *owner = result.getOwner();
        if (llvm::isa<dataflow::SyncOp>(owner) &&
            reachesAnyResult(owner, state))
          return true;
        if (mlir::Value done = dataflow::semantics::getMemoryActorDone(owner);
            done && value != done && reaches(done, state))
          return true;
      }
    }

    for (mlir::OpOperand &use : value.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (auto mux = llvm::dyn_cast<dataflow::MuxOp>(user)) {
        if (use.getOperandNumber() == 0) {
          if (reachesAnyResult(user, state))
            return true;
          continue;
        }
        const unsigned lane = use.getOperandNumber() - 1;
        auto constrained = constrain(state, mux.getSel(), lane);
        if (constrained && reachesAnyResult(user, *constrained))
          return true;
        continue;
      }
      if (reachesAnyResult(user, state))
        return true;
    }
    return false;
  }
};

bool causallyDependsOn(mlir::Value event, mlir::Value prerequisite) {
  return CausalDependencyAnalysis(prerequisite, event).dependsOn();
}

bool isCovered(mlir::Value prerequisite, mlir::ValueRange completion) {
  return llvm::any_of(completion, [&](mlir::Value witness) {
    return CausalDependencyAnalysis(prerequisite, witness).dependsOn() ||
           isRetirementPublication(witness, prerequisite);
  });
}

bool coversFalseClose(mlir::Value witness, mlir::Value closeSignal,
                      llvm::DenseSet<mlir::Value> &visited,
                      SelectorLanes &selectorLanes) {
  if (!witness)
    return false;

  std::optional<mlir::Value> insertedSelector;
  if (auto result = llvm::dyn_cast<mlir::OpResult>(witness)) {
    if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(result.getOwner())) {
      auto [it, inserted] =
          selectorLanes.try_emplace(demux.getSel(), result.getResultNumber());
      if (!inserted && it->second != result.getResultNumber())
        return false;
      if (inserted)
        insertedSelector = demux.getSel();
    }
  }
  llvm::scope_exit selectorCleanup([&] {
    if (insertedSelector)
      selectorLanes.erase(*insertedSelector);
  });

  if (!visited.insert(witness).second)
    return false;
  llvm::scope_exit visitedCleanup([&] { visited.erase(witness); });
  mlir::Operation *def = witness.getDefiningOp();
  if (!def)
    return false;

  auto covers = [&](mlir::Value value) {
    return coversFalseClose(value, closeSignal, visited, selectorLanes);
  };
  auto coversInLane = [&](mlir::Value value, mlir::Value selector,
                          unsigned lane) {
    auto insertion = selectorLanes.try_emplace(selector, lane);
    auto it = insertion.first;
    bool inserted = insertion.second;
    if (!inserted && it->second != lane)
      return false;
    llvm::scope_exit cleanup([&] {
      if (inserted)
        selectorLanes.erase(selector);
    });
    return covers(value);
  };
  if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(def)) {
    auto result = llvm::dyn_cast<mlir::OpResult>(witness);
    if (result && result.getResultNumber() == 0) {
      mlir::Value phase = demux.getSel();
      llvm::DenseSet<mlir::Value> phaseVisited;
      while (phase && phaseVisited.insert(phase).second) {
        if (phase == closeSignal)
          return true;
        mlir::Operation *phaseDef = phase.getDefiningOp();
        auto output =
            dataflow::semantics::getVectorBoundaryOutputPhase(phaseDef);
        auto input = dataflow::semantics::getVectorBoundaryInputPhase(phaseDef);
        if (!output || !input || *output != phase)
          break;
        phase = *input;
      }
    }
    return covers(demux.getInput());
  }
  if (auto sync = llvm::dyn_cast<dataflow::SyncOp>(def)) {
    return llvm::any_of(sync.getInputs(), covers);
  }
  if (auto mux = llvm::dyn_cast<dataflow::MuxOp>(def)) {
    if (auto gate = closeSignal.getDefiningOp<dataflow::GateOp>()) {
      bool conditional = true;
      for (auto [lane, input] : llvm::enumerate(mux.getInputs())) {
        auto closes = dataflow::semantics::gateClosesWhenSelected(
            gate, mux.getSel(), lane);
        if (!closes) {
          conditional = false;
          break;
        }
        if (!*closes)
          continue;
        if (!coversInLane(input, mux.getSel(), lane))
          return false;
      }
      if (conditional)
        return true;
    }
    for (auto [lane, input] : llvm::enumerate(mux.getInputs())) {
      if (coversInLane(input, mux.getSel(), lane))
        return true;
    }
    return false;
  }
  if (auto carry = llvm::dyn_cast<dataflow::CarryOp>(def)) {
    return covers(carry.getInit()) || covers(carry.getCarry());
  }
  if (mlir::Value done = dataflow::semantics::getMemoryActorDone(def)) {
    return witness == done &&
           covers(dataflow::semantics::getMemoryActorControl(def));
  }
  return false;
}

bool coversFalseClose(mlir::Value closeSignal, mlir::ValueRange completion) {
  return llvm::any_of(completion, [&](mlir::Value witness) {
    llvm::DenseSet<mlir::Value> visited;
    SelectorLanes selectorLanes;
    return coversFalseClose(witness, closeSignal, visited, selectorLanes);
  });
}

mlir::Value statefulCloseSignal(mlir::Operation *op) {
  if (auto stream = llvm::dyn_cast<dataflow::StreamOp>(op))
    return stream.getPhase();
  if (auto carry = llvm::dyn_cast<dataflow::CarryOp>(op))
    return carry.getCond();
  if (auto invariant = llvm::dyn_cast<dataflow::InvariantOp>(op))
    return invariant.getCond();
  if (auto gate = llvm::dyn_cast<dataflow::GateOp>(op))
    return gate.getAfterCond();
  if (auto phase = dataflow::semantics::getVectorBoundaryOutputPhase(op))
    return *phase;
  return {};
}

bool hasPhaseAlignedGateValue(dataflow::GateOp gate) {
  mlir::Operation *def = gate.getBeforeValue().getDefiningOp();
  if (auto carry = llvm::dyn_cast_or_null<dataflow::CarryOp>(def))
    return carry.getOutput() == gate.getBeforeValue() &&
           carry.getCond() == gate.getBeforeCond();
  if (auto invariant = llvm::dyn_cast_or_null<dataflow::InvariantOp>(def))
    return invariant.getOutput() == gate.getBeforeValue() &&
           invariant.getCond() == gate.getBeforeCond();
  return false;
}

bool hasObservableEffect(mlir::Operation *op) {
  auto effects = llvm::dyn_cast<mlir::MemoryEffectOpInterface>(op);
  if (effects)
    return effects.hasEffect<mlir::MemoryEffects::Write>() ||
           effects.hasEffect<mlir::MemoryEffects::Allocate>() ||
           effects.hasEffect<mlir::MemoryEffects::Free>();
  if (mlir::isPure(op))
    return false;
  llvm::StringRef dialect = op->getName().getDialectNamespace();
  if (dialect == "arith" || dialect == "math" || dialect == "ub")
    return false;
  if (dialect == "llvm" &&
      !llvm::isa<mlir::LLVM::CallOp, mlir::LLVM::StoreOp, mlir::LLVM::MemcpyOp,
                 mlir::LLVM::MemmoveOp, mlir::LLVM::MemsetOp>(op))
    return false;
  return true;
}

enum class AlignmentQueryKind : uint8_t {
  Close,
  TruePhaseClose,
  GateClose,
  SiblingGateClose,
  SelectedClose,
  SelectedTruePhaseClose,
  SelectedAvailability,
  SelectedAlignedAvailability,
  SelectedTruePhaseAlignedAvailability,
  Aligned,
  TruePhaseAligned,
};

struct AlignmentQuery {
  uint64_t alignmentRevision;
  mlir::Value value;
  mlir::Value parentPhase;
  mlir::Value parentAssumption;
  mlir::Value selector;
  unsigned lane;
  AlignmentQueryKind kind;

  bool operator==(const AlignmentQuery &other) const {
    return alignmentRevision == other.alignmentRevision &&
           value == other.value && parentPhase == other.parentPhase &&
           parentAssumption == other.parentAssumption &&
           selector == other.selector && lane == other.lane &&
           kind == other.kind;
  }
};

struct AlignmentQueryInfo {
  static AlignmentQuery getEmptyKey() {
    return {std::numeric_limits<uint64_t>::max(),
            {},
            {},
            {},
            {},
            0,
            AlignmentQueryKind::Close};
  }

  static AlignmentQuery getTombstoneKey() {
    return {std::numeric_limits<uint64_t>::max() - 1,
            {},
            {},
            {},
            {},
            0,
            AlignmentQueryKind::Close};
  }

  static unsigned getHashValue(const AlignmentQuery &query) {
    auto opaque = [](mlir::Value value) {
      return value ? value.getAsOpaquePointer() : nullptr;
    };
    return llvm::hash_combine(query.alignmentRevision, opaque(query.value),
                              opaque(query.parentPhase),
                              opaque(query.parentAssumption),
                              opaque(query.selector), query.lane, query.kind);
  }

  static bool isEqual(const AlignmentQuery &lhs, const AlignmentQuery &rhs) {
    return lhs == rhs;
  }
};

struct CardinalityGraphIndex {
  explicit CardinalityGraphIndex(dataflow::GraphOp graph) {
    for (mlir::Operation &op : graph.getBody().front().without_terminator()) {
      if (auto carry = llvm::dyn_cast<dataflow::CarryOp>(op)) {
        carriesByPhase[carry.getCond()].push_back(carry);
        activationInputsByPhase[carry.getCond()].push_back(carry.getInit());
        continue;
      }
      if (auto invariant = llvm::dyn_cast<dataflow::InvariantOp>(op)) {
        activationInputsByPhase[invariant.getCond()].push_back(
            invariant.getInit());
        continue;
      }
      if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(op))
        demuxesBySelector[demux.getSel()].push_back(demux);
    }
  }

  llvm::DenseMap<mlir::Value, llvm::SmallVector<dataflow::CarryOp, 4>>
      carriesByPhase;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value, 4>>
      activationInputsByPhase;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<dataflow::DemuxOp, 4>>
      demuxesBySelector;
};

using CardinalityAssumptionSetId = unsigned;

// Nested selected-close proofs create analysis views with different local
// assumptions. Intern the complete assumption sets so equivalent views share
// proof results and recursive activity without conflating distinct contexts.
struct SelectedNestedQuery {
  CardinalityAssumptionSetId exactOneAssumptions;
  CardinalityAssumptionSetId alignedCarryAssumptions;
  mlir::Value result;
  mlir::Value parentPhase;
  mlir::Value parentAssumption;
  mlir::Value selector;
  unsigned lane;
  bool truePhaseOnly;

  bool operator==(const SelectedNestedQuery &other) const {
    return exactOneAssumptions == other.exactOneAssumptions &&
           alignedCarryAssumptions == other.alignedCarryAssumptions &&
           result == other.result && parentPhase == other.parentPhase &&
           parentAssumption == other.parentAssumption &&
           selector == other.selector && lane == other.lane &&
           truePhaseOnly == other.truePhaseOnly;
  }
};

struct SelectedNestedQueryInfo {
  static SelectedNestedQuery getEmptyKey() {
    return {std::numeric_limits<unsigned>::max(),
            std::numeric_limits<unsigned>::max(),
            {},
            {},
            {},
            {},
            0,
            false};
  }

  static SelectedNestedQuery getTombstoneKey() {
    return {std::numeric_limits<unsigned>::max() - 1,
            std::numeric_limits<unsigned>::max() - 1,
            {},
            {},
            {},
            {},
            0,
            false};
  }

  static unsigned getHashValue(const SelectedNestedQuery &query) {
    auto opaque = [](mlir::Value value) {
      return value ? value.getAsOpaquePointer() : nullptr;
    };
    return llvm::hash_combine(
        query.exactOneAssumptions, query.alignedCarryAssumptions,
        opaque(query.result), opaque(query.parentPhase),
        opaque(query.parentAssumption), opaque(query.selector), query.lane,
        query.truePhaseOnly);
  }

  static bool isEqual(const SelectedNestedQuery &lhs,
                      const SelectedNestedQuery &rhs) {
    return lhs == rhs;
  }
};

struct CardinalitySharedState {
  explicit CardinalitySharedState(dataflow::GraphOp graph)
      : graphIndex(std::make_shared<CardinalityGraphIndex>(graph)) {
    assumptionSets.emplace_back();
    assumptionBuckets[hashAssumptions(assumptionSets.front())].push_back(0);
  }

  CardinalityAssumptionSetId
  internAssumptions(const llvm::DenseSet<mlir::Value> &assumptions) {
    llvm::SmallVector<mlir::Value, 8> normalized(assumptions.begin(),
                                                 assumptions.end());
    llvm::sort(normalized, [](mlir::Value lhs, mlir::Value rhs) {
      return reinterpret_cast<std::uintptr_t>(lhs.getAsOpaquePointer()) <
             reinterpret_cast<std::uintptr_t>(rhs.getAsOpaquePointer());
    });
    const std::uint64_t hash = hashAssumptions(normalized);
    auto bucket = assumptionBuckets.find(hash);
    if (bucket != assumptionBuckets.end())
      for (CardinalityAssumptionSetId candidate : bucket->second)
        if (llvm::ArrayRef<mlir::Value>(assumptionSets[candidate]) ==
            llvm::ArrayRef<mlir::Value>(normalized))
          return candidate;

    const CardinalityAssumptionSetId id = assumptionSets.size();
    assumptionSets.push_back(std::move(normalized));
    assumptionBuckets[hash].push_back(id);
    return id;
  }

  std::shared_ptr<CardinalityGraphIndex> graphIndex;
  llvm::DenseMap<SelectedNestedQuery, bool, SelectedNestedQueryInfo>
      selectedNested;
  llvm::DenseSet<SelectedNestedQuery, SelectedNestedQueryInfo>
      selectedNestedActive;

private:
  static std::uint64_t
  hashAssumptions(llvm::ArrayRef<mlir::Value> assumptions) {
    llvm::hash_code hash = assumptions.size();
    for (mlir::Value assumption : assumptions)
      hash = llvm::hash_combine(hash, assumption.getAsOpaquePointer());
    return static_cast<std::uint64_t>(hash);
  }

  llvm::SmallVector<llvm::SmallVector<mlir::Value, 8>, 8> assumptionSets;
  llvm::DenseMap<std::uint64_t,
                 llvm::SmallVector<CardinalityAssumptionSetId, 1>>
      assumptionBuckets;
};

class GraphCardinalityAnalysis {
public:
  explicit GraphCardinalityAnalysis(dataflow::GraphOp graph)
      : graph(graph),
        sharedState(std::make_shared<CardinalitySharedState>(graph)),
        graphIndex(sharedState->graphIndex) {}

  bool isExactOne(mlir::Value value) {
    auto known = exactOne.find(value);
    if (known != exactOne.end())
      return known->second;
    if (!exactOneActive.insert(value).second)
      return false;
    bool result = computeExactOne(value);
    exactOneActive.erase(value);
    exactOne.try_emplace(value, result);
    return result;
  }

  bool hasProvenStreamCommit(mlir::Value value) {
    if (isExactOne(value))
      return true;
    llvm::SmallVector<mlir::Value, 2> closeSignals;
    collectStreamCloseSignals(value, closeSignals);
    return !closeSignals.empty() &&
           llvm::all_of(closeSignals, [&](mlir::Value signal) {
             return isOneClosePhase(signal);
           });
  }

  void collectStreamCloseSignals(mlir::Value value,
                                 llvm::SmallVectorImpl<mlir::Value> &signals) {
    llvm::DenseSet<mlir::Value> visited;
    collectStreamCloseSignals(value, visited, signals);
  }

private:
  GraphCardinalityAnalysis(dataflow::GraphOp graph,
                           std::shared_ptr<CardinalitySharedState> sharedState)
      : graph(graph), sharedState(std::move(sharedState)),
        graphIndex(this->sharedState->graphIndex) {}

  dataflow::GraphOp graph;
  std::shared_ptr<CardinalitySharedState> sharedState;
  std::shared_ptr<CardinalityGraphIndex> graphIndex;
  uint64_t alignmentRevision = 0;
  llvm::DenseMap<AlignmentQuery, bool, AlignmentQueryInfo> alignment;
  llvm::DenseSet<AlignmentQuery, AlignmentQueryInfo> alignmentActive;
  llvm::DenseMap<mlir::Value, bool> exactOne;
  llvm::DenseSet<mlir::Value> exactOneActive;
  llvm::DenseMap<mlir::Value, bool> oneClosePhase;
  llvm::DenseSet<mlir::Value> oneCloseActive;
  llvm::DenseSet<mlir::Value> alignedCarryAssumptions;
  llvm::DenseMap<mlir::Value, bool> alignedCarrySystems;
  llvm::DenseSet<mlir::Value> alignedCarrySystemsActive;
  llvm::DenseSet<mlir::Value> exactOneAssumptions;

  bool insertAlignedCarryAssumption(mlir::Value value) {
    bool inserted = alignedCarryAssumptions.insert(value).second;
    if (inserted)
      ++alignmentRevision;
    return inserted;
  }

  void eraseAlignedCarryAssumption(mlir::Value value) {
    if (alignedCarryAssumptions.erase(value))
      ++alignmentRevision;
  }

  template <typename Compute>
  bool evaluateAlignment(const AlignmentQuery &query, Compute &&compute) {
    auto known = alignment.find(query);
    if (known != alignment.end())
      return known->second;
    if (!alignmentActive.insert(query).second)
      return false;
    auto eraseActive = llvm::scope_exit([&] { alignmentActive.erase(query); });
    bool result = compute();
    alignment.try_emplace(query, result);
    return result;
  }

  bool allOperandsExact(mlir::Operation *op) {
    return llvm::all_of(op->getOperands(),
                        [&](mlir::Value value) { return isExactOne(value); });
  }

  bool isGraphStreamInput(mlir::Value value) {
    auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
    return argument && argument.getOwner() == &graph.getBody().front() &&
           argument.getArgNumber() != 0 &&
           graph.getInputPortKind(argument.getArgNumber() - 1) ==
               dataflow::GraphPortKind::Stream;
  }

  std::optional<uint64_t> getKnownUnsigned(mlir::Value value) {
    if (auto constant = value.getDefiningOp<dataflow::ConstantOp>()) {
      if (auto boolean =
              llvm::dyn_cast<mlir::BoolAttr>(constant.getConstValue()))
        return boolean.getValue();
      auto integer =
          llvm::dyn_cast<mlir::IntegerAttr>(constant.getConstValue());
      if (integer && !integer.getValue().isNegative())
        return integer.getValue().getZExtValue();
    }
    mlir::APInt constant;
    if (mlir::matchPattern(value, mlir::m_ConstantInt(&constant)) &&
        !constant.isNegative())
      return constant.getZExtValue();
    return std::nullopt;
  }

  void collectStreamCloseSignals(mlir::Value value,
                                 llvm::DenseSet<mlir::Value> &visited,
                                 llvm::SmallVectorImpl<mlir::Value> &signals) {
    if (!value || !visited.insert(value).second)
      return;
    mlir::Operation *def = value.getDefiningOp();
    if (!def)
      return;
    mlir::Value signal = statefulCloseSignal(def);
    if (auto gate = llvm::dyn_cast<dataflow::GateOp>(def))
      if (hasPhaseAlignedGateValue(gate))
        signal = gate.getBeforeCond();
    if (signal) {
      auto stream = signal.getDefiningOp<dataflow::StreamOp>();
      if (stream && signal == stream.getPhase())
        if (auto activation = dataflow::semantics::getStreamActivation(stream);
            activation && !isExactOne(*activation)) {
          collectStreamCloseSignals(*activation, visited, signals);
          return;
        }
      if (!llvm::is_contained(signals, signal))
        signals.push_back(signal);
      return;
    }
    for (mlir::Value operand : def->getOperands())
      collectStreamCloseSignals(operand, visited, signals);
  }

  bool isExactOneWhenSelected(mlir::Value value, mlir::Value selector,
                              unsigned lane) {
    GraphCardinalityAnalysis branch(graph, sharedState);
    auto demuxes = graphIndex->demuxesBySelector.find(selector);
    if (demuxes != graphIndex->demuxesBySelector.end()) {
      for (dataflow::DemuxOp demux : demuxes->second) {
        if (lane >= demux.getOutputs().size() || !isExactOne(demux.getInput()))
          continue;
        branch.exactOneAssumptions.insert(demux.getOutputs()[lane]);
      }
    }
    return branch.isExactOne(value);
  }

  bool isExactOneWhenSelectedAndAligned(mlir::Value value, mlir::Value selector,
                                        unsigned lane, mlir::Value phase,
                                        mlir::Value assumption,
                                        bool truePhaseOnly) {
    GraphCardinalityAnalysis branch(graph, sharedState);
    auto demuxes = graphIndex->demuxesBySelector.find(selector);
    if (demuxes != graphIndex->demuxesBySelector.end()) {
      for (dataflow::DemuxOp demux : demuxes->second) {
        if (lane >= demux.getOutputs().size())
          continue;
        llvm::DenseSet<mlir::Value> visited;
        if (!isAligned(demux.getInput(), phase, assumption, truePhaseOnly,
                       visited))
          continue;
        branch.exactOneAssumptions.insert(demux.getOutputs()[lane]);
      }
    }
    return branch.isExactOne(value);
  }

  bool computeAvailableWhenSelected(mlir::Value value, mlir::Value selector,
                                    unsigned lane,
                                    llvm::DenseSet<mlir::Value> &visited) {
    auto result = llvm::dyn_cast<mlir::OpResult>(value);
    mlir::Operation *def = result ? result.getOwner() : nullptr;
    if (!def)
      return false;
    if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(def)) {
      if (demux.getSel() == selector)
        return result.getResultNumber() == lane &&
               (isExactOne(demux.getInput()) ||
                isGraphStreamInput(demux.getInput()));
      if (isGraphStreamInput(demux.getInput())) {
        auto activation = dataflow::semantics::getSelectorActivation(
            demux.getSel(), demux.getOutputs().size());
        return activation && isExactOne(*activation) &&
               dataflow::semantics::selectorLaneActiveWhenSelected(
                   demux.getSel(), demux.getOutputs().size(),
                   result.getResultNumber(), selector, lane);
      }
    }
    if (auto gate = dataflow::semantics::getGateCloseProjection(value)) {
      auto closes =
          dataflow::semantics::gateClosesWhenSelected(*gate, selector, lane);
      if (closes && *closes)
        return isOneClosePhase(gate->getBeforeCond()) &&
               isPhaseAligned(gate->getBeforeValue(), gate->getBeforeCond());
    }
    if (isExactOneWhenSelected(value, selector, lane))
      return true;
    if (llvm::isa<dataflow::StreamOp, dataflow::CarryOp, dataflow::InvariantOp,
                  dataflow::GateOp, dataflow::ParallelizeOp, dataflow::PackOp,
                  dataflow::UnpackOp, dataflow::SerializeOp, dataflow::MuxOp>(
            def))
      return false;
    if (!dataflow::isCanonicalDataflowActor(def) &&
        !llvm::isa<mlir::memref::CastOp>(def))
      return false;
    return llvm::all_of(def->getOperands(), [&](mlir::Value operand) {
      return availableWhenSelected(operand, selector, lane, visited);
    });
  }

  bool availableWhenSelected(mlir::Value value, mlir::Value selector,
                             unsigned lane,
                             llvm::DenseSet<mlir::Value> &visited) {
    if (isExactOne(value))
      return true;
    if (!visited.insert(value).second)
      return false;
    auto eraseVisited = llvm::scope_exit([&] { visited.erase(value); });
    AlignmentQuery query{alignmentRevision,
                         value,
                         {},
                         {},
                         selector,
                         lane,
                         AlignmentQueryKind::SelectedAvailability};
    return evaluateAlignment(query, [&] {
      return computeAvailableWhenSelected(value, selector, lane, visited);
    });
  }

  bool availableWhenSelected(mlir::Value value, mlir::Value selector,
                             unsigned lane) {
    llvm::DenseSet<mlir::Value> visited;
    return availableWhenSelected(value, selector, lane, visited);
  }

  bool computeAvailableWhenSelectedAndAligned(
      mlir::Value value, mlir::Value selector, unsigned lane, mlir::Value phase,
      mlir::Value assumption, bool truePhaseOnly,
      llvm::DenseSet<mlir::Value> &visited) {
    auto result = llvm::dyn_cast<mlir::OpResult>(value);
    mlir::Operation *def = result ? result.getOwner() : nullptr;
    if (!def)
      return false;
    if (truePhaseOnly &&
        (isSiblingGateCloseAligned(value, selector, lane, phase, assumption) ||
         isNestedGateCloseAligned(value, selector, lane, phase, assumption)))
      return true;
    if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(def)) {
      if (demux.getSel() == selector && result.getResultNumber() == lane) {
        if (!isGraphStreamInput(demux.getInput())) {
          return isAligned(demux.getInput(), phase, assumption, truePhaseOnly,
                           visited);
        }
        return isAligned(selector, phase, assumption, truePhaseOnly, visited);
      }
      if (result.getResultNumber() == 0 &&
          isNestedCloseAlignedWhenSelected(demux, result, selector, lane, phase,
                                           assumption, truePhaseOnly))
        return true;
      if (isGraphStreamInput(demux.getInput())) {
        auto activation = dataflow::semantics::getSelectorActivation(
            demux.getSel(), demux.getOutputs().size());
        if (!activation || !dataflow::semantics::selectorLaneActiveWhenSelected(
                               demux.getSel(), demux.getOutputs().size(),
                               result.getResultNumber(), selector, lane))
          return false;
        mlir::Value alignment = *activation;
        if (auto synchronization =
                dataflow::semantics::getSelectorLaneSynchronization(
                    demux.getSel(), demux.getOutputs().size(),
                    result.getResultNumber(), selector, lane))
          alignment = *synchronization;
        return isAligned(alignment, phase, assumption, truePhaseOnly, visited);
      }
      return false;
    }
    if (llvm::isa<dataflow::MuxOp>(def))
      return isExactOneWhenSelectedAndAligned(value, selector, lane, phase,
                                              assumption, truePhaseOnly);
    if (llvm::isa<dataflow::StreamOp, dataflow::CarryOp, dataflow::InvariantOp,
                  dataflow::GateOp, dataflow::ParallelizeOp, dataflow::PackOp,
                  dataflow::UnpackOp, dataflow::SerializeOp>(def))
      return false;
    if (!dataflow::isCanonicalDataflowActor(def) &&
        !llvm::isa<mlir::memref::CastOp>(def))
      return false;
    bool hasSelectedOperand = false;
    for (mlir::Value operand : def->getOperands()) {
      if (isMemoryCapabilityType(operand.getType()))
        continue;
      if (availableWhenSelectedAndAligned(operand, selector, lane, phase,
                                          assumption, truePhaseOnly, visited)) {
        hasSelectedOperand = true;
        continue;
      }
      if (!isAligned(operand, phase, assumption, truePhaseOnly, visited))
        return false;
    }
    return hasSelectedOperand;
  }

  bool availableWhenSelectedAndAligned(mlir::Value value, mlir::Value selector,
                                       unsigned lane, mlir::Value phase,
                                       mlir::Value assumption,
                                       bool truePhaseOnly,
                                       llvm::DenseSet<mlir::Value> &visited) {
    if (!visited.insert(value).second)
      return false;
    auto eraseVisited = llvm::scope_exit([&] { visited.erase(value); });
    AlignmentQuery query{
        alignmentRevision,
        value,
        phase,
        assumption,
        selector,
        lane,
        truePhaseOnly ? AlignmentQueryKind::SelectedTruePhaseAlignedAvailability
                      : AlignmentQueryKind::SelectedAlignedAvailability};
    return evaluateAlignment(query, [&] {
      return computeAvailableWhenSelectedAndAligned(
          value, selector, lane, phase, assumption, truePhaseOnly, visited);
    });
  }

  bool initializeSelectedNestedActivation(
      mlir::Value childPhase, mlir::Value selector, unsigned lane,
      mlir::Value parentPhase, mlir::Value parentAssumption, bool truePhaseOnly,
      GraphCardinalityAnalysis &activation) {
    auto assumeExact = [&](mlir::Value value) {
      llvm::DenseSet<mlir::Value> visited;
      if (!availableWhenSelectedAndAligned(value, selector, lane, parentPhase,
                                           parentAssumption, truePhaseOnly,
                                           visited))
        return false;
      activation.exactOneAssumptions.insert(value);
      return true;
    };

    if (auto stream = childPhase.getDefiningOp<dataflow::StreamOp>()) {
      if (childPhase != stream.getPhase() || !assumeExact(stream.getInit()) ||
          !assumeExact(stream.getLimit()) || !assumeExact(stream.getStep()))
        return false;
    } else {
      auto carries = graphIndex->carriesByPhase.find(childPhase);
      if (carries == graphIndex->carriesByPhase.end() ||
          carries->second.empty())
        return false;
    }

    auto inputs = graphIndex->activationInputsByPhase.find(childPhase);
    if (inputs != graphIndex->activationInputsByPhase.end())
      for (mlir::Value input : inputs->second)
        if (!assumeExact(input))
          return false;
    return true;
  }

  bool isNestedCloseAlignedWhenSelected(dataflow::DemuxOp close,
                                        mlir::OpResult result,
                                        mlir::Value selector, unsigned lane,
                                        mlir::Value parentPhase,
                                        mlir::Value parentAssumption,
                                        bool truePhaseOnly) {
    if (result.getResultNumber() != 0 || close.getOutputs().size() != 2 ||
        close.getSel() == selector || close.getSel() == parentPhase)
      return false;

    AlignmentQuery query{alignmentRevision,
                         result,
                         parentPhase,
                         parentAssumption,
                         selector,
                         lane,
                         truePhaseOnly
                             ? AlignmentQueryKind::SelectedTruePhaseClose
                             : AlignmentQueryKind::SelectedClose};
    return evaluateAlignment(query, [&] {
      SelectedNestedQuery sharedQuery{
          sharedState->internAssumptions(exactOneAssumptions),
          sharedState->internAssumptions(alignedCarryAssumptions),
          result,
          parentPhase,
          parentAssumption,
          selector,
          lane,
          truePhaseOnly};
      auto known = sharedState->selectedNested.find(sharedQuery);
      if (known != sharedState->selectedNested.end())
        return known->second;
      if (!sharedState->selectedNestedActive.insert(sharedQuery).second)
        return false;
      auto eraseActive = llvm::scope_exit(
          [&] { sharedState->selectedNestedActive.erase(sharedQuery); });

      GraphCardinalityAnalysis activation(graph, sharedState);
      bool proven = initializeSelectedNestedActivation(
                        close.getSel(), selector, lane, parentPhase,
                        parentAssumption, truePhaseOnly, activation) &&
                    activation.isExactOne(result);
      sharedState->selectedNested.try_emplace(sharedQuery, proven);
      return proven;
    });
  }

  bool initializeNestedActivation(mlir::Value childPhase,
                                  mlir::Value parentPhase,
                                  mlir::Value parentAssumption,
                                  bool truePhaseOnly,
                                  GraphCardinalityAnalysis &activation) {
    // Parent-aligned inputs are exact-one within each child activation.
    auto assumeExact = [&](mlir::Value value) {
      llvm::DenseSet<mlir::Value> visited;
      if (!isAligned(value, parentPhase, parentAssumption, truePhaseOnly,
                     visited))
        return false;
      activation.exactOneAssumptions.insert(value);
      return true;
    };

    if (auto stream = childPhase.getDefiningOp<dataflow::StreamOp>()) {
      if (childPhase != stream.getPhase() || !assumeExact(stream.getInit()) ||
          !assumeExact(stream.getLimit()) || !assumeExact(stream.getStep()))
        return false;
    } else {
      auto carries = graphIndex->carriesByPhase.find(childPhase);
      if (carries == graphIndex->carriesByPhase.end() ||
          carries->second.empty())
        return false;
    }

    auto inputs = graphIndex->activationInputsByPhase.find(childPhase);
    if (inputs != graphIndex->activationInputsByPhase.end())
      for (mlir::Value input : inputs->second)
        if (!assumeExact(input))
          return false;
    return true;
  }

  bool isNestedCloseAligned(dataflow::DemuxOp close, mlir::OpResult result,
                            mlir::Value parentPhase,
                            mlir::Value parentAssumption, bool truePhaseOnly) {
    if (result.getResultNumber() != 0 || close.getOutputs().size() != 2 ||
        close.getSel() == parentPhase)
      return false;

    AlignmentQuery query{alignmentRevision,
                         result,
                         parentPhase,
                         parentAssumption,
                         {},
                         0,
                         truePhaseOnly ? AlignmentQueryKind::TruePhaseClose
                                       : AlignmentQueryKind::Close};
    return evaluateAlignment(query, [&] {
      GraphCardinalityAnalysis activation(graph, sharedState);
      return initializeNestedActivation(close.getSel(), parentPhase,
                                        parentAssumption, truePhaseOnly,
                                        activation) &&
             activation.isExactOne(result);
    });
  }

  bool isNestedGateCloseAligned(mlir::Value value, mlir::Value selector,
                                unsigned lane, mlir::Value parentPhase,
                                mlir::Value parentAssumption) {
    auto gate = dataflow::semantics::getGateCloseProjection(value);
    if (!gate)
      return false;
    auto closes =
        dataflow::semantics::gateClosesWhenSelected(*gate, selector, lane);
    if (!closes || !*closes)
      return false;
    auto stream = gate->getBeforeCond().getDefiningOp<dataflow::StreamOp>();
    if (!stream || gate->getBeforeCond() != stream.getPhase())
      return false;

    AlignmentQuery query{alignmentRevision,
                         value,
                         parentPhase,
                         parentAssumption,
                         selector,
                         lane,
                         AlignmentQueryKind::GateClose};
    return evaluateAlignment(query, [&] {
      GraphCardinalityAnalysis activation(graph, sharedState);
      return initializeNestedActivation(stream.getPhase(), parentPhase,
                                        parentAssumption,
                                        /*truePhaseOnly=*/true, activation) &&
             activation.isOneClosePhase(stream.getPhase()) &&
             activation.isPhaseAligned(gate->getBeforeValue(),
                                       stream.getPhase());
    });
  }

  bool isSiblingGateCloseAligned(mlir::Value value, mlir::Value selector,
                                 unsigned lane, mlir::Value parentPhase,
                                 mlir::Value parentAssumption) {
    // Phase-aligned sibling gates open and close on the same parent firings.
    auto gate = dataflow::semantics::getGateCloseProjection(value);
    auto selectorGate = selector.getDefiningOp<dataflow::GateOp>();
    if (!gate || lane != 0 || !selectorGate ||
        selector != selectorGate.getAfterCond() ||
        gate->getBeforeCond() != parentPhase ||
        selectorGate.getBeforeCond() != parentPhase)
      return false;

    AlignmentQuery query{alignmentRevision,
                         value,
                         parentPhase,
                         parentAssumption,
                         selector,
                         lane,
                         AlignmentQueryKind::SiblingGateClose};
    return evaluateAlignment(query, [&] {
      llvm::DenseSet<mlir::Value> gateVisited;
      if (!isAligned(gate->getBeforeValue(), parentPhase, parentAssumption,
                     /*truePhaseOnly=*/false, gateVisited))
        return false;
      llvm::DenseSet<mlir::Value> selectorVisited;
      return isAligned(selectorGate.getBeforeValue(), parentPhase,
                       parentAssumption, /*truePhaseOnly=*/false,
                       selectorVisited);
    });
  }

  bool isAligned(mlir::Value value, mlir::Value phase, mlir::Value assumption,
                 bool truePhaseOnly, llvm::DenseSet<mlir::Value> &visited) {
    if (value == assumption || alignedCarryAssumptions.contains(value))
      return true;
    if (value == phase)
      return !truePhaseOnly;
    auto cycleResult = [&] {
      return truePhaseOnly &&
             llvm::any_of(alignedCarryAssumptions,
                          [&](mlir::Value carryOutput) {
                            return causallyDependsOn(value, carryOutput);
                          });
    };
    if (!visited.insert(value).second)
      return cycleResult();
    auto eraseVisited = llvm::scope_exit([&] { visited.erase(value); });
    AlignmentQuery query{alignmentRevision,
                         value,
                         phase,
                         assumption,
                         {},
                         0,
                         truePhaseOnly ? AlignmentQueryKind::TruePhaseAligned
                                       : AlignmentQueryKind::Aligned};
    auto known = alignment.find(query);
    if (known != alignment.end())
      return known->second;
    if (!alignmentActive.insert(query).second)
      return cycleResult();
    auto eraseActive = llvm::scope_exit([&] { alignmentActive.erase(query); });
    bool result = [&]() -> bool {
      auto result = llvm::dyn_cast<mlir::OpResult>(value);
      mlir::Operation *def = result ? result.getOwner() : nullptr;
      if (!def)
        return false;
      if (truePhaseOnly)
        if (auto activation = dataflow::semantics::getCloseActivation(value))
          return isAligned(*activation, phase, assumption,
                           /*truePhaseOnly=*/true, visited);
      if (truePhaseOnly)
        if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(def);
            demux && isGraphStreamInput(demux.getInput()))
          if (auto activation = dataflow::semantics::getSelectorActivation(
                  demux.getSel(), demux.getOutputs().size())) {
            mlir::Value alignment = *activation;
            if (auto synchronization =
                    dataflow::semantics::getSelectorLaneSynchronization(
                        demux.getSel(), demux.getOutputs().size(),
                        result.getResultNumber()))
              alignment = *synchronization;
            return isAligned(alignment, phase, assumption,
                             /*truePhaseOnly=*/true, visited);
          }
      if (auto sync = llvm::dyn_cast<dataflow::SyncOp>(def)) {
        bool hasAlignedInput = false;
        for (mlir::Value input : sync.getInputs()) {
          if (isAligned(input, phase, assumption, truePhaseOnly, visited)) {
            hasAlignedInput = true;
            continue;
          }
          return false;
        }
        return hasAlignedInput;
      }
      if (auto stream = llvm::dyn_cast<dataflow::StreamOp>(def))
        return truePhaseOnly && value == stream.getIv() &&
               phase == stream.getPhase();
      if (auto invariant = llvm::dyn_cast<dataflow::InvariantOp>(def))
        return !truePhaseOnly && value == invariant.getOutput() &&
               invariant.getCond() == phase && isExactOne(invariant.getInit());
      if (auto carry = llvm::dyn_cast<dataflow::CarryOp>(def)) {
        if (truePhaseOnly || value != carry.getOutput() ||
            carry.getCond() != phase || !isExactOne(carry.getInit()))
          return false;
        bool inserted = insertAlignedCarryAssumption(carry.getOutput());
        if (!inserted)
          return true;
        bool aligned = isAligned(carry.getCarry(), phase, carry.getOutput(),
                                 /*truePhaseOnly=*/true, visited);
        eraseAlignedCarryAssumption(carry.getOutput());
        return aligned;
      }
      if (auto gate = llvm::dyn_cast<dataflow::GateOp>(def)) {
        if (!truePhaseOnly || gate.getBeforeCond() != phase ||
            (value != gate.getAfterCond() && value != gate.getAfterValue()))
          return false;
        return isAligned(gate.getBeforeValue(), phase, assumption,
                         /*truePhaseOnly=*/false, visited);
      }
      if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(def)) {
        if (result.getResultNumber() == 0 &&
            isNestedCloseAligned(demux, result, phase, assumption,
                                 truePhaseOnly))
          return true;
        if (!truePhaseOnly)
          return false;
        if (demux.getSel() != phase || result.getResultNumber() != 1)
          return false;
        return isAligned(demux.getInput(), phase, assumption,
                         /*truePhaseOnly=*/false, visited);
      }
      if (auto mux = llvm::dyn_cast<dataflow::MuxOp>(def)) {
        if (!isAligned(mux.getSel(), phase, assumption, truePhaseOnly, visited))
          return false;
        for (auto [lane, input] : llvm::enumerate(mux.getInputs())) {
          if (!availableWhenSelectedAndAligned(input, mux.getSel(), lane, phase,
                                               assumption, truePhaseOnly,
                                               visited))
            return false;
        }
        return true;
      }
      if (dataflow::semantics::isVectorBoundaryTruePhaseOutputPayload(value,
                                                                      phase))
        return truePhaseOnly;
      if (dataflow::semantics::isStatelessOneTokenVectorBoundary(def)) {
        if (result.getResultNumber() != 0 || def->getNumOperands() != 1)
          return false;
        return isAligned(def->getOperand(0), phase, assumption, truePhaseOnly,
                         visited);
      }
      if (llvm::isa<dataflow::StreamOp, dataflow::GateOp,
                    dataflow::ParallelizeOp, dataflow::PackOp,
                    dataflow::UnpackOp, dataflow::SerializeOp,
                    dataflow::DemuxOp>(def))
        return false;
      if (!dataflow::isCanonicalDataflowActor(def) &&
          !llvm::isa<mlir::memref::CastOp>(def))
        return false;

      bool hasRequiredOperand = false;
      for (mlir::Value operand : def->getOperands()) {
        if (isMemoryCapabilityType(operand.getType()))
          continue;
        if (isAligned(operand, phase, assumption, truePhaseOnly, visited)) {
          hasRequiredOperand = true;
          continue;
        }
        if (!truePhaseOnly)
          return false;
        if (!isAligned(operand, phase, assumption,
                       /*truePhaseOnly=*/false, visited))
          return false;
      }
      return hasRequiredOperand;
    }();
    alignment.try_emplace(query, result);
    return result;
  }

  void
  collectAlignedCarries(mlir::Value phase,
                        llvm::SmallVectorImpl<dataflow::CarryOp> &carries) {
    auto phaseCarries = graphIndex->carriesByPhase.find(phase);
    if (phaseCarries == graphIndex->carriesByPhase.end())
      return;
    for (dataflow::CarryOp carry : phaseCarries->second)
      if (isExactOne(carry.getInit()))
        carries.push_back(carry);
  }

  bool isCarrySystemAligned(mlir::Value phase) {
    auto known = alignedCarrySystems.find(phase);
    if (known != alignedCarrySystems.end())
      return known->second;
    if (!alignedCarrySystemsActive.insert(phase).second)
      return true;

    llvm::SmallVector<dataflow::CarryOp, 4> carries;
    collectAlignedCarries(phase, carries);
    for (dataflow::CarryOp carry : carries)
      insertAlignedCarryAssumption(carry.getOutput());

    bool aligned = llvm::all_of(carries, [&](dataflow::CarryOp carry) {
      llvm::DenseSet<mlir::Value> visited;
      return isAligned(carry.getCarry(), phase, carry.getOutput(),
                       /*truePhaseOnly=*/true, visited);
    });

    for (dataflow::CarryOp carry : carries)
      eraseAlignedCarryAssumption(carry.getOutput());
    alignedCarrySystemsActive.erase(phase);
    alignedCarrySystems.try_emplace(phase, aligned);
    return aligned;
  }

  bool isAlignedToPhase(mlir::Value value, mlir::Value phase,
                        bool truePhaseOnly) {
    if (!isCarrySystemAligned(phase))
      return false;
    llvm::SmallVector<dataflow::CarryOp, 4> carries;
    collectAlignedCarries(phase, carries);
    for (dataflow::CarryOp carry : carries)
      insertAlignedCarryAssumption(carry.getOutput());
    llvm::DenseSet<mlir::Value> visited;
    bool aligned = isAligned(value, phase, {}, truePhaseOnly, visited);
    for (dataflow::CarryOp carry : carries)
      eraseAlignedCarryAssumption(carry.getOutput());
    return aligned;
  }

  bool isTruePhaseAligned(mlir::Value value, mlir::Value phase) {
    return isAlignedToPhase(value, phase, /*truePhaseOnly=*/true);
  }

  bool isPhaseAligned(mlir::Value value, mlir::Value phase) {
    return isAlignedToPhase(value, phase, /*truePhaseOnly=*/false);
  }

  bool isOneClosePhase(mlir::Value value) {
    auto known = oneClosePhase.find(value);
    if (known != oneClosePhase.end())
      return known->second;
    if (!oneCloseActive.insert(value).second)
      return false;
    bool result = false;
    if (auto stream = value.getDefiningOp<dataflow::StreamOp>()) {
      result = value == stream.getPhase() && isExactOne(stream.getInit()) &&
               isExactOne(stream.getLimit()) && isExactOne(stream.getStep());
    } else if (auto gate = value.getDefiningOp<dataflow::GateOp>()) {
      result = value == gate.getAfterCond() &&
               dataflow::semantics::gateAlwaysCloses(gate) &&
               isOneClosePhase(gate.getBeforeCond()) &&
               isPhaseAligned(gate.getBeforeValue(), gate.getBeforeCond());
    } else if (auto output = dataflow::semantics::getVectorBoundaryOutputPhase(
                   value.getDefiningOp());
               output && value == *output) {
      auto input = dataflow::semantics::getVectorBoundaryInputPhase(
          value.getDefiningOp());
      result = input && isOneClosePhase(*input) &&
               llvm::all_of(
                   dataflow::semantics::getVectorBoundaryTruePhaseInputPayloads(
                       value.getDefiningOp()),
                   [&](mlir::Value payload) {
                     return isTruePhaseAligned(payload, *input);
                   });
    } else if (auto sync = value.getDefiningOp<dataflow::SyncOp>()) {
      unsigned closeInputs = 0;
      result = true;
      for (mlir::Value input : sync.getInputs()) {
        if (isOneClosePhase(input)) {
          ++closeInputs;
          continue;
        }
        if (!isExactOne(input)) {
          result = false;
          break;
        }
      }
      result &= closeInputs == 1;
    } else {
      llvm::SmallVector<dataflow::CarryOp, 4> carries;
      collectAlignedCarries(value, carries);
      result = !carries.empty() && isCarrySystemAligned(value);
    }
    oneCloseActive.erase(value);
    oneClosePhase.try_emplace(value, result);
    return result;
  }

  bool computeExactOne(mlir::Value value) {
    if (exactOneAssumptions.contains(value))
      return true;
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value)) {
      if (argument.getOwner() != &graph.getBody().front())
        return false;
      if (argument.getArgNumber() == 0)
        return true;
      return graph.getInputPortKind(argument.getArgNumber() - 1) !=
             dataflow::GraphPortKind::Stream;
    }

    auto result = llvm::dyn_cast<mlir::OpResult>(value);
    mlir::Operation *def = result ? result.getOwner() : nullptr;
    if (!def)
      return false;
    if (auto constant = llvm::dyn_cast<dataflow::ConstantOp>(def))
      return isExactOne(constant.getCtrl());
    if (llvm::isa<dataflow::SyncOp>(def))
      return allOperandsExact(def);
    if (auto mux = llvm::dyn_cast<dataflow::MuxOp>(def)) {
      if (!isExactOne(mux.getSel()))
        return false;
      for (auto [lane, input] : llvm::enumerate(mux.getInputs()))
        if (!availableWhenSelected(input, mux.getSel(), lane))
          return false;
      return true;
    }
    if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(def)) {
      unsigned lane = result.getResultNumber();
      if (auto selected = getKnownUnsigned(demux.getSel())) {
        if (isGraphStreamInput(demux.getInput()))
          return *selected == lane && isExactOne(demux.getSel());
        return *selected == lane && isExactOne(demux.getInput());
      }
      if (isGraphStreamInput(demux.getInput())) {
        auto activation = dataflow::semantics::getSelectorActivation(
            demux.getSel(), demux.getOutputs().size());
        if (!activation || !isExactOne(*activation))
          return false;
        if (dataflow::semantics::selectorSelectsEveryLaneOncePerActivation(
                demux.getSel(), demux.getOutputs().size()))
          return true;
        return dataflow::semantics::selectorSelectsLaneOncePerActivation(
            demux.getSel(), demux.getOutputs().size(), lane);
      }
      return lane == 0 && isOneClosePhase(demux.getSel()) &&
             isPhaseAligned(demux.getInput(), demux.getSel());
    }
    if (dataflow::semantics::isStatelessOneTokenVectorBoundary(def))
      return result.getResultNumber() == 0 && def->getNumOperands() == 1 &&
             isExactOne(def->getOperand(0));
    if (llvm::isa<dataflow::StreamOp, dataflow::CarryOp, dataflow::InvariantOp,
                  dataflow::GateOp, dataflow::ParallelizeOp,
                  dataflow::SerializeOp>(def))
      return false;
    if (llvm::isa<mlir::memref::AllocOp>(def))
      return true;
    if (auto cast = llvm::dyn_cast<mlir::memref::CastOp>(def))
      return isExactOne(cast.getSource());
    if (dataflow::isCanonicalDataflowActor(def))
      return allOperandsExact(def);
    return false;
  }
};

} // namespace

llvm::Error dataflow::validateFinalizedGraph(GraphOp graph) {
  if (!graph || graph.isExternal())
    return graphError("finalized graph must have a body");
  mlir::Block &entry = graph.getBody().front();
  auto ret = llvm::dyn_cast<GraphReturnOp>(entry.getTerminator());
  if (!ret)
    return graphError("finalized graph is missing dataflow.graph.return");

  llvm::Error structuralError = llvm::Error::success();
  graph.getBody().walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (structuralError || llvm::isa<GraphReturnOp>(op))
      return mlir::WalkResult::interrupt();
    if (llvm::any_of(op->getResultTypes(), containsChannelType)) {
      structuralError = graphError(
          llvm::Twine("finalized graph contains channel-typed result produced "
                      "by '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
    if (llvm::any_of(op->getOperandTypes(), containsChannelType)) {
      structuralError = graphError(
          llvm::Twine("finalized graph contains channel-typed operand consumed "
                      "by '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
    if (op->getName().getDialectNamespace() == "scf" ||
        op->getName().getDialectNamespace() == "cf" ||
        op->getNumRegions() != 0 || op->getNumSuccessors() != 0) {
      structuralError = graphError(
          llvm::Twine(
              "finalized graph contains residual structured operation '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
    if (llvm::isa<CarryOp, MuxOp, DemuxOp, GateOp, InvariantOp>(op) &&
        (llvm::any_of(op->getOperandTypes(), isMemoryCapabilityType) ||
         llvm::any_of(op->getResultTypes(), isMemoryCapabilityType))) {
      structuralError = graphError(
          llvm::Twine("finalized graph routes memory capability through '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
    if (llvm::isa<mlir::memref::GetGlobalOp>(op)) {
      structuralError = graphError(
          "finalized graph contains forbidden memory root 'memref.get_global'");
      return mlir::WalkResult::interrupt();
    }
    if (llvm::isa<mlir::memref::LoadOp, mlir::memref::StoreOp>(op) ||
        isResidualLLVMMemoryOperation(op)) {
      structuralError = graphError(
          llvm::Twine("finalized graph contains residual memory operation '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
    if (auto alloc = llvm::dyn_cast<mlir::memref::AllocOp>(op)) {
      if (!llvm::all_of(alloc.getDynamicSizes(), [&](mlir::Value extent) {
            return isLaunchAvailableValueInput(graph, extent);
          })) {
        structuralError = graphError(
            "memref.alloc dynamic extent must be a graph value input");
        return mlir::WalkResult::interrupt();
      }
    } else if (op->getDialect() &&
               op->getDialect()->getNamespace() == "memref" &&
               !isSupportedMemoryView(op)) {
      structuralError = graphError(
          llvm::Twine("finalized graph contains unsupported memory capability "
                      "operation '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
    if (llvm::isa<mlir::UnrealizedConversionCastOp>(op)) {
      structuralError =
          graphError("finalized graph contains forbidden operation "
                     "'builtin.unrealized_conversion_cast'");
      return mlir::WalkResult::interrupt();
    }
    if (hasRawPointerUse(op)) {
      structuralError = graphError(
          llvm::Twine("finalized graph contains residual pointer operation '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
    if (llvm::any_of(op->getResultTypes(), isMemoryCapabilityType) &&
        !llvm::isa<mlir::memref::AllocOp>(op) && !isSupportedMemoryView(op)) {
      structuralError = graphError(
          llvm::Twine("finalized graph contains unsupported memory capability "
                      "producer '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
    if (!dataflow::isCanonicalDataflowActor(op) &&
        !llvm::isa<mlir::memref::AllocOp, mlir::memref::CastOp>(op)) {
      structuralError = graphError(
          llvm::Twine("finalized graph contains unregistered actor '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (structuralError)
    return structuralError;

  bool hasRealWork =
      llvm::any_of(entry.without_terminator(), [&](mlir::Operation &op) {
        return !llvm::isa<mlir::memref::AllocOp, mlir::memref::CastOp>(op);
      });
  if (hasRealWork && llvm::is_contained(ret.getComplete(), graph.getStart()))
    return graphError(
        "nontrivial graph uses raw start as a retirement completion witness");

  GraphCardinalityAnalysis cardinality(graph);
  for (auto [index, value] : llvm::enumerate(ret.getValues()))
    if (!cardinality.isExactOne(value))
      return graphError(llvm::Twine("graph @") + graph.getSymName() +
                        " value output #" + llvm::Twine(index) +
                        " is not statically exact-one");
  for (auto [index, stream] : llvm::enumerate(ret.getStreams()))
    if (!cardinality.hasProvenStreamCommit(stream))
      return graphError(llvm::Twine("graph @") + graph.getSymName() +
                        " stream output #" + llvm::Twine(index) +
                        " has no statically proven close/commit");
  for (auto [index, witness] : llvm::enumerate(ret.getComplete()))
    if (!cardinality.isExactOne(witness))
      return graphError(llvm::Twine("graph @") + graph.getSymName() +
                        " completion witness #" + llvm::Twine(index) +
                        " is not statically one-shot");

  for (auto [index, value] : llvm::enumerate(ret.getValues()))
    if (!isCovered(value, ret.getComplete()))
      return graphError(llvm::Twine("retirement frontier does not causally ") +
                        "cover value output #" + llvm::Twine(index));

  for (auto [index, stream] : llvm::enumerate(ret.getStreams())) {
    llvm::SmallVector<mlir::Value, 2> closeSignals;
    cardinality.collectStreamCloseSignals(stream, closeSignals);
    bool covered = closeSignals.empty()
                       ? isCovered(stream, ret.getComplete())
                       : llvm::all_of(closeSignals, [&](mlir::Value signal) {
                           return coversFalseClose(signal, ret.getComplete());
                         });
    if (!covered)
      return graphError(llvm::Twine("retirement frontier does not causally ") +
                        "cover stream output #" + llvm::Twine(index));
  }

  for (auto [index, memory] : llvm::enumerate(ret.getMemories())) {
    if (isFreshMemoryRoot(memory) &&
        !mlir::isa<mlir::MemRefType, mlir::UnrankedMemRefType>(
            memory.getType()))
      return graphError("fresh memory export must use a memref result");
    if (isProtocolEstablishedMemory(graph, memory)) {
      if (!isCovered(graph.getStart(), ret.getComplete()))
        return graphError(
            llvm::Twine(
                "retirement frontier does not cover establishment of ") +
            "memory output #" + llvm::Twine(index));
      continue;
    }
    if (!isCovered(memory, ret.getComplete()))
      return graphError(
          llvm::Twine("retirement frontier does not causally cover memory ") +
          "output #" + llvm::Twine(index));
  }

  for (mlir::Operation &op : entry.without_terminator()) {
    if (auto gate = llvm::dyn_cast<dataflow::GateOp>(op)) {
      bool covered = coversFalseClose(gate.getAfterCond(), ret.getComplete());
      if (!covered)
        return graphError("retirement frontier does not cover close/reset of "
                          "'dataflow.gate'");
      continue;
    }
    mlir::Value closeSignal = statefulCloseSignal(&op);
    if (!closeSignal)
      continue;
    llvm::SmallVector<mlir::Value, 2> sourceCloses;
    cardinality.collectStreamCloseSignals(closeSignal, sourceCloses);
    if (sourceCloses.empty())
      sourceCloses.push_back(closeSignal);
    if (!llvm::all_of(sourceCloses, [&](mlir::Value signal) {
          return coversFalseClose(signal, ret.getComplete());
        }))
      return graphError(
          llvm::Twine("retirement frontier does not cover close/reset of '") +
          op.getName().getStringRef() + "'");
  }

  llvm::Error effectError = llvm::Error::success();
  graph.getBody().walk([&](mlir::Operation *op) {
    if (effectError)
      return mlir::WalkResult::interrupt();
    if (mlir::Value done = semantics::getMemoryActorDone(op)) {
      if (!isCovered(done, ret.getComplete()))
        effectError = graphError(
            llvm::Twine("retirement frontier does not causally cover ") +
            op->getName().getStringRef() + " done");
      return effectError ? mlir::WalkResult::interrupt()
                         : mlir::WalkResult::advance();
    }
    if (auto call = llvm::dyn_cast<mlir::LLVM::CallOp>(op)) {
      bool covered = !call.getResults().empty() &&
                     llvm::any_of(call.getResults(), [&](mlir::Value result) {
                       return isCovered(result, ret.getComplete());
                     });
      if (covered)
        return mlir::WalkResult::advance();
    }
    if (llvm::isa<GraphReturnOp, mlir::memref::AllocOp>(op) || mlir::isPure(op))
      return mlir::WalkResult::advance();
    if (hasObservableEffect(op)) {
      effectError = graphError(
          llvm::Twine(
              "finalized graph contains unsupported effect operation '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return effectError;
}
