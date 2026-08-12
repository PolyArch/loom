#include "Dataflow/IR/DataflowGraphValidation.h"

#include "DataflowGraphCausality.h"

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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>

namespace {

template <typename... Ts> unsigned denseMapKeyHash(const Ts &...values) {
  using Key = std::tuple<Ts...>;
  return llvm::DenseMapInfo<Key>::getHashValue(Key(values...));
}

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

bool haveEquivalentCorrespondence(mlir::Value lhs, mlir::Value rhs) {
  return dataflow::haveEquivalentDeterministicComputeCorrespondence(lhs, rhs);
}

bool haveEquivalentSelectorCorrespondence(mlir::Value lhs, mlir::Value rhs) {
  return haveEquivalentCorrespondence(lhs, rhs) ||
         dataflow::semantics::haveEquivalentSynchronizedSelectionCorrespondence(
             lhs, rhs);
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

using SelectorLanes = llvm::DenseMap<mlir::Value, unsigned>;

SelectorLanes::iterator findEquivalentSelector(SelectorLanes &lanes,
                                               mlir::Value selector) {
  return llvm::find_if(lanes, [&](const auto &entry) {
    return haveEquivalentSelectorCorrespondence(entry.first, selector);
  });
}

/// A canonical memory actor publishes its remaining results together with
/// `done` as one retirement event.
bool isRetirementPublication(mlir::Value witness, mlir::Value prerequisite) {
  mlir::Operation *def = witness.getDefiningOp();
  return def && witness != prerequisite &&
         witness == dataflow::semantics::getMemoryActorDone(def) &&
         prerequisite.getDefiningOp() == def;
}

bool isCovered(dataflow::detail::GraphCausalDependencyCache &causalDependencies,
               mlir::Value prerequisite, mlir::ValueRange completion) {
  return llvm::any_of(completion, [&](mlir::Value witness) {
    return causalDependencies.dependsOn(witness, prerequisite) ||
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
      auto it = findEquivalentSelector(selectorLanes, demux.getSel());
      if (it != selectorLanes.end()) {
        if (it->second != result.getResultNumber())
          return false;
      } else {
        selectorLanes.try_emplace(demux.getSel(), result.getResultNumber());
        insertedSelector = demux.getSel();
      }
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
    auto it = findEquivalentSelector(selectorLanes, selector);
    bool inserted = it == selectorLanes.end();
    if (!inserted) {
      if (it->second != lane)
        return false;
    } else {
      selectorLanes.try_emplace(selector, lane);
    }
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
        if (haveEquivalentCorrespondence(phase, closeSignal) ||
            dataflow::semantics::haveEquivalentOrderedCardinality(phase,
                                                                  closeSignal))
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

using CardinalityAssumptionSetId = unsigned;

struct AlignmentQuery {
  CardinalityAssumptionSetId exactOneAssumptions;
  CardinalityAssumptionSetId alignedCarryAssumptions;
  mlir::Value value;
  mlir::Value parentPhase;
  mlir::Value parentAssumption;
  mlir::Value selector;
  unsigned lane;
  AlignmentQueryKind kind;

  bool operator==(const AlignmentQuery &other) const {
    return exactOneAssumptions == other.exactOneAssumptions &&
           alignedCarryAssumptions == other.alignedCarryAssumptions &&
           value == other.value && parentPhase == other.parentPhase &&
           parentAssumption == other.parentAssumption &&
           selector == other.selector && lane == other.lane &&
           kind == other.kind;
  }
};

struct AlignmentQueryInfo {
  static AlignmentQuery getEmptyKey() {
    return {std::numeric_limits<CardinalityAssumptionSetId>::max(),
            std::numeric_limits<CardinalityAssumptionSetId>::max(),
            {},
            {},
            {},
            {},
            0,
            AlignmentQueryKind::Close};
  }

  static AlignmentQuery getTombstoneKey() {
    return {std::numeric_limits<CardinalityAssumptionSetId>::max() - 1,
            std::numeric_limits<CardinalityAssumptionSetId>::max() - 1,
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
    return denseMapKeyHash(query.exactOneAssumptions,
                           query.alignedCarryAssumptions, opaque(query.value),
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

  void collectCarries(mlir::Value phase,
                      llvm::SmallVectorImpl<dataflow::CarryOp> &result) const {
    for (const auto &entry : carriesByPhase)
      if (haveEquivalentCorrespondence(entry.first, phase))
        result.append(entry.second);
  }

  void
  collectActivationInputs(mlir::Value phase,
                          llvm::SmallVectorImpl<mlir::Value> &result) const {
    for (const auto &entry : activationInputsByPhase)
      if (haveEquivalentCorrespondence(entry.first, phase))
        result.append(entry.second);
  }

  void collectDemuxes(mlir::Value selector,
                      llvm::SmallVectorImpl<dataflow::DemuxOp> &result) const {
    for (const auto &entry : demuxesBySelector)
      if (haveEquivalentSelectorCorrespondence(entry.first, selector))
        result.append(entry.second);
  }

  llvm::DenseMap<mlir::Value, llvm::SmallVector<dataflow::CarryOp, 4>>
      carriesByPhase;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value, 4>>
      activationInputsByPhase;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<dataflow::DemuxOp, 4>>
      demuxesBySelector;
};

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
    return denseMapKeyHash(
        query.exactOneAssumptions, query.alignedCarryAssumptions,
        opaque(query.result), opaque(query.parentPhase),
        opaque(query.parentAssumption), opaque(query.selector), query.lane,
        static_cast<unsigned>(query.truePhaseOnly));
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
  llvm::DenseMap<AlignmentQuery, bool, AlignmentQueryInfo> alignment;
  llvm::DenseSet<AlignmentQuery, AlignmentQueryInfo> alignmentActive;
  llvm::DenseMap<std::pair<CardinalityAssumptionSetId, mlir::Value>, bool>
      exactOne;
  llvm::DenseSet<std::pair<CardinalityAssumptionSetId, mlir::Value>>
      exactOneActive;

private:
  static std::uint64_t
  hashAssumptions(llvm::ArrayRef<mlir::Value> assumptions) {
    unsigned hash = denseMapKeyHash(assumptions.size());
    for (mlir::Value assumption : assumptions)
      hash = denseMapKeyHash(hash, assumption.getAsOpaquePointer());
    return hash;
  }

  llvm::SmallVector<llvm::SmallVector<mlir::Value, 8>, 8> assumptionSets;
  llvm::DenseMap<std::uint64_t,
                 llvm::SmallVector<CardinalityAssumptionSetId, 1>>
      assumptionBuckets;
};

class GraphCardinalityAnalysis {
public:
  GraphCardinalityAnalysis(
      dataflow::GraphOp graph,
      dataflow::detail::GraphCausalDependencyCache &causalDependencies)
      : graph(graph),
        sharedState(std::make_shared<CardinalitySharedState>(graph)),
        graphIndex(sharedState->graphIndex),
        causalDependencies(causalDependencies) {}

  bool isExactOne(mlir::Value value) {
    auto query = std::make_pair(internedExactOneAssumptions(), value);
    auto known = sharedState->exactOne.find(query);
    if (known != sharedState->exactOne.end())
      return known->second;
    if (!sharedState->exactOneActive.insert(query).second)
      return false;
    bool result = computeExactOne(value);
    sharedState->exactOneActive.erase(query);
    sharedState->exactOne.try_emplace(query, result);
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
  GraphCardinalityAnalysis(
      dataflow::GraphOp graph,
      std::shared_ptr<CardinalitySharedState> sharedState,
      dataflow::detail::GraphCausalDependencyCache &causalDependencies)
      : graph(graph), sharedState(std::move(sharedState)),
        graphIndex(this->sharedState->graphIndex),
        causalDependencies(causalDependencies) {}

  dataflow::GraphOp graph;
  std::shared_ptr<CardinalitySharedState> sharedState;
  std::shared_ptr<CardinalityGraphIndex> graphIndex;
  dataflow::detail::GraphCausalDependencyCache &causalDependencies;
  llvm::DenseMap<mlir::Value, bool> oneClosePhase;
  llvm::DenseSet<mlir::Value> oneCloseActive;
  llvm::DenseSet<mlir::Value> alignedCarryAssumptions;
  std::optional<CardinalityAssumptionSetId> alignedCarryAssumptionSet = 0;
  llvm::DenseMap<mlir::Value, bool> alignedCarrySystems;
  llvm::DenseSet<mlir::Value> alignedCarrySystemsActive;
  llvm::DenseSet<mlir::Value> exactOneAssumptions;
  std::optional<CardinalityAssumptionSetId> exactOneAssumptionSet = 0;

  CardinalityAssumptionSetId internedAlignedCarryAssumptions() {
    if (!alignedCarryAssumptionSet)
      alignedCarryAssumptionSet =
          sharedState->internAssumptions(alignedCarryAssumptions);
    return *alignedCarryAssumptionSet;
  }

  CardinalityAssumptionSetId internedExactOneAssumptions() {
    if (!exactOneAssumptionSet)
      exactOneAssumptionSet =
          sharedState->internAssumptions(exactOneAssumptions);
    return *exactOneAssumptionSet;
  }

  void insertExactOneAssumption(mlir::Value value) {
    if (exactOneAssumptions.insert(value).second)
      exactOneAssumptionSet.reset();
  }

  bool insertAlignedCarryAssumption(mlir::Value value) {
    bool inserted = alignedCarryAssumptions.insert(value).second;
    if (inserted)
      alignedCarryAssumptionSet.reset();
    return inserted;
  }

  void eraseAlignedCarryAssumption(mlir::Value value) {
    if (alignedCarryAssumptions.erase(value))
      alignedCarryAssumptionSet.reset();
  }

  void inheritAssumptions(const GraphCardinalityAnalysis &parent) {
    for (mlir::Value value : parent.exactOneAssumptions)
      insertExactOneAssumption(value);
    for (mlir::Value value : parent.alignedCarryAssumptions)
      insertAlignedCarryAssumption(value);
  }

  template <typename Compute>
  bool evaluateAlignment(const AlignmentQuery &query, Compute &&compute) {
    auto known = sharedState->alignment.find(query);
    if (known != sharedState->alignment.end())
      return known->second;
    if (!sharedState->alignmentActive.insert(query).second)
      return false;
    auto eraseActive =
        llvm::scope_exit([&] { sharedState->alignmentActive.erase(query); });
    bool result = compute();
    sharedState->alignment.try_emplace(query, result);
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
    GraphCardinalityAnalysis branch(graph, sharedState, causalDependencies);
    branch.inheritAssumptions(*this);
    llvm::SmallVector<dataflow::DemuxOp, 4> demuxes;
    graphIndex->collectDemuxes(selector, demuxes);
    for (dataflow::DemuxOp demux : demuxes) {
      if (lane >= demux.getOutputs().size() || !isExactOne(demux.getInput()))
        continue;
      branch.insertExactOneAssumption(demux.getOutputs()[lane]);
    }
    return branch.isExactOne(value);
  }

  bool isExactOneWhenSelectedAndAligned(mlir::Value value, mlir::Value selector,
                                        unsigned lane, mlir::Value phase,
                                        mlir::Value assumption,
                                        bool truePhaseOnly) {
    GraphCardinalityAnalysis branch(graph, sharedState, causalDependencies);
    branch.inheritAssumptions(*this);
    llvm::SmallVector<dataflow::DemuxOp, 4> demuxes;
    graphIndex->collectDemuxes(selector, demuxes);
    for (dataflow::DemuxOp demux : demuxes) {
      if (lane >= demux.getOutputs().size())
        continue;
      llvm::DenseSet<mlir::Value> visited;
      if (!isAligned(demux.getInput(), phase, assumption, truePhaseOnly,
                     visited))
        continue;
      branch.insertExactOneAssumption(demux.getOutputs()[lane]);
    }
    return branch.isExactOne(value);
  }

  bool computeAvailableWhenSelected(mlir::Value value, mlir::Value selector,
                                    unsigned lane,
                                    llvm::DenseSet<mlir::Value> &visited) {
    if (auto activation = dataflow::semantics::getSelectiveRouterLeafActivation(
            value, selector, lane))
      return isExactOne(*activation);
    auto result = llvm::dyn_cast<mlir::OpResult>(value);
    mlir::Operation *def = result ? result.getOwner() : nullptr;
    if (!def)
      return false;
    if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(def)) {
      if (haveEquivalentSelectorCorrespondence(demux.getSel(), selector))
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
    AlignmentQuery query{internedExactOneAssumptions(),
                         internedAlignedCarryAssumptions(),
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

  bool isExactOnePerSelectorActivation(mlir::Value value, mlir::Value selector,
                                       unsigned arity) {
    if (isExactOne(value))
      return true;
    auto activation =
        dataflow::semantics::getSelectorActivation(selector, arity);
    auto result = activation ? llvm::dyn_cast<mlir::OpResult>(*activation)
                             : mlir::OpResult{};
    auto projection =
        result ? llvm::dyn_cast<dataflow::DemuxOp>(result.getOwner()) : nullptr;
    if (!projection || projection.getOutputs().size() != 2 ||
        result.getResultNumber() != 1)
      return false;
    llvm::DenseSet<mlir::Value> visited;
    return isAligned(value, projection.getSel(), *activation,
                     /*truePhaseOnly=*/true, visited);
  }

  bool computeAvailableWhenSelectedAndAligned(
      mlir::Value value, mlir::Value selector, unsigned lane, mlir::Value phase,
      mlir::Value assumption, bool truePhaseOnly,
      llvm::DenseSet<mlir::Value> &visited) {
    if (auto event = dataflow::semantics::getStreamPublicationEvent(value))
      return availableWhenSelectedAndAligned(
          *event, selector, lane, phase, assumption, truePhaseOnly, visited);
    if (auto synchronization =
            dataflow::semantics::getSelectiveRouterLeafSynchronization(
                value, selector, lane))
      return isAligned(*synchronization, phase, assumption, truePhaseOnly,
                       visited);
    if (auto activation = dataflow::semantics::getSelectiveRouterLeafActivation(
            value, selector, lane))
      return isAligned(*activation, phase, assumption, truePhaseOnly, visited);
    auto result = llvm::dyn_cast<mlir::OpResult>(value);
    mlir::Operation *def = result ? result.getOwner() : nullptr;
    if (!def)
      return false;
    if (truePhaseOnly &&
        (isSiblingGateCloseAligned(value, selector, lane, phase, assumption) ||
         isNestedGateCloseAligned(value, selector, lane, phase, assumption)))
      return true;
    if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(def)) {
      if (haveEquivalentSelectorCorrespondence(demux.getSel(), selector) &&
          result.getResultNumber() == lane) {
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
    const bool laneSelectedOnce =
        dataflow::semantics::selectorSelectsLaneOncePerActivation(selector, 2,
                                                                  lane);
    for (mlir::Value operand : def->getOperands()) {
      if (isMemoryCapabilityType(operand.getType()))
        continue;
      if (availableWhenSelectedAndAligned(operand, selector, lane, phase,
                                          assumption, truePhaseOnly, visited)) {
        hasSelectedOperand = true;
        continue;
      }
      if (laneSelectedOnce &&
          isExactOnePerSelectorActivation(operand, selector, 2))
        continue;
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
        internedExactOneAssumptions(),
        internedAlignedCarryAssumptions(),
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
    activation.inheritAssumptions(*this);
    auto assumeExact = [&](mlir::Value value) {
      llvm::DenseSet<mlir::Value> visited;
      if (!availableWhenSelectedAndAligned(value, selector, lane, parentPhase,
                                           parentAssumption, truePhaseOnly,
                                           visited))
        return false;
      activation.insertExactOneAssumption(value);
      return true;
    };

    if (auto stream = childPhase.getDefiningOp<dataflow::StreamOp>()) {
      if (childPhase != stream.getPhase() || !assumeExact(stream.getInit()) ||
          !assumeExact(stream.getLimit()) || !assumeExact(stream.getStep()))
        return false;
    } else {
      llvm::SmallVector<dataflow::CarryOp, 4> carries;
      graphIndex->collectCarries(childPhase, carries);
      if (carries.empty())
        return false;
    }

    llvm::SmallVector<mlir::Value, 4> inputs;
    graphIndex->collectActivationInputs(childPhase, inputs);
    for (mlir::Value input : inputs)
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
        haveEquivalentCorrespondence(close.getSel(), selector) ||
        haveEquivalentCorrespondence(close.getSel(), parentPhase))
      return false;

    AlignmentQuery query{internedExactOneAssumptions(),
                         internedAlignedCarryAssumptions(),
                         result,
                         parentPhase,
                         parentAssumption,
                         selector,
                         lane,
                         truePhaseOnly
                             ? AlignmentQueryKind::SelectedTruePhaseClose
                             : AlignmentQueryKind::SelectedClose};
    return evaluateAlignment(query, [&] {
      SelectedNestedQuery sharedQuery{internedExactOneAssumptions(),
                                      internedAlignedCarryAssumptions(),
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

      GraphCardinalityAnalysis activation(graph, sharedState,
                                          causalDependencies);
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
    activation.inheritAssumptions(*this);
    // Parent-aligned inputs are exact-one within each child activation.
    auto assumeExact = [&](mlir::Value value) {
      llvm::DenseSet<mlir::Value> visited;
      if (!isAligned(value, parentPhase, parentAssumption, truePhaseOnly,
                     visited))
        return false;
      activation.insertExactOneAssumption(value);
      return true;
    };

    if (auto stream = childPhase.getDefiningOp<dataflow::StreamOp>()) {
      if (childPhase != stream.getPhase() || !assumeExact(stream.getInit()) ||
          !assumeExact(stream.getLimit()) || !assumeExact(stream.getStep()))
        return false;
    } else {
      llvm::SmallVector<dataflow::CarryOp, 4> carries;
      graphIndex->collectCarries(childPhase, carries);
      if (carries.empty())
        return false;
    }

    llvm::SmallVector<mlir::Value, 4> inputs;
    graphIndex->collectActivationInputs(childPhase, inputs);
    for (mlir::Value input : inputs)
      if (!assumeExact(input))
        return false;
    return true;
  }

  bool isNestedCloseAligned(dataflow::DemuxOp close, mlir::OpResult result,
                            mlir::Value parentPhase,
                            mlir::Value parentAssumption, bool truePhaseOnly) {
    if (result.getResultNumber() != 0 || close.getOutputs().size() != 2 ||
        haveEquivalentCorrespondence(close.getSel(), parentPhase))
      return false;

    AlignmentQuery query{internedExactOneAssumptions(),
                         internedAlignedCarryAssumptions(),
                         result,
                         parentPhase,
                         parentAssumption,
                         {},
                         0,
                         truePhaseOnly ? AlignmentQueryKind::TruePhaseClose
                                       : AlignmentQueryKind::Close};
    return evaluateAlignment(query, [&] {
      GraphCardinalityAnalysis activation(graph, sharedState,
                                          causalDependencies);
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

    AlignmentQuery query{internedExactOneAssumptions(),
                         internedAlignedCarryAssumptions(),
                         value,
                         parentPhase,
                         parentAssumption,
                         selector,
                         lane,
                         AlignmentQueryKind::GateClose};
    return evaluateAlignment(query, [&] {
      GraphCardinalityAnalysis activation(graph, sharedState,
                                          causalDependencies);
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
        !haveEquivalentCorrespondence(selector, selectorGate.getAfterCond()) ||
        !haveEquivalentCorrespondence(gate->getBeforeCond(), parentPhase) ||
        !haveEquivalentCorrespondence(selectorGate.getBeforeCond(),
                                      parentPhase))
      return false;

    AlignmentQuery query{internedExactOneAssumptions(),
                         internedAlignedCarryAssumptions(),
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
    if (haveEquivalentCorrespondence(value, phase))
      return !truePhaseOnly;
    auto cycleResult = [&] {
      return truePhaseOnly &&
             llvm::any_of(
                 alignedCarryAssumptions, [&](mlir::Value carryOutput) {
                   return causalDependencies.dependsOn(value, carryOutput);
                 });
    };
    if (!visited.insert(value).second)
      return cycleResult();
    auto eraseVisited = llvm::scope_exit([&] { visited.erase(value); });
    AlignmentQuery query{internedExactOneAssumptions(),
                         internedAlignedCarryAssumptions(),
                         value,
                         phase,
                         assumption,
                         {},
                         0,
                         truePhaseOnly ? AlignmentQueryKind::TruePhaseAligned
                                       : AlignmentQueryKind::Aligned};
    auto known = sharedState->alignment.find(query);
    if (known != sharedState->alignment.end())
      return known->second;
    if (!sharedState->alignmentActive.insert(query).second)
      return cycleResult();
    auto eraseActive =
        llvm::scope_exit([&] { sharedState->alignmentActive.erase(query); });
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
        if (auto activation =
                dataflow::semantics::getSelectiveRouterLeafActivation(value))
          return isAligned(*activation, phase, assumption,
                           /*truePhaseOnly=*/true, visited);
      if (truePhaseOnly)
        if (auto synchronization =
                dataflow::semantics::getSelectiveRouterLeafSynchronization(
                    value))
          return isAligned(*synchronization, phase, assumption,
                           /*truePhaseOnly=*/true, visited);
      if (truePhaseOnly)
        if (auto event = dataflow::semantics::getStreamActivityEvent(value))
          return isAligned(*event, phase, assumption,
                           /*truePhaseOnly=*/true, visited);
      if (truePhaseOnly)
        if (auto event = dataflow::semantics::getStreamPublicationEvent(value))
          return isAligned(*event, phase, assumption,
                           /*truePhaseOnly=*/true, visited);
      if (truePhaseOnly)
        if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(def);
            demux && isGraphStreamInput(demux.getInput())) {
          if (auto selected = getKnownUnsigned(demux.getSel());
              selected && *selected == result.getResultNumber() &&
              isAligned(demux.getSel(), phase, assumption,
                        /*truePhaseOnly=*/true, visited))
            return true;
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
               haveEquivalentCorrespondence(invariant.getCond(), phase) &&
               isExactOne(invariant.getInit());
      if (auto carry = llvm::dyn_cast<dataflow::CarryOp>(def)) {
        if (truePhaseOnly || value != carry.getOutput() ||
            !haveEquivalentCorrespondence(carry.getCond(), phase) ||
            !isExactOne(carry.getInit()))
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
        if (!truePhaseOnly ||
            !haveEquivalentCorrespondence(gate.getBeforeCond(), phase) ||
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
        if (auto activation =
                dataflow::semantics::getSelectorLaneEventActivation(
                    demux.getSel(), demux.getOutputs().size(),
                    result.getResultNumber(), demux.getInput()))
          return isAligned(*activation, phase, assumption,
                           /*truePhaseOnly=*/true, visited);
        if (!haveEquivalentCorrespondence(demux.getSel(), phase) ||
            result.getResultNumber() != 1)
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
      if (truePhaseOnly)
        if (auto ownPhase =
                dataflow::semantics::getVectorBoundaryOutputPhase(def);
            ownPhase &&
            dataflow::semantics::isVectorBoundaryTruePhaseOutputPayload(
                value, *ownPhase) &&
            dataflow::semantics::haveEquivalentOrderedCardinality(*ownPhase,
                                                                  phase))
          return true;
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
    sharedState->alignment.try_emplace(query, result);
    return result;
  }

  void
  collectAlignedCarries(mlir::Value phase,
                        llvm::SmallVectorImpl<dataflow::CarryOp> &carries) {
    llvm::SmallVector<dataflow::CarryOp, 4> phaseCarries;
    graphIndex->collectCarries(phase, phaseCarries);
    for (dataflow::CarryOp carry : phaseCarries)
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
      if (auto activation =
              dataflow::semantics::getSelectiveRouterLeafActivation(value))
        return isExactOne(*activation);
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

  dataflow::detail::GraphCausalDependencyCache causalDependencies;
  GraphCardinalityAnalysis cardinality(graph, causalDependencies);
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
    if (!isCovered(causalDependencies, value, ret.getComplete()))
      return graphError(llvm::Twine("retirement frontier does not causally ") +
                        "cover value output #" + llvm::Twine(index));

  for (auto [index, stream] : llvm::enumerate(ret.getStreams())) {
    llvm::SmallVector<mlir::Value, 2> closeSignals;
    cardinality.collectStreamCloseSignals(stream, closeSignals);
    bool covered =
        closeSignals.empty()
            ? isCovered(causalDependencies, stream, ret.getComplete())
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
      if (!isCovered(causalDependencies, graph.getStart(), ret.getComplete()))
        return graphError(
            llvm::Twine(
                "retirement frontier does not cover establishment of ") +
            "memory output #" + llvm::Twine(index));
      continue;
    }
    if (!isCovered(causalDependencies, memory, ret.getComplete()))
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
      if (!isCovered(causalDependencies, done, ret.getComplete()))
        effectError = graphError(
            llvm::Twine("retirement frontier does not causally cover ") +
            op->getName().getStringRef() + " done");
      return effectError ? mlir::WalkResult::interrupt()
                         : mlir::WalkResult::advance();
    }
    if (auto call = llvm::dyn_cast<mlir::LLVM::CallOp>(op)) {
      bool covered =
          !call.getResults().empty() &&
          llvm::any_of(call.getResults(), [&](mlir::Value result) {
            return isCovered(causalDependencies, result, ret.getComplete());
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
