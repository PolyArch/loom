#include "DataflowGraphCausality.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>

namespace {

template <typename... Ts> unsigned denseMapKeyHash(const Ts &...values) {
  using Key = std::tuple<Ts...>;
  return llvm::DenseMapInfo<Key>::getHashValue(Key(values...));
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
    return denseMapKeyHash(key.parent, key.selector.getAsOpaquePointer(),
                           key.lane);
  }
  static bool isEqual(const CausalConstraintTransition &lhs,
                      const CausalConstraintTransition &rhs) {
    return lhs == rhs;
  }
};

struct CausalMemoKey {
  mlir::Value value;
  unsigned constraints;

  bool operator==(const CausalMemoKey &other) const {
    return value == other.value && constraints == other.constraints;
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
    return denseMapKeyHash(key.value.getAsOpaquePointer(), key.constraints);
  }
  static bool isEqual(const CausalMemoKey &lhs, const CausalMemoKey &rhs) {
    return lhs == rhs;
  }
};

class CausalDependencyAnalysis {
public:
  explicit CausalDependencyAnalysis(mlir::Value event) : event(event) {
    constraintStates.emplace_back();
    constraintStatesByHash[hashConstraintState(constraintStates.front())]
        .push_back(0);
  }

  bool dependsOn(mlir::Value prerequisite) {
    if (queryReaches(prerequisite, 0))
      return true;

    auto result = llvm::dyn_cast<mlir::OpResult>(prerequisite);
    if (!result)
      return false;
    mlir::Operation *owner = result.getOwner();
    if (llvm::isa<dataflow::SyncOp>(owner) && queryReachesAnyResult(owner, 0))
      return true;
    mlir::Value done = dataflow::semantics::getMemoryActorDone(owner);
    return done && prerequisite != done && queryReaches(done, 0);
  }

private:
  enum class MemoState : uint8_t {
    Visiting,
    ProvisionalFalse,
    StableFalse,
    True
  };
  struct MemoEntry {
    std::uint64_t generation;
    MemoState state;
  };

  mlir::Value event;
  std::uint64_t generation = 0;
  llvm::DenseMap<mlir::Value, unsigned> selectorIds;
  llvm::SmallVector<llvm::SmallVector<std::uint64_t, 4>, 8> constraintStates;
  llvm::DenseMap<std::uint64_t, llvm::SmallVector<unsigned, 1>>
      constraintStatesByHash;
  llvm::DenseMap<CausalConstraintTransition, unsigned,
                 CausalConstraintTransitionInfo>
      constraintTransitions;
  llvm::DenseMap<CausalMemoKey, MemoEntry, CausalMemoKeyInfo> memo;
  llvm::SmallVector<CausalMemoKey, 64> queryKeys;

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
    for (unsigned candidate : bucket)
      if (constraintStates[candidate] == assignments)
        return candidate;
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
    if (position != current.size() && (current[position] >> 32) == selectorId)
      return current[position] == assignment ? std::optional<unsigned>(state)
                                             : std::nullopt;

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

  bool queryReaches(mlir::Value value, unsigned state) {
    if (++generation == 0) {
      memo.clear();
      generation = 1;
    }
    queryKeys.clear();
    bool result = reaches(value, state);
    // A failed root query proves its entire explored closure unreachable. A
    // successful query only proves the positive path: provisional failures
    // may have observed an ancestor before that ancestor reached the event.
    for (const CausalMemoKey &key : queryKeys) {
      auto known = memo.find(key);
      if (known == memo.end() || known->second.generation != generation ||
          known->second.state == MemoState::True)
        continue;
      if (result)
        memo.erase(known);
      else
        known->second.state = MemoState::StableFalse;
    }
    return result;
  }

  bool queryReachesAnyResult(mlir::Operation *operation, unsigned state) {
    return llvm::any_of(operation->getResults(), [&](mlir::Value result) {
      return queryReaches(result, state);
    });
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
    if (known != memo.end()) {
      if (known->second.state == MemoState::True)
        return true;
      if (known->second.state == MemoState::StableFalse ||
          known->second.generation == generation)
        return false;
    }
    memo[key] = MemoEntry{generation, MemoState::Visiting};
    queryKeys.push_back(key);
    bool result = compute(value, state);
    memo[key] = MemoEntry{generation, result ? MemoState::True
                                             : MemoState::ProvisionalFalse};
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

} // namespace

namespace dataflow::detail {

class GraphCausalDependencyCache::Impl {
public:
  bool dependsOn(mlir::Value event, mlir::Value prerequisite) {
    if (!event || !prerequisite)
      return false;
    auto [it, inserted] = analyses.try_emplace(event);
    if (inserted)
      it->second = std::make_unique<CausalDependencyAnalysis>(event);
    return it->second->dependsOn(prerequisite);
  }

private:
  llvm::DenseMap<mlir::Value, std::unique_ptr<CausalDependencyAnalysis>>
      analyses;
};

GraphCausalDependencyCache::GraphCausalDependencyCache()
    : impl(std::make_unique<Impl>()) {}

GraphCausalDependencyCache::~GraphCausalDependencyCache() = default;

bool GraphCausalDependencyCache::dependsOn(mlir::Value event,
                                           mlir::Value prerequisite) {
  return impl->dependsOn(event, prerequisite);
}

} // namespace dataflow::detail
