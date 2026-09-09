#include "DataflowGraphCausality.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

namespace {

// A reduced ordered decision diagram shares selector conditions across paths.
// Each decision has one successor per lane, so a selector always denotes one
// lane even when several muxes and demuxes consume the same value.
class CausalConditions {
public:
  using Condition = unsigned;
  static constexpr Condition impossible = 0;
  static constexpr Condition unconditional = 1;

  void registerSelector(mlir::Value selector, unsigned lanes) {
    auto [it, inserted] = selectorIds.try_emplace(selector, laneCounts.size());
    if (inserted)
      laneCounts.push_back(lanes);
    else
      laneCounts[it->second] = std::max(laneCounts[it->second], lanes);
  }

  Condition lane(mlir::Value selector, unsigned selectedLane) {
    unsigned id = selectorIds.lookup(selector);
    std::vector<Condition> children(laneCounts[id], impossible);
    children[selectedLane] = unconditional;
    return intern(id, std::move(children));
  }

  Condition intersect(Condition lhs, Condition rhs) {
    return combine(Connective::And, lhs, rhs);
  }

  Condition unite(Condition lhs, Condition rhs) {
    return combine(Connective::Or, lhs, rhs);
  }

private:
  enum class Connective : unsigned { And, Or };
  static constexpr unsigned terminal = std::numeric_limits<unsigned>::max();
  struct Node {
    unsigned selector;
    std::vector<Condition> children;
  };
  llvm::DenseMap<mlir::Value, unsigned> selectorIds;
  std::vector<unsigned> laneCounts;
  std::vector<Node> nodes{{terminal, {}}, {terminal, {}}};
  llvm::DenseMap<std::uint64_t, llvm::SmallVector<Condition, 1>> uniqueNodes;
  llvm::DenseMap<std::tuple<unsigned, Condition, Condition>, Condition>
      combined;

  Condition intern(unsigned selector, std::vector<Condition> children) {
    if (llvm::all_of(children, [&](Condition child) {
          return child == children.front();
        }))
      return children.front();
    std::uint64_t hash = llvm::hash_combine(
        selector, llvm::hash_combine_range(children.begin(), children.end()));
    if (hash >= std::numeric_limits<std::uint64_t>::max() - 1)
      hash -= 2;
    auto &bucket = uniqueNodes[hash];
    for (Condition candidate : bucket)
      if (nodes[candidate].selector == selector &&
          nodes[candidate].children == children)
        return candidate;
    Condition result = nodes.size();
    nodes.push_back({selector, std::move(children)});
    bucket.push_back(result);
    return result;
  }

  Condition combine(Connective connective, Condition lhs, Condition rhs) {
    if (lhs > rhs)
      std::swap(lhs, rhs);
    if (lhs == rhs)
      return lhs;
    if (connective == Connective::And) {
      if (lhs == impossible)
        return impossible;
      if (lhs == unconditional)
        return rhs;
    } else {
      if (lhs == impossible)
        return rhs;
      if (lhs == unconditional)
        return unconditional;
    }
    auto key = std::make_tuple(static_cast<unsigned>(connective), lhs, rhs);
    auto known = combined.find(key);
    if (known != combined.end())
      return known->second;
    unsigned selector = std::min(nodes[lhs].selector, nodes[rhs].selector);
    std::vector<Condition> children;
    children.reserve(laneCounts[selector]);
    for (unsigned lane = 0; lane < laneCounts[selector]; ++lane) {
      Condition left =
          nodes[lhs].selector == selector ? nodes[lhs].children[lane] : lhs;
      Condition right =
          nodes[rhs].selector == selector ? nodes[rhs].children[lane] : rhs;
      children.push_back(combine(connective, left, right));
    }
    Condition result = intern(selector, std::move(children));
    combined.try_emplace(key, result);
    return result;
  }
};

class CausalDependencyAnalysis {
public:
  explicit CausalDependencyAnalysis(mlir::Value event) {
    CausalConditions conditions;
    llvm::DenseMap<mlir::Value, CausalConditions::Condition> reachable;
    llvm::DenseSet<mlir::Value> ancestors;
    llvm::SmallVector<mlir::Value, 64> pending{event};
    while (!pending.empty()) {
      mlir::Value value = pending.pop_back_val();
      if (!value || !ancestors.insert(value).second)
        continue;
      mlir::Operation *owner = value.getDefiningOp();
      if (!owner)
        continue;
      if (auto mux = llvm::dyn_cast<dataflow::MuxOp>(owner))
        conditions.registerSelector(mux.getSel(), mux->getNumOperands() - 1);
      if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(owner))
        conditions.registerSelector(demux.getSel(), demux->getNumResults());
      pending.append(owner->getOperands().begin(), owner->getOperands().end());
    }

    // The least fixed point describes all finite feasible paths to the event.
    // Sharing their conditions avoids enumerating the Cartesian product of
    // lane assignments. Cycles add paths but cannot invent a causal witness.
    reachable[event] = outputCondition(conditions, event);
    std::deque<mlir::Value> worklist{event};
    llvm::DenseSet<mlir::Value> queued{event};
    while (!worklist.empty()) {
      mlir::Value value = worklist.front();
      worklist.pop_front();
      queued.erase(value);
      mlir::Operation *owner = value.getDefiningOp();
      if (!owner)
        continue;
      auto mux = llvm::dyn_cast<dataflow::MuxOp>(owner);
      for (mlir::OpOperand &operand : owner->getOpOperands()) {
        mlir::Value prerequisite = operand.get();
        auto candidate = reachable.lookup(value);
        if (mux && operand.getOperandNumber() != 0)
          candidate = conditions.intersect(
              candidate,
              conditions.lane(mux.getSel(), operand.getOperandNumber() - 1));
        candidate = conditions.intersect(
            candidate, outputCondition(conditions, prerequisite));
        auto previous = reachable.lookup(prerequisite);
        auto updated = conditions.unite(previous, candidate);
        if (updated == previous)
          continue;
        reachable[prerequisite] = updated;
        if (queued.insert(prerequisite).second)
          worklist.push_back(prerequisite);
      }
    }
    for (auto [value, condition] : reachable)
      if (condition != CausalConditions::impossible)
        prerequisites.insert(value);
  }

  bool dependsOn(mlir::Value prerequisite) const {
    if (reaches(prerequisite))
      return true;
    auto result = llvm::dyn_cast<mlir::OpResult>(prerequisite);
    if (!result)
      return false;
    mlir::Operation *owner = result.getOwner();
    if (llvm::isa<dataflow::SyncOp>(owner) &&
        llvm::any_of(owner->getResults(),
                     [&](mlir::Value output) { return reaches(output); }))
      return true;
    mlir::Value done = dataflow::semantics::getMemoryActorDone(owner);
    return done && prerequisite != done && reaches(done);
  }

private:
  llvm::DenseSet<mlir::Value> prerequisites;

  static CausalConditions::Condition
  outputCondition(CausalConditions &conditions, mlir::Value value) {
    if (auto result = llvm::dyn_cast<mlir::OpResult>(value))
      if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(result.getOwner()))
        return conditions.lane(demux.getSel(), result.getResultNumber());
    return CausalConditions::unconditional;
  }

  bool reaches(mlir::Value value) const {
    return prerequisites.contains(value);
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
