#include "Dataflow/IR/DataflowThreadCompletion.h"

#include "Dataflow/IR/DataflowOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

namespace {

struct RegionValueSource {
  mlir::RegionBranchOpInterface branch;
  mlir::RegionSuccessor successor;
  unsigned inputIndex;
};

std::optional<RegionValueSource> getRegionValueSource(mlir::Value value) {
  if (auto result = llvm::dyn_cast<mlir::OpResult>(value)) {
    auto branch =
        llvm::dyn_cast<mlir::RegionBranchOpInterface>(result.getOwner());
    if (!branch)
      return std::nullopt;
    mlir::RegionSuccessor successor(result.getOwner());
    mlir::ValueRange inputs = branch.getSuccessorInputs(successor);
    auto position = llvm::find(inputs, value);
    if (position == inputs.end())
      return std::nullopt;
    return RegionValueSource{
        branch, successor,
        static_cast<unsigned>(std::distance(inputs.begin(), position))};
  }

  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (!argument)
    return std::nullopt;
  auto branch = llvm::dyn_cast_or_null<mlir::RegionBranchOpInterface>(
      argument.getOwner()->getParentOp());
  if (!branch)
    return std::nullopt;
  mlir::RegionSuccessor successor(argument.getOwner()->getParent());
  mlir::ValueRange inputs = branch.getSuccessorInputs(successor);
  auto position = llvm::find(inputs, value);
  if (position == inputs.end())
    return std::nullopt;
  return RegionValueSource{
      branch, successor,
      static_cast<unsigned>(std::distance(inputs.begin(), position))};
}

bool isNestedIn(mlir::Operation *op, mlir::Operation *ancestor) {
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (parent == ancestor)
      return true;
  return false;
}

bool eventMayExistOnRegionPredecessor(mlir::Value event,
                                      mlir::RegionBranchOpInterface branch,
                                      mlir::RegionBranchPoint predecessor) {
  mlir::Operation *definition = event.getDefiningOp();
  if (!definition)
    return true;
  if (predecessor.isParent())
    return !isNestedIn(definition, branch.getOperation());
  auto terminator = predecessor.getTerminatorPredecessorOrNull();
  return !mlir::insideMutuallyExclusiveRegions(definition,
                                               terminator.getOperation());
}

void collectRequiredSelectorLanes(mlir::Value value, mlir::Value selector,
                                  llvm::DenseSet<mlir::Value> &visited,
                                  llvm::DenseSet<unsigned> &lanes) {
  if (!value || !visited.insert(value).second)
    return;
  auto result = llvm::dyn_cast<mlir::OpResult>(value);
  if (!result)
    return;

  mlir::Operation *definition = result.getOwner();
  if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(definition)) {
    if (demux.getSel() == selector) {
      lanes.insert(result.getResultNumber());
      return;
    }
    collectRequiredSelectorLanes(demux.getInput(), selector, visited, lanes);
    return;
  }

  if (auto launch = llvm::dyn_cast<dataflow::GraphLaunchOp>(definition)) {
    if (value != launch.getDone())
      return;
    for (mlir::Value dependency : launch.getDependencies())
      collectRequiredSelectorLanes(dependency, selector, visited, lanes);
    return;
  }

  if (auto sync = llvm::dyn_cast<dataflow::SyncOp>(definition)) {
    for (mlir::Value input : sync.getInputs())
      collectRequiredSelectorLanes(input, selector, visited, lanes);
    return;
  }
}

bool eventMayExistOnMuxLane(mlir::Value event, dataflow::MuxOp mux,
                            unsigned lane) {
  llvm::DenseSet<mlir::Value> visited;
  llvm::DenseSet<unsigned> requiredLanes;
  collectRequiredSelectorLanes(event, mux.getSel(), visited, requiredLanes);
  return requiredLanes.size() != 1 || requiredLanes.contains(lane);
}

using CoverageKey = std::pair<mlir::Value, mlir::Value>;
using CoverageCache = llvm::DenseMap<CoverageKey, bool>;

enum class CoverageNodeKind { False, True, Any, All };

struct CoverageNode {
  CoverageNodeKind kind = CoverageNodeKind::False;
  llvm::SmallVector<unsigned, 4> dependencies;
};

bool isRepetitiveEntry(mlir::RegionBranchOpInterface branch,
                       mlir::RegionSuccessor successor,
                       llvm::ArrayRef<mlir::RegionBranchPoint> predecessors) {
  mlir::Region *region = successor.getSuccessor();
  return region && branch.isRepetitiveRegion(region->getRegionNumber()) &&
         llvm::any_of(predecessors, [](mlir::RegionBranchPoint predecessor) {
           return predecessor.isParent();
         });
}

llvm::DenseSet<mlir::Region *>
collectReachableRegions(mlir::RegionBranchOpInterface branch) {
  llvm::DenseSet<mlir::Region *> reachable;
  llvm::SmallVector<mlir::Region *, 4> worklist;
  auto enqueue = [&](llvm::ArrayRef<mlir::RegionSuccessor> successors) {
    for (mlir::RegionSuccessor successor : successors) {
      mlir::Region *region = successor.getSuccessor();
      if (region && reachable.insert(region).second)
        worklist.push_back(region);
    }
  };

  llvm::SmallVector<mlir::RegionSuccessor, 4> successors;
  branch.getSuccessorRegions(mlir::RegionBranchPoint::parent(), successors);
  enqueue(successors);
  while (!worklist.empty()) {
    mlir::Region *region = worklist.pop_back_val();
    for (mlir::Block &block : *region) {
      auto terminator = llvm::dyn_cast<mlir::RegionBranchTerminatorOpInterface>(
          block.getTerminator());
      if (!terminator)
        continue;
      successors.clear();
      branch.getSuccessorRegions(mlir::RegionBranchPoint(terminator),
                                 successors);
      enqueue(successors);
    }
  }
  return reachable;
}

bool isReachablePredecessor(
    mlir::RegionBranchPoint predecessor,
    const llvm::DenseSet<mlir::Region *> &reachableRegions) {
  if (predecessor.isParent())
    return true;
  auto terminator = predecessor.getTerminatorPredecessorOrNull();
  return reachableRegions.contains(terminator->getParentRegion());
}

bool isRepetitiveExit(RegionValueSource source,
                      mlir::RegionBranchPoint predecessor) {
  if (!source.successor.isOperation() || predecessor.isParent())
    return false;
  auto terminator = predecessor.getTerminatorPredecessorOrNull();
  mlir::Region *region = terminator->getParentRegion();
  return source.branch.isRepetitiveRegion(region->getRegionNumber());
}

llvm::SmallVector<mlir::Value, 4>
getRepetitiveEntryCarries(mlir::RegionBranchOpInterface branch) {
  llvm::SmallVector<mlir::Value, 4> carries;
  for (mlir::Region &region : branch.getOperation()->getRegions()) {
    mlir::RegionSuccessor successor(&region);
    llvm::SmallVector<mlir::RegionBranchPoint, 4> predecessors;
    branch.getPredecessors(successor, predecessors);
    if (!isRepetitiveEntry(branch, successor, predecessors))
      continue;
    for (mlir::Value input : branch.getSuccessorInputs(successor))
      if (llvm::isa<mlir::NoneType>(input.getType()))
        carries.push_back(input);
  }
  return carries;
}

class CompletionCoverageSolver {
public:
  explicit CompletionCoverageSolver(CoverageCache &known) : known(known) {}

  bool solve(mlir::Value terminal, mlir::Value event) {
    unsigned root = getNode(terminal, event);
    llvm::SmallVector<unsigned char, 16> covered(nodes.size(), false);
    bool changed;
    do {
      changed = false;
      for (auto [index, node] : llvm::enumerate(nodes)) {
        if (covered[index] || !evaluate(node, covered))
          continue;
        covered[index] = true;
        changed = true;
      }
    } while (changed);

    for (auto [index, key] : keyedNodes)
      known.try_emplace(key, covered[index]);
    return covered[root];
  }

private:
  unsigned getNode(mlir::Value terminal, mlir::Value event) {
    CoverageKey key{terminal, event};
    auto existing = nodeIndices.find(key);
    if (existing != nodeIndices.end())
      return existing->second;

    unsigned index = nodes.size();
    nodeIndices.insert({key, index});
    keyedNodes.push_back({index, key});
    nodes.emplace_back();

    auto knownResult = known.find(key);
    if (knownResult != known.end()) {
      nodes[index].kind = knownResult->second ? CoverageNodeKind::True
                                              : CoverageNodeKind::False;
      return index;
    }
    nodes[index] = buildNode(terminal, event);
    return index;
  }

  CoverageNode buildNode(mlir::Value terminal, mlir::Value event) {
    if (terminal == event)
      return CoverageNode{CoverageNodeKind::True, {}};
    if (!terminal || !llvm::isa<mlir::NoneType>(terminal.getType()))
      return CoverageNode{};

    if (std::optional<RegionValueSource> source =
            getRegionValueSource(terminal))
      return buildRegionNode(terminal, event, *source);

    auto result = llvm::dyn_cast<mlir::OpResult>(terminal);
    if (!result)
      return CoverageNode{};
    mlir::Operation *definition = result.getOwner();
    if (auto launch = llvm::dyn_cast<dataflow::GraphLaunchOp>(definition)) {
      if (terminal != launch.getDone())
        return CoverageNode{};
      return buildAnyNode(launch.getDependencies(), event);
    }
    if (auto sync = llvm::dyn_cast<dataflow::SyncOp>(definition))
      return buildAnyNode(sync.getInputs(), event);
    if (auto mux = llvm::dyn_cast<dataflow::MuxOp>(definition)) {
      CoverageNode node{CoverageNodeKind::All, {}};
      for (auto [lane, input] : llvm::enumerate(mux.getInputs())) {
        if (eventMayExistOnMuxLane(event, mux, lane))
          addDependency(node, input, event);
      }
      if (node.dependencies.empty())
        node.kind = CoverageNodeKind::False;
      return node;
    }
    return CoverageNode{};
  }

  CoverageNode buildRegionNode(mlir::Value terminal, mlir::Value event,
                               RegionValueSource source) {
    llvm::SmallVector<mlir::RegionBranchPoint, 4> predecessors;
    llvm::SmallVector<mlir::Value, 4> values;
    source.branch.getPredecessors(source.successor, predecessors);
    source.branch.getPredecessorValues(source.successor, source.inputIndex,
                                       values);
    if (predecessors.size() != values.size())
      return CoverageNode{};

    llvm::DenseSet<mlir::Region *> reachableRegions =
        collectReachableRegions(source.branch);
    CoverageNode node{CoverageNodeKind::All, {}};
    if (!isRepetitiveEntry(source.branch, source.successor, predecessors)) {
      for (auto [predecessor, value] : llvm::zip_equal(predecessors, values)) {
        if (!isReachablePredecessor(predecessor, reachableRegions))
          continue;
        if (isRepetitiveExit(source, predecessor)) {
          CoverageNode alternatives{CoverageNodeKind::Any, {}};
          for (mlir::Value carry : getRepetitiveEntryCarries(source.branch)) {
            CoverageNode candidate{CoverageNodeKind::All, {}};
            addDependency(candidate, carry, event);
            addDependency(candidate, value, carry);
            if (eventMayExistOnRegionPredecessor(event, source.branch,
                                                 predecessor))
              addDependency(candidate, value, event);
            alternatives.dependencies.push_back(addNode(std::move(candidate)));
          }
          if (alternatives.dependencies.empty())
            return CoverageNode{};
          node.dependencies.push_back(addNode(std::move(alternatives)));
          continue;
        }
        if (eventMayExistOnRegionPredecessor(event, source.branch, predecessor))
          addDependency(node, value, event);
      }
    } else {
      // Parent entry establishes zero-trip coverage. Every backedge must
      // preserve that carried frontier, and loop-local events must additionally
      // be covered on each backedge where the current occurrence may exist.
      bool eventExistsAtEntry =
          llvm::any_of(predecessors, [&](mlir::RegionBranchPoint predecessor) {
            return predecessor.isParent() &&
                   eventMayExistOnRegionPredecessor(event, source.branch,
                                                    predecessor);
          });
      bool sawEventPath = false;
      for (auto [predecessor, value] : llvm::zip_equal(predecessors, values)) {
        if (!isReachablePredecessor(predecessor, reachableRegions))
          continue;
        if (predecessor.isParent()) {
          if (!eventMayExistOnRegionPredecessor(event, source.branch,
                                                predecessor))
            continue;
          addDependency(node, value, event);
          sawEventPath = true;
          continue;
        }

        addDependency(node, value, terminal);
        if (!eventExistsAtEntry && eventMayExistOnRegionPredecessor(
                                       event, source.branch, predecessor)) {
          addDependency(node, value, event);
          sawEventPath = true;
        }
      }
      if (!sawEventPath)
        node.kind = CoverageNodeKind::False;
    }

    if (node.dependencies.empty())
      node.kind = CoverageNodeKind::False;
    return node;
  }

  CoverageNode buildAnyNode(mlir::ValueRange inputs, mlir::Value event) {
    CoverageNode node{CoverageNodeKind::Any, {}};
    for (mlir::Value input : inputs)
      addDependency(node, input, event);
    if (node.dependencies.empty())
      node.kind = CoverageNodeKind::False;
    return node;
  }

  unsigned addNode(CoverageNode node) {
    unsigned index = nodes.size();
    nodes.push_back(std::move(node));
    return index;
  }

  void addDependency(CoverageNode &node, mlir::Value terminal,
                     mlir::Value event) {
    unsigned dependency = getNode(terminal, event);
    if (!llvm::is_contained(node.dependencies, dependency))
      node.dependencies.push_back(dependency);
  }

  bool evaluate(const CoverageNode &node,
                llvm::ArrayRef<unsigned char> covered) const {
    switch (node.kind) {
    case CoverageNodeKind::False:
      return false;
    case CoverageNodeKind::True:
      return true;
    case CoverageNodeKind::Any:
      return llvm::any_of(node.dependencies,
                          [&](unsigned index) { return covered[index]; });
    case CoverageNodeKind::All:
      return llvm::all_of(node.dependencies,
                          [&](unsigned index) { return covered[index]; });
    }
    llvm_unreachable("unknown completion coverage node kind");
  }

  CoverageCache &known;
  llvm::DenseMap<CoverageKey, unsigned> nodeIndices;
  llvm::SmallVector<std::pair<unsigned, CoverageKey>, 16> keyedNodes;
  llvm::SmallVector<CoverageNode, 16> nodes;
};

} // namespace

bool dataflow::ThreadCompletionCoverageAnalysis::covers(mlir::Value terminal,
                                                        mlir::Value event) {
  if (!event || !llvm::isa<mlir::NoneType>(event.getType()))
    return false;
  CompletionCoverageSolver solver(coverage);
  return solver.solve(terminal, event);
}

bool dataflow::ThreadCompletionCoverageAnalysis::isFrontierMemberNecessary(
    mlir::ValueRange frontier, unsigned memberIndex,
    mlir::ValueRange graphLaunchCompletions) {
  if (memberIndex >= frontier.size())
    return false;
  for (mlir::Value completion : graphLaunchCompletions) {
    if (!covers(frontier[memberIndex], completion))
      continue;
    bool coveredByOther = false;
    for (auto [otherIndex, terminal] : llvm::enumerate(frontier)) {
      if (otherIndex == memberIndex)
        continue;
      if (covers(terminal, completion)) {
        coveredByOther = true;
        break;
      }
    }
    if (!coveredByOther)
      return true;
  }
  return false;
}

llvm::SmallVector<mlir::Value, 4>
dataflow::ThreadCompletionCoverageAnalysis::computeMinimalFrontier(
    mlir::ValueRange candidates, mlir::ValueRange graphLaunchCompletions) {
  llvm::SmallVector<mlir::Value, 4> unique;
  for (mlir::Value candidate : candidates)
    if (!llvm::is_contained(unique, candidate))
      unique.push_back(candidate);

  llvm::SmallVector<mlir::Value, 4> frontier;
  for (auto [index, candidate] : llvm::enumerate(unique)) {
    bool covered = false;
    for (auto [otherIndex, other] : llvm::enumerate(unique)) {
      if (index == otherIndex || !covers(other, candidate))
        continue;
      if (!covers(candidate, other) || otherIndex < index) {
        covered = true;
        break;
      }
    }
    if (!covered)
      frontier.push_back(candidate);
  }

  for (unsigned index = frontier.size(); index > 0; --index)
    if (!isFrontierMemberNecessary(frontier, index - 1, graphLaunchCompletions))
      frontier.erase(frontier.begin() + index - 1);
  return frontier;
}
