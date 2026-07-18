#include "Dataflow/IR/DataflowThreadCompletion.h"

#include "Dataflow/IR/DataflowOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
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

enum class CoverageResult { Uncovered, Recurrence, Covered };

CoverageResult completionEventCoversImpl(mlir::Value terminal,
                                         mlir::Value event,
                                         llvm::DenseSet<mlir::Value> &active) {
  if (terminal == event)
    return CoverageResult::Covered;
  if (!terminal || !llvm::isa<mlir::NoneType>(terminal.getType()))
    return CoverageResult::Uncovered;
  if (!active.insert(terminal).second)
    return CoverageResult::Recurrence;

  CoverageResult coverage = CoverageResult::Uncovered;
  if (std::optional<RegionValueSource> source =
          getRegionValueSource(terminal)) {
    llvm::SmallVector<mlir::RegionBranchPoint, 4> points;
    llvm::SmallVector<mlir::Value, 4> values;
    source->branch.getPredecessors(source->successor, points);
    source->branch.getPredecessorValues(source->successor, source->inputIndex,
                                        values);
    if (points.size() == values.size()) {
      bool sawRelevantPath = false;
      bool sawCoveredPath = false;
      coverage = CoverageResult::Covered;
      for (auto [point, value] : llvm::zip_equal(points, values)) {
        if (!eventMayExistOnRegionPredecessor(event, source->branch, point))
          continue;
        sawRelevantPath = true;
        CoverageResult predecessorCoverage =
            completionEventCoversImpl(value, event, active);
        if (predecessorCoverage == CoverageResult::Uncovered) {
          coverage = CoverageResult::Uncovered;
          break;
        }
        sawCoveredPath |= predecessorCoverage == CoverageResult::Covered;
      }
      if (coverage != CoverageResult::Uncovered)
        coverage = !sawRelevantPath
                       ? CoverageResult::Uncovered
                       : sawCoveredPath ? CoverageResult::Covered
                                        : CoverageResult::Recurrence;
    }
  } else if (auto result = llvm::dyn_cast<mlir::OpResult>(terminal)) {
    mlir::Operation *definition = result.getOwner();
    if (auto launch = llvm::dyn_cast<dataflow::GraphLaunchOp>(definition)) {
      if (terminal == launch.getDone()) {
        for (mlir::Value input : launch.getDependencies()) {
          CoverageResult inputCoverage =
              completionEventCoversImpl(input, event, active);
          if (inputCoverage == CoverageResult::Covered) {
            coverage = CoverageResult::Covered;
            break;
          }
          if (inputCoverage == CoverageResult::Recurrence)
            coverage = CoverageResult::Recurrence;
        }
      }
    } else if (auto sync = llvm::dyn_cast<dataflow::SyncOp>(definition)) {
      for (mlir::Value input : sync.getInputs()) {
        CoverageResult inputCoverage =
            completionEventCoversImpl(input, event, active);
        if (inputCoverage == CoverageResult::Covered) {
          coverage = CoverageResult::Covered;
          break;
        }
        if (inputCoverage == CoverageResult::Recurrence)
          coverage = CoverageResult::Recurrence;
      }
    } else if (auto mux = llvm::dyn_cast<dataflow::MuxOp>(definition)) {
      bool sawRelevantPath = false;
      bool sawCoveredPath = false;
      coverage = CoverageResult::Covered;
      for (auto [lane, input] : llvm::enumerate(mux.getInputs())) {
        if (!eventMayExistOnMuxLane(event, mux, lane))
          continue;
        sawRelevantPath = true;
        CoverageResult inputCoverage =
            completionEventCoversImpl(input, event, active);
        if (inputCoverage == CoverageResult::Uncovered) {
          coverage = CoverageResult::Uncovered;
          break;
        }
        sawCoveredPath |= inputCoverage == CoverageResult::Covered;
      }
      if (coverage != CoverageResult::Uncovered)
        coverage = !sawRelevantPath
                       ? CoverageResult::Uncovered
                       : sawCoveredPath ? CoverageResult::Covered
                                        : CoverageResult::Recurrence;
    }
  }

  active.erase(terminal);
  return coverage;
}

} // namespace

bool dataflow::completionEventCovers(mlir::Value terminal, mlir::Value event) {
  if (!event || !llvm::isa<mlir::NoneType>(event.getType()))
    return false;
  llvm::DenseSet<mlir::Value> active;
  return completionEventCoversImpl(terminal, event, active) ==
         CoverageResult::Covered;
}

bool dataflow::isThreadCompletionFrontierMemberNecessary(
    mlir::ValueRange frontier, unsigned memberIndex,
    mlir::ValueRange graphLaunchCompletions) {
  if (memberIndex >= frontier.size())
    return false;
  for (mlir::Value completion : graphLaunchCompletions) {
    if (!completionEventCovers(frontier[memberIndex], completion))
      continue;
    bool coveredByOther = false;
    for (auto [otherIndex, terminal] : llvm::enumerate(frontier)) {
      if (otherIndex == memberIndex)
        continue;
      if (completionEventCovers(terminal, completion)) {
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
dataflow::computeMinimalThreadCompletionFrontier(
    mlir::ValueRange candidates, mlir::ValueRange graphLaunchCompletions) {
  llvm::SmallVector<mlir::Value, 4> unique;
  for (mlir::Value candidate : candidates)
    if (!llvm::is_contained(unique, candidate))
      unique.push_back(candidate);

  llvm::SmallVector<mlir::Value, 4> frontier;
  for (auto [index, candidate] : llvm::enumerate(unique)) {
    bool covered = false;
    for (auto [otherIndex, other] : llvm::enumerate(unique)) {
      if (index == otherIndex || !completionEventCovers(other, candidate))
        continue;
      if (!completionEventCovers(candidate, other) || otherIndex < index) {
        covered = true;
        break;
      }
    }
    if (!covered)
      frontier.push_back(candidate);
  }

  for (unsigned index = frontier.size(); index > 0; --index)
    if (!isThreadCompletionFrontierMemberNecessary(frontier, index - 1,
                                                   graphLaunchCompletions))
      frontier.erase(frontier.begin() + index - 1);
  return frontier;
}
