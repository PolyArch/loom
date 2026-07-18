#include "Dataflow/IR/DataflowThreadCompletion.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace {

void appendRegionBranchPredecessors(
    mlir::RegionBranchOpInterface branch, mlir::RegionSuccessor successor,
    mlir::Value value, llvm::SmallVectorImpl<mlir::Value> &predecessors) {
  mlir::ValueRange inputs = branch.getSuccessorInputs(successor);
  auto position = llvm::find(inputs, value);
  if (position == inputs.end())
    return;
  branch.getPredecessorValues(
      successor, std::distance(inputs.begin(), position), predecessors);
}

void getCausalPredecessors(mlir::Value value,
                           llvm::SmallVectorImpl<mlir::Value> &predecessors) {
  if (auto result = llvm::dyn_cast<mlir::OpResult>(value)) {
    mlir::Operation *definition = result.getOwner();
    if (auto branch =
            llvm::dyn_cast<mlir::RegionBranchOpInterface>(definition)) {
      appendRegionBranchPredecessors(branch, mlir::RegionSuccessor(definition),
                                     value, predecessors);
      if (!predecessors.empty())
        return;
    }
    llvm::append_range(predecessors, definition->getOperands());
    return;
  }

  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (!argument)
    return;
  mlir::Operation *parent = argument.getOwner()->getParentOp();
  auto branch = llvm::dyn_cast_or_null<mlir::RegionBranchOpInterface>(parent);
  if (!branch)
    return;
  appendRegionBranchPredecessors(
      branch, mlir::RegionSuccessor(argument.getOwner()->getParent()), value,
      predecessors);
}

bool completionEventCoversImpl(mlir::Value terminal, mlir::Value event,
                               llvm::DenseSet<mlir::Value> &visited) {
  if (terminal == event)
    return true;
  if (!terminal || !llvm::isa<mlir::NoneType>(terminal.getType()) ||
      !visited.insert(terminal).second)
    return false;

  llvm::SmallVector<mlir::Value, 4> predecessors;
  getCausalPredecessors(terminal, predecessors);
  return llvm::any_of(predecessors, [&](mlir::Value predecessor) {
    return llvm::isa<mlir::NoneType>(predecessor.getType()) &&
           completionEventCoversImpl(predecessor, event, visited);
  });
}

} // namespace

bool dataflow::completionEventCovers(mlir::Value terminal, mlir::Value event) {
  if (!event || !llvm::isa<mlir::NoneType>(event.getType()))
    return false;
  llvm::DenseSet<mlir::Value> visited;
  return completionEventCoversImpl(terminal, event, visited);
}

llvm::SmallVector<mlir::Value, 4>
dataflow::computeMinimalThreadCompletionFrontier(mlir::ValueRange candidates) {
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
  return frontier;
}
