#include "DataflowGraphCausality.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <random>

namespace {

enum class ActorKind { Add, Mux, Demux };

// Enumerate concrete selector assignments, then use ordinary reachability.
// This oracle has no symbolic conditions, memoized constraints, or fixed-point
// formulas in common with the production analysis.
bool concretelyDependsOn(mlir::Value event, mlir::Value prerequisite,
                         llvm::ArrayRef<mlir::Value> selectors) {
  for (unsigned first = 0; first != 3; ++first) {
    for (unsigned second = 0; second != 3; ++second) {
      std::array<unsigned, 2> assignment{first, second};
      auto selectedLane = [&](mlir::Value selector) {
        return assignment[selector == selectors[0] ? 0 : 1];
      };
      auto isActive = [&](mlir::Value value) {
        auto result = llvm::dyn_cast<mlir::OpResult>(value);
        if (!result)
          return true;
        auto demux = llvm::dyn_cast<dataflow::DemuxOp>(result.getOwner());
        return !demux ||
               selectedLane(demux.getSel()) == result.getResultNumber();
      };
      llvm::DenseSet<mlir::Value> visited;
      llvm::SmallVector<mlir::Value> pending{prerequisite};
      while (!pending.empty()) {
        mlir::Value value = pending.pop_back_val();
        if (!isActive(value) || !visited.insert(value).second)
          continue;
        if (value == event)
          return true;
        for (mlir::OpOperand &use : value.getUses()) {
          auto mux = llvm::dyn_cast<dataflow::MuxOp>(use.getOwner());
          if (mux && use.getOperandNumber() != 0 &&
              selectedLane(mux.getSel()) != use.getOperandNumber() - 1)
            continue;
          for (mlir::Value result : use.getOwner()->getResults())
            pending.push_back(result);
        }
      }
    }
  }
  return false;
}

} // namespace

int main() {
  mlir::MLIRContext context;
  context.loadDialect<dataflow::DataflowDialect, mlir::arith::ArithDialect>();
  mlir::OpBuilder builder(&context);
  auto location = builder.getUnknownLoc();
  auto type = builder.getIndexType();
  std::minstd_rand random(731);
  unsigned positive = 0, negative = 0;
  for (unsigned graph = 0; graph != 128; ++graph) {
    mlir::Block block;
    llvm::SmallVector<mlir::Value> selectors;
    for (unsigned index = 0; index != 2; ++index)
      selectors.push_back(block.addArgument(type, location));
    llvm::SmallVector<mlir::Value> values(selectors);
    values.push_back(block.addArgument(type, location));
    llvm::SmallVector<mlir::Operation *> actors;
    for (unsigned index = 0; index != 10; ++index) {
      auto kind = static_cast<ActorKind>(index % 3);
      mlir::OperationState state(location, kind == ActorKind::Add ? "arith.addi"
                                           : kind == ActorKind::Mux
                                               ? "dataflow.mux"
                                               : "dataflow.demux");
      if (kind == ActorKind::Add)
        state.addOperands({values.back(), values.back()});
      else if (kind == ActorKind::Mux)
        state.addOperands({selectors[random() % selectors.size()],
                           values.back(), values.back(), values.back()});
      else
        state.addOperands(
            {selectors[random() % selectors.size()], values.back()});
      state.addTypes(llvm::SmallVector<mlir::Type>(
          kind == ActorKind::Demux ? 3 : 1, type));
      auto *actor = mlir::Operation::create(state);
      block.push_back(actor);
      actors.push_back(actor);
      values.append(actor->getResults().begin(), actor->getResults().end());
    }
    // Forward references and self edges exercise cycles as well as reconvergent
    // paths. Muxes and demuxes share three-lane selectors.
    for (mlir::Operation *actor : actors)
      for (mlir::OpOperand &operand : actor->getOpOperands()) {
        if (operand.getOperandNumber() == 0 &&
            llvm::isa<dataflow::MuxOp, dataflow::DemuxOp>(actor))
          continue;
        operand.set(values[random() % values.size()]);
      }
    dataflow::detail::GraphCausalDependencyCache cache;
    for (mlir::Value event : values)
      for (mlir::Value prerequisite : values) {
        bool expected = concretelyDependsOn(event, prerequisite, selectors);
        if (cache.dependsOn(event, prerequisite) != expected) {
          llvm::errs() << "causal dependency disagrees with concrete selector "
                          "assignments in graph "
                       << graph << '\n';
          return EXIT_FAILURE;
        }
        expected ? ++positive : ++negative;
      }
  }
  if (!positive || !negative)
    return EXIT_FAILURE;
  llvm::outs() << "Concrete causal checks: " << positive << " reachable, "
               << negative << " unreachable\n";
  return EXIT_SUCCESS;
}
