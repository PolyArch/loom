#include "Fabric/Tech/Partitioner/Materializer.h"

#include "Dataflow/IR/DataflowOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace fabric {

namespace {

// Materialize one Block into a dataflow.subgraph.
//
// The subgraph is inserted immediately before the block's first op in
// program order. External operands (whose defining op is outside the
// block, or which are block args of the enclosing dataflow.graph) become
// subgraph inputs. External uses of block-op results become subgraph
// results.
void materializeBlock(const Block &block, ::mlir::OpBuilder &builder) {
  if (block.ops.empty())
    return;

  // Block-membership lookup for testing operand/use locality.
  ::llvm::DenseSet<::mlir::Operation *> inBlock;
  for (::mlir::Operation *op : block.ops)
    inBlock.insert(op);

  // 1. Collect external operands (deduped, deterministic by first use).
  ::llvm::SetVector<::mlir::Value> externalOperands;
  for (::mlir::Operation *op : block.ops) {
    for (::mlir::Value v : op->getOperands()) {
      ::mlir::Operation *def = v.getDefiningOp();
      if (def && inBlock.contains(def))
        continue; // produced inside the block
      externalOperands.insert(v);
    }
  }

  // 2. Collect external uses: results of block ops that have any user
  //    outside the block. Track them in deterministic op-then-result order.
  ::llvm::SmallVector<::mlir::Value> externalResults;
  for (::mlir::Operation *op : block.ops) {
    for (::mlir::Value res : op->getResults()) {
      bool hasExternal = false;
      for (::mlir::Operation *user : res.getUsers()) {
        if (!inBlock.contains(user)) {
          hasExternal = true;
          break;
        }
      }
      if (hasExternal)
        externalResults.push_back(res);
    }
  }

  // 3. Insert the new dataflow.subgraph right before the block's first op
  //    in program order so SSA defs land before any later user we leave at
  //    graph level. The body of the new subgraph is initialized with an
  //    implicit dataflow.yield by SingleBlockImplicitTerminator.
  ::mlir::Operation *firstOp = block.ops.front();
  for (::mlir::Operation *op : block.ops)
    if (op->isBeforeInBlock(firstOp))
      firstOp = op;

  ::mlir::SmallVector<::mlir::Type> resultTypes;
  resultTypes.reserve(externalResults.size());
  for (::mlir::Value v : externalResults)
    resultTypes.push_back(v.getType());

  ::mlir::SmallVector<::mlir::Value> operandValues(externalOperands.begin(),
                                                   externalOperands.end());

  // Use a raw OperationState so we can populate the body region with an
  // explicit entry block before the op is created. SubgraphOp's builder
  // does not auto-create the entry block.
  builder.setInsertionPoint(firstOp);
  ::mlir::Location loc = firstOp->getLoc();
  ::mlir::OperationState state(loc, ::dataflow::SubgraphOp::getOperationName());
  state.addOperands(operandValues);
  state.addTypes(resultTypes);
  ::mlir::Region *body = state.addRegion();
  ::mlir::Block *bodyBlock = new ::mlir::Block();
  body->push_back(bodyBlock);
  ::llvm::SmallVector<::mlir::Location> argLocs(operandValues.size(), loc);
  ::llvm::SmallVector<::mlir::Type> argTypes;
  argTypes.reserve(operandValues.size());
  for (::mlir::Value v : operandValues)
    argTypes.push_back(v.getType());
  bodyBlock->addArguments(argTypes, argLocs);

  auto subgraph =
      ::mlir::cast<::dataflow::SubgraphOp>(builder.create(state));

  // 4. Populate the subgraph body. We map external operands to the new
  //    block arguments, clone block ops in source program order via the
  //    mapping, then add an explicit dataflow.yield.
  ::mlir::Block &sgBlock = subgraph.getBody().front();
  ::mlir::IRMapping mapping;
  for (auto [extVal, blockArg] :
       ::llvm::zip(operandValues, sgBlock.getArguments()))
    mapping.map(extVal, blockArg);

  ::mlir::OpBuilder bodyBuilder(&sgBlock, sgBlock.end());

  // Clone in source program order so internal SSA edges stay valid.
  ::llvm::SmallVector<::mlir::Operation *> sortedBlockOps(block.ops.begin(),
                                                          block.ops.end());
  std::sort(sortedBlockOps.begin(), sortedBlockOps.end(),
            [](::mlir::Operation *a, ::mlir::Operation *b) {
              return a->isBeforeInBlock(b);
            });
  for (::mlir::Operation *op : sortedBlockOps)
    bodyBuilder.clone(*op, mapping);

  // 5. Build the explicit yield using mapped values for external results.
  ::mlir::SmallVector<::mlir::Value> yieldValues;
  yieldValues.reserve(externalResults.size());
  for (::mlir::Value v : externalResults)
    yieldValues.push_back(mapping.lookup(v));

  ::dataflow::YieldOp::create(bodyBuilder, subgraph.getLoc(), yieldValues);

  // 6. Replace external uses of block-op results with the new subgraph's
  //    results. Internal uses (within the cloned body) are not touched.
  for (auto [origRes, newRes] :
       ::llvm::zip(externalResults, subgraph.getResults())) {
    origRes.replaceUsesWithIf(newRes, [&](::mlir::OpOperand &use) {
      ::mlir::Operation *owner = use.getOwner();
      // Don't redirect uses that we just cloned into the subgraph body.
      if (owner->getParentOp() == subgraph.getOperation())
        return false;
      return !inBlock.contains(owner);
    });
  }

  // 7. Erase original block ops. Reverse program order avoids dangling
  //    uses among internal SSA edges.
  for (auto it = sortedBlockOps.rbegin(); it != sortedBlockOps.rend(); ++it)
    (*it)->erase();
}

} // namespace

void applyPartition(::dataflow::GraphOp graph,
                    const PartitionResult &partition,
                    ::mlir::OpBuilder &builder) {
  for (const Block &b : partition.blocks) {
    if (b.tpl == nullptr)
      continue;
    materializeBlock(b, builder);
  }
  (void)graph;
}

} // namespace fabric
