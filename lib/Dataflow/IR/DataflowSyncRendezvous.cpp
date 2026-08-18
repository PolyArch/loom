#include "Dataflow/IR/DataflowSyncRendezvous.h"

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/Builders.h"

#include <cassert>

namespace {

mlir::Value buildTree(mlir::OpBuilder &builder, mlir::Location location,
                      mlir::ValueRange inputs, std::size_t firstLeaf,
                      std::size_t carrierLeaf) {
  if (inputs.size() == 1)
    return inputs.front();

  const std::size_t leftCount = (inputs.size() + 1) / 2;
  mlir::Value left = buildTree(builder, location, inputs.take_front(leftCount),
                               firstLeaf, carrierLeaf);
  mlir::Value right = buildTree(builder, location, inputs.drop_front(leftCount),
                                firstLeaf + leftCount, carrierLeaf);
  auto sync = dataflow::SyncOp::create(
      builder, location, mlir::TypeRange{left.getType(), right.getType()},
      mlir::ValueRange{left, right});
  const bool carrierInRight = carrierLeaf >= firstLeaf + leftCount &&
                              carrierLeaf < firstLeaf + inputs.size();
  return sync.getOutputs()[carrierInRight ? 1 : 0];
}

} // namespace

mlir::Value dataflow::buildCanonicalSyncRendezvousTree(
    mlir::OpBuilder &builder, mlir::Location location, mlir::ValueRange inputs,
    std::size_t carrierLeaf) {
  assert(!inputs.empty() && "rendezvous tree requires at least one input");
  assert(carrierLeaf < inputs.size() &&
         "rendezvous carrier must name an input leaf");
  return buildTree(builder, location, inputs, 0, carrierLeaf);
}
