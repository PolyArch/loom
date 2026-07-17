#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

using namespace mlir;
using namespace dataflow;

namespace {

LogicalResult verifyChannelEndpoint(Operation *op) {
  if (op->getParentOfType<GraphOp>())
    return op->emitOpError(
        "must not appear inside a dataflow.graph definition");
  if (!op->getParentOfType<ThreadOp>())
    return op->emitOpError("must appear inside a dataflow.thread body");
  return success();
}

} // namespace

bool DataflowDialect::containsChannelOrThreadToken(Type type) {
  return type
      .walk<WalkOrder::PreOrder>([](Type nested) -> WalkResult {
        return isa<ChannelType, ThreadTokenType>(nested)
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

LogicalResult
ChannelType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                    Type elementType) {
  if (!DataflowDialect::containsChannelOrThreadToken(elementType))
    return success();
  if (isa<ChannelType>(elementType))
    return emitError()
           << "channel element type must not be another dataflow channel";
  if (isa<ThreadTokenType>(elementType))
    return emitError()
           << "channel element type must not be !dataflow.thread_token";
  return emitError() << "channel element type must not contain "
                        "!dataflow.channel or !dataflow.thread_token";
}

LogicalResult ChannelSendOp::verify() {
  return verifyChannelEndpoint(getOperation());
}

LogicalResult ChannelReceiveOp::verify() {
  return verifyChannelEndpoint(getOperation());
}
