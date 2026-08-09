#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

using namespace mlir;
using namespace dataflow;

namespace {

LogicalResult verifyChannelEndpoint(Operation *op) {
  if (op->getParentOfType<GraphOp>())
    return op->emitOpError(
        "must not appear inside a dataflow.graph definition");
  ThreadOp thread = op->getParentOfType<ThreadOp>();
  if (!thread)
    return op->emitOpError("must appear inside a dataflow.thread body");
  if (thread.getDomain().getKind() == ThreadDomainKind::DynamicWork)
    return op->emitOpError("must not appear inside a dynamic-work thread body");
  return success();
}

llvm::Error payloadError(const llvm::Twine &message) {
  return llvm::createStringError(std::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

} // namespace

bool DataflowDialect::isMemoryCapabilityType(Type type) {
  return isa<MemRefType, UnrankedMemRefType>(type);
}

bool DataflowDialect::containsMemoryCapability(Type type) {
  return type
      .walk<WalkOrder::PreOrder>([](Type nested) -> WalkResult {
        return DataflowDialect::isMemoryCapabilityType(nested)
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

bool DataflowDialect::isPointerValueType(Type type) {
  return isa<LLVM::LLVMPointerType>(type);
}

bool DataflowDialect::containsPointerValue(Type type) {
  return type
      .walk<WalkOrder::PreOrder>([](Type nested) -> WalkResult {
        return DataflowDialect::isPointerValueType(nested)
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

bool DataflowDialect::containsChannelOrThreadToken(Type type) {
  return type
      .walk<WalkOrder::PreOrder>([](Type nested) -> WalkResult {
        return isa<ChannelType, ThreadTokenType>(nested)
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

llvm::Error DataflowDialect::validateTransferPayloadType(Type type,
                                                         StringRef subject) {
  if (isa<ChannelType>(type))
    return payloadError(subject + " must not be another dataflow channel");
  if (isa<ThreadTokenType>(type))
    return payloadError(subject + " must not be !dataflow.thread_token");
  if (containsChannelOrThreadToken(type))
    return payloadError(subject + " must not contain !dataflow.channel or "
                                  "!dataflow.thread_token");
  if (containsMemoryCapability(type))
    return payloadError(subject + " must not contain a memory capability");
  if (!isPointerValueType(type) && containsPointerValue(type))
    return payloadError(subject +
                        " must not contain a nested LLVM pointer value");
  return llvm::Error::success();
}

LogicalResult
ChannelType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                    Type elementType) {
  if (llvm::Error error = DataflowDialect::validateTransferPayloadType(
          elementType, "channel element type"))
    return emitError() << llvm::toString(std::move(error));
  return success();
}

LogicalResult ChannelCreateOp::verify() {
  if ((*this)->getParentOfType<GraphOp>())
    return emitOpError("must not appear inside a dataflow.graph definition");

  for (Operation *parent = (*this)->getParentOp(); parent;
       parent = parent->getParentOp())
    if (parent->getName().getStringRef() == "loom.spatial_region")
      return emitOpError("must not appear inside loom.spatial_region");

  if (auto thread = (*this)->getParentOfType<ThreadOp>();
      thread && thread.getDomain().getKind() == ThreadDomainKind::DynamicWork)
    return emitOpError(
        "must not appear inside a DynamicWork dataflow.thread definition");
  return success();
}

LogicalResult ChannelSendOp::verify() {
  return verifyChannelEndpoint(getOperation());
}

LogicalResult ChannelReceiveOp::verify() {
  return verifyChannelEndpoint(getOperation());
}
