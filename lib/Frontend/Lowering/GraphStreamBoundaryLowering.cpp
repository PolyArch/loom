#include "GraphStreamBoundaryLowering.h"

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"

#include <cassert>

namespace {

using ::loom::lowering::detail::StreamBindingPlan;
using ::loom::lowering::detail::StreamScheduleNode;

bool isStreamEndpoint(::mlir::Operation *op, ::mlir::Value channel,
                      bool input) {
  if (input) {
    auto receive = ::llvm::dyn_cast<::dataflow::ChannelReceiveOp>(op);
    return receive && receive.getChannel() == channel;
  }
  auto send = ::llvm::dyn_cast<::dataflow::ChannelSendOp>(op);
  return send && send.getChannel() == channel;
}

bool isNestedWithinBlock(::mlir::Operation *op, ::mlir::Block *block) {
  for (::mlir::Operation *current = op; current;
       current = current->getParentOp())
    if (current->getBlock() == block)
      return true;
  return false;
}

bool containsStreamEndpoint(::mlir::Operation *op, ::mlir::Value channel,
                            bool input) {
  for (::mlir::Operation *endpoint :
       ::loom::lowering::detail::collectStreamEndpoints(channel, input))
    for (::mlir::Operation *current = endpoint; current;
         current = current->getParentOp())
      if (current == op)
        return true;
  return false;
}

std::unique_ptr<StreamScheduleNode>
makeEndpointSchedule(::mlir::Operation *endpoint) {
  auto node = std::make_unique<StreamScheduleNode>(
      StreamScheduleNode::Kind::Endpoint, endpoint->getLoc());
  node->width = 1;
  node->endpoint = endpoint;
  return node;
}

::mlir::LogicalResult
buildStreamSchedule(::mlir::Block &block, ::mlir::Value channel, bool input,
                    std::unique_ptr<StreamScheduleNode> &schedule) {
  std::vector<std::unique_ptr<StreamScheduleNode>> children;
  unsigned width = 0;
  for (::mlir::Operation &op : block.without_terminator()) {
    if (isStreamEndpoint(&op, channel, input)) {
      children.push_back(makeEndpointSchedule(&op));
      ++width;
      continue;
    }

    auto ifOp = ::llvm::dyn_cast<::mlir::scf::IfOp>(&op);
    if (ifOp && containsStreamEndpoint(&op, channel, input)) {
      std::unique_ptr<StreamScheduleNode> falseSchedule;
      std::unique_ptr<StreamScheduleNode> trueSchedule;
      if (::mlir::failed(buildStreamSchedule(ifOp.getThenRegion().front(),
                                             channel, input, trueSchedule)))
        return ::mlir::failure();
      if (!ifOp.getElseRegion().empty() &&
          ::mlir::failed(buildStreamSchedule(ifOp.getElseRegion().front(),
                                             channel, input, falseSchedule)))
        return ::mlir::failure();
      if (!falseSchedule || !trueSchedule ||
          falseSchedule->width != trueSchedule->width)
        return ifOp.emitError(
            "loom-lower-graph-memory: mutually exclusive stream endpoint "
            "paths must have the same fixed site width");

      auto choice = std::make_unique<StreamScheduleNode>(
          StreamScheduleNode::Kind::Choice, ifOp.getLoc());
      choice->width = trueSchedule->width;
      choice->selector = ifOp.getCondition();
      choice->children.push_back(std::move(falseSchedule));
      choice->children.push_back(std::move(trueSchedule));
      width += choice->width;
      children.push_back(std::move(choice));
      continue;
    }

    if (containsStreamEndpoint(&op, channel, input))
      return op.emitError(
          "loom-lower-graph-memory: multiple stream endpoint sites must "
          "share one fixed-width sequential or mutually exclusive scope");
  }

  if (children.empty())
    return ::mlir::success();
  if (children.size() == 1) {
    schedule = std::move(children.front());
    return ::mlir::success();
  }

  schedule = std::make_unique<StreamScheduleNode>(
      StreamScheduleNode::Kind::Sequence, block.getParentOp()->getLoc());
  schedule->width = width;
  schedule->children = std::move(children);
  return ::mlir::success();
}

} // namespace

namespace loom {
namespace lowering {
namespace detail {

::llvm::SmallVector<::mlir::Operation *, 4>
collectStreamEndpoints(::mlir::Value channel, bool input) {
  ::llvm::SmallVector<::mlir::Operation *, 4> endpoints;
  for (::mlir::OpOperand &use : channel.getUses())
    if (isStreamEndpoint(use.getOwner(), channel, input))
      endpoints.push_back(use.getOwner());
  return endpoints;
}

::mlir::FailureOr<std::unique_ptr<StreamBindingPlan>>
analyzeStreamBinding(::mlir::Value channel, bool input) {
  ::llvm::SmallVector<::mlir::Operation *, 4> endpoints =
      collectStreamEndpoints(channel, input);
  assert(endpoints.size() > 1 &&
         "multi-site stream analysis requires multiple endpoints");

  ::llvm::SmallVector<::mlir::Block *, 4> candidateScopes;
  for (::mlir::Block *scope = endpoints.front()->getBlock(); scope;) {
    candidateScopes.push_back(scope);
    ::mlir::Operation *parent = scope->getParentOp();
    scope = parent ? parent->getBlock() : nullptr;
  }

  ::mlir::Block *scope = nullptr;
  for (::mlir::Block *candidate : candidateScopes)
    if (::llvm::all_of(endpoints, [&](::mlir::Operation *endpoint) {
          return isNestedWithinBlock(endpoint, candidate);
        })) {
      scope = candidate;
      break;
    }
  assert(scope && "stream endpoints must share their graph entry block");

  auto plan = std::make_unique<StreamBindingPlan>();
  plan->scope = scope;
  plan->channel = channel;
  plan->input = input;
  if (::mlir::failed(
          buildStreamSchedule(*scope, channel, input, plan->schedule)))
    return ::mlir::failure();
  assert(plan->schedule && "common stream scope must contain every endpoint");
  return plan;
}

} // namespace detail
} // namespace lowering
} // namespace loom
