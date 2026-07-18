#include "GraphStreamBoundaryLowering.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"

#include <cassert>
#include <cstdint>
#include <utility>

namespace {

using ::loom::lowering::detail::StreamBindingPlan;
using ::loom::lowering::detail::StreamScheduleNode;

struct StreamChoiceLeg {
  ::mlir::Operation *choice;
  bool onTrue;
};

struct ScheduledStreamEndpoint {
  ::mlir::Operation *endpoint;
  ::llvm::SmallVector<StreamChoiceLeg, 4> path;
};

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

bool isNestedInParallelRegion(::mlir::Operation *op,
                              ::dataflow::GraphOp graph) {
  for (::mlir::Operation *parent = op->getParentOp();
       parent && parent != graph.getOperation(); parent = parent->getParentOp())
    if (::llvm::isa<::mlir::scf::ParallelOp, ::mlir::scf::ForallOp>(parent))
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
  node->siteCount = 1;
  node->endpoint = endpoint;
  return node;
}

std::unique_ptr<StreamScheduleNode> makeEmptySchedule(::mlir::Location loc) {
  return std::make_unique<StreamScheduleNode>(StreamScheduleNode::Kind::Empty,
                                              loc);
}

void setInsertionPoint(::mlir::OpBuilder &builder, ::mlir::Operation *anchor) {
  builder.setInsertionPoint(anchor);
}

::mlir::Value scheduleConstant(::mlir::Value ctrl, ::mlir::IntegerType type,
                               int64_t value, ::mlir::Location loc,
                               ::mlir::OpBuilder &builder,
                               ::mlir::Operation *anchor) {
  setInsertionPoint(builder, anchor);
  return ::dataflow::ConstantOp::create(builder, loc, type, ctrl,
                                        builder.getIntegerAttr(type, value))
      .getValue();
}

::llvm::SmallVector<::mlir::Value, 4> demux(::mlir::Value selector,
                                            ::mlir::Value input, unsigned width,
                                            ::mlir::Location loc,
                                            ::mlir::OpBuilder &builder,
                                            ::mlir::Operation *anchor) {
  assert(width > 1 && "multi-lane demux requires multiple outputs");
  setInsertionPoint(builder, anchor);
  ::llvm::SmallVector<::mlir::Type, 4> types(width, input.getType());
  auto op = ::dataflow::DemuxOp::create(builder, loc, types, selector, input);
  return {op.getOutputs().begin(), op.getOutputs().end()};
}

std::pair<::mlir::Value, ::mlir::Value>
demux(::mlir::Value selector, ::mlir::Value input, ::mlir::Location loc,
      ::mlir::OpBuilder &builder, ::mlir::Operation *anchor) {
  setInsertionPoint(builder, anchor);
  auto op = ::dataflow::DemuxOp::create(
      builder, loc, ::mlir::TypeRange{input.getType(), input.getType()},
      selector, input);
  return {op.getOutputs()[0], op.getOutputs()[1]};
}

::mlir::Value mux(::mlir::Value selector, ::mlir::ValueRange inputs,
                  ::mlir::Location loc, ::mlir::OpBuilder &builder,
                  ::mlir::Operation *anchor) {
  assert(inputs.size() > 1 && "multi-lane mux requires multiple inputs");
  setInsertionPoint(builder, anchor);
  return ::dataflow::MuxOp::create(builder, loc, inputs.front().getType(),
                                   selector, inputs)
      .getOutput();
}

::mlir::Value scheduleSelector(::mlir::Value ordinal, unsigned siteCount,
                               ::mlir::Location loc, ::mlir::OpBuilder &builder,
                               ::mlir::Operation *anchor) {
  assert(siteCount > 1 && "one site needs no schedule selector");
  setInsertionPoint(builder, anchor);
  if (siteCount == 2)
    return ::mlir::arith::TruncIOp::create(builder, loc, builder.getI1Type(),
                                           ordinal)
        .getResult();
  return ::mlir::arith::IndexCastOp::create(builder, loc,
                                            builder.getIndexType(), ordinal)
      .getResult();
}

void collectScheduledEndpoints(
    const StreamScheduleNode &schedule,
    ::llvm::SmallVectorImpl<StreamChoiceLeg> &path,
    ::llvm::SmallVectorImpl<ScheduledStreamEndpoint> &endpoints) {
  if (schedule.kind == StreamScheduleNode::Kind::Endpoint) {
    endpoints.push_back({schedule.endpoint, {path.begin(), path.end()}});
    return;
  }
  if (schedule.kind == StreamScheduleNode::Kind::Choice) {
    assert(schedule.children.size() == 2 &&
           "stream choice must have false and true paths");
    path.push_back({schedule.choice, false});
    collectScheduledEndpoints(*schedule.children[0], path, endpoints);
    path.back().onTrue = true;
    collectScheduledEndpoints(*schedule.children[1], path, endpoints);
    path.pop_back();
    return;
  }
  for (const auto &child : schedule.children)
    collectScheduledEndpoints(*child, path, endpoints);
}

::mlir::Value
materializeEndpointActivity(const ScheduledStreamEndpoint &endpoint,
                            ::mlir::Value event, ::mlir::OpBuilder &builder,
                            ::mlir::Operation *anchor) {
  auto i1 = builder.getI1Type();
  ::mlir::Location loc = endpoint.endpoint->getLoc();
  ::mlir::Value active = scheduleConstant(event, i1, 1, loc, builder, anchor);
  for (const StreamChoiceLeg &leg : ::llvm::reverse(endpoint.path)) {
    ::mlir::Value condition =
        ::mlir::cast<::mlir::scf::IfOp>(leg.choice).getCondition();
    ::mlir::Value inactive =
        scheduleConstant(event, i1, 0, loc, builder, anchor);
    active = leg.onTrue ? mux(condition, ::mlir::ValueRange{inactive, active},
                              loc, builder, anchor)
                        : mux(condition, ::mlir::ValueRange{active, inactive},
                              loc, builder, anchor);
  }
  return active;
}

::mlir::LogicalResult
buildStreamSchedule(::mlir::Block &block, ::mlir::Value channel, bool input,
                    std::unique_ptr<StreamScheduleNode> &schedule) {
  std::vector<std::unique_ptr<StreamScheduleNode>> children;
  unsigned siteCount = 0;
  for (::mlir::Operation &op : block.without_terminator()) {
    if (isStreamEndpoint(&op, channel, input)) {
      children.push_back(makeEndpointSchedule(&op));
      ++siteCount;
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
      if (!falseSchedule)
        falseSchedule = makeEmptySchedule(ifOp.getLoc());
      if (!trueSchedule)
        trueSchedule = makeEmptySchedule(ifOp.getLoc());

      auto choice = std::make_unique<StreamScheduleNode>(
          StreamScheduleNode::Kind::Choice, ifOp.getLoc());
      choice->siteCount = falseSchedule->siteCount + trueSchedule->siteCount;
      choice->choice = ifOp;
      choice->children.push_back(std::move(falseSchedule));
      choice->children.push_back(std::move(trueSchedule));
      siteCount += choice->siteCount;
      children.push_back(std::move(choice));
      continue;
    }

    if (containsStreamEndpoint(&op, channel, input))
      return op.emitError(
          "loom-lower-graph-memory: multiple stream endpoint sites must "
          "share one structured sequential or mutually exclusive scope");
  }

  if (children.empty()) {
    schedule = makeEmptySchedule(block.getParentOp()->getLoc());
    return ::mlir::success();
  }
  if (children.size() == 1) {
    schedule = std::move(children.front());
    return ::mlir::success();
  }

  schedule = std::make_unique<StreamScheduleNode>(
      StreamScheduleNode::Kind::Sequence, block.getParentOp()->getLoc());
  schedule->siteCount = siteCount;
  schedule->children = std::move(children);
  return ::mlir::success();
}

} // namespace

namespace loom {
namespace lowering {
namespace detail {

::mlir::FailureOr<StreamBoundaryInfo>
analyzeStreamBoundary(::dataflow::GraphOp graph) {
  StreamBoundaryInfo info;
  ::mlir::Block &entry = graph.getBody().front();
  size_t canonicalArgumentCount = graph.getFunctionType().getNumInputs() + 1;
  if (entry.getNumArguments() == canonicalArgumentCount)
    return info;

  ::llvm::ArrayRef<int32_t> inputSegments = graph.getInputSegmentSizes();
  ::llvm::ArrayRef<int32_t> resultSegments = graph.getResultSegmentSizes();
  size_t inputCount = static_cast<size_t>(inputSegments[1]);
  size_t outputCount = static_cast<size_t>(resultSegments[1]);
  if (entry.getNumArguments() !=
      canonicalArgumentCount + inputCount + outputCount)
    return graph.emitError(
        "loom-lower-graph-memory: graph entry argument count does not match "
        "either the canonical ABI or a transient stream boundary");

  ::llvm::ArrayRef<::mlir::Type> inputTypes =
      graph.getFunctionType().getInputs();
  ::llvm::ArrayRef<::mlir::Type> resultTypes =
      graph.getFunctionType().getResults();
  size_t inputPayloadBegin = static_cast<size_t>(inputSegments[0]);
  size_t outputPayloadBegin = static_cast<size_t>(resultSegments[0]);
  for (size_t index = 0; index < inputCount; ++index) {
    auto channel = ::llvm::dyn_cast<::mlir::BlockArgument>(
        entry.getArgument(canonicalArgumentCount + index));
    auto channelType =
        ::llvm::dyn_cast<::dataflow::ChannelType>(channel.getType());
    ::mlir::Type payloadType = inputTypes[inputPayloadBegin + index];
    if (!channelType || channelType.getElementType() != payloadType)
      return graph.emitError(
          "loom-lower-graph-memory: transient stream input channel type "
          "does not match its graph payload port");
    info.inputChannels.push_back(channel);
    info.inputPayloads.push_back(
        entry.getArgument(inputPayloadBegin + index + 1));
  }
  for (size_t index = 0; index < outputCount; ++index) {
    auto channel = ::llvm::dyn_cast<::mlir::BlockArgument>(
        entry.getArgument(canonicalArgumentCount + inputCount + index));
    auto channelType =
        ::llvm::dyn_cast<::dataflow::ChannelType>(channel.getType());
    ::mlir::Type payloadType = resultTypes[outputPayloadBegin + index];
    if (!channelType || channelType.getElementType() != payloadType)
      return graph.emitError(
          "loom-lower-graph-memory: transient stream output channel type "
          "does not match its graph payload port");
    info.outputChannels.push_back(channel);
  }
  return info;
}

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
  assert(!endpoints.empty() &&
         "stream analysis requires at least one endpoint");

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
  while (::llvm::isa_and_nonnull<::mlir::scf::IfOp>(scope->getParentOp()))
    scope = scope->getParentOp()->getBlock();

  auto plan = std::make_unique<StreamBindingPlan>();
  plan->scope = scope;
  if (::mlir::failed(
          buildStreamSchedule(*scope, channel, input, plan->schedule)))
    return ::mlir::failure();
  assert(plan->schedule && "common stream scope must contain every endpoint");
  assert(plan->schedule->siteCount == endpoints.size() &&
         "stream schedule must own every endpoint exactly once");
  return plan;
}

::mlir::LogicalResult
checkStreamBoundaryUses(::dataflow::GraphOp graph,
                        const StreamBoundaryInfo &boundary) {
  auto checkBinding = [&](::mlir::BlockArgument channel,
                          bool input) -> ::mlir::LogicalResult {
    unsigned endpointCount = 0;
    for (::mlir::OpOperand &use : channel.getUses()) {
      ::mlir::Operation *owner = use.getOwner();
      bool endpoint = input ? ::llvm::isa<::dataflow::ChannelReceiveOp>(owner)
                            : ::llvm::isa<::dataflow::ChannelSendOp>(owner);
      if (!endpoint || use.getOperandNumber() != 0)
        return owner->emitError()
               << "loom-lower-graph-memory: transient stream "
               << (input ? "input" : "output")
               << " channel has a non-endpoint use";
      if (isNestedInParallelRegion(owner, graph))
        return owner->emitError(
            "loom-lower-graph-memory: one stream binding cannot contain "
            "parallel endpoint sites without a deterministic merge");
      ++endpointCount;
    }
    if (endpointCount == 0)
      return graph.emitError() << "loom-lower-graph-memory: stream "
                               << (input ? "input" : "output")
                               << " binding requires at least one static "
                               << (input ? "receive" : "send")
                               << " site for mechanical publication";
    if (::mlir::failed(analyzeStreamBinding(channel, input)))
      return ::mlir::failure();
    return ::mlir::success();
  };

  for (::mlir::BlockArgument channel : boundary.inputChannels)
    if (::mlir::failed(checkBinding(channel, true)))
      return ::mlir::failure();
  for (::mlir::BlockArgument channel : boundary.outputChannels)
    if (::mlir::failed(checkBinding(channel, false)))
      return ::mlir::failure();
  return ::mlir::success();
}

StreamScheduleMaterialization
materializeStreamSchedule(const StreamScheduleNode &schedule,
                          ::mlir::Value execution, ::mlir::OpBuilder &builder,
                          ::mlir::Operation *anchor) {
  StreamScheduleMaterialization result;
  ::llvm::SmallVector<StreamChoiceLeg, 4> path;
  ::llvm::SmallVector<ScheduledStreamEndpoint, 4> endpoints;
  collectScheduledEndpoints(schedule, path, endpoints);
  assert(endpoints.size() == schedule.siteCount &&
         "stream schedule must contain every endpoint");
  for (const ScheduledStreamEndpoint &endpoint : endpoints)
    result.endpoints.push_back(endpoint.endpoint);

  if (schedule.kind == StreamScheduleNode::Kind::Endpoint) {
    result.event = execution;
    result.selector = scheduleConstant(execution, builder.getI1Type(), 1,
                                       schedule.loc, builder, anchor);
    return result;
  }

  auto recurrenceType =
      ::mlir::IntegerType::get(builder.getContext(), ::loom::getIndexWidth());
  ::mlir::Value init = scheduleConstant(execution, recurrenceType, 0,
                                        schedule.loc, builder, anchor);
  ::mlir::Value limit =
      scheduleConstant(execution, recurrenceType, schedule.siteCount,
                       schedule.loc, builder, anchor);
  ::mlir::Value step = scheduleConstant(execution, recurrenceType, 1,
                                        schedule.loc, builder, anchor);
  setInsertionPoint(builder, anchor);
  auto stream = ::dataflow::StreamOp::create(
      builder, schedule.loc, recurrenceType, builder.getI1Type(), init, limit,
      step, ::dataflow::StreamStepKind::Add, ::mlir::arith::CmpIPredicate::slt);
  auto activations = ::dataflow::InvariantOp::create(
      builder, schedule.loc, builder.getNoneType(), stream.getPhase(),
      execution);
  auto phases = ::dataflow::DemuxOp::create(
      builder, schedule.loc,
      ::mlir::TypeRange{builder.getNoneType(), builder.getNoneType()},
      stream.getPhase(), activations.getOutput());

  ::mlir::Value ordinal = stream.getIv();
  result.event = phases.getOutputs()[1];
  result.close = phases.getOutputs()[0];
  if (schedule.siteCount > 1)
    result.selector = scheduleSelector(ordinal, schedule.siteCount,
                                       schedule.loc, builder, anchor);

  if (!::llvm::any_of(endpoints, [](const ScheduledStreamEndpoint &endpoint) {
        return !endpoint.path.empty();
      }))
    return result;

  ::llvm::SmallVector<::mlir::Value, 4> events;
  if (endpoints.size() == 1)
    events.push_back(result.event);
  else
    events = demux(result.selector, result.event, endpoints.size(),
                   schedule.loc, builder, anchor);

  ::llvm::SmallVector<::mlir::Value, 4> activities;
  for (auto [endpoint, event] : ::llvm::zip_equal(endpoints, events))
    activities.push_back(
        materializeEndpointActivity(endpoint, event, builder, anchor));
  ::mlir::Value active =
      activities.size() == 1
          ? activities.front()
          : mux(result.selector, activities, schedule.loc, builder, anchor);

  ordinal = demux(active, ordinal, schedule.loc, builder, anchor).second;
  result.event =
      demux(active, result.event, schedule.loc, builder, anchor).second;
  if (schedule.siteCount > 1)
    result.selector = scheduleSelector(ordinal, schedule.siteCount,
                                       schedule.loc, builder, anchor);
  return result;
}

} // namespace detail
} // namespace lowering
} // namespace loom
