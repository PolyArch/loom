#include "GraphStreamBoundaryLowering.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/Lowering/StreamLoopAttrs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <cassert>
#include <cstdint>
#include <utility>

namespace {

using ::loom::lowering::detail::StreamBindingPlan;
using ::loom::lowering::detail::StreamScheduleMaterialization;
using ::loom::lowering::detail::StreamScheduleNode;

struct StreamChoiceLeg {
  ::mlir::Operation *choice;
  bool onTrue;
};

struct ScheduledStreamEndpoint {
  ::mlir::Operation *endpoint;
  ::llvm::SmallVector<StreamChoiceLeg, 4> path;
  ::mlir::Operation *repeat = nullptr;
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
    endpoints.push_back(
        {schedule.endpoint, {path.begin(), path.end()}, nullptr});
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
  if (schedule.kind == StreamScheduleNode::Kind::Repeat) {
    assert(schedule.children.size() == 1 && "stream repeat must have one body");
    size_t begin = endpoints.size();
    collectScheduledEndpoints(*schedule.children.front(), path, endpoints);
    // The innermost repeat selector already implies all structured ancestors.
    for (size_t index = begin; index < endpoints.size(); ++index)
      if (!endpoints[index].repeat)
        endpoints[index].repeat = schedule.repeat;
    return;
  }
  for (const auto &child : schedule.children)
    collectScheduledEndpoints(*child, path, endpoints);
}

::mlir::Value
materializeEndpointActivity(const ScheduledStreamEndpoint &endpoint,
                            ::mlir::Value event, ::mlir::OpBuilder &builder,
                            ::mlir::Operation *anchor,
                            StreamScheduleMaterialization &materialization) {
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
    materialization.choiceSelectorUses.push_back(
        {leg.choice, active.getDefiningOp()});
  }
  if (endpoint.repeat) {
    setInsertionPoint(builder, anchor);
    auto placeholder = ::mlir::arith::ConstantOp::create(
        builder, loc, i1, builder.getBoolAttr(false));
    ::mlir::Value inactive =
        scheduleConstant(event, i1, 0, loc, builder, anchor);
    active = mux(placeholder, ::mlir::ValueRange{inactive, active}, loc,
                 builder, anchor);
    materialization.repeatSelectorUses.push_back(
        {endpoint.repeat, active.getDefiningOp(), placeholder});
  }
  return active;
}

::mlir::LogicalResult
buildStreamSchedule(::mlir::Block &block, ::mlir::Value channel, bool input,
                    std::unique_ptr<StreamScheduleNode> &schedule,
                    ::llvm::ArrayRef<::mlir::scf::ForOp> repetitions = {}) {
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
                                             channel, input, trueSchedule,
                                             repetitions)))
        return ::mlir::failure();
      if (!ifOp.getElseRegion().empty() &&
          ::mlir::failed(buildStreamSchedule(ifOp.getElseRegion().front(),
                                             channel, input, falseSchedule,
                                             repetitions)))
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

    auto forOp = ::llvm::dyn_cast<::mlir::scf::ForOp>(&op);
    if (forOp && containsStreamEndpoint(&op, channel, input)) {
      auto stepKind = ::loom::lowering::inferStreamStepKind(forOp);
      auto predicate = ::loom::lowering::inferStreamPredicate(forOp);
      if (::mlir::failed(stepKind) || ::mlir::failed(predicate) ||
          *stepKind != ::dataflow::StreamStepKind::Add ||
          *predicate != ::mlir::arith::CmpIPredicate::slt)
        return forOp.emitError(
            "loom-lower-graph-memory: cross-scope stream repetition requires "
            "an additive increasing scf.for domain");
      for (::mlir::scf::ForOp repetition : repetitions)
        if (!repetition.isDefinedOutsideOfLoop(forOp.getLowerBound()) ||
            !repetition.isDefinedOutsideOfLoop(forOp.getUpperBound()) ||
            !repetition.isDefinedOutsideOfLoop(forOp.getStep()))
          return forOp.emitError(
              "loom-lower-graph-memory: nested cross-scope stream repetition "
              "requires a loop-invariant inner domain");

      std::unique_ptr<StreamScheduleNode> bodySchedule;
      ::llvm::SmallVector<::mlir::scf::ForOp, 4> nested(repetitions);
      nested.push_back(forOp);
      if (::mlir::failed(buildStreamSchedule(forOp.getRegion().front(), channel,
                                             input, bodySchedule, nested)))
        return ::mlir::failure();
      auto repeat = std::make_unique<StreamScheduleNode>(
          StreamScheduleNode::Kind::Repeat, forOp.getLoc());
      repeat->siteCount = bodySchedule->siteCount;
      repeat->repeat = forOp;
      repeat->children.push_back(std::move(bodySchedule));
      siteCount += repeat->siteCount;
      children.push_back(std::move(repeat));
      continue;
    }

    if (::llvm::isa<::mlir::scf::WhileOp>(&op) &&
        containsStreamEndpoint(&op, channel, input))
      return op.emitError(
          "loom-lower-graph-memory: cross-scope stream repetition through "
          "scf.while requires an online ordered event transfer");

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

bool containsRepeat(const StreamScheduleNode &schedule) {
  if (schedule.kind == StreamScheduleNode::Kind::Repeat)
    return true;
  return ::llvm::any_of(schedule.children, [](const auto &child) {
    return containsRepeat(*child);
  });
}

class ScheduleNetworkBuilder {
public:
  ScheduleNetworkBuilder(const StreamScheduleNode &schedule,
                         ::mlir::Value execution, ::mlir::OpBuilder &builder,
                         ::mlir::Operation *anchor,
                         ::llvm::ArrayRef<ScheduledStreamEndpoint> endpoints)
      : execution(execution), builder(builder), anchor(anchor),
        recurrenceType(::mlir::IntegerType::get(builder.getContext(),
                                                ::loom::getIndexWidth())) {
    for (auto [lane, endpoint] : ::llvm::enumerate(endpoints))
      laneByEndpoint.try_emplace(endpoint.endpoint, lane);
    (void)count(schedule);
  }

  ::mlir::Value count(const StreamScheduleNode &schedule) {
    auto known = counts.find(&schedule);
    if (known != counts.end())
      return known->second;

    if (!containsRepeat(schedule)) {
      ::mlir::Value result = constant(schedule.siteCount, schedule.loc);
      counts.try_emplace(&schedule, result);
      return result;
    }

    ::mlir::Value result;
    switch (schedule.kind) {
    case StreamScheduleNode::Kind::Empty:
    case StreamScheduleNode::Kind::Endpoint:
      llvm_unreachable("static stream schedule handled above");
    case StreamScheduleNode::Kind::Sequence:
    case StreamScheduleNode::Kind::Choice: {
      uint64_t staticCount = 0;
      for (const auto &child : schedule.children) {
        if (!containsRepeat(*child)) {
          staticCount += child->siteCount;
          continue;
        }
        ::mlir::Value childCount = count(*child);
        result = result ? add(result, childCount, schedule.loc) : childCount;
      }
      if (staticCount != 0) {
        ::mlir::Value fixed = constant(staticCount, schedule.loc);
        result = result ? add(result, fixed, schedule.loc) : fixed;
      }
      if (!result)
        result = constant(0, schedule.loc);
      break;
    }
    case StreamScheduleNode::Kind::Repeat: {
      assert(schedule.children.size() == 1 &&
             "stream repeat must have one body");
      result = multiply(repeatCount(schedule), count(*schedule.children[0]),
                        schedule.loc);
      break;
    }
    }
    counts.try_emplace(&schedule, result);
    return result;
  }

  void setPhase(::mlir::Value value) { phase = value; }

  ::mlir::Value route(const StreamScheduleNode &schedule, ::mlir::Value ordinal,
                      ::mlir::Value event) {
    switch (schedule.kind) {
    case StreamScheduleNode::Kind::Empty:
      return eventConstant(event, 0, schedule.loc);
    case StreamScheduleNode::Kind::Endpoint:
      return eventConstant(event, laneByEndpoint.lookup(schedule.endpoint),
                           schedule.loc);
    case StreamScheduleNode::Kind::Repeat: {
      assert(schedule.children.size() == 1 &&
             "stream repeat must have one body");
      ::mlir::Value childCount = project(count(*schedule.children[0]));
      ::mlir::Value zero = eventConstant(event, 0, schedule.loc);
      ::mlir::Value one = eventConstant(event, 1, schedule.loc);
      setInsertionPoint(builder, anchor);
      ::mlir::Value isEmpty =
          ::mlir::arith::CmpIOp::create(builder, schedule.loc,
                                        ::mlir::arith::CmpIPredicate::eq,
                                        childCount, zero)
              .getResult();
      ::mlir::Value divisor =
          ::mlir::arith::SelectOp::create(builder, schedule.loc, isEmpty, one,
                                          childCount)
              .getResult();
      ::mlir::Value local = ::mlir::arith::RemUIOp::create(
                                builder, schedule.loc, ordinal, divisor)
                                .getResult();
      return route(*schedule.children[0], local, event);
    }
    case StreamScheduleNode::Kind::Sequence:
    case StreamScheduleNode::Kind::Choice:
      break;
    }

    assert(!schedule.children.empty() &&
           "non-empty schedule composition needs children");
    ::llvm::SmallVector<::mlir::Value, 4> routes;
    ::llvm::SmallVector<::mlir::Value, 4> boundaries;
    ::mlir::Value prefix = constant(0, schedule.loc);
    for (const auto &child : schedule.children) {
      ::mlir::Value local = ordinal;
      if (routes.size() != 0) {
        setInsertionPoint(builder, anchor);
        local = ::mlir::arith::SubIOp::create(builder, schedule.loc, ordinal,
                                              project(prefix))
                    .getResult();
      }
      routes.push_back(route(*child, local, event));
      prefix = add(prefix, count(*child), schedule.loc);
      boundaries.push_back(prefix);
    }

    ::mlir::Value selected = routes.back();
    for (size_t index = routes.size() - 1; index != 0; --index) {
      setInsertionPoint(builder, anchor);
      ::mlir::Value before =
          ::mlir::arith::CmpIOp::create(builder, schedule.loc,
                                        ::mlir::arith::CmpIPredicate::ult,
                                        ordinal, project(boundaries[index - 1]))
              .getResult();
      selected = ::mlir::arith::SelectOp::create(builder, schedule.loc, before,
                                                 routes[index - 1], selected)
                     .getResult();
    }
    return selected;
  }

private:
  ::mlir::Value execution;
  ::mlir::OpBuilder &builder;
  ::mlir::Operation *anchor;
  ::mlir::IntegerType recurrenceType;
  ::mlir::Value phase;
  ::llvm::DenseMap<const StreamScheduleNode *, ::mlir::Value> counts;
  ::llvm::DenseMap<int64_t, ::mlir::Value> constants;
  ::llvm::DenseMap<::mlir::Value, ::mlir::Value> projections;
  ::llvm::DenseMap<::mlir::Operation *, unsigned> laneByEndpoint;

  ::mlir::Value constant(int64_t value, ::mlir::Location loc) {
    auto known = constants.find(value);
    if (known != constants.end())
      return known->second;
    ::mlir::Value result = scheduleConstant(execution, recurrenceType, value,
                                            loc, builder, anchor);
    constants.try_emplace(value, result);
    return result;
  }

  ::mlir::Value eventConstant(::mlir::Value event, int64_t value,
                              ::mlir::Location loc) {
    return scheduleConstant(event, recurrenceType, value, loc, builder, anchor);
  }

  ::mlir::Value add(::mlir::Value lhs, ::mlir::Value rhs,
                    ::mlir::Location loc) {
    setInsertionPoint(builder, anchor);
    return ::mlir::arith::AddIOp::create(builder, loc, lhs, rhs).getResult();
  }

  ::mlir::Value multiply(::mlir::Value lhs, ::mlir::Value rhs,
                         ::mlir::Location loc) {
    setInsertionPoint(builder, anchor);
    return ::mlir::arith::MulIOp::create(builder, loc, lhs, rhs).getResult();
  }

  ::mlir::Value castToRecurrence(::mlir::Value value, ::mlir::Location loc) {
    if (value.getType() == recurrenceType)
      return value;
    assert(::llvm::isa<::mlir::IndexType>(value.getType()) &&
           "scf.for schedule domains must use index values");
    setInsertionPoint(builder, anchor);
    return ::mlir::arith::IndexCastOp::create(builder, loc, recurrenceType,
                                              value)
        .getResult();
  }

  ::mlir::Value repeatCount(const StreamScheduleNode &schedule) {
    auto forOp = ::llvm::cast<::mlir::scf::ForOp>(schedule.repeat);
    ::mlir::Location loc = schedule.loc;
    ::mlir::Value lower = castToRecurrence(forOp.getLowerBound(), loc);
    ::mlir::Value upper = castToRecurrence(forOp.getUpperBound(), loc);
    ::mlir::Value step = castToRecurrence(forOp.getStep(), loc);
    ::mlir::Value zero = constant(0, loc);
    ::mlir::Value one = constant(1, loc);

    setInsertionPoint(builder, anchor);
    ::mlir::Value nonEmpty =
        ::mlir::arith::CmpIOp::create(
            builder, loc, ::mlir::arith::CmpIPredicate::slt, lower, upper)
            .getResult();
    ::mlir::Value delta =
        ::mlir::arith::SubIOp::create(builder, loc, upper, lower).getResult();
    delta = ::mlir::arith::SelectOp::create(builder, loc, nonEmpty, delta, zero)
                .getResult();
    ::mlir::Value quotient =
        ::mlir::arith::DivUIOp::create(builder, loc, delta, step).getResult();
    ::mlir::Value remainder =
        ::mlir::arith::RemUIOp::create(builder, loc, delta, step).getResult();
    ::mlir::Value hasRemainder =
        ::mlir::arith::CmpIOp::create(
            builder, loc, ::mlir::arith::CmpIPredicate::ne, remainder, zero)
            .getResult();
    ::mlir::Value extra =
        ::mlir::arith::SelectOp::create(builder, loc, hasRemainder, one, zero)
            .getResult();
    return add(quotient, extra, loc);
  }

  ::mlir::Value project(::mlir::Value value) {
    assert(phase && "schedule phase must exist before projection");
    auto known = projections.find(value);
    if (known != projections.end())
      return known->second;
    setInsertionPoint(builder, anchor);
    auto invariant = ::dataflow::InvariantOp::create(
        builder, value.getLoc(), value.getType(), phase, value);
    auto lanes = ::dataflow::DemuxOp::create(
        builder, value.getLoc(),
        ::mlir::TypeRange{value.getType(), value.getType()}, phase,
        invariant.getOutput());
    ::mlir::Value projected = lanes.getOutputs()[1];
    projections.try_emplace(value, projected);
    return projected;
  }
};

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

  bool repeated = containsRepeat(schedule);
  std::unique_ptr<ScheduleNetworkBuilder> network;
  if (repeated)
    network = std::make_unique<ScheduleNetworkBuilder>(
        schedule, execution, builder, anchor, endpoints);
  auto recurrenceType =
      ::mlir::IntegerType::get(builder.getContext(), ::loom::getIndexWidth());
  ::mlir::Value init = scheduleConstant(execution, recurrenceType, 0,
                                        schedule.loc, builder, anchor);
  ::mlir::Value limit =
      repeated ? network->count(schedule)
               : scheduleConstant(execution, recurrenceType, schedule.siteCount,
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
  ::mlir::Value route = ordinal;
  if (repeated) {
    network->setPhase(stream.getPhase());
    route = network->route(schedule, ordinal, result.event);
  }
  if (schedule.siteCount > 1) {
    result.selector = scheduleSelector(route, schedule.siteCount, schedule.loc,
                                       builder, anchor);
  } else {
    result.selector = scheduleConstant(result.event, builder.getI1Type(), 1,
                                       schedule.loc, builder, anchor);
  }

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
        materializeEndpointActivity(endpoint, event, builder, anchor, result));
  ::mlir::Value active =
      activities.size() == 1
          ? activities.front()
          : mux(result.selector, activities, schedule.loc, builder, anchor);

  route = demux(active, route, schedule.loc, builder, anchor).second;
  result.event =
      demux(active, result.event, schedule.loc, builder, anchor).second;
  if (schedule.siteCount > 1) {
    result.selector = scheduleSelector(route, schedule.siteCount, schedule.loc,
                                       builder, anchor);
  } else {
    result.selector = scheduleConstant(result.event, builder.getI1Type(), 1,
                                       schedule.loc, builder, anchor);
  }
  return result;
}

} // namespace detail
} // namespace lowering
} // namespace loom
