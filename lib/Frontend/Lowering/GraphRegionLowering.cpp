#include "GraphRegionLowering.h"
#include "GraphIndexLowering.h"
#include "GraphStreamBoundaryLowering.h"

#include "Frontend/Lowering/StreamLoopAttrs.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace loom {
namespace lowering {

bool isSupportedGraphLoweringLeaf(::mlir::Operation *op) {
  return ::dataflow::isCanonicalDataflowActor(op) ||
         ::llvm::isa<::mlir::UnrealizedConversionCastOp,
                     ::mlir::memref::AllocOp, ::mlir::memref::CastOp,
                     ::mlir::memref::GetGlobalOp, ::mlir::memref::LoadOp,
                     ::mlir::memref::StoreOp, ::mlir::LLVM::AddressOfOp,
                     ::mlir::LLVM::GEPOp, ::mlir::LLVM::LoadOp,
                     ::mlir::LLVM::StoreOp, ::mlir::LLVM::MemcpyOp,
                     ::mlir::LLVM::MemsetOp>(op);
}

bool hasGraphOwnedParallelProvenance(::mlir::Operation *op) {
  return op->hasAttr("loom.parallel_group") ||
         op->hasAttr("loom.parallel_schedule");
}

} // namespace lowering
} // namespace loom

namespace {

using ::loom::lowering::detail::analyzeStreamBinding;
using ::loom::lowering::detail::analyzeStreamBoundary;
using ::loom::lowering::detail::checkStreamBoundaryUses;
using ::loom::lowering::detail::materializeStreamSchedule;
using ::loom::lowering::detail::StreamBoundaryInfo;
using ::loom::lowering::detail::StreamScheduleNode;

struct MemoryFrontier {
  ::mlir::Value write;
  ::mlir::Value read;
};

using MemoryState = ::llvm::SmallVector<MemoryFrontier, 4>;

struct RegionResult {
  ::mlir::Value execution;
  MemoryState memory;
};

struct GatedValue {
  ::mlir::Value value;
  ::mlir::Value close;
};

struct FixedParallelDomain {
  ::llvm::SmallVector<int64_t, 4> lower;
  ::llvm::SmallVector<int64_t, 4> upper;
  ::llvm::SmallVector<int64_t, 4> step;
};

struct StreamLoweringPlan {
  StreamLoweringPlan(::mlir::Block *scope, ::mlir::Value channel, bool input,
                     unsigned boundaryIndex,
                     std::unique_ptr<StreamScheduleNode> schedule)
      : scope(scope), channel(channel), input(input),
        boundaryIndex(boundaryIndex), loc(schedule->loc),
        schedule(std::move(schedule)) {}

  ::mlir::Block *scope;
  ::mlir::Value channel;
  bool input;
  unsigned boundaryIndex;
  ::mlir::Location loc;
  std::unique_ptr<StreamScheduleNode> schedule;
  ::mlir::Value selector;
  ::mlir::Value event;
  ::mlir::Value close;
  ::llvm::SmallVector<::mlir::Value, 4> outputs;
};

struct StreamOutputSlot {
  StreamLoweringPlan *plan;
  unsigned index;
};

struct StreamRepeatUse {
  ::mlir::Operation *user;
  ::mlir::Operation *placeholder;
};

std::optional<int64_t> getConstantIndex(::mlir::Value value) {
  ::mlir::APInt constant;
  if (!::mlir::matchPattern(value, ::mlir::m_ConstantInt(&constant)) ||
      !constant.isSignedIntN(64))
    return std::nullopt;
  return constant.getSExtValue();
}

std::optional<FixedParallelDomain>
getFixedParallelDomain(::mlir::scf::ForallOp forall) {
  auto lower = ::mlir::getConstantIntValues(forall.getMixedLowerBound());
  auto upper = ::mlir::getConstantIntValues(forall.getMixedUpperBound());
  auto step = ::mlir::getConstantIntValues(forall.getMixedStep());
  if (!lower || !upper || !step)
    return std::nullopt;
  return FixedParallelDomain{std::move(*lower), std::move(*upper),
                             std::move(*step)};
}

std::optional<FixedParallelDomain>
getFixedParallelDomain(::mlir::scf::ParallelOp parallel) {
  FixedParallelDomain domain;
  for (::mlir::Value value : parallel.getLowerBound()) {
    auto constant = getConstantIndex(value);
    if (!constant)
      return std::nullopt;
    domain.lower.push_back(*constant);
  }
  for (::mlir::Value value : parallel.getUpperBound()) {
    auto constant = getConstantIndex(value);
    if (!constant)
      return std::nullopt;
    domain.upper.push_back(*constant);
  }
  for (::mlir::Value value : parallel.getStep()) {
    auto constant = getConstantIndex(value);
    if (!constant)
      return std::nullopt;
    domain.step.push_back(*constant);
  }
  return domain;
}

std::optional<FixedParallelDomain>
getFixedParallelDomain(::mlir::Operation *op) {
  if (auto forall = ::llvm::dyn_cast<::mlir::scf::ForallOp>(op))
    return getFixedParallelDomain(forall);
  if (auto parallel = ::llvm::dyn_cast<::mlir::scf::ParallelOp>(op))
    return getFixedParallelDomain(parallel);
  return std::nullopt;
}

::mlir::LogicalResult checkSelectedParallel(::mlir::Operation *op) {
  if (!::loom::lowering::hasGraphOwnedParallelProvenance(op)) {
    op->emitError() << "loom-lower-graph-memory: raw "
                    << op->getName().getStringRef()
                    << " requires a selected schedule and provenance before "
                       "graph-region lowering";
    return ::mlir::failure();
  }
  if (auto mapping = op->getAttrOfType<::mlir::ArrayAttr>("mapping");
      mapping && !mapping.empty())
    return op->emitError(
        "loom-lower-graph-memory: graph-owned parallel SCF must not retain "
        "an execution-resource mapping");
  if (auto forall = ::llvm::dyn_cast<::mlir::scf::ForallOp>(op)) {
    auto inParallel = forall.getTerminator();
    if (!forall.getOutputs().empty() || forall.getNumResults() != 0 ||
        inParallel.getRegion().empty() ||
        !inParallel.getRegion().front().empty())
      return forall.emitError(
          "loom-lower-graph-memory: graph-owned scf.forall must be in "
          "effect form before fixed-lane lowering");
  }
  if (auto parallel = ::llvm::dyn_cast<::mlir::scf::ParallelOp>(op)) {
    if (!parallel.getInitVals().empty() || parallel.getNumResults() != 0)
      return parallel.emitError(
          "loom-lower-graph-memory: graph-owned scf.parallel reductions "
          "must be normalized before fixed-lane lowering");
  }
  auto domain = getFixedParallelDomain(op);
  if (!domain)
    return op->emitError(
        "loom-lower-graph-memory: selected graph-owned parallel SCF requires "
        "a fixed compile-time lane domain");
  if (::llvm::any_of(domain->step, [](int64_t step) { return step <= 0; }))
    return op->emitError(
        "loom-lower-graph-memory: selected graph-owned parallel SCF requires "
        "positive fixed lane steps");
  return ::mlir::success();
}

void forEachParallelPoint(
    const FixedParallelDomain &domain, unsigned dimension,
    ::llvm::SmallVectorImpl<int64_t> &point,
    ::llvm::function_ref<void(::llvm::ArrayRef<int64_t>)> callback) {
  if (dimension == domain.lower.size()) {
    callback(point);
    return;
  }
  int64_t value = domain.lower[dimension];
  while (value < domain.upper[dimension]) {
    point.push_back(value);
    forEachParallelPoint(domain, dimension + 1, point, callback);
    point.pop_back();
    if (value > std::numeric_limits<int64_t>::max() - domain.step[dimension])
      break;
    value += domain.step[dimension];
  }
}

bool isCompilerOwnedControlUse(::mlir::OpOperand &use) {
  ::mlir::Operation *owner = use.getOwner();
  if (auto load = ::llvm::dyn_cast<::dataflow::LoadOp>(owner))
    return &use == &load.getCtrlMutable();
  if (auto store = ::llvm::dyn_cast<::dataflow::StoreOp>(owner))
    return &use == &store.getCtrlMutable();
  if (auto constant = ::llvm::dyn_cast<::dataflow::ConstantOp>(owner))
    return &use == &constant.getCtrlMutable();
  return false;
}

bool isUseInside(::mlir::OpOperand &use, ::mlir::Region &region) {
  return region.isAncestor(use.getOwner()->getParentRegion());
}

void replaceUsesInside(::mlir::Value from, ::mlir::Value to,
                       ::mlir::Region &region) {
  from.replaceUsesWithIf(
      to, [&](::mlir::OpOperand &use) { return isUseInside(use, region); });
}

::mlir::LogicalResult checkOneGraph(::dataflow::GraphOp graph,
                                    const StreamBoundaryInfo &boundary) {
  ::mlir::Block &entry = graph.getBody().front();
  if (entry.getNumArguments() == 0 ||
      !::llvm::isa<::mlir::NoneType>(entry.getArgument(0).getType()))
    return graph.emitError(
        "loom-lower-graph-memory: graph entry must start with none");
  if (::mlir::failed(checkStreamBoundaryUses(graph, boundary)))
    return ::mlir::failure();

  ::mlir::WalkResult parallelResult =
      graph.getBody().walk([&](::mlir::Operation *op) {
        if (!::llvm::isa<::mlir::scf::ParallelOp, ::mlir::scf::ForallOp>(op))
          return ::mlir::WalkResult::advance();
        if (::mlir::failed(checkSelectedParallel(op)))
          return ::mlir::WalkResult::interrupt();
        return ::mlir::WalkResult::advance();
      });
  if (parallelResult.wasInterrupted())
    return ::mlir::failure();

  ::mlir::WalkResult result = graph.getBody().walk([&](::mlir::Operation *op)
                                                       -> ::mlir::WalkResult {
    if (auto load = ::llvm::dyn_cast<::mlir::memref::LoadOp>(op)) {
      if (load.getMemRefType().getRank() != 1 ||
          load.getIndices().size() != 1) {
        load.emitError("loom-lower-graph-memory: only rank-one memref.load "
                       "is supported by dataflow.load");
        return ::mlir::WalkResult::interrupt();
      }
    } else if (auto store = ::llvm::dyn_cast<::mlir::memref::StoreOp>(op)) {
      if (store.getMemRefType().getRank() != 1 ||
          store.getIndices().size() != 1) {
        store.emitError(
            "loom-lower-graph-memory: only rank-one memref.store is "
            "supported by dataflow.store");
        return ::mlir::WalkResult::interrupt();
      }
    }

    auto findMemoryCapability = [&](::mlir::TypeRange types) {
      for (::mlir::Type type : types)
        if (::dataflow::DataflowDialect::isMemoryCapabilityType(type))
          return type;
      return ::mlir::Type{};
    };
    if (::llvm::isa<::dataflow::CarryOp, ::dataflow::MuxOp, ::dataflow::DemuxOp,
                    ::dataflow::GateOp, ::dataflow::InvariantOp>(op)) {
      ::mlir::Type memory = findMemoryCapability(op->getOperandTypes());
      if (!memory)
        memory = findMemoryCapability(op->getResultTypes());
      if (memory) {
        op->emitError() << "cannot lower memory capability " << memory
                        << " through " << op->getName().getStringRef();
        return ::mlir::WalkResult::interrupt();
      }
    } else if (auto ifOp = ::llvm::dyn_cast<::mlir::scf::IfOp>(op)) {
      ::mlir::Type memory = findMemoryCapability(ifOp.getResultTypes());
      if (memory) {
        ifOp.emitError() << "cannot lower selected memory capability " << memory
                         << " through dataflow.mux/demux";
        return ::mlir::WalkResult::interrupt();
      }
    } else if (auto switchOp =
                   ::llvm::dyn_cast<::mlir::scf::IndexSwitchOp>(op)) {
      if (switchOp.getNumCases() == 0) {
        switchOp.emitError(
            "loom-lower-graph-memory: zero-case scf.index_switch requires "
            "upstream normalization before graph-region lowering");
        return ::mlir::WalkResult::interrupt();
      }
      ::mlir::Type memory = findMemoryCapability(switchOp.getResultTypes());
      if (memory) {
        switchOp.emitError() << "cannot lower selected memory capability "
                             << memory << " through dataflow.mux/demux";
        return ::mlir::WalkResult::interrupt();
      }
    } else if (auto forOp = ::llvm::dyn_cast<::mlir::scf::ForOp>(op)) {
      ::mlir::Type memory =
          findMemoryCapability(forOp.getInitArgs().getTypes());
      if (memory) {
        forOp.emitError() << "cannot lower loop-carried memory capability "
                          << memory << " through dataflow.carry";
        return ::mlir::WalkResult::interrupt();
      }
      if (::mlir::failed(::loom::lowering::inferStreamStepKind(forOp))) {
        forOp.emitError("loom-lower-graph-memory: scf.for has invalid "
                        "'loom.stream_step_kind'");
        return ::mlir::WalkResult::interrupt();
      }
      if (::mlir::failed(::loom::lowering::inferStreamPredicate(forOp))) {
        forOp.emitError("loom-lower-graph-memory: scf.for has invalid "
                        "'loom.stream_predicate'");
        return ::mlir::WalkResult::interrupt();
      }
    } else if (auto whileOp = ::llvm::dyn_cast<::mlir::scf::WhileOp>(op)) {
      ::mlir::Type memory = findMemoryCapability(whileOp.getInits().getTypes());
      if (memory) {
        whileOp.emitError() << "cannot lower loop-carried memory capability "
                            << memory << " through dataflow.carry";
        return ::mlir::WalkResult::interrupt();
      }
    }
    if (::llvm::isa<::mlir::scf::SCFDialect>(op->getDialect()) &&
        !::llvm::isa<::mlir::scf::IfOp, ::mlir::scf::ForOp,
                     ::mlir::scf::WhileOp, ::mlir::scf::IndexSwitchOp,
                     ::mlir::scf::ParallelOp, ::mlir::scf::ForallOp,
                     ::mlir::scf::YieldOp, ::mlir::scf::ConditionOp,
                     ::mlir::scf::ReduceOp, ::mlir::scf::InParallelOp>(op)) {
      op->emitError("loom-lower-graph-memory: unsupported residual SCF "
                    "must be normalized before graph-region lowering");
      return ::mlir::WalkResult::interrupt();
    }
    bool modeled =
        ::llvm::isa<::mlir::scf::IfOp, ::mlir::scf::ForOp, ::mlir::scf::WhileOp,
                    ::mlir::scf::IndexSwitchOp, ::mlir::scf::ParallelOp,
                    ::mlir::scf::ForallOp, ::mlir::scf::YieldOp,
                    ::mlir::scf::ConditionOp, ::mlir::scf::ReduceOp,
                    ::mlir::scf::InParallelOp, ::dataflow::GraphReturnOp>(op) ||
        ::loom::lowering::isSupportedGraphLoweringLeaf(op);
    if (::llvm::isa<::dataflow::ChannelSendOp, ::dataflow::ChannelReceiveOp>(
            op))
      modeled = boundary.isTransient();
    if (!modeled && (op->getNumRegions() != 0 || op->getNumSuccessors() != 0)) {
      op->emitError()
          << "loom-lower-graph-memory: effectful or unmodeled graph "
             "operation '"
          << op->getName().getStringRef() << "' is unsupported";
      return ::mlir::WalkResult::interrupt();
    }
    if (!modeled) {
      op->emitError()
          << "loom-lower-graph-memory: operation '"
          << op->getName().getStringRef()
          << "' is not a registered canonical Dataflow actor or a supported "
             "graph-lowering operation";
      return ::mlir::WalkResult::interrupt();
    }
    return ::mlir::WalkResult::advance();
  });
  return result.wasInterrupted() ? ::mlir::failure() : ::mlir::success();
}

class GraphRegionLowerer {
public:
  GraphRegionLowerer(::dataflow::GraphOp graph,
                     const StreamBoundaryInfo &boundary, unsigned indexBits)
      : graph(graph), builder(graph.getContext()),
        entry(graph.getBody().front()), anchor(entry.getTerminator()),
        transientStreamBoundary(boundary.isTransient()), indexBits(indexBits) {
    for (auto [channel, payload] :
         ::llvm::zip_equal(boundary.inputChannels, boundary.inputPayloads))
      streamInputByChannel.try_emplace(channel, payload);
    for (auto [index, channel] : ::llvm::enumerate(boundary.outputChannels))
      streamOutputByChannel.try_emplace(channel, index);
    streamOutputs.resize(boundary.outputChannels.size());
    for (auto [index, channel] : ::llvm::enumerate(boundary.inputChannels))
      registerStreamPlan(channel, /*input=*/true, index);
    for (auto [index, channel] : ::llvm::enumerate(boundary.outputChannels))
      registerStreamPlan(channel, /*input=*/false, index);
  }

  ::mlir::LogicalResult run() {
    collectParallelDomains();
    collectPartitions();
    MemoryState initial(partitionCount);
    for (MemoryFrontier &frontier : initial)
      frontier = {graph.getStart(), graph.getStart()};

    RegionResult result =
        lowerBlock(entry, graph.getStart(), std::move(initial));
    assert(routedStreamInputs.empty() && streamOutputSlots.empty() &&
           "stream endpoint maps must be retired before publication");
    assert(streamChoiceUsers.empty() &&
           "stream choice selectors must be bound before SCF erasure");
    assert(streamRepeatUsers.empty() &&
           "stream repeat selectors must be bound before SCF erasure");
    ::loom::lowering::lowerGraphIndexDomains(graph, indexBits);
    auto returnOp = ::llvm::cast<::dataflow::GraphReturnOp>(anchor);
    if (transientStreamBoundary) {
      if (::llvm::any_of(streamOutputs,
                         [](const ::mlir::Value &value) { return !value; }))
        return graph.emitError(
            "loom-lower-graph-memory: stream output binding was not lowered");
      returnOp.getStreamsMutable().assign(streamOutputs);
    }
    finalizeReturn(returnOp, result);
    if (transientStreamBoundary)
      eraseTransientChannelArguments();
    return ::mlir::success();
  }

private:
  ::dataflow::GraphOp graph;
  ::mlir::OpBuilder builder;
  ::mlir::Block &entry;
  ::mlir::Operation *anchor;
  bool transientStreamBoundary;
  // Resolved once at the pass boundary and read-only from here on.
  unsigned indexBits;
  ::llvm::DenseMap<::mlir::Value, ::mlir::Value> streamInputByChannel;
  ::llvm::DenseMap<::mlir::Value, unsigned> streamOutputByChannel;
  ::llvm::SmallVector<::mlir::Value, 4> streamOutputs;
  std::vector<std::unique_ptr<StreamLoweringPlan>> streamPlans;
  ::llvm::DenseMap<::mlir::Operation *, ::mlir::Value> routedStreamInputs;
  ::llvm::DenseMap<::mlir::Operation *, StreamOutputSlot> streamOutputSlots;
  ::llvm::DenseMap<::mlir::Operation *,
                   ::llvm::SmallVector<::mlir::Operation *, 2>>
      streamChoiceUsers;
  ::llvm::DenseMap<::mlir::Operation *, ::llvm::SmallVector<StreamRepeatUse, 2>>
      streamRepeatUsers;
  unsigned partitionCount = 0;
  ::llvm::DenseMap<::mlir::Value, unsigned> partitionByRoot;
  ::mlir::Value sharedBoundaryRoot;
  ::llvm::DenseMap<::mlir::FlatSymbolRefAttr, ::mlir::Value> globalRoots;
  ::llvm::DenseMap<::mlir::Operation *, ::llvm::SmallVector<unsigned, 4>>
      partitionsByAccess;
  ::llvm::DenseMap<::mlir::Operation *, FixedParallelDomain> parallelDomains;

  void setInsertionPoint(::mlir::Location) {
    builder.setInsertionPoint(anchor);
  }

  void registerStreamPlan(::mlir::Value channel, bool input,
                          unsigned boundaryIndex) {
    auto plan = analyzeStreamBinding(channel, input);
    assert(::mlir::succeeded(plan) &&
           "stream plan preflight must agree with graph lowering");
    if ((*plan)->schedule->kind == StreamScheduleNode::Kind::Endpoint && !input)
      return;
    auto loweringPlan = std::make_unique<StreamLoweringPlan>(
        (*plan)->scope, channel, input, boundaryIndex,
        std::move((*plan)->schedule));
    streamPlans.push_back(std::move(loweringPlan));
  }

  ::llvm::SmallVector<::mlir::Value, 4> demux(::mlir::Value selector,
                                              ::mlir::Value input,
                                              unsigned width,
                                              ::mlir::Location loc) {
    assert(width > 1 && "multi-lane demux requires multiple outputs");
    setInsertionPoint(loc);
    ::llvm::SmallVector<::mlir::Type, 4> types(width, input.getType());
    auto op = ::dataflow::DemuxOp::create(builder, loc, types, selector, input);
    return {op.getOutputs().begin(), op.getOutputs().end()};
  }

  ::mlir::Value mux(::mlir::Value selector, ::mlir::ValueRange inputs,
                    ::mlir::Location loc) {
    assert(inputs.size() > 1 && "multi-lane mux requires multiple inputs");
    setInsertionPoint(loc);
    return ::dataflow::MuxOp::create(builder, loc, inputs.front().getType(),
                                     selector, inputs)
        .getOutput();
  }

  void prepareStreamPlans(::mlir::Block &block, ::mlir::Value execution) {
    for (const auto &ownedPlan : streamPlans) {
      StreamLoweringPlan *plan = ownedPlan.get();
      if (plan->scope != &block)
        continue;

      auto materialization = materializeStreamSchedule(
          *plan->schedule, execution, builder, anchor);
      plan->selector = materialization.selector;
      plan->event = materialization.event;
      plan->close = materialization.close;
      for (const auto &choiceUse : materialization.choiceSelectorUses)
        streamChoiceUsers[choiceUse.choice].push_back(choiceUse.user);
      for (const auto &repeatUse : materialization.repeatSelectorUses)
        streamRepeatUsers[repeatUse.repeat].push_back(
            {repeatUse.user, repeatUse.placeholder});

      if (plan->input) {
        ::mlir::Value input = streamInputByChannel.lookup(plan->channel);
        assert(input && "stream input plan must reference a graph stream port");
        ::llvm::SmallVector<::mlir::Value, 4> routed;
        if (materialization.endpoints.size() == 1) {
          auto lanes = demux(materialization.selector, input, 2, plan->loc);
          routed.push_back(lanes[1]);
        } else {
          routed = demux(plan->selector, input,
                         materialization.endpoints.size(), plan->loc);
        }
        for (auto [endpoint, payload] :
             ::llvm::zip_equal(materialization.endpoints, routed))
          routedStreamInputs.try_emplace(endpoint, payload);
      } else {
        plan->outputs.resize(materialization.endpoints.size());
        for (auto [index, endpoint] :
             ::llvm::enumerate(materialization.endpoints))
          streamOutputSlots.try_emplace(
              endpoint, StreamOutputSlot{plan, static_cast<unsigned>(index)});
      }
      plan->schedule.reset();
    }
  }

  RegionResult finishStreamPlans(::mlir::Block &block, RegionResult result) {
    ::llvm::SmallVector<::mlir::Value, 4> closeEvents;
    for (const auto &ownedPlan : streamPlans) {
      StreamLoweringPlan *plan = ownedPlan.get();
      if (plan->scope != &block)
        continue;

      if (plan->close)
        closeEvents.push_back(plan->close);
      if (!plan->input) {
        assert(::llvm::all_of(plan->outputs,
                              [](const ::mlir::Value &value) {
                                return static_cast<bool>(value);
                              }) &&
               "every stream output endpoint must be lowered");
        ::mlir::Value output =
            plan->outputs.size() == 1
                ? plan->outputs.front()
                : mux(plan->selector, plan->outputs, plan->loc);
        if (plan->outputs.size() == 1) {
          setInsertionPoint(plan->loc);
          auto publication = ::dataflow::SyncOp::create(
              builder, plan->loc,
              ::mlir::TypeRange{builder.getNoneType(), output.getType()},
              ::mlir::ValueRange{plan->event, output});
          output = publication.getOutputs()[1];
        }
        assert(!streamOutputs[plan->boundaryIndex] &&
               "stream output binding must be materialized once");
        streamOutputs[plan->boundaryIndex] = output;
      }
      plan->scope = nullptr;
      plan->channel = {};
      plan->selector = {};
      plan->event = {};
      plan->close = {};
      plan->outputs.clear();
    }
    if (!closeEvents.empty()) {
      closeEvents.insert(closeEvents.begin(), result.execution);
      result.execution = joinEvents(closeEvents, block.getParentOp()->getLoc());
    }
    return result;
  }

  void eraseTransientChannelArguments() {
    size_t canonicalArgumentCount = graph.getFunctionType().getNumInputs() + 1;
    while (entry.getNumArguments() > canonicalArgumentCount) {
      ::mlir::BlockArgument argument = entry.getArguments().back();
      assert(argument.use_empty() &&
             "stream endpoint lowering must remove every channel use");
      entry.eraseArgument(entry.getNumArguments() - 1);
    }
  }

  std::optional<::mlir::Value> findKnownRoot(::mlir::Value value) const {
    ::llvm::DenseSet<::mlir::Value> visited;
    while (value && visited.insert(value).second) {
      if (::llvm::isa<::mlir::BlockArgument>(value))
        return value;
      ::mlir::Operation *def = value.getDefiningOp();
      if (!def)
        return value;
      if (::llvm::isa<::mlir::memref::AllocOp, ::mlir::memref::AllocaOp,
                      ::mlir::memref::GetGlobalOp>(def))
        return value;
      if (auto view = ::llvm::dyn_cast<::mlir::ViewLikeOpInterface>(def)) {
        value = view.getViewSource();
        continue;
      }
      if (auto cast =
              ::llvm::dyn_cast<::mlir::UnrealizedConversionCastOp>(def)) {
        if (cast.getInputs().size() != 1)
          return std::nullopt;
        value = cast.getInputs().front();
        continue;
      }
      return std::nullopt;
    }
    return std::nullopt;
  }

  bool isMemoryCapabilityCapture(::mlir::Value value) {
    if (!::dataflow::DataflowDialect::isMemoryCapabilityType(value.getType()))
      return false;

    ::llvm::DenseSet<::mlir::Value> visited;
    while (value && visited.insert(value).second) {
      if (auto argument = ::llvm::dyn_cast<::mlir::BlockArgument>(value)) {
        if (argument.getOwner() != &entry || argument.getArgNumber() == 0)
          return true;
        return graph.getInputPortKind(argument.getArgNumber() - 1) ==
               ::dataflow::GraphPortKind::Memory;
      }

      ::mlir::Operation *def = value.getDefiningOp();
      if (!def)
        return true;
      if (::llvm::isa<::mlir::memref::AllocOp, ::mlir::memref::AllocaOp,
                      ::mlir::memref::GetGlobalOp, ::mlir::LLVM::AddressOfOp>(
              def))
        return true;
      if (auto view = ::llvm::dyn_cast<::mlir::ViewLikeOpInterface>(def)) {
        value = view.getViewSource();
        continue;
      }
      if (auto cast =
              ::llvm::dyn_cast<::mlir::UnrealizedConversionCastOp>(def)) {
        if (cast.getInputs().size() != 1)
          return true;
        value = cast.getInputs().front();
        continue;
      }
      if (auto gep = ::llvm::dyn_cast<::mlir::LLVM::GEPOp>(def)) {
        value = gep.getBase();
        continue;
      }
      return true;
    }
    return true;
  }

  ::llvm::SmallVector<::mlir::Value, 8>
  collectProjectedCaptures(::mlir::Region &region) {
    ::llvm::SetVector<::mlir::Value> candidates;
    ::mlir::getUsedValuesDefinedAbove(region, region, candidates);

    ::llvm::SmallVector<::mlir::Value, 8> captures;
    for (::mlir::Value value : candidates) {
      if (isMemoryCapabilityCapture(value) ||
          ::llvm::isa<::dataflow::ChannelType>(value.getType()))
        continue;
      bool hasSemanticUse = false;
      for (::mlir::OpOperand &use : value.getUses()) {
        if (!isUseInside(use, region))
          continue;
        if (!isCompilerOwnedControlUse(use)) {
          hasSemanticUse = true;
          break;
        }
      }
      if (hasSemanticUse)
        captures.push_back(value);
    }
    return captures;
  }

  bool hasExplicitNoAlias(::mlir::BlockArgument argument) const {
    if (argument.getOwner() != &entry || argument.getArgNumber() == 0)
      return false;
    ::mlir::DictionaryAttr attrs =
        ::mlir::function_interface_impl::getArgAttrDict(
            graph, argument.getArgNumber() - 1);
    return attrs && attrs.contains("llvm.noalias");
  }

  ::mlir::Value canonicalizeRoot(::mlir::Value root) {
    if (auto argument = ::llvm::dyn_cast<::mlir::BlockArgument>(root)) {
      if (argument.getOwner() == &entry) {
        assert(argument.getArgNumber() > 0 &&
               "start cannot be a memory capability root");
        assert(graph.getInputPortKind(argument.getArgNumber() - 1) ==
                   ::dataflow::GraphPortKind::Memory &&
               "boundary memory root must come from the memory segment");
      }
      if (argument.getOwner() == &entry && !hasExplicitNoAlias(argument)) {
        if (!sharedBoundaryRoot)
          sharedBoundaryRoot = root;
        return sharedBoundaryRoot;
      }
      return root;
    }
    if (auto global = root.getDefiningOp<::mlir::memref::GetGlobalOp>()) {
      return globalRoots.try_emplace(global.getNameAttr(), root).first->second;
    }
    return root;
  }

  ::mlir::Value getMemoryOperand(::mlir::Operation *op) const {
    if (auto load = ::llvm::dyn_cast<::dataflow::LoadOp>(op))
      return load.getMem();
    if (auto store = ::llvm::dyn_cast<::dataflow::StoreOp>(op))
      return store.getMem();
    if (auto load = ::llvm::dyn_cast<::mlir::memref::LoadOp>(op))
      return load.getMemref();
    if (auto store = ::llvm::dyn_cast<::mlir::memref::StoreOp>(op))
      return store.getMemref();
    return {};
  }

  bool isMemoryLeaf(::mlir::Operation *op) const {
    return static_cast<bool>(getMemoryOperand(op));
  }

  void collectParallelDomains() {
    graph.getBody().walk([&](::mlir::Operation *op) {
      if (!::llvm::isa<::mlir::scf::ParallelOp, ::mlir::scf::ForallOp>(op))
        return ::mlir::WalkResult::advance();
      auto domain = getFixedParallelDomain(op);
      assert(domain && "parallel preflight must establish a fixed lane domain");
      parallelDomains.try_emplace(op, std::move(*domain));
      return ::mlir::WalkResult::advance();
    });
  }

  void collectPartitions() {
    struct AccessRoot {
      ::mlir::Operation *op;
      std::optional<::mlir::Value> root;
    };
    ::llvm::SmallVector<AccessRoot, 8> accesses;
    bool hasUnknown = false;
    graph.getBody().walk([&](::mlir::Operation *op) {
      ::mlir::Value mem = getMemoryOperand(op);
      if (!mem)
        return ::mlir::WalkResult::advance();
      std::optional<::mlir::Value> root = findKnownRoot(mem);
      if (root)
        root = canonicalizeRoot(*root);
      accesses.push_back({op, root});
      if (!root) {
        hasUnknown = true;
        return ::mlir::WalkResult::advance();
      }
      if (!partitionByRoot.contains(*root)) {
        unsigned index = partitionCount++;
        partitionByRoot.try_emplace(*root, index);
      }
      return ::mlir::WalkResult::advance();
    });
    if (hasUnknown)
      ++partitionCount;
    for (const AccessRoot &access : accesses) {
      ::llvm::SmallVector<unsigned, 4> membership;
      if (access.root) {
        membership.push_back(partitionByRoot.find(*access.root)->second);
      } else {
        membership.reserve(partitionCount);
        for (unsigned i = 0; i < partitionCount; ++i)
          membership.push_back(i);
      }
      partitionsByAccess.try_emplace(access.op, std::move(membership));
    }
  }

  ::llvm::SmallVector<unsigned, 4> partitionsFor(::mlir::Operation *op) const {
    auto it = partitionsByAccess.find(op);
    assert(it != partitionsByAccess.end() && "memory leaf was not analyzed");
    return it->second;
  }

  ::llvm::SmallBitVector touchedPartitions(::mlir::Region &region) const {
    ::llvm::SmallBitVector touched(partitionCount);
    region.walk([&](::mlir::Operation *op) {
      if (!isMemoryLeaf(op))
        return ::mlir::WalkResult::advance();
      for (unsigned partition : partitionsFor(op))
        touched.set(partition);
      return ::mlir::WalkResult::advance();
    });
    return touched;
  }

  bool causallyDependsOn(::mlir::Value event, ::mlir::Value prerequisite,
                         ::llvm::DenseSet<::mlir::Value> &visited) const {
    if (event == prerequisite)
      return true;
    if (!event || !visited.insert(event).second)
      return false;
    ::mlir::Operation *def = event.getDefiningOp();
    if (!def)
      return false;

    if (auto sync = ::llvm::dyn_cast<::dataflow::SyncOp>(def)) {
      return ::llvm::any_of(sync.getInputs(), [&](::mlir::Value input) {
        ::llvm::DenseSet<::mlir::Value> inputVisited = visited;
        return causallyDependsOn(input, prerequisite, inputVisited);
      });
    }
    if (auto load = ::llvm::dyn_cast<::dataflow::LoadOp>(def))
      return event == load.getDone() &&
             causallyDependsOn(load.getCtrl(), prerequisite, visited);
    if (auto store = ::llvm::dyn_cast<::dataflow::StoreOp>(def))
      return event == store.getDone() &&
             causallyDependsOn(store.getCtrl(), prerequisite, visited);
    if (auto demux = ::llvm::dyn_cast<::dataflow::DemuxOp>(def))
      return causallyDependsOn(demux.getInput(), prerequisite, visited);
    if (auto mux = ::llvm::dyn_cast<::dataflow::MuxOp>(def)) {
      return ::llvm::all_of(mux.getInputs(), [&](::mlir::Value input) {
        ::llvm::DenseSet<::mlir::Value> laneVisited = visited;
        return causallyDependsOn(input, prerequisite, laneVisited);
      });
    }
    if (auto carry = ::llvm::dyn_cast<::dataflow::CarryOp>(def)) {
      ::llvm::DenseSet<::mlir::Value> initVisited = visited;
      ::llvm::DenseSet<::mlir::Value> carryVisited = visited;
      return causallyDependsOn(carry.getInit(), prerequisite, initVisited) &&
             causallyDependsOn(carry.getCarry(), prerequisite, carryVisited);
    }
    if (auto invariant = ::llvm::dyn_cast<::dataflow::InvariantOp>(def))
      return causallyDependsOn(invariant.getInit(), prerequisite, visited);
    if (auto gate = ::llvm::dyn_cast<::dataflow::GateOp>(def))
      return causallyDependsOn(gate.getBeforeValue(), prerequisite, visited);
    if (auto constant = ::llvm::dyn_cast<::dataflow::ConstantOp>(def))
      return causallyDependsOn(constant.getCtrl(), prerequisite, visited);
    return false;
  }

  bool causallyDependsOn(::mlir::Value event,
                         ::mlir::Value prerequisite) const {
    ::llvm::DenseSet<::mlir::Value> visited;
    return causallyDependsOn(event, prerequisite, visited);
  }

  ::llvm::SmallVector<::mlir::Value, 4>
  reduceEvents(::mlir::ValueRange inputs) const {
    ::llvm::SmallVector<::mlir::Value, 4> unique;
    for (::mlir::Value input : inputs)
      if (input && !::llvm::is_contained(unique, input))
        unique.push_back(input);

    ::llvm::SmallVector<::mlir::Value, 4> reduced;
    for (unsigned i = 0; i < unique.size(); ++i) {
      bool covered = false;
      for (unsigned j = 0; j < unique.size(); ++j) {
        if (i != j && causallyDependsOn(unique[j], unique[i])) {
          covered = true;
          break;
        }
      }
      if (!covered)
        reduced.push_back(unique[i]);
    }
    return reduced;
  }

  ::mlir::Value joinEvents(::mlir::ValueRange inputs, ::mlir::Location loc) {
    ::llvm::SmallVector<::mlir::Value, 4> reduced = reduceEvents(inputs);
    ::mlir::Value control;
    for (::mlir::Value input : reduced)
      if (::llvm::isa<::mlir::NoneType>(input.getType())) {
        control = input;
        break;
      }
    if (!control)
      for (::mlir::Value input : inputs)
        if (input && ::llvm::isa<::mlir::NoneType>(input.getType())) {
          control = input;
          break;
        }
    assert(control && "event join requires a none-typed control token");

    ::llvm::SmallVector<::mlir::Value, 4> rendezvous{control};
    for (::mlir::Value input : reduced)
      if (input != control)
        rendezvous.push_back(input);
    if (rendezvous.size() == 1)
      return control;

    setInsertionPoint(loc);
    ::llvm::SmallVector<::mlir::Type, 4> types;
    for (::mlir::Value input : rendezvous)
      types.push_back(input.getType());
    auto sync = ::dataflow::SyncOp::create(builder, loc, types, rendezvous);
    return sync.getOutputs().front();
  }

  ::mlir::Value programOrderControl(::mlir::Value execution,
                                    const MemoryState &memory,
                                    ::mlir::Location loc) {
    ::llvm::SmallVector<::mlir::Value, 8> inputs{execution};
    for (const MemoryFrontier &frontier : memory)
      inputs.push_back(frontier.read);
    return joinEvents(inputs, loc);
  }

  void finalizeReturn(::dataflow::GraphReturnOp returnOp,
                      const RegionResult &result) {
    ::llvm::SmallVector<::mlir::Value, 8> candidates;
    ::mlir::Value start = graph.getStart();
    if (result.execution != start)
      candidates.push_back(result.execution);
    for (const MemoryFrontier &frontier : result.memory)
      if (frontier.read != start)
        candidates.push_back(frontier.read);
    for (::mlir::Value witness : returnOp.getComplete())
      if (witness != start)
        candidates.push_back(witness);

    if (candidates.empty())
      candidates.push_back(start);
    ::llvm::SmallVector<::mlir::Value, 4> reduced = reduceEvents(candidates);
    assert(!reduced.empty() && "graph retirement must have a witness");

    ::llvm::SmallVector<::mlir::Value, 4> values(returnOp.getValues().begin(),
                                                 returnOp.getValues().end());
    auto eraseUnusedWriteFrontierMuxes = [&]() {
      for (const MemoryFrontier &frontier : result.memory) {
        auto mux = frontier.write.getDefiningOp<::dataflow::MuxOp>();
        if (mux && mux.getOutput().use_empty())
          mux.erase();
      }
    };
    if (values.empty()) {
      returnOp.getCompleteMutable().assign(reduced);
      eraseUnusedWriteFrontierMuxes();
      return;
    }

    ::mlir::Value publicationBase = joinEvents(reduced, returnOp.getLoc());
    ::llvm::SmallVector<::mlir::Value, 4> publicationFrontier;
    for (auto [index, value] : ::llvm::enumerate(values)) {
      setInsertionPoint(returnOp.getLoc());
      auto sync = ::dataflow::SyncOp::create(
          builder, returnOp.getLoc(),
          ::mlir::TypeRange{builder.getNoneType(), value.getType()},
          ::mlir::ValueRange{publicationBase, value});
      publicationFrontier.push_back(sync.getOutputs().front());
      values[index] = sync.getOutputs().back();
    }
    returnOp.getValuesMutable().assign(values);
    returnOp.getCompleteMutable().assign(publicationFrontier);
    eraseUnusedWriteFrontierMuxes();
  }

  std::pair<::mlir::Value, ::mlir::Value>
  demux(::mlir::Value selector, ::mlir::Value input, ::mlir::Location loc) {
    setInsertionPoint(loc);
    auto op = ::dataflow::DemuxOp::create(
        builder, loc, ::mlir::TypeRange{input.getType(), input.getType()},
        selector, input);
    return {op.getOutputs()[0], op.getOutputs()[1]};
  }

  ::mlir::Value mux(::mlir::Value selector, ::mlir::Value falseValue,
                    ::mlir::Value trueValue, ::mlir::Location loc) {
    setInsertionPoint(loc);
    return ::dataflow::MuxOp::create(builder, loc, falseValue.getType(),
                                     selector,
                                     ::mlir::ValueRange{falseValue, trueValue})
        .getOutput();
  }

  GatedValue gateTrueLane(::mlir::Value phase, ::mlir::Value value,
                          ::mlir::Location loc) {
    setInsertionPoint(loc);
    auto gate = ::dataflow::GateOp::create(builder, loc, builder.getI1Type(),
                                           value.getType(), phase, value);
    auto close = ::dataflow::DemuxOp::create(
        builder, loc, ::mlir::TypeRange{value.getType(), value.getType()},
        gate.getAfterCond(), gate.getAfterValue());
    return {gate.getAfterValue(), close.getOutputs()[0]};
  }

  ::llvm::SmallVector<::mlir::Value, 4>
  projectForCaptures(::mlir::Region &region, ::mlir::ValueRange captures,
                     ::mlir::Value phase, ::mlir::Location loc) {
    ::llvm::SmallVector<::mlir::Value, 4> closeEvents;
    for (::mlir::Value capture : captures) {
      setInsertionPoint(loc);
      ::mlir::Value raw = ::dataflow::InvariantOp::create(
                              builder, loc, capture.getType(), phase, capture)
                              .getOutput();
      GatedValue gated = gateTrueLane(phase, raw, loc);
      replaceUsesInside(capture, gated.value, region);
      closeEvents.push_back(gated.close);
    }
    return closeEvents;
  }

  ::llvm::SmallVector<::dataflow::InvariantOp, 4>
  projectWhileBeforeCaptures(::mlir::Region &region,
                             ::mlir::ValueRange captures,
                             ::mlir::Value condition, ::mlir::Location loc) {
    ::llvm::SmallVector<::dataflow::InvariantOp, 4> invariants;
    for (::mlir::Value capture : captures) {
      setInsertionPoint(loc);
      auto invariant = ::dataflow::InvariantOp::create(
          builder, loc, capture.getType(), condition, capture);
      replaceUsesInside(capture, invariant.getOutput(), region);
      invariants.push_back(invariant);
    }
    return invariants;
  }

  RegionResult lowerOperations(::llvm::ArrayRef<::mlir::Operation *> operations,
                               ::mlir::Value execution, MemoryState memory) {
    for (::mlir::Operation *op : operations) {
      if (auto ifOp = ::llvm::dyn_cast<::mlir::scf::IfOp>(op)) {
        RegionResult result = lowerIf(ifOp, execution, std::move(memory));
        execution = result.execution;
        memory = std::move(result.memory);
        continue;
      }
      if (auto switchOp = ::llvm::dyn_cast<::mlir::scf::IndexSwitchOp>(op)) {
        RegionResult result =
            lowerIndexSwitch(switchOp, execution, std::move(memory));
        execution = result.execution;
        memory = std::move(result.memory);
        continue;
      }
      if (auto forOp = ::llvm::dyn_cast<::mlir::scf::ForOp>(op)) {
        RegionResult result = lowerFor(forOp, execution, std::move(memory));
        execution = result.execution;
        memory = std::move(result.memory);
        continue;
      }
      if (auto whileOp = ::llvm::dyn_cast<::mlir::scf::WhileOp>(op)) {
        RegionResult result = lowerWhile(whileOp, execution, std::move(memory));
        execution = result.execution;
        memory = std::move(result.memory);
        continue;
      }
      if (auto parallel = ::llvm::dyn_cast<::mlir::scf::ParallelOp>(op)) {
        RegionResult result =
            lowerParallel(parallel.getOperation(), parallel.getRegion().front(),
                          parallel.getInductionVars(), execution, memory);
        execution = result.execution;
        memory = std::move(result.memory);
        continue;
      }
      if (auto forall = ::llvm::dyn_cast<::mlir::scf::ForallOp>(op)) {
        RegionResult result =
            lowerParallel(forall.getOperation(), forall.getRegion().front(),
                          forall.getInductionVars(), execution, memory);
        execution = result.execution;
        memory = std::move(result.memory);
        continue;
      }
      if (auto receive = ::llvm::dyn_cast<::dataflow::ChannelReceiveOp>(op)) {
        ::mlir::Value payload;
        if (auto routed = routedStreamInputs.find(op);
            routed != routedStreamInputs.end()) {
          payload = routed->second;
          routedStreamInputs.erase(routed);
        } else {
          payload = streamInputByChannel.lookup(receive.getChannel());
        }
        assert(payload && "stream receive must use a published input binding");
        ::mlir::Value control =
            programOrderControl(execution, memory, receive.getLoc());
        setInsertionPoint(receive.getLoc());
        auto sync = ::dataflow::SyncOp::create(
            builder, receive.getLoc(),
            ::mlir::TypeRange{builder.getNoneType(), payload.getType()},
            ::mlir::ValueRange{control, payload});
        receive.getMessage().replaceAllUsesWith(sync.getOutputs()[1]);
        execution = sync.getOutputs()[0];
        receive.erase();
        continue;
      }
      if (auto send = ::llvm::dyn_cast<::dataflow::ChannelSendOp>(op)) {
        auto output = streamOutputByChannel.find(send.getChannel());
        assert(output != streamOutputByChannel.end() &&
               "stream send must use a published output binding");
        ::mlir::Value control =
            programOrderControl(execution, memory, send.getLoc());
        setInsertionPoint(send.getLoc());
        auto sync = ::dataflow::SyncOp::create(
            builder, send.getLoc(),
            ::mlir::TypeRange{builder.getNoneType(),
                              send.getMessage().getType()},
            ::mlir::ValueRange{control, send.getMessage()});
        if (auto slot = streamOutputSlots.find(op);
            slot != streamOutputSlots.end()) {
          StreamOutputSlot target = slot->second;
          assert(!target.plan->outputs[target.index] &&
                 "stream endpoint must be lowered once");
          target.plan->outputs[target.index] = sync.getOutputs()[1];
          streamOutputSlots.erase(slot);
        } else {
          assert(!streamOutputs[output->second] &&
                 "single-site stream binding must be materialized once");
          streamOutputs[output->second] = sync.getOutputs()[1];
        }
        execution = sync.getOutputs()[0];
        send.erase();
        continue;
      }
      if (auto load = ::llvm::dyn_cast<::mlir::memref::LoadOp>(op)) {
        lowerMemrefLoad(load, execution, memory);
        continue;
      }
      if (auto store = ::llvm::dyn_cast<::mlir::memref::StoreOp>(op)) {
        lowerMemrefStore(store, execution, memory);
        continue;
      }
      if (auto load = ::llvm::dyn_cast<::dataflow::LoadOp>(op)) {
        lowerDataflowLoad(load, execution, memory);
        continue;
      }
      if (auto store = ::llvm::dyn_cast<::dataflow::StoreOp>(op)) {
        lowerDataflowStore(store, execution, memory);
        continue;
      }
      if (auto constant = ::llvm::dyn_cast<::mlir::arith::ConstantOp>(op)) {
        if (constant->getBlock() == &entry)
          continue;
        setInsertionPoint(constant.getLoc());
        ::mlir::Value value =
            ::dataflow::ConstantOp::create(builder, constant.getLoc(),
                                           constant.getType(), execution,
                                           constant.getValue())
                .getValue();
        constant.getResult().replaceAllUsesWith(value);
        constant.erase();
        continue;
      }
      if (auto constant = ::llvm::dyn_cast<::dataflow::ConstantOp>(op))
        constant.getCtrlMutable().assign(execution);

      if (op->getBlock() != &entry) {
        assert(op->getNumRegions() == 0 && ::mlir::isPure(op) &&
               "graph preflight admitted an unmovable operation");
        op->moveBefore(anchor);
      }
    }
    return {execution, std::move(memory)};
  }

  RegionResult lowerBlock(::mlir::Block &block, ::mlir::Value execution,
                          MemoryState memory) {
    ::llvm::SmallVector<::mlir::Operation *, 16> operations;
    for (::mlir::Operation &op : block.without_terminator())
      operations.push_back(&op);
    prepareStreamPlans(block, execution);
    return finishStreamPlans(
        block, lowerOperations(operations, execution, std::move(memory)));
  }

  void registerClonedMemoryAccesses(::mlir::Block &source,
                                    ::mlir::IRMapping &mapping) {
    source.walk([&](::mlir::Operation *op) {
      if (::llvm::isa<::mlir::scf::ParallelOp, ::mlir::scf::ForallOp>(op)) {
        ::mlir::Operation *clone = mapping.lookupOrNull(op);
        assert(clone && "cloned parallel op must be present in IR mapping");
        auto domain = parallelDomains.find(op);
        assert(domain != parallelDomains.end() &&
               "cloned parallel op must have a fixed domain");
        parallelDomains.try_emplace(clone, domain->second);
      }
      if (!isMemoryLeaf(op))
        return ::mlir::WalkResult::advance();
      ::mlir::Operation *clone = mapping.lookupOrNull(op);
      assert(clone && "cloned memory access must be present in IR mapping");
      partitionsByAccess.try_emplace(clone, partitionsFor(op));
      return ::mlir::WalkResult::advance();
    });
  }

  RegionResult lowerParallel(::mlir::Operation *parallel, ::mlir::Block &body,
                             ::mlir::ValueRange inductionVars,
                             ::mlir::Value execution, MemoryState memory) {
    auto domain = parallelDomains.find(parallel);
    assert(domain != parallelDomains.end() &&
           "parallel preflight must establish a fixed lane domain");
    ::mlir::Location loc = parallel->getLoc();
    ::llvm::SmallVector<RegionResult, 4> lanes;
    ::llvm::SmallVector<int64_t, 4> point;
    forEachParallelPoint(
        domain->second, 0, point, [&](::llvm::ArrayRef<int64_t> coordinates) {
          ::mlir::IRMapping mapping;
          ::llvm::SmallVector<::mlir::Operation *, 16> laneOperations;
          builder.setInsertionPoint(parallel);
          for (auto [iv, coordinate] :
               ::llvm::zip_equal(inductionVars, coordinates)) {
            auto constant = ::mlir::arith::ConstantOp::create(
                builder, loc, builder.getIndexAttr(coordinate));
            mapping.map(iv, constant.getResult());
            laneOperations.push_back(constant);
          }
          for (::mlir::Operation &op : body.without_terminator())
            laneOperations.push_back(builder.clone(op, mapping));
          registerClonedMemoryAccesses(body, mapping);
          lanes.push_back(lowerOperations(laneOperations, execution, memory));
        });

    if (lanes.empty()) {
      parallel->erase();
      return {execution, std::move(memory)};
    }

    ::llvm::SmallVector<::mlir::Value, 8> laneEvents;
    laneEvents.reserve(lanes.size());
    for (const RegionResult &lane : lanes)
      laneEvents.push_back(lane.execution);
    ::mlir::Value outputExecution = joinEvents(laneEvents, loc);

    MemoryState output = memory;
    for (unsigned partition = 0; partition < partitionCount; ++partition) {
      ::llvm::SmallVector<::mlir::Value, 8> writes;
      ::llvm::SmallVector<::mlir::Value, 8> reads;
      writes.reserve(lanes.size());
      reads.reserve(lanes.size());
      for (const RegionResult &lane : lanes) {
        writes.push_back(lane.memory[partition].write);
        reads.push_back(lane.memory[partition].read);
      }
      ::mlir::Value write = joinEvents(writes, loc);
      ::mlir::Value read = writes == reads ? write : joinEvents(reads, loc);
      output[partition] = {write, read};
    }
    parallel->erase();
    return {outputExecution, std::move(output)};
  }

  void updateReadFrontiers(::mlir::Operation *op, ::mlir::Value done,
                           MemoryState &memory) {
    for (unsigned partition : partitionsFor(op)) {
      MemoryFrontier &frontier = memory[partition];
      frontier.read =
          joinEvents(::mlir::ValueRange{frontier.read, done}, op->getLoc());
    }
  }

  ::mlir::Value readControl(::mlir::Operation *op, ::mlir::Value execution,
                            MemoryState &memory) {
    ::llvm::SmallVector<::mlir::Value, 8> inputs{execution};
    for (unsigned partition : partitionsFor(op))
      inputs.push_back(memory[partition].write);
    return joinEvents(inputs, op->getLoc());
  }

  ::mlir::Value writeControl(::mlir::Operation *op, ::mlir::Value execution,
                             MemoryState &memory) {
    ::llvm::SmallVector<::mlir::Value, 8> inputs{execution};
    for (unsigned partition : partitionsFor(op))
      inputs.push_back(memory[partition].read);
    return joinEvents(inputs, op->getLoc());
  }

  void updateWriteFrontiers(::mlir::Operation *op, ::mlir::Value done,
                            MemoryState &memory) {
    for (unsigned partition : partitionsFor(op))
      memory[partition] = {done, done};
  }

  void lowerMemrefLoad(::mlir::memref::LoadOp load, ::mlir::Value execution,
                       MemoryState &memory) {
    ::llvm::SmallVector<unsigned, 4> membership = partitionsFor(load);
    ::mlir::Value ctrl = readControl(load, execution, memory);
    setInsertionPoint(load.getLoc());
    auto lowered = ::dataflow::LoadOp::create(
        builder, load.getLoc(), load.getType(), builder.getNoneType(),
        load.getMemref(), load.getIndices().front(), ctrl);
    partitionsByAccess.try_emplace(lowered, std::move(membership));
    load.getResult().replaceAllUsesWith(lowered.getData());
    updateReadFrontiers(lowered, lowered.getDone(), memory);
    load.erase();
  }

  void lowerMemrefStore(::mlir::memref::StoreOp store, ::mlir::Value execution,
                        MemoryState &memory) {
    ::llvm::SmallVector<unsigned, 4> membership = partitionsFor(store);
    ::mlir::Value ctrl = writeControl(store, execution, memory);
    setInsertionPoint(store.getLoc());
    auto lowered = ::dataflow::StoreOp::create(
        builder, store.getLoc(), builder.getNoneType(), store.getMemref(),
        store.getIndices().front(), store.getValue(), ctrl);
    partitionsByAccess.try_emplace(lowered, std::move(membership));
    updateWriteFrontiers(lowered, lowered.getDone(), memory);
    store.erase();
  }

  void lowerDataflowLoad(::dataflow::LoadOp load, ::mlir::Value execution,
                         MemoryState &memory) {
    load.getCtrlMutable().assign(readControl(load, execution, memory));
    updateReadFrontiers(load, load.getDone(), memory);
    if (load->getBlock() != &entry)
      load->moveBefore(anchor);
  }

  void lowerDataflowStore(::dataflow::StoreOp store, ::mlir::Value execution,
                          MemoryState &memory) {
    store.getCtrlMutable().assign(writeControl(store, execution, memory));
    updateWriteFrontiers(store, store.getDone(), memory);
    if (store->getBlock() != &entry)
      store->moveBefore(anchor);
  }

  RegionResult lowerSelection(::mlir::Operation *selection,
                              ::mlir::Value selector,
                              ::llvm::ArrayRef<::mlir::Region *> laneRegions,
                              ::mlir::Value execution, MemoryState memory) {
    assert(laneRegions.size() > 1 && "selection requires multiple lanes");
    ::mlir::Location loc = selection->getLoc();
    if (auto uses = streamChoiceUsers.find(selection);
        uses != streamChoiceUsers.end()) {
      for (::mlir::Operation *user : uses->second)
        user->setOperand(0, selector);
      streamChoiceUsers.erase(uses);
    }
    ::llvm::SmallVector<::mlir::Value, 4> laneExecutions =
        demux(selector, execution, laneRegions.size(), loc);

    ::llvm::SmallBitVector touched(partitionCount);
    for (::mlir::Region *region : laneRegions)
      if (region)
        touched |= touchedPartitions(*region);

    std::vector<MemoryState> laneMemory(laneRegions.size(), memory);
    for (int partition = touched.find_first(); partition >= 0;
         partition = touched.find_next(partition)) {
      ::llvm::SmallVector<::mlir::Value, 4> writes =
          demux(selector, memory[partition].write, laneRegions.size(), loc);
      ::llvm::SmallVector<::mlir::Value, 4> reads =
          demux(selector, memory[partition].read, laneRegions.size(), loc);
      for (unsigned lane = 0; lane < laneRegions.size(); ++lane)
        laneMemory[lane][partition] = {writes[lane], reads[lane]};
    }

    ::llvm::SetVector<::mlir::Value> captures;
    for (::mlir::Region &region : selection->getRegions())
      if (!region.empty())
        for (::mlir::Value value : collectProjectedCaptures(region))
          captures.insert(value);
    for (::mlir::Value capture : captures) {
      ::llvm::SmallVector<::mlir::Value, 4> projected =
          demux(selector, capture, laneRegions.size(), loc);
      for (auto [lane, region] : ::llvm::enumerate(laneRegions))
        if (region)
          replaceUsesInside(capture, projected[lane], *region);
    }

    ::llvm::SmallVector<RegionResult, 4> laneResults;
    laneResults.reserve(laneRegions.size());
    for (auto [lane, region] : ::llvm::enumerate(laneRegions)) {
      if (!region) {
        laneResults.push_back(
            {laneExecutions[lane], std::move(laneMemory[lane])});
        continue;
      }
      laneResults.push_back(lowerBlock(region->front(), laneExecutions[lane],
                                       std::move(laneMemory[lane])));
    }

    for (unsigned resultIndex = 0; resultIndex < selection->getNumResults();
         ++resultIndex) {
      ::llvm::SmallVector<::mlir::Value, 4> inputs;
      for (::mlir::Region *region : laneRegions) {
        assert(region && "value-producing selection requires every lane");
        auto yield =
            ::llvm::cast<::mlir::scf::YieldOp>(region->front().getTerminator());
        inputs.push_back(yield.getOperand(resultIndex));
      }
      selection->getResult(resultIndex)
          .replaceAllUsesWith(mux(selector, inputs, loc));
    }

    MemoryState output = memory;
    for (int partition = touched.find_first(); partition >= 0;
         partition = touched.find_next(partition)) {
      ::llvm::SmallVector<::mlir::Value, 4> writes;
      ::llvm::SmallVector<::mlir::Value, 4> reads;
      for (const RegionResult &lane : laneResults) {
        writes.push_back(lane.memory[partition].write);
        reads.push_back(lane.memory[partition].read);
      }
      output[partition] = {mux(selector, writes, loc),
                           mux(selector, reads, loc)};
    }
    ::llvm::SmallVector<::mlir::Value, 4> completions;
    for (const RegionResult &lane : laneResults)
      completions.push_back(lane.execution);
    ::mlir::Value outputExecution = mux(selector, completions, loc);
    selection->erase();
    return {outputExecution, std::move(output)};
  }

  ::mlir::Value
  normalizeIndexSwitchSelector(::mlir::scf::IndexSwitchOp switchOp,
                               ::mlir::Value execution) {
    assert(switchOp.getNumCases() > 0 &&
           "zero-case switch must fail graph preflight");
    ::mlir::Location loc = switchOp.getLoc();
    auto controlledIndex = [&](int64_t value) {
      setInsertionPoint(loc);
      return ::dataflow::ConstantOp::create(builder, loc,
                                            builder.getIndexType(), execution,
                                            builder.getIndexAttr(value))
          .getValue();
    };
    auto caseMatch = [&](int64_t value) {
      ::mlir::Value constant = controlledIndex(value);
      setInsertionPoint(loc);
      return ::mlir::arith::CmpIOp::create(builder, loc,
                                           ::mlir::arith::CmpIPredicate::eq,
                                           switchOp.getArg(), constant)
          .getResult();
    };

    if (switchOp.getNumCases() == 1)
      return caseMatch(switchOp.getCases().front());

    ::mlir::Value selector = controlledIndex(0);
    for (auto [caseIndex, caseValue] : ::llvm::enumerate(switchOp.getCases())) {
      ::mlir::Value match = caseMatch(caseValue);
      ::mlir::Value lane = controlledIndex(caseIndex + 1);
      setInsertionPoint(loc);
      selector =
          ::mlir::arith::SelectOp::create(builder, loc, match, lane, selector)
              .getResult();
    }
    return selector;
  }

  RegionResult lowerIf(::mlir::scf::IfOp ifOp, ::mlir::Value execution,
                       MemoryState memory) {
    ::llvm::SmallVector<::mlir::Region *, 2> lanes;
    lanes.push_back(ifOp.getElseRegion().empty() ? nullptr
                                                 : &ifOp.getElseRegion());
    lanes.push_back(&ifOp.getThenRegion());
    return lowerSelection(ifOp, ifOp.getCondition(), lanes, execution,
                          std::move(memory));
  }

  RegionResult lowerIndexSwitch(::mlir::scf::IndexSwitchOp switchOp,
                                ::mlir::Value execution, MemoryState memory) {
    ::mlir::Value selector = normalizeIndexSwitchSelector(switchOp, execution);
    ::llvm::SmallVector<::mlir::Region *, 4> lanes;
    lanes.push_back(&switchOp.getDefaultRegion());
    for (::mlir::Region &region : switchOp.getCaseRegions())
      lanes.push_back(&region);
    return lowerSelection(switchOp, selector, lanes, execution,
                          std::move(memory));
  }

  RegionResult lowerFor(::mlir::scf::ForOp forOp, ::mlir::Value execution,
                        MemoryState memory) {
    ::mlir::Location loc = forOp.getLoc();
    setInsertionPoint(loc);

    ::mlir::Value lower = forOp.getLowerBound();
    ::mlir::Value upper = forOp.getUpperBound();
    ::mlir::Value step = forOp.getStep();
    ::mlir::Type streamType = lower.getType();
    bool indexLoop = ::llvm::isa<::mlir::IndexType>(streamType);
    if (indexLoop) {
      streamType = ::mlir::IntegerType::get(graph.getContext(), indexBits);
      lower =
          ::mlir::arith::IndexCastOp::create(builder, loc, streamType, lower)
              .getResult();
      upper =
          ::mlir::arith::IndexCastOp::create(builder, loc, streamType, upper)
              .getResult();
      step = ::mlir::arith::IndexCastOp::create(builder, loc, streamType, step)
                 .getResult();
    }
    auto stepKind = ::loom::lowering::inferStreamStepKind(forOp);
    auto predicate = ::loom::lowering::inferStreamPredicate(forOp);
    auto stream = ::dataflow::StreamOp::create(
        builder, loc, streamType, builder.getI1Type(), lower, upper, step,
        *stepKind, *predicate);
    ::mlir::Value phase = stream.getPhase();
    ::mlir::Value bodyIv = stream.getIv();
    if (indexLoop)
      bodyIv = ::mlir::arith::IndexCastOp::create(
                   builder, loc, builder.getIndexType(), bodyIv)
                   .getResult();

    auto executionCarry = ::dataflow::CarryOp::create(
        builder, loc, builder.getNoneType(), phase, execution, execution);
    auto [executionExit, executionBody] =
        demux(phase, executionCarry.getOutput(), loc);

    ::llvm::SmallVector<::mlir::Value, 8> captures =
        collectProjectedCaptures(forOp.getRegion());
    ::llvm::SmallVector<::dataflow::CarryOp, 4> valueCarries;
    ::llvm::SmallVector<::mlir::Value, 4> valueExits;
    for (::mlir::Value init : forOp.getInitArgs()) {
      setInsertionPoint(loc);
      auto carry = ::dataflow::CarryOp::create(builder, loc, init.getType(),
                                               phase, init, init);
      auto [exit, body] = demux(phase, carry.getOutput(), loc);
      valueCarries.push_back(carry);
      valueExits.push_back(exit);
      replaceUsesInside(forOp.getRegionIterArgs()[valueCarries.size() - 1],
                        body, forOp.getRegion());
    }
    replaceUsesInside(forOp.getInductionVar(), bodyIv, forOp.getRegion());
    ::llvm::SmallVector<::mlir::Value, 4> captureCloses =
        projectForCaptures(forOp.getRegion(), captures, phase, loc);
    if (auto uses = streamRepeatUsers.find(forOp);
        uses != streamRepeatUsers.end()) {
      setInsertionPoint(loc);
      auto active =
          ::dataflow::ConstantOp::create(builder, loc, builder.getI1Type(),
                                         execution, builder.getBoolAttr(true));
      auto invariant = ::dataflow::InvariantOp::create(
          builder, loc, builder.getI1Type(), phase, active.getValue());
      auto projected = ::dataflow::DemuxOp::create(
          builder, loc,
          ::mlir::TypeRange{builder.getI1Type(), builder.getI1Type()}, phase,
          invariant.getOutput());
      for (const StreamRepeatUse &use : uses->second) {
        use.user->setOperand(0, projected.getOutputs()[1]);
        assert(use.placeholder->use_empty() &&
               "stream repeat placeholder must be fully retired");
        use.placeholder->erase();
      }
      auto close = ::dataflow::SyncOp::create(
          builder, loc,
          ::mlir::TypeRange{builder.getNoneType(), builder.getI1Type()},
          ::mlir::ValueRange{executionExit, projected.getOutputs()[0]});
      executionExit = close.getOutputs()[0];
      streamRepeatUsers.erase(uses);
    }

    ::llvm::SmallBitVector touched = touchedPartitions(forOp.getRegion());
    MemoryState bodyMemory = memory;
    ::llvm::SmallVector<std::optional<::dataflow::CarryOp>, 4> writeCarries(
        partitionCount);
    ::llvm::SmallVector<std::optional<::dataflow::CarryOp>, 4> readCarries(
        partitionCount);
    ::llvm::SmallVector<::mlir::Value, 4> writeExits(partitionCount);
    ::llvm::SmallVector<::mlir::Value, 4> readExits(partitionCount);
    for (int partition = touched.find_first(); partition >= 0;
         partition = touched.find_next(partition)) {
      setInsertionPoint(loc);
      auto writeCarry = ::dataflow::CarryOp::create(
          builder, loc, builder.getNoneType(), phase, memory[partition].write,
          memory[partition].write);
      auto readCarry = ::dataflow::CarryOp::create(
          builder, loc, builder.getNoneType(), phase, memory[partition].read,
          memory[partition].read);
      auto [writeExit, writeBody] = demux(phase, writeCarry.getOutput(), loc);
      auto [readExit, readBody] = demux(phase, readCarry.getOutput(), loc);
      writeCarries[partition] = writeCarry;
      readCarries[partition] = readCarry;
      writeExits[partition] = writeExit;
      readExits[partition] = readExit;
      bodyMemory[partition] = {writeBody, readBody};
    }

    RegionResult bodyResult = lowerBlock(forOp.getRegion().front(),
                                         executionBody, std::move(bodyMemory));
    auto yield = ::llvm::cast<::mlir::scf::YieldOp>(
        forOp.getRegion().front().getTerminator());
    executionCarry.getCarryMutable().assign(bodyResult.execution);
    for (unsigned i = 0; i < valueCarries.size(); ++i)
      valueCarries[i].getCarryMutable().assign(yield.getOperand(i));

    MemoryState output = memory;
    for (int partition = touched.find_first(); partition >= 0;
         partition = touched.find_next(partition)) {
      writeCarries[partition]->getCarryMutable().assign(
          bodyResult.memory[partition].write);
      readCarries[partition]->getCarryMutable().assign(
          bodyResult.memory[partition].read);
      output[partition] = {writeExits[partition], readExits[partition]};
    }
    for (unsigned i = 0; i < forOp.getNumResults(); ++i)
      forOp.getResult(i).replaceAllUsesWith(valueExits[i]);
    if (!captureCloses.empty()) {
      setInsertionPoint(loc);
      ::mlir::Value nonEmpty =
          ::mlir::arith::CmpIOp::create(builder, loc, *predicate, lower, upper);
      auto [emptyExit, activeExit] = demux(nonEmpty, executionExit, loc);
      captureCloses.insert(captureCloses.begin(), activeExit);
      executionExit =
          mux(nonEmpty, emptyExit, joinEvents(captureCloses, loc), loc);
    }
    forOp.erase();
    return {executionExit, std::move(output)};
  }

  RegionResult lowerWhile(::mlir::scf::WhileOp whileOp, ::mlir::Value execution,
                          MemoryState memory) {
    ::mlir::Location loc = whileOp.getLoc();
    auto condition = ::llvm::cast<::mlir::scf::ConditionOp>(
        whileOp.getBefore().front().getTerminator());

    ::llvm::SmallVector<::mlir::Value, 8> beforeCaptures =
        collectProjectedCaptures(whileOp.getBefore());
    ::llvm::SmallVector<::mlir::Value, 8> afterCaptures =
        collectProjectedCaptures(whileOp.getAfter());

    setInsertionPoint(loc);
    ::mlir::Value pendingSelector =
        ::mlir::arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                          builder.getBoolAttr(false))
            .getResult();
    auto executionCarry =
        ::dataflow::CarryOp::create(builder, loc, builder.getNoneType(),
                                    pendingSelector, execution, execution);

    ::llvm::SmallVector<::dataflow::CarryOp, 4> valueCarries;
    for (::mlir::Value init : whileOp.getInits()) {
      auto carry = ::dataflow::CarryOp::create(builder, loc, init.getType(),
                                               pendingSelector, init, init);
      valueCarries.push_back(carry);
    }
    for (unsigned i = 0; i < valueCarries.size(); ++i)
      replaceUsesInside(whileOp.getBeforeArguments()[i],
                        valueCarries[i].getOutput(), whileOp.getBefore());
    ::llvm::SmallVector<::dataflow::InvariantOp, 4> beforeInvariants =
        projectWhileBeforeCaptures(whileOp.getBefore(), beforeCaptures,
                                   pendingSelector, loc);

    ::llvm::SmallBitVector touched = touchedPartitions(whileOp.getBefore());
    touched |= touchedPartitions(whileOp.getAfter());
    MemoryState beforeMemory = memory;
    ::llvm::SmallVector<std::optional<::dataflow::CarryOp>, 4> writeCarries(
        partitionCount);
    ::llvm::SmallVector<std::optional<::dataflow::CarryOp>, 4> readCarries(
        partitionCount);
    for (int partition = touched.find_first(); partition >= 0;
         partition = touched.find_next(partition)) {
      setInsertionPoint(loc);
      auto writeCarry = ::dataflow::CarryOp::create(
          builder, loc, builder.getNoneType(), pendingSelector,
          memory[partition].write, memory[partition].write);
      auto readCarry = ::dataflow::CarryOp::create(
          builder, loc, builder.getNoneType(), pendingSelector,
          memory[partition].read, memory[partition].read);
      writeCarries[partition] = writeCarry;
      readCarries[partition] = readCarry;
      beforeMemory[partition] = {writeCarry.getOutput(), readCarry.getOutput()};
    }

    RegionResult beforeResult =
        lowerBlock(whileOp.getBefore().front(), executionCarry.getOutput(),
                   std::move(beforeMemory));
    ::mlir::Value selector = condition.getCondition();
    executionCarry.getCondMutable().assign(selector);
    for (::dataflow::CarryOp carry : valueCarries)
      carry.getCondMutable().assign(selector);
    for (::dataflow::InvariantOp invariant : beforeInvariants)
      invariant.getCondMutable().assign(selector);
    for (int partition = touched.find_first(); partition >= 0;
         partition = touched.find_next(partition)) {
      writeCarries[partition]->getCondMutable().assign(selector);
      readCarries[partition]->getCondMutable().assign(selector);
    }
    pendingSelector.getDefiningOp()->erase();

    auto [executionExit, unusedExecution] =
        demux(selector, beforeResult.execution, loc);
    (void)unusedExecution;
    GatedValue gatedExecution =
        gateTrueLane(selector, beforeResult.execution, loc);
    ::mlir::Value executionAfter = gatedExecution.value;
    ::llvm::SmallVector<::mlir::Value, 4> closeEvents{gatedExecution.close};

    MemoryState afterMemory = beforeResult.memory;
    MemoryState output = memory;
    for (int partition = touched.find_first(); partition >= 0;
         partition = touched.find_next(partition)) {
      auto [writeExit, writeAfter] =
          demux(selector, beforeResult.memory[partition].write, loc);
      auto [readExit, readAfter] =
          demux(selector, beforeResult.memory[partition].read, loc);
      output[partition] = {writeExit, readExit};
      afterMemory[partition] = {writeAfter, readAfter};
    }

    ::llvm::SmallVector<::mlir::Value, 4> resultValues;
    for (::mlir::Value value : condition.getArgs()) {
      auto [exit, after] = demux(selector, value, loc);
      resultValues.push_back(exit);
      replaceUsesInside(whileOp.getAfterArguments()[resultValues.size() - 1],
                        after, whileOp.getAfter());
    }
    ::llvm::SmallVector<::mlir::Value, 4> afterCaptureCloses =
        projectForCaptures(whileOp.getAfter(), afterCaptures, selector, loc);
    closeEvents.append(afterCaptureCloses);

    RegionResult afterResult = lowerBlock(
        whileOp.getAfter().front(), executionAfter, std::move(afterMemory));
    auto yield = ::llvm::cast<::mlir::scf::YieldOp>(
        whileOp.getAfter().front().getTerminator());
    executionCarry.getCarryMutable().assign(afterResult.execution);
    for (unsigned i = 0; i < valueCarries.size(); ++i)
      valueCarries[i].getCarryMutable().assign(yield.getOperand(i));
    for (int partition = touched.find_first(); partition >= 0;
         partition = touched.find_next(partition)) {
      writeCarries[partition]->getCarryMutable().assign(
          afterResult.memory[partition].write);
      readCarries[partition]->getCarryMutable().assign(
          afterResult.memory[partition].read);
    }
    for (unsigned i = 0; i < whileOp.getNumResults(); ++i)
      whileOp.getResult(i).replaceAllUsesWith(resultValues[i]);
    whileOp.erase();
    closeEvents.insert(closeEvents.begin(), executionExit);
    return {joinEvents(closeEvents, loc), std::move(output)};
  }
};

} // namespace

namespace loom {
namespace lowering {

::mlir::LogicalResult
checkGraphRegionLoweringPreconditions(::mlir::ModuleOp module) {
  ::mlir::WalkResult result =
      module.walk([&](::dataflow::GraphOp graph) -> ::mlir::WalkResult {
        if (graph.isExternal())
          return ::mlir::WalkResult::advance();
        auto boundary = analyzeStreamBoundary(graph);
        if (::mlir::failed(boundary) ||
            ::mlir::failed(checkOneGraph(graph, *boundary)))
          return ::mlir::WalkResult::interrupt();
        return ::mlir::WalkResult::advance();
      });
  return result.wasInterrupted() ? ::mlir::failure() : ::mlir::success();
}

::mlir::LogicalResult lowerGraphRegions(::dataflow::GraphOp graph,
                                        unsigned indexBits) {
  auto boundary = analyzeStreamBoundary(graph);
  if (::mlir::failed(boundary))
    return ::mlir::failure();
  return GraphRegionLowerer(graph, *boundary, indexBits).run();
}

} // namespace lowering
} // namespace loom
