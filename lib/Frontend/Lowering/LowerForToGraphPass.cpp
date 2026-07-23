// Publish explicit `loom.spatial_region` operations inside dataflow.thread
// bodies as canonical `dataflow.graph` definitions and matching launches.
// The graph function_type contains only normalized application payload ports.
// Start/done remain explicit launch protocol endpoints, and graph.return owns
// the segmented payload boundary plus retirement frontier.

#include "Frontend/Lowering/Passes.h"
#include "GraphRegionLowering.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowGraphValidation.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/DataflowThreadCompletion.h"
#include "Frontend/IR/LoomDialect.h"
#include "Frontend/IR/LoomOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <array>

namespace {

bool followsOnSameStructuredPath(::mlir::Operation *first,
                                 ::mlir::Operation *second,
                                 ::mlir::Operation *scope) {
  if (::mlir::insideMutuallyExclusiveRegions(first, second))
    return false;

  for (::mlir::Operation *firstAncestor = first;
       firstAncestor && firstAncestor != scope;
       firstAncestor = firstAncestor->getParentOp()) {
    ::mlir::Block *block = firstAncestor->getBlock();
    for (::mlir::Operation *secondAncestor = second;
         secondAncestor && secondAncestor != scope;
         secondAncestor = secondAncestor->getParentOp()) {
      if (secondAncestor->getBlock() != block)
        continue;
      return firstAncestor != secondAncestor &&
             firstAncestor->isBeforeInBlock(secondAncestor);
    }
  }
  return false;
}

bool valueDependsOnSpatialResult(::mlir::Value value,
                                 ::loom::SpatialRegionOp spatial,
                                 ::llvm::DenseSet<::mlir::Value> &visited);

bool isNestedIn(::mlir::Operation *op, ::mlir::Operation *ancestor) {
  for (::mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (parent == ancestor)
      return true;
  return false;
}

bool spatialMayExistOnRegionPredecessor(::loom::SpatialRegionOp spatial,
                                        ::mlir::RegionBranchOpInterface branch,
                                        ::mlir::RegionBranchPoint predecessor) {
  if (predecessor.isParent())
    return !isNestedIn(spatial, branch.getOperation());
  auto terminator = predecessor.getTerminatorPredecessorOrNull();
  return !::mlir::insideMutuallyExclusiveRegions(spatial,
                                                 terminator.getOperation());
}

bool predecessorValuesDependOnSpatialResult(
    ::mlir::RegionBranchOpInterface branch, ::mlir::RegionSuccessor successor,
    ::mlir::Value value, ::loom::SpatialRegionOp spatial,
    ::llvm::DenseSet<::mlir::Value> &visited) {
  ::mlir::ValueRange inputs = branch.getSuccessorInputs(successor);
  auto position = ::llvm::find(inputs, value);
  if (position == inputs.end())
    return false;

  ::llvm::SmallVector<::mlir::Value, 4> predecessors;
  ::llvm::SmallVector<::mlir::RegionBranchPoint, 4> predecessorPoints;
  branch.getPredecessors(successor, predecessorPoints);
  branch.getPredecessorValues(
      successor, static_cast<unsigned>(std::distance(inputs.begin(), position)),
      predecessors);
  if (predecessorPoints.size() != predecessors.size())
    return false;

  bool sawSpatialPath = false;
  for (auto [predecessorPoint, predecessor] :
       ::llvm::zip_equal(predecessorPoints, predecessors)) {
    if (!spatialMayExistOnRegionPredecessor(spatial, branch, predecessorPoint))
      continue;
    sawSpatialPath = true;
    ::llvm::DenseSet<::mlir::Value> pathVisited = visited;
    if (!valueDependsOnSpatialResult(predecessor, spatial, pathVisited))
      return false;
  }
  return sawSpatialPath;
}

bool valueDependsOnSpatialResult(::mlir::Value value,
                                 ::loom::SpatialRegionOp spatial,
                                 ::llvm::DenseSet<::mlir::Value> &visited) {
  if (!value || !visited.insert(value).second)
    return false;

  if (auto result = ::llvm::dyn_cast<::mlir::OpResult>(value)) {
    ::mlir::Operation *definition = result.getOwner();
    if (definition == spatial.getOperation())
      return ::llvm::is_contained(spatial.getValueResults(), value);
    if (auto branch =
            ::llvm::dyn_cast<::mlir::RegionBranchOpInterface>(definition))
      return predecessorValuesDependOnSpatialResult(
          branch, ::mlir::RegionSuccessor(definition), value, spatial, visited);
    if (auto mux = ::llvm::dyn_cast<::dataflow::MuxOp>(definition)) {
      ::llvm::DenseSet<::mlir::Value> selectorVisited = visited;
      if (valueDependsOnSpatialResult(mux.getSel(), spatial, selectorVisited))
        return true;
      return !mux.getInputs().empty() &&
             ::llvm::all_of(mux.getInputs(), [&](::mlir::Value input) {
               ::llvm::DenseSet<::mlir::Value> pathVisited = visited;
               return valueDependsOnSpatialResult(input, spatial, pathVisited);
             });
    }
    if (auto carry = ::llvm::dyn_cast<::dataflow::CarryOp>(definition)) {
      ::llvm::DenseSet<::mlir::Value> pathVisited = visited;
      return valueDependsOnSpatialResult(carry.getInit(), spatial, pathVisited);
    }
    if (auto invariant =
            ::llvm::dyn_cast<::dataflow::InvariantOp>(definition)) {
      ::llvm::DenseSet<::mlir::Value> pathVisited = visited;
      return valueDependsOnSpatialResult(invariant.getInit(), spatial,
                                         pathVisited);
    }
    auto inputPhase =
        ::dataflow::semantics::getVectorBoundaryInputPhase(definition);
    auto outputPhase =
        ::dataflow::semantics::getVectorBoundaryOutputPhase(definition);
    if (inputPhase && outputPhase && value == *outputPhase) {
      ::llvm::DenseSet<::mlir::Value> pathVisited = visited;
      return valueDependsOnSpatialResult(*inputPhase, spatial, pathVisited);
    }
    return ::llvm::any_of(definition->getOperands(), [&](::mlir::Value input) {
      ::llvm::DenseSet<::mlir::Value> pathVisited = visited;
      return valueDependsOnSpatialResult(input, spatial, pathVisited);
    });
  }

  auto argument = ::llvm::dyn_cast<::mlir::BlockArgument>(value);
  auto branch = ::llvm::dyn_cast_or_null<::mlir::RegionBranchOpInterface>(
      argument.getOwner()->getParentOp());
  return branch &&
         predecessorValuesDependOnSpatialResult(
             branch, ::mlir::RegionSuccessor(argument.getOwner()->getParent()),
             value, spatial, visited);
}

bool isOrderedBySpatialResult(::mlir::Operation *continuation,
                              ::loom::SpatialRegionOp spatial) {
  auto thread = spatial->getParentOfType<::dataflow::ThreadOp>();
  for (::mlir::Operation *op = continuation; op && op != thread.getOperation();
       op = op->getParentOp())
    if (::llvm::any_of(op->getOperands(), [&](::mlir::Value operand) {
          ::llvm::DenseSet<::mlir::Value> visited;
          return valueDependsOnSpatialResult(operand, spatial, visited);
        }))
      return true;
  return false;
}

bool regionAlwaysPublishes(
    ::mlir::Region &region,
    ::llvm::ArrayRef<::mlir::Operation *> continuations) {
  if (region.empty())
    return false;
  for (::mlir::Operation &op : region.front()) {
    if (::llvm::is_contained(continuations, &op))
      return true;
    auto ifOp = ::llvm::dyn_cast<::mlir::scf::IfOp>(op);
    if (ifOp && ::llvm::all_of(ifOp->getRegions(), [&](::mlir::Region &branch) {
          return regionAlwaysPublishes(branch, continuations);
        }))
      return true;
  }
  return false;
}

::llvm::SmallVector<::mlir::Operation *, 4>
findChannelRetirementAnchors(::loom::SpatialRegionOp spatial) {
  if (spatial.getStreamInputs().empty())
    return {};

  auto thread = spatial->getParentOfType<::dataflow::ThreadOp>();
  ::llvm::SmallVector<::mlir::Operation *, 4> continuations;
  thread.walk([&](::mlir::Operation *op) {
    bool publishesChannel = false;
    if (auto send = ::llvm::dyn_cast<::dataflow::ChannelSendOp>(op)) {
      publishesChannel =
          send->getParentOfType<::dataflow::ThreadOp>() == thread &&
          !send->getParentOfType<::loom::SpatialRegionOp>() &&
          !send->getParentOfType<::dataflow::GraphOp>();
    } else if (auto target = ::llvm::dyn_cast<::loom::SpatialRegionOp>(op)) {
      publishesChannel =
          target != spatial && !target.getStreamOutputs().empty();
    } else if (auto launch = ::llvm::dyn_cast<::dataflow::GraphLaunchOp>(op)) {
      publishesChannel =
          launch->getParentOfType<::dataflow::ThreadOp>() == thread &&
          !launch->getParentOfType<::loom::SpatialRegionOp>() &&
          !launch->getParentOfType<::dataflow::GraphOp>() &&
          !launch.getStreamOutputs().empty();
    }
    if (publishesChannel && followsOnSameStructuredPath(spatial, op, thread))
      continuations.push_back(op);
  });

  // A preceding publication with deferred SSA readiness, a wait, or a launch
  // dependency orders later publications on the same path.
  ::mlir::DominanceInfo dominance(thread);
  ::mlir::PostDominanceInfo postDominance(thread);
  ::llvm::SmallVector<::mlir::scf::IfOp, 4> branches;
  thread.walk([&](::mlir::scf::IfOp ifOp) { branches.push_back(ifOp); });
  ::llvm::SmallVector<::mlir::Operation *, 4> anchors;
  for (::mlir::Operation *continuation : continuations) {
    if (isOrderedBySpatialResult(continuation, spatial))
      continue;
    if (::llvm::any_of(continuations, [&](::mlir::Operation *predecessor) {
          if (predecessor == continuation ||
              !followsOnSameStructuredPath(predecessor, continuation, thread))
            return false;
          return dominance.properlyDominates(predecessor, continuation) ||
                 postDominance.properlyPostDominates(predecessor, spatial);
        }))
      continue;
    if (::llvm::any_of(branches, [&](::mlir::scf::IfOp branch) {
          ::mlir::Operation *branchOp = branch.getOperation();
          if (!followsOnSameStructuredPath(spatial, branchOp, thread) ||
              !followsOnSameStructuredPath(branchOp, continuation, thread))
            return false;
          if (!dominance.properlyDominates(branchOp, continuation) &&
              !postDominance.properlyPostDominates(branchOp, spatial))
            return false;
          return ::llvm::all_of(
              branchOp->getRegions(), [&](::mlir::Region &region) {
                return regionAlwaysPublishes(region, continuations);
              });
        }))
      continue;
    anchors.push_back(continuation);
  }
  return anchors;
}

using PendingLaunchDependencies =
    ::llvm::DenseMap<::mlir::Operation *,
                     ::llvm::SmallVector<::mlir::Value, 2>>;

void addCompletionDependency(
    ::llvm::SmallVectorImpl<::mlir::Value> &dependencies,
    ::mlir::Value completion) {
  ::dataflow::ThreadCompletionCoverageAnalysis coverage;
  if (::llvm::any_of(dependencies, [&](::mlir::Value dependency) {
        return coverage.covers(dependency, completion);
      }))
    return;
  ::llvm::erase_if(dependencies, [&](::mlir::Value dependency) {
    return coverage.covers(completion, dependency);
  });
  dependencies.push_back(completion);
}

void replacePendingLaunchDependency(PendingLaunchDependencies &pending,
                                    ::mlir::Value oldValue,
                                    ::mlir::Value newValue) {
  for (auto &entry : pending)
    for (::mlir::Value &dependency : entry.second)
      if (dependency == oldValue)
        dependency = newValue;
}

void addGraphLaunchDependency(::dataflow::GraphLaunchOp launch,
                              ::mlir::Value completion) {
  ::llvm::SmallVector<::mlir::Value, 4> dependencies(
      launch.getDependencies().begin(), launch.getDependencies().end());
  addCompletionDependency(dependencies, completion);
  launch.getDependenciesMutable().assign(dependencies);
}

std::string uniqueSymbol(::mlir::ModuleOp module, ::llvm::StringRef stem) {
  ::mlir::SymbolTable st(module);
  std::string base = stem.str();
  if (!st.lookup(base))
    return base;
  unsigned suffix = 1;
  while (true) {
    std::string candidate = base + "_" + std::to_string(suffix);
    if (!st.lookup(candidate))
      return candidate;
    ++suffix;
  }
}

void addThreadCompletionFrontier(::dataflow::ThreadOp thread,
                                 ::mlir::Value completion) {
  auto yield = ::mlir::cast<::dataflow::ThreadYieldOp>(
      thread.getBody().front().getTerminator());
  ::llvm::SmallVector<::mlir::Value, 4> candidates(
      yield.getCompletionFrontier().begin(),
      yield.getCompletionFrontier().end());
  if (!::llvm::is_contained(candidates, completion))
    candidates.push_back(completion);
  ::llvm::SmallVector<::mlir::Value, 4> graphLaunchCompletions;
  thread.walk([&](::dataflow::GraphLaunchOp launch) {
    graphLaunchCompletions.push_back(launch.getDone());
  });
  ::dataflow::ThreadCompletionCoverageAnalysis coverage;
  ::llvm::SmallVector<::mlir::Value, 4> frontier =
      coverage.computeMinimalFrontier(candidates, graphLaunchCompletions);
  yield.getCompletionFrontierMutable().assign(frontier);
}

::mlir::FailureOr<::mlir::Value>
propagateIfCompletion(::mlir::scf::IfOp ifOp, ::mlir::Value completion,
                      ::mlir::Value fallback,
                      PendingLaunchDependencies &pendingLaunchDependencies,
                      ::mlir::OpBuilder &builder) {
  ::mlir::Region *completionRegion =
      completion.getDefiningOp()->getParentRegion();
  bool inThen = ifOp.getThenRegion().isAncestor(completionRegion);
  bool inElse = ifOp.getElseRegion().isAncestor(completionRegion);
  if (inThen == inElse)
    return ::mlir::failure();

  ::llvm::SmallVector<::mlir::Type, 4> resultTypes(
      ifOp.getResultTypes().begin(), ifOp.getResultTypes().end());
  resultTypes.push_back(builder.getNoneType());
  builder.setInsertionPoint(ifOp);
  auto replacement = ::mlir::scf::IfOp::create(
      builder, ifOp.getLoc(), resultTypes, ifOp.getCondition(),
      /*withElseRegion=*/true);

  auto moveBranch = [&](::mlir::Region &source, ::mlir::Region &target,
                        ::mlir::Value branchCompletion) {
    ::mlir::Block &targetBlock = target.front();
    ::mlir::scf::YieldOp defaultYield;
    if (!targetBlock.empty())
      defaultYield = ::llvm::dyn_cast<::mlir::scf::YieldOp>(
          targetBlock.back());
    ::llvm::SmallVector<::mlir::Value, 4> yielded;
    if (!source.empty()) {
      ::mlir::Block &sourceBlock = source.front();
      auto sourceYield = ::mlir::cast<::mlir::scf::YieldOp>(
          sourceBlock.getTerminator());
      yielded.append(sourceYield.getResults().begin(),
                     sourceYield.getResults().end());
      targetBlock.getOperations().splice(
          defaultYield ? defaultYield->getIterator() : targetBlock.end(),
          sourceBlock.getOperations(),
          sourceBlock.begin(), sourceYield->getIterator());
      sourceYield.erase();
    }
    yielded.push_back(branchCompletion);
    if (defaultYield)
      builder.setInsertionPoint(defaultYield);
    else
      builder.setInsertionPointToEnd(&targetBlock);
    ::mlir::scf::YieldOp::create(builder, ifOp.getLoc(), yielded);
    if (defaultYield)
      defaultYield.erase();
  };

  moveBranch(ifOp.getThenRegion(), replacement.getThenRegion(),
             inThen ? completion : fallback);
  moveBranch(ifOp.getElseRegion(), replacement.getElseRegion(),
             inElse ? completion : fallback);

  for (auto [oldResult, newResult] : ::llvm::zip_equal(
           ifOp.getResults(),
           replacement.getResults().take_front(ifOp.getNumResults()))) {
    replacePendingLaunchDependency(pendingLaunchDependencies, oldResult,
                                   newResult);
    oldResult.replaceAllUsesWith(newResult);
  }
  ::mlir::Value propagated = replacement.getResult(ifOp.getNumResults());
  ifOp.erase();
  return propagated;
}

::mlir::FailureOr<::mlir::Value> propagateCompletionToThread(
    ::dataflow::ThreadOp thread, ::mlir::Value completion,
    ::mlir::Value fallback,
    ::llvm::MutableArrayRef<::mlir::Operation *> channelRetirementAnchors,
    PendingLaunchDependencies &pendingLaunchDependencies,
    ::mlir::Location waitLoc, ::mlir::OpBuilder &builder) {
  auto placeDominatedRetirementAnchors = [&]() {
    ::mlir::DominanceInfo dominance(thread);
    for (::mlir::Operation *&anchor : channelRetirementAnchors) {
      if (!anchor || !dominance.dominates(completion, anchor))
        continue;
      if (auto target = ::llvm::dyn_cast<::loom::SpatialRegionOp>(anchor)) {
        addCompletionDependency(pendingLaunchDependencies[target], completion);
      } else if (auto launch =
                     ::llvm::dyn_cast<::dataflow::GraphLaunchOp>(anchor)) {
        addGraphLaunchDependency(launch, completion);
      } else {
        builder.setInsertionPoint(anchor);
        ::dataflow::GraphWaitOp::create(builder, waitLoc,
                                        ::mlir::ValueRange{completion});
      }
      anchor = nullptr;
    }
  };

  placeDominatedRetirementAnchors();
  while (completion.getDefiningOp()->getParentOfType<::dataflow::ThreadOp>() ==
             thread &&
         completion.getDefiningOp()->getBlock() != &thread.getBody().front()) {
    ::mlir::Operation *parent = completion.getDefiningOp()->getParentOp();
    while (parent && parent != thread.getOperation() &&
           !::llvm::isa<::mlir::scf::IfOp>(parent))
      parent = parent->getParentOp();
    auto ifOp = ::llvm::dyn_cast_or_null<::mlir::scf::IfOp>(parent);
    if (!ifOp)
      return ::mlir::failure();
    auto propagated = propagateIfCompletion(ifOp, completion, fallback,
                                            pendingLaunchDependencies, builder);
    if (::mlir::failed(propagated))
      return ::mlir::failure();
    completion = *propagated;
    placeDominatedRetirementAnchors();
  }
  if (::llvm::any_of(channelRetirementAnchors,
                     [](auto *anchor) { return anchor != nullptr; }))
    return ::mlir::failure();
  return completion;
}

struct ClassifiedGraphValues {
  ::llvm::SmallVector<::mlir::Value, 8> values;
  ::llvm::SmallVector<::mlir::Value, 4> memories;

  std::array<int32_t, 3> segments() const {
    return {static_cast<int32_t>(values.size()), 0,
            static_cast<int32_t>(memories.size())};
  }
};

::mlir::Value findGraphPublicationMemoryRoot(::mlir::Value value,
                                             ::mlir::Block &threadEntry) {
  ::llvm::DenseSet<::mlir::Value> visited;
  while (value && visited.insert(value).second) {
    if (auto argument = ::llvm::dyn_cast<::mlir::BlockArgument>(value))
      return argument.getOwner() == &threadEntry ? value : ::mlir::Value{};
    ::mlir::Operation *def = value.getDefiningOp();
    if (!def)
      return {};
    if (::llvm::isa<::mlir::memref::AllocOp, ::mlir::memref::AllocaOp,
                    ::mlir::memref::GetGlobalOp, ::mlir::LLVM::AddressOfOp>(
            def))
      return value;
    if (auto view = ::llvm::dyn_cast<::mlir::ViewLikeOpInterface>(def)) {
      if (value != view.getViewDest())
        return {};
      value = view.getViewSource();
      continue;
    }
    if (auto cast = ::llvm::dyn_cast<::mlir::UnrealizedConversionCastOp>(def)) {
      if (cast.getInputs().size() != 1)
        return {};
      value = cast.getInputs().front();
      continue;
    }
    if (auto gep = ::llvm::dyn_cast<::mlir::LLVM::GEPOp>(def)) {
      value = gep.getBase();
      continue;
    }
    if (auto bitcast = ::llvm::dyn_cast<::mlir::LLVM::BitcastOp>(def)) {
      value = bitcast.getArg();
      continue;
    }
    return {};
  }
  return {};
}

::llvm::SmallVector<::mlir::NamedAttribute, 2>
graphSegmentAttrs(::mlir::OpBuilder &builder,
                  const ClassifiedGraphValues &inputs,
                  const ClassifiedGraphValues &results,
                  int32_t inputStreamCount = 0, int32_t resultStreamCount = 0) {
  std::array<int32_t, 3> inputSegments = inputs.segments();
  std::array<int32_t, 3> resultSegments = results.segments();
  inputSegments[1] = inputStreamCount;
  resultSegments[1] = resultStreamCount;
  return {
      builder.getNamedAttr("input_segments",
                           builder.getDenseI32ArrayAttr(inputSegments)),
      builder.getNamedAttr("result_segments",
                           builder.getDenseI32ArrayAttr(resultSegments)),
  };
}

struct LowerForToGraphPass
    : public ::mlir::PassWrapper<LowerForToGraphPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerForToGraphPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-lower-for-to-graph";
  }
  ::llvm::StringRef getDescription() const final {
    return "Publish explicit loom.spatial_region operations as "
           "dataflow.graph definitions plus dataflow.graph.launch ops.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::func::FuncDialect,
                    ::mlir::LLVM::LLVMDialect, ::mlir::math::MathDialect,
                    ::mlir::memref::MemRefDialect, ::mlir::scf::SCFDialect,
                    ::mlir::ub::UBDialect, ::dataflow::DataflowDialect,
                    ::loom::LoomDialect>();
  }

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();
    ::mlir::MLIRContext *ctx = &getContext();
    ::mlir::OwningOpRef<::mlir::ModuleOp> scratch(
        ::mlir::cast<::mlir::ModuleOp>(module->clone()));
    ::mlir::OpBuilder scratchBuilder(ctx);
    if (::mlir::failed(publishSpatialRegions(*scratch, scratchBuilder)) ||
        ::mlir::failed(finalizePublishedModule(*scratch))) {
      signalPassFailure();
      return;
    }

    module->setAttrs((*scratch)->getAttrs());
    module.getBodyRegion().takeBody(scratch->getBodyRegion());
  }

  // Finalization applies to graphs that are not canonical Dataflow yet: the
  // ones just published and any pre-final graph the module was given.
  // `dataflow::validateFinalizedGraph` is that structural authority, so
  // finality is not decided a second time here and needs no marker. A graph it
  // accepts already owns its ctrl/done memory-event network; lowering it again
  // would drive canonical dataflow.load/store back through the graph-memory
  // owner and rebuild that network.
  //
  // Those graphs are therefore staged: a temporary module carries the original
  // module attributes, so the data layout that resolves each graph's index
  // width is the same one, and clones of just the graphs still to finalize. A
  // staged graph body can name a module-scope symbol, so the symbol providers
  // come too, as declarations: only their signature is needed to resolve a use,
  // and a retained host body could launch a thread this module deliberately
  // does not hold. Nothing else is cloned, so no top-level value is separated
  // from its uses. Only when the existing pipeline succeeds does each staged
  // graph replace its original, which keeps the signature changes finalization
  // makes. The complete outer module is validated afterwards.
  ::mlir::LogicalResult finalizePublishedModule(::mlir::ModuleOp module) {
    ::llvm::SmallVector<::dataflow::GraphOp, 4> pending;
    for (auto graph : module.getOps<::dataflow::GraphOp>()) {
      if (graph.isExternal())
        continue;
      ::llvm::Error error = ::dataflow::validateFinalizedGraph(graph);
      if (!error)
        continue;
      ::llvm::consumeError(std::move(error));
      pending.push_back(graph);
    }

    if (!pending.empty()) {
      ::mlir::OpBuilder builder(module.getContext());
      ::mlir::OwningOpRef<::mlir::ModuleOp> staging =
          ::mlir::ModuleOp::create(builder, module.getLoc());
      (*staging)->setAttrs(module->getAttrs());
      builder.setInsertionPointToEnd(staging->getBody());
      for (::mlir::Operation &op : *module.getBody()) {
        if (!::llvm::isa<::mlir::SymbolOpInterface>(op) ||
            ::llvm::isa<::dataflow::GraphOp, ::dataflow::ThreadOp>(op))
          continue;
        ::mlir::Operation *declaration = builder.clone(op);
        if (auto callable =
                ::llvm::dyn_cast<::mlir::FunctionOpInterface>(declaration)) {
          callable.getFunctionBody().getBlocks().clear();
          // A body-less callable is a declaration, which cannot be public.
          ::mlir::SymbolTable::setSymbolVisibility(
              declaration, ::mlir::SymbolTable::Visibility::Private);
        }
      }
      ::llvm::SmallVector<::dataflow::GraphOp, 4> staged;
      staged.reserve(pending.size());
      for (::dataflow::GraphOp graph : pending)
        staged.push_back(::mlir::cast<::dataflow::GraphOp>(
            builder.clone(*graph.getOperation())));

      if (::mlir::failed(lowerPendingGraphs(*staging)))
        return ::mlir::failure();

      for (auto [graph, finalized] : ::llvm::zip_equal(pending, staged)) {
        finalized->moveBefore(graph);
        graph.erase();
      }
    }

    if (auto error = ::dataflow::validateFinalizedProgram(module)) {
      module.emitError("canonical Dataflow publication failed: ")
          << ::llvm::toString(std::move(error));
      return ::mlir::failure();
    }
    return ::mlir::success();
  }

  ::mlir::LogicalResult lowerPendingGraphs(::mlir::ModuleOp module) {
    // Stream endpoints temporarily retain channel block arguments in the
    // scratch module until graph-region lowering replaces them with ports.
    // The first canonicalizer owns the upstream memref.copy folds, so the
    // expansion that follows only sees copies with a live extent. The second
    // one canonicalizes the expanded loops together with the structured loops
    // already in the body, which keeps one canonical set of index constants
    // instead of a second set per expanded copy. Both precede the graph-memory
    // owner, which consumes graph accesses.
    ::mlir::PassManager lowerer(module.getContext());
    lowerer.enableVerifier(false);
    lowerer.addPass(::mlir::createCanonicalizerPass());
    lowerer.addPass(::loom::lowering::createLowerKnownLibraryCallsPass());
    lowerer.addPass(::loom::lowering::createExpandGraphMemrefCopyPass());
    lowerer.addPass(::mlir::createCanonicalizerPass());
    lowerer.addPass(::loom::lowering::createLowerGraphMemoryPass());
    if (::mlir::failed(lowerer.run(module)) || ::mlir::failed(verify(module)))
      return ::mlir::failure();

    ::mlir::PassManager finalizer(module.getContext());
    finalizer.enableVerifier(true);
    finalizer.addPass(::loom::lowering::createLowerGraphConstantsPass());
    finalizer.addPass(::mlir::createCanonicalizerPass());
    return finalizer.run(module);
  }

  ::mlir::LogicalResult publishSpatialRegions(::mlir::ModuleOp module,
                                              ::mlir::OpBuilder &builder) {
    ::llvm::SmallVector<::loom::SpatialRegionOp, 8> regions;
    module.walk([&](::loom::SpatialRegionOp spatial) {
      regions.push_back(spatial);
    });

    for (::loom::SpatialRegionOp spatial : regions) {
      auto thread = spatial->getParentOfType<::dataflow::ThreadOp>();
      if (!thread)
        return spatial.emitOpError(
            "expected loom.spatial_region inside dataflow.thread");
      for (::mlir::Operation *parent = spatial->getParentOp();
           parent && parent != thread.getOperation();
           parent = parent->getParentOp()) {
        if (!::llvm::isa<::mlir::scf::IfOp>(parent))
          return spatial.emitOpError(
              "completion propagation through enclosing '")
                 << parent->getName()
                 << "' is not implemented; spatial candidate cannot be "
                    "published";
      }
    }

    PendingLaunchDependencies pendingLaunchDependencies;
    for (::loom::SpatialRegionOp spatial : regions) {
      auto thread = spatial->getParentOfType<::dataflow::ThreadOp>();
      ::mlir::Block &threadEntry = thread.getBody().front();
      ::llvm::SmallVector<::mlir::Operation *, 4> channelRetirementAnchors =
          findChannelRetirementAnchors(spatial);
      size_t ctrlIndex = thread.getFunctionType().getInputs().size();
      if (threadEntry.getNumArguments() <= ctrlIndex)
        return thread.emitOpError("is missing thread control block argument");
      ::mlir::Value threadCtrl = threadEntry.getArgument(ctrlIndex);
      ::llvm::SmallVector<::mlir::Value, 4> launchDependencies{threadCtrl};
      auto pending = pendingLaunchDependencies.find(spatial);
      if (pending != pendingLaunchDependencies.end()) {
        for (::mlir::Value dependency : pending->second)
          addCompletionDependency(launchDependencies, dependency);
        pendingLaunchDependencies.erase(pending);
      }

      ClassifiedGraphValues graphInputs;
      graphInputs.values.append(spatial.getValueInputs().begin(),
                                spatial.getValueInputs().end());
      graphInputs.memories.append(spatial.getMemoryInputs().begin(),
                                  spatial.getMemoryInputs().end());
      ClassifiedGraphValues graphResults;
      graphResults.values.append(spatial.getValueResults().begin(),
                                 spatial.getValueResults().end());
      graphResults.memories.append(spatial.getMemoryResults().begin(),
                                   spatial.getMemoryResults().end());

      ::llvm::SmallVector<::mlir::Type, 8> inputTypes;
      for (::mlir::Value value : graphInputs.values)
        inputTypes.push_back(value.getType());
      for (::mlir::Value channel : spatial.getStreamInputs())
        inputTypes.push_back(
            ::llvm::cast<::dataflow::ChannelType>(channel.getType())
                .getElementType());
      for (::mlir::Value memory : graphInputs.memories)
        inputTypes.push_back(memory.getType());
      ::llvm::SmallVector<::mlir::Type, 4> resultTypes;
      for (::mlir::Value value : graphResults.values)
        resultTypes.push_back(value.getType());
      for (::mlir::Value channel : spatial.getStreamOutputs())
        resultTypes.push_back(
            ::llvm::cast<::dataflow::ChannelType>(channel.getType())
                .getElementType());
      for (::mlir::Value memory : graphResults.memories)
        resultTypes.push_back(memory.getType());
      auto functionType = builder.getFunctionType(inputTypes, resultTypes);
      auto segmentAttrs = graphSegmentAttrs(
          builder, graphInputs, graphResults,
          static_cast<int32_t>(spatial.getStreamInputs().size()),
          static_cast<int32_t>(spatial.getStreamOutputs().size()));

      std::string graphName = uniqueSymbol(
          module, spatial.getGraphName().value_or("g_spatial_candidate"));
      ::mlir::Location loc = spatial.getLoc();
      builder.setInsertionPointToEnd(module.getBody());
      auto graph = ::dataflow::GraphOp::create(
          builder, loc, graphName, functionType, segmentAttrs);
      graph.setSymVisibilityAttr(builder.getStringAttr("private"));

      ::llvm::SmallVector<::mlir::Value, 8> memoryRoots(
          graphInputs.memories.size());
      ::llvm::DenseMap<::mlir::Value, unsigned> capturedRootCounts;
      bool hasUnknownMemoryRoot = false;
      for (auto [index, memory] : ::llvm::enumerate(graphInputs.memories)) {
        ::mlir::Value root =
            findGraphPublicationMemoryRoot(memory, threadEntry);
        memoryRoots[index] = root;
        if (root)
          ++capturedRootCounts[root];
        else
          hasUnknownMemoryRoot = true;
      }

      auto getThreadArgAttrs = [&](::mlir::Value value) {
        if (!value)
          return ::mlir::DictionaryAttr{};
        auto argument = ::llvm::dyn_cast<::mlir::BlockArgument>(value);
        if (!argument || argument.getOwner() != &threadEntry ||
            argument.getArgNumber() >= thread.getFunctionType().getNumInputs())
          return ::mlir::DictionaryAttr{};
        return ::mlir::function_interface_impl::getArgAttrDict(
            thread, argument.getArgNumber());
      };

      ::llvm::SmallVector<::mlir::DictionaryAttr, 8> graphArgAttrs;
      graphArgAttrs.reserve(inputTypes.size());
      for (::mlir::Value input : graphInputs.values) {
        ::mlir::NamedAttrList attrs(getThreadArgAttrs(input));
        graphArgAttrs.push_back(attrs.getDictionary(builder.getContext()));
      }
      for ([[maybe_unused]] ::mlir::Value channel : spatial.getStreamInputs())
        graphArgAttrs.push_back(builder.getDictionaryAttr({}));
      for (auto [index, memory] : ::llvm::enumerate(graphInputs.memories)) {
        ::mlir::NamedAttrList attrs(getThreadArgAttrs(memory));
        ::mlir::Value root = memoryRoots[index];
        ::mlir::DictionaryAttr rootAttrs = getThreadArgAttrs(root);
        ::mlir::Attribute noAlias =
            rootAttrs ? rootAttrs.get("llvm.noalias") : ::mlir::Attribute{};
        bool uniqueKnownRoot = !hasUnknownMemoryRoot && root &&
                               capturedRootCounts.lookup(root) == 1;
        if (uniqueKnownRoot && noAlias)
          attrs.set("llvm.noalias", noAlias);
        else
          attrs.erase("llvm.noalias");
        graphArgAttrs.push_back(attrs.getDictionary(builder.getContext()));
      }
      ::mlir::function_interface_impl::setAllArgAttrDicts(graph, graphArgAttrs);

      ::mlir::Block *graphEntry = builder.createBlock(&graph.getBody());
      graphEntry->addArgument(builder.getNoneType(), loc);
      for (::mlir::Type type : inputTypes)
        graphEntry->addArgument(type, loc);

      ::mlir::IRMapping mapping;
      ::mlir::Block &spatialEntry = spatial.getBody().front();
      size_t spatialArgument = 0;
      size_t graphArgument = 1;
      for ([[maybe_unused]] ::mlir::Value input : spatial.getValueInputs())
        mapping.map(spatialEntry.getArgument(spatialArgument++),
                    graphEntry->getArgument(graphArgument++));
      graphArgument += spatial.getStreamInputs().size();
      size_t inputChannelArgument = graphEntry->getNumArguments();
      for (::mlir::Value channel : spatial.getStreamInputs()) {
        graphEntry->addArgument(channel.getType(), loc);
        mapping.map(spatialEntry.getArgument(spatialArgument++),
                    graphEntry->getArgument(inputChannelArgument++));
      }
      for ([[maybe_unused]] ::mlir::Value memory : spatial.getMemoryInputs())
        mapping.map(spatialEntry.getArgument(spatialArgument++),
                    graphEntry->getArgument(graphArgument++));
      size_t outputChannelArgument = graphEntry->getNumArguments();
      for (::mlir::Value channel : spatial.getStreamOutputs()) {
        graphEntry->addArgument(channel.getType(), loc);
        mapping.map(spatialEntry.getArgument(spatialArgument++),
                    graphEntry->getArgument(outputChannelArgument++));
      }

      builder.setInsertionPointToEnd(graphEntry);
      for (::mlir::Operation &op : spatialEntry.without_terminator())
        builder.clone(op, mapping);
      auto spatialYield =
          ::mlir::cast<::loom::SpatialYieldOp>(spatialEntry.getTerminator());
      ::llvm::SmallVector<::mlir::Value, 4> returnValues;
      for (::mlir::Value value : spatialYield.getValues())
        returnValues.push_back(mapping.lookup(value));
      ::llvm::SmallVector<::mlir::Value, 4> returnMemories;
      for (::mlir::Value value : spatialYield.getMemories())
        returnMemories.push_back(mapping.lookup(value));
      ::dataflow::GraphReturnOp::create(
          builder, loc, returnValues, ::mlir::ValueRange{}, returnMemories,
          ::mlir::ValueRange{graphEntry->getArgument(0)});

      builder.setInsertionPoint(spatial);
      auto callee =
          ::mlir::FlatSymbolRefAttr::get(builder.getContext(), graphName);
      ::llvm::SmallVector<::mlir::Type, 4> valueResultTypes;
      for (::mlir::Value result : spatial.getValueResults())
        valueResultTypes.push_back(result.getType());
      ::llvm::SmallVector<::mlir::Type, 4> memoryResultTypes;
      for (::mlir::Value result : spatial.getMemoryResults())
        memoryResultTypes.push_back(result.getType());
      auto launch = ::dataflow::GraphLaunchOp::create(
          builder, loc, valueResultTypes, memoryResultTypes,
          builder.getNoneType(), callee, spatial.getSourceMaps(),
          launchDependencies, spatial.getValueInputs(),
          spatial.getStreamInputs(), spatial.getMemoryInputs(),
          spatial.getStreamOutputs());
      auto propagated = propagateCompletionToThread(
          thread, launch.getDone(), threadCtrl, channelRetirementAnchors,
          pendingLaunchDependencies, loc, builder);
      if (::mlir::failed(propagated))
        return launch.emitOpError(
            "failed to propagate completion through enclosing structured "
            "control");
      addThreadCompletionFrontier(thread, *propagated);
      for (auto [index, result] :
           ::llvm::enumerate(spatial.getValueResults()))
        result.replaceAllUsesWith(launch.getValueResults()[index]);
      for (auto [index, result] :
           ::llvm::enumerate(spatial.getMemoryResults()))
        result.replaceAllUsesWith(launch.getMemoryResults()[index]);
      spatial.erase();
    }
    if (!pendingLaunchDependencies.empty())
      return module.emitError(
          "channel retirement dependency target was not published");
    return ::mlir::success();
  }
};

} // namespace

namespace loom {
namespace lowering {

std::unique_ptr<::mlir::Pass> createLowerForToGraphPass() {
  return std::make_unique<LowerForToGraphPass>();
}

void registerLowerForToGraphPass() {
  static bool once = []() {
    ::mlir::PassRegistration<LowerForToGraphPass>();
    return true;
  }();
  (void)once;
}

} // namespace lowering
} // namespace loom
