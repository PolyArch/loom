// Publish explicit `loom.spatial_region` operations inside dataflow.thread
// bodies as canonical `dataflow.graph` definitions and matching launches.
// The graph function_type contains only normalized application payload ports.
// Start/done remain explicit launch protocol endpoints, and graph.return owns
// the segmented payload boundary plus retirement frontier.

#include "Frontend/Lowering/GraphParallelLowering.h"
#include "Frontend/Lowering/Passes.h"
#include "GraphMemoryLowering.h"
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
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

#include <algorithm>
#include <array>
#include <deque>
#include <optional>

namespace {

using ::loom::lowering::FixedParallelDomain;
using ::loom::lowering::forEachParallelPoint;
using ::loom::lowering::getFixedParallelDomain;

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

void addThreadCompletionFrontiers(::dataflow::ThreadOp thread,
                                  ::mlir::ValueRange completions) {
  auto yield = ::mlir::cast<::dataflow::ThreadYieldOp>(
      thread.getBody().front().getTerminator());
  ::llvm::SmallVector<::mlir::Value, 4> candidates(
      yield.getCompletionFrontier().begin(),
      yield.getCompletionFrontier().end());
  for (::mlir::Value completion : completions)
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

struct PendingGraphCompletion {
  ::dataflow::GraphLaunchOp launch;
  ::llvm::SmallVector<::mlir::Operation *, 4> channelRetirementAnchors;
  unsigned parallelGroup = 0;
};

struct ParallelCompletionCandidate {
  ::dataflow::ThreadOp thread;
  ::mlir::Value completion;
  unsigned parallelGroup;
};

unsigned publishParallelCompletionCandidates(
    ::llvm::ArrayRef<ParallelCompletionCandidate> candidates,
    ::mlir::OpBuilder &builder) {
  struct CompletionGroup {
    ::dataflow::ThreadOp thread;
    ::llvm::SmallVector<::mlir::Value, 4> completions;
  };
  ::llvm::DenseMap<unsigned, unsigned> groupIndices;
  ::llvm::SmallVector<CompletionGroup, 4> groups;
  unsigned inspections = 0;
  for (const ParallelCompletionCandidate &candidate : candidates) {
    ++inspections;
    auto [entry, inserted] =
        groupIndices.try_emplace(candidate.parallelGroup, groups.size());
    if (inserted)
      groups.push_back(CompletionGroup{candidate.thread, {}});
    CompletionGroup &group = groups[entry->second];
    assert(group.thread == candidate.thread &&
           "one parallel completion group cannot cross threads");
    addCompletionDependency(group.completions, candidate.completion);
  }

  struct ThreadCompletions {
    ::dataflow::ThreadOp thread;
    ::llvm::SmallVector<::mlir::Value, 4> completions;
  };
  ::llvm::DenseMap<::mlir::Operation *, unsigned> threadIndices;
  ::llvm::SmallVector<ThreadCompletions, 4> publications;
  for (CompletionGroup &group : groups) {
    ::mlir::Value completion = group.completions.front();
    if (group.completions.size() > 1) {
      ::dataflow::ThreadOp thread = group.thread;
      auto yield = ::mlir::cast<::dataflow::ThreadYieldOp>(
          thread.getBody().front().getTerminator());
      builder.setInsertionPoint(yield);
      ::llvm::SmallVector<::mlir::Type, 4> types(group.completions.size(),
                                                 builder.getNoneType());
      completion = ::dataflow::SyncOp::create(builder, yield.getLoc(), types,
                                              group.completions)
                       .getOutputs()
                       .front();
    }
    auto [entry, inserted] = threadIndices.try_emplace(
        group.thread.getOperation(), publications.size());
    if (inserted)
      publications.push_back(ThreadCompletions{group.thread, {}});
    addCompletionDependency(publications[entry->second].completions,
                            completion);
  }
  for (const ThreadCompletions &publication : publications)
    addThreadCompletionFrontiers(publication.thread, publication.completions);
  return inspections;
}

void placeDominatedRetirementAnchors(
    ::dataflow::ThreadOp thread, ::mlir::Value completion,
    ::llvm::MutableArrayRef<::mlir::Operation *> channelRetirementAnchors,
    PendingLaunchDependencies &pendingLaunchDependencies,
    ::mlir::OpBuilder &builder) {
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
      ::dataflow::GraphWaitOp::create(builder, anchor->getLoc(),
                                      ::mlir::ValueRange{completion});
    }
    anchor = nullptr;
  }
}

void replaceStructuredResults(
    ::mlir::Operation *oldOp, ::mlir::Operation *replacement,
    unsigned oldResultCount,
    PendingLaunchDependencies &pendingLaunchDependencies) {
  for (auto [oldResult, newResult] : ::llvm::zip_equal(
           oldOp->getResults(),
           replacement->getResults().take_front(oldResultCount))) {
    replacePendingLaunchDependency(pendingLaunchDependencies, oldResult,
                                   newResult);
    oldResult.replaceAllUsesWith(newResult);
  }
}

void moveRegionBody(::mlir::Region &source, ::mlir::Region &target) {
  if (source.empty())
    return;
  target.takeBody(source);
}

std::optional<unsigned> findContainingRegion(::mlir::Operation *parent,
                                             ::mlir::Operation *nested) {
  ::mlir::Region *nestedRegion = nested->getParentRegion();
  for (auto [index, region] : ::llvm::enumerate(parent->getRegions()))
    if (&region == nestedRegion || region.isAncestor(nestedRegion))
      return index;
  return std::nullopt;
}

void replaceLaunchEntryDependency(::dataflow::GraphLaunchOp launch,
                                  ::mlir::Value fallback,
                                  ::mlir::Value incoming) {
  ::llvm::SmallVector<::mlir::Value, 4> dependencies;
  for (::mlir::Value dependency : launch.getDependencies()) {
    if (dependency == fallback)
      dependency = incoming;
    if (!::llvm::is_contained(dependencies, dependency))
      dependencies.push_back(dependency);
  }
  if (!::llvm::is_contained(dependencies, incoming))
    dependencies.push_back(incoming);
  if (incoming == fallback) {
    ::dataflow::ThreadCompletionCoverageAnalysis coverage;
    bool fallbackCovered =
        ::llvm::any_of(dependencies, [&](::mlir::Value dependency) {
          return dependency != fallback &&
                 coverage.covers(dependency, fallback);
        });
    if (fallbackCovered)
      ::llvm::erase(dependencies, fallback);
  }
  launch.getDependenciesMutable().assign(dependencies);
}

::mlir::FailureOr<::mlir::Value> propagateCompletionPath(
    ::llvm::ArrayRef<::mlir::Operation *> path,
    ::dataflow::GraphLaunchOp launch, ::mlir::Value incoming,
    ::mlir::Value threadFallback,
    ::llvm::MutableArrayRef<::mlir::Operation *> channelRetirementAnchors,
    PendingLaunchDependencies &pendingLaunchDependencies,
    std::deque<PendingGraphCompletion> &pendingCompletions,
    unsigned &parallelGroup, unsigned &nextParallelGroup,
    ::dataflow::ThreadOp thread, ::mlir::OpBuilder &builder) {
  if (path.empty()) {
    replaceLaunchEntryDependency(launch, threadFallback, incoming);
    placeDominatedRetirementAnchors(thread, launch.getDone(),
                                    channelRetirementAnchors,
                                    pendingLaunchDependencies, builder);
    return launch.getDone();
  }

  ::mlir::Operation *op = path.front();
  ::mlir::Operation *nested =
      path.size() == 1 ? launch.getOperation() : path[1];
  auto containingRegion = findContainingRegion(op, nested);
  if (!containingRegion)
    return ::mlir::failure();

  auto propagateSelection =
      [&](::mlir::Operation *replacement) -> ::mlir::FailureOr<::mlir::Value> {
    unsigned oldResultCount = op->getNumResults();
    replacement->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    for (auto [source, target] :
         ::llvm::zip_equal(op->getRegions(), replacement->getRegions()))
      moveRegionBody(source, target);
    replaceStructuredResults(op, replacement, oldResultCount,
                             pendingLaunchDependencies);
    op->erase();

    auto selected = propagateCompletionPath(
        path.drop_front(), launch, incoming, threadFallback,
        channelRetirementAnchors, pendingLaunchDependencies, pendingCompletions,
        parallelGroup, nextParallelGroup, thread, builder);
    if (::mlir::failed(selected))
      return ::mlir::failure();
    for (auto [index, region] : ::llvm::enumerate(replacement->getRegions())) {
      ::mlir::Value regionCompletion =
          index == *containingRegion ? *selected : incoming;
      if (region.empty())
        builder.createBlock(&region);
      ::mlir::Block &block = region.front();
      if (block.empty()) {
        builder.setInsertionPointToEnd(&block);
        ::mlir::scf::YieldOp::create(builder, replacement->getLoc(),
                                     regionCompletion);
        continue;
      }
      auto yield =
          ::llvm::dyn_cast<::mlir::scf::YieldOp>(block.getTerminator());
      if (!yield)
        return ::mlir::failure();
      yield.getResultsMutable().append(regionCompletion);
    }
    ::mlir::Value output = replacement->getResult(oldResultCount);
    placeDominatedRetirementAnchors(thread, output, channelRetirementAnchors,
                                    pendingLaunchDependencies, builder);
    return output;
  };

  if (auto ifOp = ::llvm::dyn_cast<::mlir::scf::IfOp>(op)) {
    ::llvm::SmallVector<::mlir::Type, 4> resultTypes(ifOp.getResultTypes());
    resultTypes.push_back(builder.getNoneType());
    builder.setInsertionPoint(ifOp);
    auto replacement = ::mlir::scf::IfOp::create(
        builder, ifOp.getLoc(), resultTypes, ifOp.getCondition(),
        /*withElseRegion=*/true);
    return propagateSelection(replacement);
  }

  if (auto switchOp = ::llvm::dyn_cast<::mlir::scf::IndexSwitchOp>(op)) {
    ::llvm::SmallVector<::mlir::Type, 4> resultTypes(switchOp.getResultTypes());
    resultTypes.push_back(builder.getNoneType());
    builder.setInsertionPoint(switchOp);
    auto replacement = ::mlir::scf::IndexSwitchOp::create(
        builder, switchOp.getLoc(), resultTypes, switchOp.getArg(),
        switchOp.getCasesAttr(), switchOp.getNumCases());
    return propagateSelection(replacement);
  }

  if (auto forOp = ::llvm::dyn_cast<::mlir::scf::ForOp>(op)) {
    ::mlir::Location loc = forOp.getLoc();
    unsigned oldResultCount = forOp.getNumResults();
    ::llvm::SmallVector<::mlir::Value, 4> initArgs(forOp.getInitArgs());
    initArgs.push_back(incoming);
    builder.setInsertionPoint(forOp);
    auto replacement = ::mlir::scf::ForOp::create(
        builder, loc, forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), initArgs, nullptr, forOp.getUnsignedCmp());
    replacement->setDiscardableAttrs(forOp->getDiscardableAttrDictionary());
    moveRegionBody(forOp.getRegion(), replacement.getRegion());
    ::mlir::Block &body = replacement.getRegion().front();
    ::mlir::Value bodyIncoming = body.addArgument(builder.getNoneType(), loc);
    replaceStructuredResults(forOp, replacement, oldResultCount,
                             pendingLaunchDependencies);
    forOp.erase();

    auto bodyCompletion = propagateCompletionPath(
        path.drop_front(), launch, bodyIncoming, threadFallback,
        channelRetirementAnchors, pendingLaunchDependencies, pendingCompletions,
        parallelGroup, nextParallelGroup, thread, builder);
    if (::mlir::failed(bodyCompletion))
      return ::mlir::failure();
    auto yield = ::mlir::cast<::mlir::scf::YieldOp>(body.getTerminator());
    yield.getResultsMutable().append(*bodyCompletion);
    ::mlir::Value output = replacement.getResult(oldResultCount);
    placeDominatedRetirementAnchors(thread, output, channelRetirementAnchors,
                                    pendingLaunchDependencies, builder);
    return output;
  }

  if (auto whileOp = ::llvm::dyn_cast<::mlir::scf::WhileOp>(op)) {
    ::mlir::Location loc = whileOp.getLoc();
    unsigned oldResultCount = whileOp.getNumResults();
    ::llvm::SmallVector<::mlir::Type, 4> resultTypes(whileOp.getResultTypes());
    resultTypes.push_back(builder.getNoneType());
    ::llvm::SmallVector<::mlir::Value, 4> inits(whileOp.getInits());
    inits.push_back(incoming);
    builder.setInsertionPoint(whileOp);
    auto replacement = ::mlir::scf::WhileOp::create(builder, loc, resultTypes,
                                                    inits, nullptr, nullptr);
    replacement->setDiscardableAttrs(whileOp->getDiscardableAttrDictionary());
    moveRegionBody(whileOp.getBefore(), replacement.getBefore());
    moveRegionBody(whileOp.getAfter(), replacement.getAfter());
    ::mlir::Value beforeIncoming =
        replacement.getBefore().front().addArgument(builder.getNoneType(), loc);
    ::mlir::Value afterIncoming =
        replacement.getAfter().front().addArgument(builder.getNoneType(), loc);
    replaceStructuredResults(whileOp, replacement, oldResultCount,
                             pendingLaunchDependencies);
    whileOp.erase();

    auto condition = replacement.getConditionOp();
    auto yield = replacement.getYieldOp();
    if (*containingRegion == 0) {
      auto beforeCompletion = propagateCompletionPath(
          path.drop_front(), launch, beforeIncoming, threadFallback,
          channelRetirementAnchors, pendingLaunchDependencies,
          pendingCompletions, parallelGroup, nextParallelGroup, thread,
          builder);
      if (::mlir::failed(beforeCompletion))
        return ::mlir::failure();
      condition.getArgsMutable().append(*beforeCompletion);
      yield.getResultsMutable().append(afterIncoming);
    } else {
      condition.getArgsMutable().append(beforeIncoming);
      auto afterCompletion = propagateCompletionPath(
          path.drop_front(), launch, afterIncoming, threadFallback,
          channelRetirementAnchors, pendingLaunchDependencies,
          pendingCompletions, parallelGroup, nextParallelGroup, thread,
          builder);
      if (::mlir::failed(afterCompletion))
        return ::mlir::failure();
      yield.getResultsMutable().append(*afterCompletion);
    }
    ::mlir::Value output = replacement.getResult(oldResultCount);
    placeDominatedRetirementAnchors(thread, output, channelRetirementAnchors,
                                    pendingLaunchDependencies, builder);
    return output;
  }

  if (::llvm::isa<::mlir::scf::ParallelOp, ::mlir::scf::ForallOp>(op)) {
    auto domain = getFixedParallelDomain(op);
    if (!domain)
      return ::mlir::failure();
    if (parallelGroup == 0)
      parallelGroup = nextParallelGroup++;
    ::mlir::Block *body;
    ::llvm::SmallVector<::mlir::Value, 4> inductionVars;
    if (auto parallel = ::llvm::dyn_cast<::mlir::scf::ParallelOp>(op)) {
      body = parallel.getBody();
      inductionVars = parallel.getInductionVars();
    } else {
      auto forall = ::mlir::cast<::mlir::scf::ForallOp>(op);
      body = forall.getBody();
      inductionVars = forall.getInductionVars();
    }

    ::llvm::SmallVector<PendingGraphCompletion, 4> nestedPending;
    ::llvm::SmallVector<PendingGraphCompletion, 4> remainingPending;
    for (PendingGraphCompletion &pending : pendingCompletions) {
      if (isNestedIn(pending.launch, op))
        nestedPending.push_back(std::move(pending));
      else
        remainingPending.push_back(std::move(pending));
    }
    pendingCompletions.assign(std::make_move_iterator(remainingPending.begin()),
                              std::make_move_iterator(remainingPending.end()));

    ::llvm::SmallVector<::mlir::Operation *, 4> internalAnchors;
    for (::mlir::Operation *&anchor : channelRetirementAnchors) {
      if (!anchor || !isNestedIn(anchor, op))
        continue;
      internalAnchors.push_back(anchor);
      anchor = nullptr;
    }

    ::mlir::LogicalResult status = ::mlir::success();
    ::llvm::SmallVector<::mlir::Value, 4> laneCompletions;
    forEachParallelPoint(*domain, [&](::llvm::ArrayRef<int64_t> coordinates) {
      if (::mlir::failed(status))
        return;
      ::mlir::IRMapping mapping;
      builder.setInsertionPoint(op);
      for (auto [iv, coordinate] :
           ::llvm::zip_equal(inductionVars, coordinates)) {
        auto constant = ::mlir::arith::ConstantOp::create(
            builder, op->getLoc(), builder.getIndexAttr(coordinate));
        mapping.map(iv, constant.getResult());
      }
      for (::mlir::Operation &bodyOp : body->without_terminator())
        builder.clone(bodyOp, mapping);

      for (PendingGraphCompletion &pending : nestedPending) {
        auto mappedLaunch = ::llvm::dyn_cast_or_null<::dataflow::GraphLaunchOp>(
            mapping.lookupOrNull(pending.launch.getOperation()));
        if (!mappedLaunch) {
          status = ::mlir::failure();
          return;
        }
        PendingGraphCompletion mapped{mappedLaunch, {}, parallelGroup};
        for (::mlir::Operation *anchor : pending.channelRetirementAnchors) {
          if (anchor && isNestedIn(anchor, op))
            anchor = mapping.lookupOrNull(anchor);
          mapped.channelRetirementAnchors.push_back(anchor);
        }
        pendingCompletions.push_back(std::move(mapped));
      }

      auto mappedLaunch = ::llvm::dyn_cast_or_null<::dataflow::GraphLaunchOp>(
          mapping.lookupOrNull(launch.getOperation()));
      if (!mappedLaunch) {
        status = ::mlir::failure();
        return;
      }
      ::llvm::SmallVector<::mlir::Operation *, 4> mappedPath;
      for (::mlir::Operation *pathOp : path.drop_front()) {
        ::mlir::Operation *mapped = mapping.lookupOrNull(pathOp);
        if (!mapped) {
          status = ::mlir::failure();
          return;
        }
        mappedPath.push_back(mapped);
      }
      ::llvm::SmallVector<::mlir::Operation *, 4> mappedAnchors;
      for (::mlir::Operation *anchor : internalAnchors) {
        anchor = mapping.lookupOrNull(anchor);
        if (!anchor) {
          status = ::mlir::failure();
          return;
        }
        mappedAnchors.push_back(anchor);
      }
      auto laneCompletion = propagateCompletionPath(
          mappedPath, mappedLaunch, incoming, threadFallback, mappedAnchors,
          pendingLaunchDependencies, pendingCompletions, parallelGroup,
          nextParallelGroup, thread, builder);
      if (::mlir::failed(laneCompletion) ||
          ::llvm::any_of(mappedAnchors,
                         [](auto *anchor) { return anchor != nullptr; })) {
        status = ::mlir::failure();
        return;
      }
      laneCompletions.push_back(*laneCompletion);
    });
    if (::mlir::failed(status))
      return ::mlir::failure();

    builder.setInsertionPoint(op);
    ::mlir::Value output = incoming;
    if (laneCompletions.size() == 1) {
      output = laneCompletions.front();
    } else if (!laneCompletions.empty()) {
      ::llvm::SmallVector<::mlir::Type, 4> types(laneCompletions.size(),
                                                 builder.getNoneType());
      output = ::dataflow::SyncOp::create(builder, op->getLoc(), types,
                                          laneCompletions)
                   .getOutputs()
                   .front();
    }
    op->erase();
    placeDominatedRetirementAnchors(thread, output, channelRetirementAnchors,
                                    pendingLaunchDependencies, builder);
    return output;
  }

  return ::mlir::failure();
}

::mlir::FailureOr<::mlir::Value> propagateCompletionToThread(
    ::dataflow::ThreadOp thread, ::dataflow::GraphLaunchOp launch,
    ::mlir::Value fallback,
    ::llvm::MutableArrayRef<::mlir::Operation *> channelRetirementAnchors,
    PendingLaunchDependencies &pendingLaunchDependencies,
    std::deque<PendingGraphCompletion> &pendingCompletions,
    unsigned &parallelGroup, unsigned &nextParallelGroup,
    ::mlir::OpBuilder &builder) {
  ::llvm::SmallVector<::mlir::Operation *, 4> path;
  for (::mlir::Operation *parent = launch->getParentOp();
       parent && parent != thread.getOperation();
       parent = parent->getParentOp())
    path.push_back(parent);
  std::reverse(path.begin(), path.end());

  auto completion = propagateCompletionPath(
      path, launch, fallback, fallback, channelRetirementAnchors,
      pendingLaunchDependencies, pendingCompletions, parallelGroup,
      nextParallelGroup, thread, builder);
  if (::mlir::failed(completion) ||
      ::llvm::any_of(channelRetirementAnchors,
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

  LowerForToGraphPass() = default;
  LowerForToGraphPass(const LowerForToGraphPass &) : PassWrapper() {}

  Statistic parallelCompletionCandidateInspections{
      this, "parallel-completion-candidate-inspections",
      "Number of parallel completion candidates inspected for publication"};

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
  ::mlir::LogicalResult updateGraphLaunches(
      ::mlir::ModuleOp module, ::dataflow::GraphOp original,
      ::dataflow::GraphOp finalized,
      ::llvm::ArrayRef<::loom::lowering::GraphMemoryInputSource> sources) {
    ::llvm::ArrayRef<int32_t> finalizedInputs =
        finalized.getInputSegmentSizes();
    ::llvm::ArrayRef<int32_t> finalizedResults =
        finalized.getResultSegmentSizes();
    if (sources.size() != static_cast<size_t>(finalizedInputs[2]))
      return finalized.emitError(
          "graph memory projection does not cover the finalized signature");

    ::llvm::SmallVector<::dataflow::GraphLaunchOp, 4> launches;
    module.walk([&](::dataflow::GraphLaunchOp launch) {
      if (launch.getCallee() == original.getSymName())
        launches.push_back(launch);
    });

    ::llvm::ArrayRef<::mlir::Type> resultTypes =
        finalized.getFunctionType().getResults();
    ::llvm::ArrayRef<::mlir::Type> inputTypes =
        finalized.getFunctionType().getInputs();
    const unsigned valueInputCount = static_cast<unsigned>(finalizedInputs[0]);
    const unsigned streamInputCount = static_cast<unsigned>(finalizedInputs[1]);
    ::mlir::TypeRange memoryInputTypes =
        inputTypes.slice(valueInputCount + streamInputCount,
                         static_cast<unsigned>(finalizedInputs[2]));
    unsigned valueResultCount = static_cast<unsigned>(finalizedResults[0]);
    unsigned streamResultCount = static_cast<unsigned>(finalizedResults[1]);
    unsigned memoryResultCount = static_cast<unsigned>(finalizedResults[2]);
    ::mlir::TypeRange valueResultTypes =
        resultTypes.take_front(valueResultCount);
    ::mlir::TypeRange memoryResultTypes = resultTypes.slice(
        valueResultCount + streamResultCount, memoryResultCount);

    ::mlir::OpBuilder builder(module.getContext());
    for (::dataflow::GraphLaunchOp launch : launches) {
      ::llvm::SmallVector<::mlir::Value, 4> memoryInputs;
      memoryInputs.reserve(sources.size());
      builder.setInsertionPoint(launch);
      for (auto [ordinal, source] : ::llvm::enumerate(sources)) {
        if (source.kind ==
            ::loom::lowering::GraphMemoryInputSourceKind::ExistingMemory) {
          if (source.sourceOrdinal >= launch.getMemoryInputs().size())
            return launch.emitOpError("graph memory projection source #")
                   << source.sourceOrdinal << " is out of range";
          memoryInputs.push_back(
              launch.getMemoryInputs()[source.sourceOrdinal]);
          continue;
        }
        if (source.sourceOrdinal >= launch.getValueInputs().size())
          return launch.emitOpError("pointer service value source #")
                 << source.sourceOrdinal << " is out of range";
        ::mlir::Value pointer = launch.getValueInputs()[source.sourceOrdinal];
        if (!::llvm::isa<::mlir::LLVM::LLVMPointerType>(pointer.getType()))
          return launch.emitOpError("pointer service source #")
                 << source.sourceOrdinal << " has non-pointer type "
                 << pointer.getType();
        auto service = ::dataflow::MemoryServiceOp::create(
            builder, launch.getLoc(), memoryInputTypes[ordinal], pointer);
        memoryInputs.push_back(service.getMemory());
      }

      auto replacement = ::dataflow::GraphLaunchOp::create(
          builder, launch.getLoc(), valueResultTypes, memoryResultTypes,
          builder.getNoneType(), launch.getCalleeAttr(), launch.getSourceMaps(),
          launch.getDependencies(), launch.getValueInputs(),
          launch.getStreamInputs(), memoryInputs, launch.getStreamOutputs());
      for (::mlir::NamedAttribute attr : launch->getAttrs())
        if (!replacement->hasAttr(attr.getName()))
          replacement->setAttr(attr.getName(), attr.getValue());

      if (launch->getNumResults() != replacement->getNumResults())
        return launch.emitOpError(
            "finalized graph changed launch result cardinality");
      for (auto [oldResult, newResult] :
           ::llvm::zip_equal(launch->getResults(), replacement->getResults())) {
        if (oldResult.getType() != newResult.getType())
          return launch.emitOpError("finalized graph changed launch result ")
                 << oldResult.getResultNumber() << " type from "
                 << oldResult.getType() << " to " << newResult.getType();
        oldResult.replaceAllUsesWith(newResult);
      }
      launch.erase();
    }
    return ::mlir::success();
  }

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
          callable.getFunctionBody().dropAllReferences();
          callable.getFunctionBody().getBlocks().clear();
          // A body-less callable is a declaration, which cannot be public.
          ::mlir::SymbolTable::setSymbolVisibility(
              declaration, ::mlir::SymbolTable::Visibility::Private);
          if (auto llvmFunction =
                  ::llvm::dyn_cast<::mlir::LLVM::LLVMFuncOp>(declaration))
            llvmFunction.setLinkage(::mlir::LLVM::Linkage::External);
        }
      }
      ::llvm::SmallVector<::dataflow::GraphOp, 4> staged;
      staged.reserve(pending.size());
      for (::dataflow::GraphOp graph : pending)
        staged.push_back(::mlir::cast<::dataflow::GraphOp>(
            builder.clone(*graph.getOperation())));

      ::llvm::SmallVector<::loom::lowering::GraphMemoryInputProjection, 4>
          projections;
      if (::mlir::failed(lowerPendingGraphs(*staging, projections)))
        return ::mlir::failure();

      for (auto [graph, finalized] : ::llvm::zip_equal(pending, staged)) {
        ::dataflow::GraphOp finalizedGraph = finalized;
        auto projection =
            ::llvm::find_if(projections, [&](const auto &candidate) {
              return candidate.graph == finalizedGraph;
            });
        if (projection == projections.end())
          return finalizedGraph.emitError(
              "graph memory finalization produced no input projection");
        if (::mlir::failed(updateGraphLaunches(module, graph, finalizedGraph,
                                               projection->sources)))
          return ::mlir::failure();
        finalizedGraph->moveBefore(graph);
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

  ::mlir::LogicalResult lowerPendingGraphs(
      ::mlir::ModuleOp module,
      ::llvm::SmallVectorImpl<::loom::lowering::GraphMemoryInputProjection>
          &projections) {
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
    lowerer.addPass(::loom::lowering::createExpandGraphMemrefCopyPass());
    lowerer.addPass(::mlir::createCanonicalizerPass());
    if (::mlir::failed(lowerer.run(module)) ||
        ::mlir::failed(
            ::loom::lowering::lowerGraphMemory(module, &projections)) ||
        ::mlir::failed(verify(module)))
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
    module.walk(
        [&](::loom::SpatialRegionOp spatial) { regions.push_back(spatial); });

    ::llvm::DenseSet<::mlir::Operation *> parallelSet;
    ::llvm::SmallVector<::mlir::Operation *, 8> parallelOps;
    for (::loom::SpatialRegionOp spatial : regions) {
      auto thread = spatial->getParentOfType<::dataflow::ThreadOp>();
      if (!thread)
        return spatial.emitOpError(
            "expected loom.spatial_region inside dataflow.thread");
      for (::mlir::Operation *parent = spatial->getParentOp();
           parent && parent != thread.getOperation();
           parent = parent->getParentOp()) {
        if (::llvm::isa<::mlir::scf::IfOp, ::mlir::scf::ForOp,
                        ::mlir::scf::WhileOp, ::mlir::scf::IndexSwitchOp>(
                parent))
          continue;
        if (::llvm::isa<::mlir::scf::ParallelOp, ::mlir::scf::ForallOp>(
                parent)) {
          if (parallelSet.insert(parent).second)
            parallelOps.push_back(parent);
          continue;
        }
        return spatial.emitOpError("completion propagation through enclosing '")
               << parent->getName()
               << "' is not supported; spatial candidate cannot be "
                  "published";
      }
    }
    if (::mlir::failed(::loom::lowering::checkGraphOwnedParallelPreconditions(
            parallelOps)))
      return ::mlir::failure();

    PendingLaunchDependencies pendingLaunchDependencies;
    std::deque<PendingGraphCompletion> pendingCompletions;
    ::llvm::SmallVector<ParallelCompletionCandidate, 8> parallelCompletions;
    unsigned nextParallelGroup = 1;
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
      auto graph = ::dataflow::GraphOp::create(builder, loc, graphName,
                                               functionType, segmentAttrs);
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
      for (auto [index, result] : ::llvm::enumerate(spatial.getValueResults()))
        result.replaceAllUsesWith(launch.getValueResults()[index]);
      for (auto [index, result] : ::llvm::enumerate(spatial.getMemoryResults()))
        result.replaceAllUsesWith(launch.getMemoryResults()[index]);
      for (PendingGraphCompletion &pending : pendingCompletions)
        for (::mlir::Operation *&anchor : pending.channelRetirementAnchors)
          if (anchor == spatial.getOperation())
            anchor = launch.getOperation();
      spatial.erase();
      pendingCompletions.push_back(PendingGraphCompletion{
          launch, std::move(channelRetirementAnchors), 0});
    }
    if (!pendingLaunchDependencies.empty())
      return module.emitError(
          "channel retirement dependency target was not published");

    while (!pendingCompletions.empty()) {
      PendingGraphCompletion pending = std::move(pendingCompletions.front());
      pendingCompletions.pop_front();
      auto thread = pending.launch->getParentOfType<::dataflow::ThreadOp>();
      ::mlir::Block &threadEntry = thread.getBody().front();
      size_t ctrlIndex = thread.getFunctionType().getInputs().size();
      ::mlir::Value threadCtrl = threadEntry.getArgument(ctrlIndex);
      auto propagated = propagateCompletionToThread(
          thread, pending.launch, threadCtrl, pending.channelRetirementAnchors,
          pendingLaunchDependencies, pendingCompletions, pending.parallelGroup,
          nextParallelGroup, builder);
      if (::mlir::failed(propagated))
        return pending.launch.emitOpError(
            "failed to propagate completion through enclosing structured "
            "control");
      if (pending.parallelGroup == 0) {
        addThreadCompletionFrontiers(thread, ::mlir::ValueRange{*propagated});
      } else {
        parallelCompletions.push_back(ParallelCompletionCandidate{
            thread, *propagated, pending.parallelGroup});
      }
    }
    parallelCompletionCandidateInspections +=
        publishParallelCompletionCandidates(parallelCompletions, builder);
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
