#include "DataflowRewriteInternal.h"

#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace dataflow::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "dataflow_graph_definition_rewrite_invalid: " +
                                     message);
}

void stripEntityIds(mlir::Operation *root) {
  root->walk([](mlir::Operation *op) { op->removeAttr(kEntityIdAttrName); });
}

bool areAlphaIsomorphic(GraphOp lhs, GraphOp rhs) {
  if (!lhs || !rhs || lhs.isExternal() || rhs.isExternal())
    return false;
  mlir::OwningOpRef<GraphOp> left(llvm::cast<GraphOp>(lhs->clone()));
  mlir::OwningOpRef<GraphOp> right(llvm::cast<GraphOp>(rhs->clone()));
  stripEntityIds(left->getOperation());
  stripEntityIds(right->getOperation());
  left->setSymName("__loom_alpha_graph");
  right->setSymName("__loom_alpha_graph");
  const auto flags = static_cast<mlir::OperationEquivalence::Flags>(
      mlir::OperationEquivalence::IgnoreLocations |
      mlir::OperationEquivalence::IgnoreCommutativity);
  return mlir::OperationEquivalence::isEquivalentTo(
      left->getOperation(), right->getOperation(), flags);
}

std::vector<CanonicalStaticGraphLaunchView>
callersOf(const CanonicalDataflowProgramView &view, GraphRef graph) {
  std::vector<CanonicalStaticGraphLaunchView> callers;
  for (const CanonicalStaticGraphLaunchView &launch :
       view.staticGraphLaunches())
    if (launch.callee == graph)
      callers.push_back(launch);
  llvm::sort(callers, [](const CanonicalStaticGraphLaunchView &lhs,
                         const CanonicalStaticGraphLaunchView &rhs) {
    return lhs.ref.entity.value() < rhs.ref.entity.value();
  });
  return callers;
}

bool idSequenceLess(llvm::ArrayRef<StaticGraphLaunchId> lhs,
                    llvm::ArrayRef<StaticGraphLaunchId> rhs) {
  return std::lexicographical_compare(
      lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
      [](StaticGraphLaunchId left, StaticGraphLaunchId right) {
        return left.value() < right.value();
      });
}

void enumerateNormalizedSplits(
    GraphId graph, llvm::ArrayRef<CanonicalStaticGraphLaunchView> callers,
    std::size_t ordinal, std::vector<StaticGraphLaunchId> &selected,
    std::vector<DataflowRewriteDecision> &decisions) {
  if (ordinal == callers.size()) {
    if (selected.size() != callers.size())
      decisions.emplace_back(GraphDefinitionSplitRewrite{graph, selected});
    return;
  }
  selected.push_back(callers[ordinal].ref.entity);
  enumerateNormalizedSplits(graph, callers, ordinal + 1, selected, decisions);
  selected.pop_back();
  enumerateNormalizedSplits(graph, callers, ordinal + 1, selected, decisions);
}

llvm::Expected<std::pair<std::vector<CanonicalStaticGraphLaunchView>,
                         std::vector<CanonicalStaticGraphLaunchView>>>
validateSplit(const CanonicalDataflowProgramView &view,
              const GraphDefinitionSplitRewrite &decision) {
  auto graph = view.resolve(GraphRef{view.identity(), decision.graph});
  if (!graph)
    return graph.takeError();
  auto callers = callersOf(view, graph->ref);
  if (callers.size() < 2)
    return invalid("split graph has fewer than two static launches");

  std::vector<CanonicalStaticGraphLaunchView> selected;
  std::vector<CanonicalStaticGraphLaunchView> complement;
  std::size_t selectedOrdinal = 0;
  for (const CanonicalStaticGraphLaunchView &caller : callers) {
    if (selectedOrdinal < decision.launches.size() &&
        decision.launches[selectedOrdinal] == caller.ref.entity) {
      selected.push_back(caller);
      ++selectedOrdinal;
    } else {
      complement.push_back(caller);
    }
  }
  if (selectedOrdinal != decision.launches.size() || selected.empty() ||
      complement.empty())
    return invalid("split does not name a nonempty proper caller partition");

  std::vector<StaticGraphLaunchId> complementIds;
  llvm::transform(complement, std::back_inserter(complementIds),
                  [](const CanonicalStaticGraphLaunchView &launch) {
                    return launch.ref.entity;
                  });
  if (!idSequenceLess(decision.launches, complementIds))
    return invalid("split selects the noncanonical side of its bipartition");
  return std::make_pair(std::move(selected), std::move(complement));
}

std::string freshGraphName(mlir::ModuleOp module) {
  mlir::SymbolTable symbols(module);
  std::string name = "__loom_graph_split";
  unsigned suffix = 0;
  while (symbols.lookup(name))
    name = (llvm::Twine("__loom_graph_split_") + llvm::Twine(++suffix)).str();
  return name;
}

llvm::Expected<std::optional<CanonicalDataflowArtifact>>
finalizeChanged(const CanonicalDataflowArtifact &parent,
                mlir::ModuleOp candidate) {
  auto finalized = finalizeCanonicalDataflow(candidate);
  if (!finalized)
    return finalized.takeError();
  if (finalized->identity() == parent.identity())
    return std::optional<CanonicalDataflowArtifact>{};
  return std::optional<CanonicalDataflowArtifact>(std::move(*finalized));
}

} // namespace

llvm::Expected<std::vector<DataflowRewriteDecision>>
enumerateGraphDefinitionRefactorDecisions(
    const CanonicalDataflowArtifact &parent) {
  auto view = parent.view();
  if (!view)
    return view.takeError();
  std::vector<DataflowRewriteDecision> decisions;
  for (const CanonicalGraphView &graph : view->graphs()) {
    auto callers = callersOf(*view, graph.ref);
    if (callers.size() < 2)
      continue;
    std::vector<StaticGraphLaunchId> selected{callers.front().ref.entity};
    enumerateNormalizedSplits(graph.ref.entity, callers, 1, selected,
                              decisions);
  }

  for (std::size_t lower = 0; lower != view->graphs().size(); ++lower) {
    for (std::size_t higher = lower + 1; higher != view->graphs().size();
         ++higher) {
      const CanonicalGraphView *first = &view->graphs()[lower];
      const CanonicalGraphView *second = &view->graphs()[higher];
      if (second->ref.entity.value() < first->ref.entity.value())
        std::swap(first, second);
      if (areAlphaIsomorphic(llvm::cast<GraphOp>(first->op),
                             llvm::cast<GraphOp>(second->op)))
        decisions.emplace_back(
            GraphDefinitionMergeRewrite{first->ref.entity, second->ref.entity});
    }
  }
  llvm::sort(decisions, dataflowRewriteDecisionLess);
  return decisions;
}

llvm::Expected<std::optional<CanonicalDataflowArtifact>>
materializeGraphDefinitionRefactor(const CanonicalDataflowArtifact &parent,
                                   const DataflowRewriteDecision &decision) {
  auto view = parent.view();
  if (!view)
    return view.takeError();
  mlir::IRMapping mapping;
  mlir::OwningOpRef<mlir::ModuleOp> candidate(
      mlir::cast<mlir::ModuleOp>(parent.module()->clone(mapping)));

  if (const auto *split = std::get_if<GraphDefinitionSplitRewrite>(&decision)) {
    auto partition = validateSplit(*view, *split);
    if (!partition)
      return partition.takeError();
    auto sourceView = view->resolve(GraphRef{parent.identity(), split->graph});
    if (!sourceView)
      return sourceView.takeError();
    auto source =
        llvm::dyn_cast_or_null<GraphOp>(mapping.lookupOrNull(sourceView->op));
    if (!source)
      return invalid("split graph was not cloned");

    auto clone = llvm::cast<GraphOp>(source->clone());
    stripEntityIds(clone.getOperation());
    clone.setSymName(freshGraphName(candidate.get()));
    mlir::OpBuilder builder(source);
    builder.setInsertionPointAfter(source);
    builder.insert(clone.getOperation());
    for (const CanonicalStaticGraphLaunchView &launch : partition->first) {
      auto clonedLaunch = llvm::dyn_cast_or_null<GraphLaunchOp>(
          mapping.lookupOrNull(launch.op));
      if (!clonedLaunch)
        return invalid("selected static launch was not cloned");
      clonedLaunch.setCallee(clone.getSymName());
    }
    return finalizeChanged(parent, candidate.get());
  }

  const auto *merge = std::get_if<GraphDefinitionMergeRewrite>(&decision);
  if (!merge)
    return invalid("decision is not a graph-definition variant");
  auto lowerView =
      view->resolve(GraphRef{parent.identity(), merge->lowerGraph});
  if (!lowerView)
    return lowerView.takeError();
  auto higherView =
      view->resolve(GraphRef{parent.identity(), merge->higherGraph});
  if (!higherView)
    return higherView.takeError();
  auto lower = llvm::dyn_cast<GraphOp>(lowerView->op);
  auto higher = llvm::dyn_cast<GraphOp>(higherView->op);
  if (!areAlphaIsomorphic(lower, higher))
    return invalid("merge graphs are not alpha-isomorphic");
  auto clonedLower = llvm::dyn_cast_or_null<GraphOp>(
      mapping.lookupOrNull(lower.getOperation()));
  auto clonedHigher = llvm::dyn_cast_or_null<GraphOp>(
      mapping.lookupOrNull(higher.getOperation()));
  if (!clonedLower || !clonedHigher)
    return invalid("merge graphs were not cloned");
  for (const CanonicalStaticGraphLaunchView &launch :
       callersOf(*view, higherView->ref)) {
    auto clonedLaunch =
        llvm::dyn_cast_or_null<GraphLaunchOp>(mapping.lookupOrNull(launch.op));
    if (!clonedLaunch)
      return invalid("higher graph static launch was not cloned");
    clonedLaunch.setCallee(clonedLower.getSymName());
  }
  clonedHigher.erase();
  return finalizeChanged(parent, candidate.get());
}

} // namespace dataflow::detail
