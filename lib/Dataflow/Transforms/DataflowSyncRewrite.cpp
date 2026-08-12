#include "DataflowRewriteInternal.h"

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace dataflow::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "dataflow_sync_rewrite_invalid: " + message);
}

std::optional<std::size_t> directLiveResult(SyncOp sync) {
  if (sync.getInputs().size() <= 2 ||
      sync.getOutputs().size() != sync.getInputs().size())
    return std::nullopt;
  std::optional<std::size_t> live;
  for (auto [ordinal, result] : llvm::enumerate(sync.getOutputs())) {
    if (result.use_empty())
      continue;
    if (live)
      return std::nullopt;
    live = ordinal;
  }
  return live;
}

struct TreeNode final {
  SyncOp sync;
  std::size_t firstLeaf;
  std::size_t leafCount;
  std::size_t leftLeafCount;
  unsigned selectedResult;
};

struct TreeAnalysis final {
  llvm::SmallVector<mlir::Value, 8> leaves;
  std::size_t carrierLeaf = 0;
  std::vector<TreeNode> nodes;
};

llvm::Expected<std::optional<TreeAnalysis>>
analyzeTreeNode(SyncOp sync, unsigned selectedResult, bool root) {
  if (sync.getInputs().size() != 2 || sync.getOutputs().size() != 2 ||
      selectedResult >= 2)
    return std::optional<TreeAnalysis>{};
  for (unsigned result = 0; result != 2; ++result) {
    mlir::Value output = sync.getOutputs()[result];
    if (result != selectedResult) {
      if (!output.use_empty())
        return std::optional<TreeAnalysis>{};
    } else if (root ? output.use_empty() : !output.hasOneUse()) {
      return std::optional<TreeAnalysis>{};
    }
  }

  std::optional<TreeAnalysis> children[2];
  for (unsigned side = 0; side != 2; ++side) {
    auto result = llvm::dyn_cast<mlir::OpResult>(sync.getInputs()[side]);
    auto child = result ? llvm::dyn_cast<SyncOp>(result.getOwner()) : SyncOp{};
    if (!child)
      continue;
    auto analyzed = analyzeTreeNode(child, result.getResultNumber(), false);
    if (!analyzed)
      return analyzed.takeError();
    if (*analyzed)
      children[side] = std::move(**analyzed);
  }

  TreeAnalysis analysis;
  std::size_t childCounts[2] = {1, 1};
  for (unsigned side = 0; side != 2; ++side) {
    if (children[side]) {
      childCounts[side] = children[side]->leaves.size();
      analysis.leaves.append(children[side]->leaves);
      const std::size_t offset = side == 0 ? 0 : childCounts[0];
      for (TreeNode node : children[side]->nodes) {
        node.firstLeaf += offset;
        analysis.nodes.push_back(node);
      }
    } else {
      analysis.leaves.push_back(sync.getInputs()[side]);
    }
  }
  const std::size_t total = childCounts[0] + childCounts[1];
  if (childCounts[0] != (total + 1) / 2 || childCounts[1] != total / 2)
    return std::optional<TreeAnalysis>{};
  analysis.carrierLeaf =
      selectedResult == 0
          ? (children[0] ? children[0]->carrierLeaf : 0)
          : childCounts[0] + (children[1] ? children[1]->carrierLeaf : 0);
  analysis.nodes.push_back(
      TreeNode{sync, 0, total, childCounts[0], selectedResult});
  return std::optional<TreeAnalysis>(std::move(analysis));
}

std::optional<TreeAnalysis> analyzeCanonicalTree(SyncOp root) {
  std::optional<unsigned> selected;
  for (unsigned result = 0; result != root.getOutputs().size(); ++result) {
    if (root.getOutputs()[result].use_empty())
      continue;
    if (selected)
      return std::nullopt;
    selected = result;
  }
  if (!selected)
    return std::nullopt;
  auto analyzed = analyzeTreeNode(root, *selected, true);
  if (!analyzed) {
    llvm::consumeError(analyzed.takeError());
    return std::nullopt;
  }
  if (!*analyzed || (*analyzed)->leaves.size() <= 2)
    return std::nullopt;

  const std::size_t liveLeaf = (*analyzed)->carrierLeaf;
  for (const TreeNode &node : (*analyzed)->nodes) {
    const bool containsLive = liveLeaf >= node.firstLeaf &&
                              liveLeaf < node.firstLeaf + node.leafCount;
    const unsigned expected =
        containsLive && liveLeaf >= node.firstLeaf + node.leftLeafCount ? 1 : 0;
    if (node.selectedResult != expected)
      return std::nullopt;
  }
  return std::move(**analyzed);
}

mlir::Value buildCanonicalTree(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::ValueRange inputs, std::size_t firstLeaf,
                               std::size_t liveLeaf) {
  if (inputs.size() == 1)
    return inputs.front();
  const std::size_t leftCount = (inputs.size() + 1) / 2;
  mlir::Value left = buildCanonicalTree(
      builder, loc, inputs.take_front(leftCount), firstLeaf, liveLeaf);
  mlir::Value right =
      buildCanonicalTree(builder, loc, inputs.drop_front(leftCount),
                         firstLeaf + leftCount, liveLeaf);
  auto sync = SyncOp::create(builder, loc,
                             mlir::TypeRange{left.getType(), right.getType()},
                             mlir::ValueRange{left, right});
  const bool liveInRight =
      liveLeaf >= firstLeaf + leftCount && liveLeaf < firstLeaf + inputs.size();
  return sync.getOutputs()[liveInRight ? 1 : 0];
}

llvm::Expected<CanonicalActorView>
resolveRoot(const CanonicalDataflowArtifact &parent, ActorId root) {
  auto view = parent.view();
  if (!view)
    return view.takeError();
  auto resolved = view->resolve(ActorRef{parent.identity(), root});
  if (!resolved)
    return resolved.takeError();
  if (!llvm::isa<SyncOp>(resolved->op))
    return invalid("root is not dataflow.sync");
  return *resolved;
}

} // namespace

llvm::Expected<std::vector<DataflowRewriteDecision>>
enumerateSyncRendezvousDecisions(const CanonicalDataflowArtifact &parent) {
  auto view = parent.view();
  if (!view)
    return view.takeError();
  std::vector<DataflowRewriteDecision> decisions;
  for (const CanonicalActorView &actor : view->actors()) {
    auto sync = llvm::dyn_cast<SyncOp>(actor.op);
    if (!sync)
      continue;
    if (directLiveResult(sync))
      decisions.emplace_back(SyncRendezvousRewrite{
          actor.ref.entity, SyncRendezvousDirection::DirectToTree});
    if (analyzeCanonicalTree(sync))
      decisions.emplace_back(SyncRendezvousRewrite{
          actor.ref.entity, SyncRendezvousDirection::TreeToDirect});
  }
  llvm::sort(decisions, dataflowRewriteDecisionLess);
  return decisions;
}

llvm::Expected<std::optional<MaterializedDataflowRewriteProjection>>
materializeSyncRendezvousRewriteProjection(
    const CanonicalDataflowArtifact &parent,
    const SyncRendezvousRewrite &decision,
    llvm::ArrayRef<StaticGraphLaunchRef> trackedStaticGraphLaunches) {
  auto resolved = resolveRoot(parent, decision.root);
  if (!resolved)
    return resolved.takeError();
  auto source = llvm::cast<SyncOp>(resolved->op);
  std::optional<std::size_t> live;
  std::optional<TreeAnalysis> tree;
  if (decision.direction == SyncRendezvousDirection::DirectToTree) {
    live = directLiveResult(source);
    if (!live)
      return invalid("root is not a legal direct rendezvous");
  } else if (decision.direction == SyncRendezvousDirection::TreeToDirect) {
    tree = analyzeCanonicalTree(source);
    if (!tree)
      return invalid("root is not the canonical rendezvous tree");
  } else {
    return invalid("direction is unknown");
  }

  mlir::IRMapping mapping;
  mlir::OwningOpRef<mlir::ModuleOp> candidate(
      mlir::cast<mlir::ModuleOp>(parent.module()->clone(mapping)));
  auto root = llvm::dyn_cast_or_null<SyncOp>(mapping.lookupOrNull(source));
  if (!root)
    return invalid("root was not cloned into the candidate");
  mlir::OpBuilder builder(root);

  if (decision.direction == SyncRendezvousDirection::DirectToTree) {
    mlir::Value replacement =
        buildCanonicalTree(builder, root.getLoc(), root.getInputs(), 0, *live);
    root.getOutputs()[*live].replaceAllUsesWith(replacement);
    root.erase();
  } else {
    std::optional<TreeAnalysis> cloned = analyzeCanonicalTree(root);
    if (!cloned)
      return invalid("cloned rendezvous tree changed shape");
    llvm::SmallVector<mlir::Type, 8> types;
    for (mlir::Value leaf : cloned->leaves)
      types.push_back(leaf.getType());
    auto direct = SyncOp::create(builder, root.getLoc(), types, cloned->leaves);
    const unsigned rootCarrier = cloned->nodes.back().selectedResult;
    root.getOutputs()[rootCarrier].replaceAllUsesWith(
        direct.getOutputs()[cloned->carrierLeaf]);
    for (TreeNode &node : llvm::reverse(cloned->nodes))
      node.sync.erase();
  }

  return finalizeDataflowRewriteCandidate(parent, candidate.get(), mapping,
                                          trackedStaticGraphLaunches);
}

llvm::Expected<std::optional<CanonicalDataflowArtifact>>
materializeSyncRendezvousRewrite(const CanonicalDataflowArtifact &parent,
                                 const SyncRendezvousRewrite &decision) {
  auto projected =
      materializeSyncRendezvousRewriteProjection(parent, decision, {});
  if (!projected)
    return projected.takeError();
  if (!*projected)
    return std::optional<CanonicalDataflowArtifact>{};
  return std::optional<CanonicalDataflowArtifact>(
      std::move((*projected)->artifact));
}

} // namespace dataflow::detail
