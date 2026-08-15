#include "SpatialProgressAnalysis.h"

#include "PnR/RouteTreeState.h"
#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialPnrProblem.h"

#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <system_error>
#include <vector>

using namespace loom::fabric;
using namespace loom::pnr;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_progress_analysis_invalid: " + message);
}

bool isBufferedTraversal(const FrozenSpatialRoutingGraph &routing,
                         PnrIndex traversal) {
  if (traversal >= routing.traversals().size())
    return false;
  const auto *fifo = std::get_if<FabricFifoTraversalPayload>(
      &routing.traversals()[traversal].reference.payload);
  return fifo && fifo->mode == FabricFifoTraversalMode::Buffered;
}

llvm::Expected<PnrIndex> parentSlot(const RouteTreeState &tree, PnrIndex slot) {
  if (slot >= tree.nodeStorage().size() || !tree.nodeStorage()[slot].isActive())
    return invalid("route-tree slot is absent");
  const RouteTreeNode &node = tree.node(slot);
  if (node.parentArc == getInvalidPnrIndex())
    return getInvalidPnrIndex();
  const auto arcs = tree.routingGraph().routingArcs();
  const auto sources = tree.routingGraph().arcSources();
  if (node.parentArc >= arcs.size() || node.parentArc >= sources.size())
    return invalid("route-tree parent arc is out of range");
  const auto parent = tree.findNode(sources[node.parentArc]);
  if (!parent)
    return invalid("route-tree parent endpoint is absent");
  return *parent;
}

} // namespace

llvm::Expected<bool> loom::pnr::spatialAttachmentProvidesLocalProgressBoundary(
    const FrozenSpatialPortIndex &ports, PnrIndex attachmentOption) {
  if (attachmentOption >= ports.attachmentOptions().size())
    return invalid("attachment option is out of range");
  return ports.attachmentOptions()[attachmentOption].progressBoundary !=
         ::loom::mapping::SpatialDurableProgressBoundaryKind::None;
}

llvm::Expected<bool> loom::pnr::spatialTerminalProvidesLocalProgressBoundary(
    const SpatialCandidateState &candidate,
    FrozenSpatialTerminalBinding terminal) {
  PnrIndex option = 0;
  switch (terminal.kind) {
  case FrozenSpatialTerminalBindingKind::PortDemand:
    if (terminal.index >= candidate.problem().ports().portDemands().size())
      return invalid("terminal PortDemand is out of range");
    option = candidate.portAttachment(terminal.index);
    break;
  case FrozenSpatialTerminalBindingKind::GraphBoundary:
    if (terminal.index >= candidate.problem().ports().graphBoundaries().size())
      return invalid("terminal graph boundary is out of range");
    option = candidate.graphBoundaryAttachment(terminal.index);
    break;
  }
  return spatialAttachmentProvidesLocalProgressBoundary(
      candidate.problem().ports(), option);
}

llvm::Expected<llvm::ArrayRef<FrozenSpatialProgressPrerequisite>>
loom::pnr::spatialSinkProgressDependencies(
    const FrozenSpatialPnrProblem &problem, PnrIndex logicalNet,
    PnrIndex dependentSink) {
  const FrozenSpatialTransferIndex &transfers = problem.transfers();
  if (logicalNet >= transfers.logicalNets().size())
    return invalid("logical net is out of range");
  const FrozenSpatialLogicalNet &net = transfers.logicalNets()[logicalNet];
  if (dependentSink >= net.sinkCount)
    return invalid("dependent sink is out of range");
  const std::size_t globalSink =
      static_cast<std::size_t>(net.sinkOffset) + dependentSink;
  const auto offsets = transfers.sinkProgressDependencyOffsets();
  const auto dependencies = transfers.sinkProgressDependencies();
  if (globalSink + 1 >= offsets.size() ||
      offsets[globalSink] > offsets[globalSink + 1] ||
      offsets[globalSink + 1] > dependencies.size())
    return invalid("sink progress dependency CSR is inconsistent");
  return dependencies.slice(offsets[globalSink],
                            offsets[globalSink + 1] - offsets[globalSink]);
}

llvm::Expected<bool> loom::pnr::spatialRouteProgressDependencySatisfied(
    const SpatialCandidateState &candidate, PnrIndex logicalNet,
    const FrozenSpatialProgressPrerequisite &prerequisite,
    PnrIndex dependentSink) {
  auto dependencies = spatialSinkProgressDependencies(
      candidate.problem(), logicalNet, dependentSink);
  if (!dependencies)
    return dependencies.takeError();
  if (!llvm::is_contained(*dependencies, prerequisite))
    return invalid("sink pair is not a frozen progress dependency");

  const FrozenSpatialLogicalNet &net =
      candidate.problem().transfers().logicalNets()[logicalNet];
  const auto *external =
      std::get_if<FrozenSpatialExternalSinkPrerequisite>(&prerequisite);
  if (external && external->sink >= net.sinkCount)
    return invalid("external progress prerequisite is out of range");
  auto localBoundary = spatialTerminalProvidesLocalProgressBoundary(
      candidate, candidate.problem()
                     .transfers()
                     .logicalNetSinkBindings()[net.sinkOffset + dependentSink]);
  if (!localBoundary)
    return localBoundary.takeError();
  if (*localBoundary)
    return true;

  const RouteTreeState &tree = candidate.routeTree(logicalNet);
  const auto dependentEndpoint = tree.sinkEndpoint(dependentSink);
  if (!dependentEndpoint)
    return true;
  const auto dependentSlot = tree.findNode(*dependentEndpoint);
  if (!dependentSlot)
    return true;

  if (!external) {
    PnrIndex slot = *dependentSlot;
    std::size_t visited = 0;
    while (slot != getInvalidPnrIndex()) {
      if (++visited > tree.activeNodeCount())
        return invalid("route-tree dependent ancestry is cyclic");
      const RouteTreeNode &node = tree.node(slot);
      if (node.parentArc == getInvalidPnrIndex())
        return false;
      const auto arcs = tree.routingGraph().routingArcs();
      if (node.parentArc >= arcs.size())
        return invalid("dependent branch parent arc is out of range");
      if (isBufferedTraversal(tree.routingGraph(),
                              arcs[node.parentArc].traversal))
        return true;
      auto parent = parentSlot(tree, slot);
      if (!parent)
        return parent.takeError();
      slot = *parent;
    }
    return false;
  }

  const auto prerequisiteEndpoint = tree.sinkEndpoint(external->sink);
  if (!prerequisiteEndpoint)
    return true;
  const auto prerequisiteSlot = tree.findNode(*prerequisiteEndpoint);
  if (!prerequisiteSlot)
    return true;

  std::vector<std::uint8_t> prerequisiteAncestors(tree.nodeStorage().size(), 0);
  PnrIndex slot = *prerequisiteSlot;
  while (slot != getInvalidPnrIndex()) {
    if (slot >= prerequisiteAncestors.size() || prerequisiteAncestors[slot])
      return invalid("route-tree prerequisite ancestry is cyclic");
    prerequisiteAncestors[slot] = 1;
    auto parent = parentSlot(tree, slot);
    if (!parent)
      return parent.takeError();
    slot = *parent;
  }

  bool bufferedAfterDivergence = false;
  slot = *dependentSlot;
  std::size_t visited = 0;
  while (slot != getInvalidPnrIndex() && !prerequisiteAncestors[slot]) {
    if (++visited > tree.activeNodeCount())
      return invalid("route-tree dependent ancestry is cyclic");
    const RouteTreeNode &node = tree.node(slot);
    if (node.parentArc == getInvalidPnrIndex())
      return invalid("dependent branch does not meet the prerequisite branch");
    const auto arcs = tree.routingGraph().routingArcs();
    if (node.parentArc >= arcs.size())
      return invalid("dependent branch parent arc is out of range");
    bufferedAfterDivergence |= isBufferedTraversal(
        tree.routingGraph(), arcs[node.parentArc].traversal);
    auto parent = parentSlot(tree, slot);
    if (!parent)
      return parent.takeError();
    slot = *parent;
  }
  if (slot == getInvalidPnrIndex())
    return invalid("route-tree sink branches have no common ancestor");
  return bufferedAfterDivergence;
}

llvm::Expected<std::uint64_t> loom::pnr::spatialCandidateClosedWaitCount(
    const SpatialCandidateState &candidate) {
  std::uint64_t count = 0;
  const auto nets = candidate.problem().transfers().logicalNets();
  for (PnrIndex logicalNet = 0; logicalNet < nets.size(); ++logicalNet) {
    for (PnrIndex dependent = 0; dependent < nets[logicalNet].sinkCount;
         ++dependent) {
      auto prerequisites = spatialSinkProgressDependencies(
          candidate.problem(), logicalNet, dependent);
      if (!prerequisites)
        return prerequisites.takeError();
      for (const FrozenSpatialProgressPrerequisite &prerequisite :
           *prerequisites) {
        auto satisfied = spatialRouteProgressDependencySatisfied(
            candidate, logicalNet, prerequisite, dependent);
        if (!satisfied)
          return satisfied.takeError();
        if (*satisfied)
          continue;
        if (count == std::numeric_limits<std::uint64_t>::max())
          return invalid("closed wait count exceeds u64");
        ++count;
      }
    }
  }
  return count;
}
