#include "SpatialProgressAnalysis.h"

#include "PnR/RouteTreeState.h"
#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialPnrProblem.h"

#include "Fabric/IR/PhysicalTag.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefs.h"
#include "Mapping/Artifact/MappingProgressAnalysis.h"

#include "SpatialProgressIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <set>
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

llvm::Expected<const FrozenSpatialAttachmentOption *>
selectedAttachment(const SpatialCandidateState &candidate,
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
  if (option >= candidate.problem().ports().attachmentOptions().size())
    return invalid("selected terminal attachment is out of range");
  return &candidate.problem().ports().attachmentOptions()[option];
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
  auto attachment = selectedAttachment(candidate, terminal);
  if (!attachment)
    return attachment.takeError();
  return (*attachment)->progressBoundary !=
         ::loom::mapping::SpatialDurableProgressBoundaryKind::None;
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

llvm::Expected<std::uint64_t> loom::pnr::spatialCandidateProgressWitnessCount(
    const SpatialCandidateState &candidate) {
  const std::size_t ownerCount =
      candidate.problem().progressIndex().finiteBufferOwners().size();
  std::vector<std::uint64_t> selectedNets(ownerCount, 0);
  std::vector<std::uint8_t> repeated(ownerCount, 0);
  std::vector<std::uint8_t> indeterminateQueueClass(ownerCount, 0);
  const auto nets = candidate.problem().transfers().logicalNets();
  for (PnrIndex logicalNet = 0; logicalNet < nets.size(); ++logicalNet) {
    auto capacity =
        projectSpatialNetCapacityProofInputs(candidate, logicalNet);
    if (!capacity)
      return capacity.takeError();
    for (const SpatialProgressOwnerCapacityUse &use : capacity->owners) {
      if (use.owner >= ownerCount ||
          selectedNets[use.owner] ==
              std::numeric_limits<std::uint64_t>::max())
        return invalid("cold capacity proof aggregate exceeds u64");
      ++selectedNets[use.owner];
      repeated[use.owner] |= use.repeatedWithinChannel;
      indeterminateQueueClass[use.owner] |= use.queueClassIndeterminate;
    }
  }

  std::uint64_t count = 0;
  for (std::size_t owner = 0; owner != ownerCount; ++owner) {
    const bool virtualChannel =
        candidate.problem().progressIndex().ownerQueueDisciplines()[owner] ==
        ::fabric::FifoQueueDiscipline::PerTagVirtualChannel;
    const bool debt = selectedNets[owner] != 0 &&
                      (repeated[owner] ||
                       (virtualChannel && indeterminateQueueClass[owner]));
    const bool shortfall =
        !debt && selectedNets[owner] >
                     candidate.problem()
                         .progressIndex()
                         .ownerSharedSlotCapacities()[owner];
    if (debt || shortfall) {
      if (count == std::numeric_limits<std::uint64_t>::max())
        return invalid("cold capacity proof witness count exceeds u64");
      ++count;
    }
  }

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

llvm::Expected<std::optional<SpatialFifoCapacityShortfall>>
loom::pnr::projectSpatialFifoCapacityShortfall(
    const SpatialCandidateState &candidate) {
  const auto owner = candidate.progress().firstCapacityShortfallOwner();
  if (!owner)
    return std::optional<SpatialFifoCapacityShortfall>();
  const auto capacities =
      candidate.problem().progressIndex().ownerSharedSlotCapacities();
  if (*owner >= capacities.size())
    return invalid("capacity feedback owner is out of range");
  const std::uint64_t selected = capacities[*owner];
  const std::uint64_t shortfall =
      candidate.progress().capacityShortfall(*owner);
  if (shortfall == 0 ||
      shortfall > std::numeric_limits<std::uint64_t>::max() - selected)
    return invalid("capacity feedback has an invalid shortfall");
  SpatialFiniteBufferConflictWitness witness;
  if (llvm::Error error = candidate.rebuildCapacityShortfallWitness(
          *owner, witness))
    return std::move(error);

  SpatialFifoCapacityShortfall feedback;
  feedback.owner = witness.owner;
  feedback.selectedCapacity = selected;
  feedback.minimumLegalCapacity = selected + shortfall;
  const auto logicalNets = candidate.problem().transfers().logicalNets();
  feedback.logicalNets.reserve(witness.competingLogicalNets.size());
  for (PnrIndex logicalNet : witness.competingLogicalNets) {
    if (logicalNet >= logicalNets.size())
      return invalid("capacity feedback logical net is out of range");
    feedback.logicalNets.push_back(logicalNets[logicalNet].producer);
  }
  std::map<std::string, ::loom::fabric::FabricPhysicalTraversalRef> anchors;
  const auto traversals = candidate.problem().routing().traversals();
  for (const SpatialProgressRouteAnchor &anchor : witness.routeAnchors) {
    if (anchor.traversal >= traversals.size())
      return invalid("capacity feedback traversal is out of range");
    const auto &reference = traversals[anchor.traversal].reference;
    const std::vector<std::uint8_t> bytes =
        ::loom::fabric::canonicalFabricBytes(reference);
    anchors.emplace(
        std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size()),
        reference);
  }
  feedback.routeAnchors.reserve(anchors.size());
  for (const auto &[key, anchor] : anchors) {
    (void)key;
    feedback.routeAnchors.push_back(anchor);
  }
  return std::optional<SpatialFifoCapacityShortfall>(std::move(feedback));
}

llvm::Expected<SpatialProgressNetCapacityProjection>
loom::pnr::projectSpatialNetCapacityProofInputs(
    const SpatialCandidateState &candidate, PnrIndex logicalNet,
    const RouteTreeState *projectedRoute) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const auto nets = problem.transfers().logicalNets();
  const auto sources = problem.transfers().logicalNetSourceBindings();
  const auto sinks = problem.transfers().logicalNetSinkBindings();
  if (logicalNet >= nets.size() || logicalNet >= sources.size())
    return invalid("capacity proof input names a foreign logical net");
  const RouteTreeState &tree =
      projectedRoute ? *projectedRoute : candidate.routeTree(logicalNet);
  if (&tree.routingGraph() != &problem.routing())
    return invalid("capacity proof route belongs to another frozen problem");
  if (candidate.usesRegisterFifo(logicalNet) || !tree.isRouted())
    return SpatialProgressNetCapacityProjection{};

  const FrozenSpatialLogicalNet &net = nets[logicalNet];
  if (net.sinkOffset > sinks.size() ||
      net.sinkCount > sinks.size() - net.sinkOffset)
    return invalid("capacity proof input has an invalid sink range");
  auto source = selectedAttachment(candidate, sources[logicalNet]);
  if (!source)
    return source.takeError();
  const auto tagSegments = candidate.tagSegments(logicalNet);
  const auto tagNodeSegments = candidate.tagNodeSegments(logicalNet);
  const auto tagValues = candidate.tagValues(logicalNet);
  if (tagValues.size() != tagSegments.size())
    return invalid("capacity proof tag values disagree with route segments");
  const auto sourceEndpoint = tree.sourceEndpoint();
  const auto sourceSlot =
      sourceEndpoint ? tree.findNode(*sourceEndpoint) : std::nullopt;
  if (!sourceSlot)
    return invalid("routed capacity proof channel has no source node");

  struct MutableUse final {
    std::uint64_t channels = 0;
    std::uint64_t initializedFeedbackChannels = 0;
    bool repeated = false;
    bool queueClassIndeterminate = false;
    std::vector<llvm::APInt> queueClasses;
  };
  std::map<PnrIndex, MutableUse> uses;
  for (PnrIndex sink = 0; sink < net.sinkCount; ++sink) {
    struct ChannelUse final {
      std::set<PnrIndex> traversals;
      bool queueClassIndeterminate = false;
      std::vector<llvm::APInt> queueClasses;
    };
    std::map<PnrIndex, ChannelUse> channelUses;
    const auto appendTraversal = [&](PnrIndex traversal,
                                     PnrIndex node) -> llvm::Error {
      if (traversal >= problem.routing().traversals().size())
        return invalid("capacity proof input selects a foreign traversal");
      const PnrIndex owner =
          problem.progressIndex().traversalOwner(traversal);
      if (owner == getInvalidPnrIndex())
        return llvm::Error::success();
      if (owner >= problem.progressIndex().ownerQueueDisciplines().size())
        return invalid("capacity proof traversal owner is out of range");
      ChannelUse &use = channelUses[owner];
      if (!use.traversals.insert(traversal).second)
        return llvm::Error::success();
      if (problem.progressIndex().ownerQueueDisciplines()[owner] !=
          ::fabric::FifoQueueDiscipline::PerTagVirtualChannel)
        return llvm::Error::success();
      if (node >= tagNodeSegments.size()) {
        use.queueClassIndeterminate = true;
        return llvm::Error::success();
      }
      const PnrIndex segment = tagNodeSegments[node];
      if (segment == getInvalidPnrIndex() || segment >= tagValues.size() ||
          !tagValues[segment]) {
        use.queueClassIndeterminate = true;
        return llvm::Error::success();
      }
      use.queueClasses.push_back(*tagValues[segment]);
      return llvm::Error::success();
    };

    if ((*source)->localTraversal)
      if (llvm::Error error =
              appendTraversal(*(*source)->localTraversal, *sourceSlot))
        return std::move(error);

    const auto endpoint = tree.sinkEndpoint(sink);
    if (!endpoint)
      return invalid("routed capacity proof channel has no sink endpoint");
    auto slot = tree.findNode(*endpoint);
    if (!slot)
      return invalid("routed capacity proof channel has no sink node");
    const PnrIndex sinkSlot = *slot;
    std::size_t visited = 0;
    while (*slot != getInvalidPnrIndex()) {
      if (++visited > tree.activeNodeCount())
        return invalid("capacity proof channel ancestry is cyclic");
      const RouteTreeNode &node = tree.node(*slot);
      if (node.parentArc == getInvalidPnrIndex())
        break;
      if (node.parentArc >= problem.routing().routingArcs().size())
        return invalid("capacity proof channel parent arc is out of range");
      if (llvm::Error error = appendTraversal(
              problem.routing().routingArcs()[node.parentArc].traversal,
              *slot))
        return std::move(error);
      auto parent = parentSlot(tree, *slot);
      if (!parent)
        return parent.takeError();
      *slot = *parent;
    }

    auto sinkAttachment =
        selectedAttachment(candidate, sinks[net.sinkOffset + sink]);
    if (!sinkAttachment)
      return sinkAttachment.takeError();
    if ((*sinkAttachment)->localTraversal)
      if (llvm::Error error =
              appendTraversal(*(*sinkAttachment)->localTraversal,
                              sinkSlot))
        return std::move(error);

    auto dependencies =
        spatialSinkProgressDependencies(problem, logicalNet, sink);
    if (!dependencies)
      return dependencies.takeError();
    const bool initializedFeedback = llvm::any_of(
        *dependencies, [](const FrozenSpatialProgressPrerequisite &value) {
          return std::holds_alternative<
              FrozenSpatialInitializedFeedbackPrerequisite>(value);
        });
    for (auto &[owner, channel] : channelUses) {
      MutableUse &use = uses[owner];
      if (use.channels == std::numeric_limits<std::uint64_t>::max())
        return invalid("capacity proof owner channel count exceeds u64");
      ++use.channels;
      if (initializedFeedback) {
        if (use.initializedFeedbackChannels ==
            std::numeric_limits<std::uint64_t>::max())
          return invalid("capacity proof feedback channel count exceeds u64");
        ++use.initializedFeedbackChannels;
      }
      use.repeated |= channel.traversals.size() > 1;
      use.queueClassIndeterminate |= channel.queueClassIndeterminate;
      use.queueClasses.insert(use.queueClasses.end(),
                              std::make_move_iterator(
                                  channel.queueClasses.begin()),
                              std::make_move_iterator(channel.queueClasses.end()));
    }
  }

  SpatialProgressNetCapacityProjection result;
  result.owners.reserve(uses.size());
  for (auto &[owner, use] : uses) {
    llvm::sort(use.queueClasses, [](const llvm::APInt &lhs,
                                    const llvm::APInt &rhs) {
      return ::fabric::comparePhysicalTagValues(lhs, rhs) < 0;
    });
    use.queueClasses.erase(
        std::unique(use.queueClasses.begin(), use.queueClasses.end(),
                    [](const llvm::APInt &lhs, const llvm::APInt &rhs) {
                      return ::fabric::comparePhysicalTagValues(lhs, rhs) == 0;
                    }),
        use.queueClasses.end());
    result.owners.push_back({owner, use.channels,
                             use.initializedFeedbackChannels, use.repeated,
                             use.queueClassIndeterminate,
                             std::move(use.queueClasses)});
  }
  return result;
}
