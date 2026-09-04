#include "Mapping/Artifact/SystemMappingClosureProjection.h"
#include "MappingProgressInternal.h"

#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace loom::mapping {
using namespace progress_detail;
namespace {

::dataflow::RootThreadLaunchRef
rootOf(const ::dataflow::RootThreadBoundaryTransferRef &transfer) {
  return std::visit([](const auto &typed) { return typed.launch; }, transfer);
}

::dataflow::RootedGraphLaunchRef
graphOf(const ::dataflow::GraphLaunchBoundaryTransferRef &transfer) {
  return std::visit([](const auto &typed) { return typed.launch; }, transfer);
}

struct EventLocation final {
  ::dataflow::RootThreadLaunchRef root;
  std::optional<::dataflow::RootedGraphLaunchRef> graph;
  bool rootCompletion = false;
};

EventLocation
producerLocation(const ::dataflow::CanonicalProducerTerminalRef &terminal) {
  return std::visit(
      [](const auto &typed) -> EventLocation {
        using Terminal = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Terminal,
                                     ::dataflow::RootThreadBoundarySourceRef>) {
          const auto root = rootOf(typed.transfer);
          const bool completion = std::holds_alternative<
              ::dataflow::RootThreadCompletionTransferRef>(typed.transfer);
          return {root, std::nullopt, completion};
        } else if constexpr (std::is_same_v<
                                 Terminal,
                                 ::dataflow::GraphLaunchBoundarySourceRef>) {
          const auto graph = graphOf(typed.transfer);
          return {graph.rootThreadLaunch, graph, false};
        } else {
          return std::visit(
              [](const auto &producer) -> EventLocation {
                using Producer = std::decay_t<decltype(producer)>;
                if constexpr (std::is_same_v<
                                  Producer,
                                  ::dataflow::GraphStreamOutputProducerRef>)
                  return {producer.launch.rootThreadLaunch, producer.launch,
                          false};
                else
                  return {producer.launch, std::nullopt, false};
              },
              typed.producer);
        }
      },
      terminal);
}

EventLocation
consumerLocation(const ::dataflow::CanonicalSinkTerminalRef &terminal) {
  return std::visit(
      [](const auto &typed) -> EventLocation {
        using Terminal = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Terminal,
                                     ::dataflow::RootThreadBoundarySinkRef>) {
          return {rootOf(typed.transfer), std::nullopt, false};
        } else if constexpr (std::is_same_v<
                                 Terminal,
                                 ::dataflow::GraphLaunchBoundarySinkRef>) {
          const auto graph = graphOf(typed.transfer);
          return {graph.rootThreadLaunch, graph, false};
        } else {
          return std::visit(
              [](const auto &consumer) -> EventLocation {
                using Consumer = std::decay_t<decltype(consumer)>;
                if constexpr (std::is_same_v<
                                  Consumer,
                                  ::dataflow::GraphStreamInputConsumerRef>)
                  return {consumer.launch.rootThreadLaunch, consumer.launch,
                          false};
                else
                  return {consumer.launch, std::nullopt, false};
              },
              typed.consumer);
        }
      },
      terminal);
}

EventLocation eventLocation(const ::dataflow::EventFamilyKey &event) {
  return std::visit(
      [](const auto &typed) -> EventLocation {
        using Event = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Event,
                                     ::dataflow::StaticTransferEventRef>) {
          return std::visit(
              [](const auto &transfer) -> EventLocation {
                using Transfer = std::decay_t<decltype(transfer)>;
                if constexpr (std::is_same_v<
                                  Transfer,
                                  ::dataflow::ProducedTransferEventRef>)
                  return producerLocation(transfer.terminal);
                else
                  return consumerLocation(transfer.terminal);
              },
              typed);
        } else {
          return {typed.actor.launch.rootThreadLaunch, typed.actor.launch,
                  false};
        }
      },
      event);
}

class SystemEventDependencyGraph final {
public:
  explicit SystemEventDependencyGraph(
      const ::dataflow::CanonicalDataflowProgramView &dataflow)
      : dataflow_(dataflow) {}

  llvm::Expected<std::uint32_t>
  intern(const ::dataflow::EventFamilyKey &event) {
    if (llvm::Error error = dataflow_.validate(event))
      return std::move(error);
    auto key = eventKey(dataflow_.identity(), event);
    if (!key)
      return key.takeError();
    const auto found = ordinals_.find(*key);
    if (found != ordinals_.end())
      return found->second;
    if (events_.size() >= std::numeric_limits<std::uint32_t>::max())
      return invalid("System event dependency inventory exceeds u32");
    const std::uint32_t ordinal = static_cast<std::uint32_t>(events_.size());
    ordinals_.emplace(std::move(*key), ordinal);
    events_.push_back(event);
    edges_.emplace_back();
    reverseEdges_.emplace_back();
    return ordinal;
  }

  llvm::Error addEdge(const ::dataflow::EventFamilyKey &source,
                      const ::dataflow::EventFamilyKey &sink) {
    auto sourceOrdinal = intern(source);
    if (!sourceOrdinal)
      return sourceOrdinal.takeError();
    auto sinkOrdinal = intern(sink);
    if (!sinkOrdinal)
      return sinkOrdinal.takeError();
    if (*sourceOrdinal == *sinkOrdinal)
      return llvm::Error::success();
    edges_[*sourceOrdinal].push_back(*sinkOrdinal);
    reverseEdges_[*sinkOrdinal].push_back(*sourceOrdinal);
    return llvm::Error::success();
  }

  llvm::Error closeRootCompletionDependencies() {
    const std::size_t originalCount = events_.size();
    for (std::size_t completion = 0; completion != originalCount;
         ++completion) {
      const EventLocation location = eventLocation(events_[completion]);
      if (!location.rootCompletion)
        continue;
      for (std::size_t source = 0; source != originalCount; ++source) {
        if (source == completion ||
            eventLocation(events_[source]).root != location.root)
          continue;
        edges_[source].push_back(static_cast<std::uint32_t>(completion));
        reverseEdges_[completion].push_back(static_cast<std::uint32_t>(source));
      }
    }
    canonicalize();
    return llvm::Error::success();
  }

  std::vector<bool> ancestors(std::uint32_t event) const {
    std::vector<bool> result(events_.size(), false);
    if (event >= events_.size())
      return result;
    std::vector<std::uint32_t> worklist{event};
    result[event] = true;
    for (std::size_t cursor = 0; cursor != worklist.size(); ++cursor)
      for (std::uint32_t predecessor : reverseEdges_[worklist[cursor]])
        if (!result[predecessor]) {
          result[predecessor] = true;
          worklist.push_back(predecessor);
        }
    return result;
  }

  std::map<std::string, std::uint32_t> takeOrdinals() {
    return std::move(ordinals_);
  }

  std::vector<std::vector<std::uint32_t>> takeReverseEdges() {
    return std::move(reverseEdges_);
  }

private:
  void canonicalize() {
    for (auto &successors : edges_) {
      llvm::sort(successors);
      successors.erase(std::unique(successors.begin(), successors.end()),
                       successors.end());
    }
    for (auto &predecessors : reverseEdges_) {
      llvm::sort(predecessors);
      predecessors.erase(std::unique(predecessors.begin(), predecessors.end()),
                         predecessors.end());
    }
  }

  const ::dataflow::CanonicalDataflowProgramView &dataflow_;
  std::map<std::string, std::uint32_t> ordinals_;
  std::vector<::dataflow::EventFamilyKey> events_;
  std::vector<std::vector<std::uint32_t>> edges_;
  std::vector<std::vector<std::uint32_t>> reverseEdges_;
};

llvm::Error buildSystemEventDependencies(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    SystemEventDependencyGraph &graph) {
  std::map<mlir::Operation *, ::dataflow::RootThreadLaunchRef> rootByLaunch;
  for (const auto &root : dataflow.rootThreadLaunches())
    rootByLaunch.emplace(root.op, root.ref);
  for (const auto &root : dataflow.rootThreadLaunches()) {
    auto launch = mlir::dyn_cast<::dataflow::ThreadLaunchOp>(root.op);
    if (!launch)
      return invalid("root thread inventory contains a non-launch operation");
    for (mlir::Value dependency : launch.getAsyncDependencies()) {
      auto producer = dependency.getDefiningOp<::dataflow::ThreadLaunchOp>();
      if (!producer)
        continue;
      const auto producerRoot = rootByLaunch.find(producer.getOperation());
      if (producerRoot == rootByLaunch.end())
        continue;
      if (llvm::Error error = graph.addEdge(
              ::dataflow::rootThreadCompletionEventFamily(producerRoot->second),
              ::dataflow::rootThreadStartEventFamily(root.ref)))
        return error;
    }
  }

  std::set<std::pair<std::uint64_t, std::uint32_t>> initializedFeedbackInputs;
  for (const ::dataflow::CanonicalActorView &actor : dataflow.actors()) {
    auto inputs = initializedFeedbackInputOrdinals(actor);
    if (!inputs)
      return inputs.takeError();
    for (std::uint32_t input : *inputs)
      initializedFeedbackInputs.emplace(actor.ref.entity.value(), input);
  }
  std::vector<std::pair<::dataflow::CanonicalGraphProducerEndpointRef,
                        ::dataflow::CanonicalGraphConsumerEndpointRef>>
      graphEdges;
  if (llvm::Error error = dataflow.forEachGraphEdge(
          [&](const auto &producer, const auto &consumer) -> llvm::Error {
            graphEdges.emplace_back(producer, consumer);
            return llvm::Error::success();
          }))
    return error;

  std::vector<::dataflow::RootedGraphLaunchRef> launches;
  dataflow.forEachRootedGraphLaunch(
      [&](::dataflow::RootedGraphLaunchRef launch) {
        launches.push_back(launch);
      });
  for (const ::dataflow::RootedGraphLaunchRef &launch : launches) {
    auto launchedGraph = dataflow.resolve(launch);
    if (!launchedGraph)
      return launchedGraph.takeError();
    for (const auto &[producer, consumer] : graphEdges) {
      auto owner = dataflow.graphOf(producer);
      if (!owner)
        return owner.takeError();
      if (*owner != *launchedGraph)
        continue;
      // Initialized feedback carries a prior-iteration token. Its durable
      // disposition is verified separately; it is not same-coordinate event
      // precedence.
      if (const auto *operand =
              std::get_if<::dataflow::ActorTokenOperandRef>(&consumer);
          operand &&
          initializedFeedbackInputs.count(
              {operand->actor.entity.value(), operand->ordinal}) != 0)
        continue;
      auto produced =
          dataflow.projectRootedGraphEndpointEventFamilies(launch, producer);
      if (!produced)
        return produced.takeError();
      auto consumed =
          dataflow.projectRootedGraphEndpointEventFamilies(launch, consumer);
      if (!consumed)
        return consumed.takeError();
      for (const auto &source : *produced)
        for (const auto &sink : *consumed)
          if (llvm::Error error = graph.addEdge(source, sink))
            return error;
    }
  }

  for (const auto &root : dataflow.rootThreadLaunches()) {
    if (llvm::Error error = dataflow.forEachProducerTerminal(
            root.ref, [&](const auto &producer) -> llvm::Error {
              std::vector<::dataflow::CanonicalSinkTerminalRef> sinks;
              if (llvm::Error pairError = dataflow.pairedSinks(
                      producer.terminal,
                      [&](const auto &sink) { sinks.push_back(sink); }))
                return pairError;
              const ::dataflow::EventFamilyKey produced(
                  ::dataflow::StaticTransferEventRef(
                      ::dataflow::ProducedTransferEventRef{producer.terminal}));
              for (const auto &sink : sinks) {
                const ::dataflow::EventFamilyKey consumed(
                    ::dataflow::StaticTransferEventRef(
                        ::dataflow::ConsumedTransferEventRef{sink}));
                if (llvm::Error edgeError = graph.addEdge(produced, consumed))
                  return edgeError;
              }
              return llvm::Error::success();
            }))
      return error;
  }
  return graph.closeRootCompletionDependencies();
}

llvm::Expected<bool>
systemDependentBranchHasDurableBoundary(const SystemTransferLegView &route,
                                        std::uint64_t prerequisiteNode,
                                        std::uint64_t dependentNode) {
  std::map<std::uint64_t, const SystemTransferRouteNodeView *> nodes;
  for (const SystemTransferRouteNodeView &node : route.nodes) {
    if (node.ordinal == 0 || !nodes.emplace(node.ordinal, &node).second)
      return invalid("System route has an invalid node identity");
  }
  const auto parent = [&](std::uint64_t node) -> llvm::Expected<std::uint64_t> {
    if (node == 0)
      return 0;
    const auto found = nodes.find(node);
    if (found == nodes.end())
      return invalid("System route progress dependency names an absent node");
    return found->second->parentOrdinal;
  };

  std::set<std::uint64_t> prerequisiteAncestors;
  std::uint64_t node = prerequisiteNode;
  for (std::size_t visited = 0;; ++visited) {
    if (visited > nodes.size() || !prerequisiteAncestors.insert(node).second)
      return invalid("System route prerequisite ancestry is cyclic");
    if (node == 0)
      break;
    auto next = parent(node);
    if (!next)
      return next.takeError();
    node = *next;
  }

  node = dependentNode;
  for (std::size_t visited = 0; prerequisiteAncestors.count(node) == 0;
       ++visited) {
    if (visited > nodes.size() || node == 0)
      return invalid("System route branches have no common ancestor");
    const auto found = nodes.find(node);
    if (found == nodes.end())
      return invalid("System route dependent branch names an absent node");
    if (isBuffered(std::optional{found->second->incomingTraversal}))
      return true;
    node = found->second->parentOrdinal;
  }
  return false;
}

} // namespace

llvm::Expected<FrozenMappingProgressModel> freezeMappingProgressModel(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::EventFamilyKey> activationEvents) {
  SystemEventDependencyGraph graph(dataflow);
  for (const auto &event : activationEvents) {
    auto ordinal = graph.intern(event);
    if (!ordinal)
      return ordinal.takeError();
  }
  if (llvm::Error error = buildSystemEventDependencies(dataflow, graph))
    return std::move(error);
  return FrozenMappingProgressModel(dataflow.identity(), graph.takeOrdinals(),
                                    graph.takeReverseEdges());
}

llvm::Expected<bool>
mappingEventPrecedes(const FrozenMappingProgressModel &model,
                     const ::dataflow::EventFamilyKey &predecessor,
                     const ::dataflow::EventFamilyKey &dependent) {
  auto predecessorKey = eventKey(model.dataflowIdentity_, predecessor);
  if (!predecessorKey)
    return predecessorKey.takeError();
  auto dependentKey = eventKey(model.dataflowIdentity_, dependent);
  if (!dependentKey)
    return dependentKey.takeError();
  const auto predecessorOrdinal = model.eventOrdinals_.find(*predecessorKey);
  const auto dependentOrdinal = model.eventOrdinals_.find(*dependentKey);
  if (predecessorOrdinal == model.eventOrdinals_.end() ||
      dependentOrdinal == model.eventOrdinals_.end())
    return invalid("resource-time event is absent from the frozen Dataflow "
                   "causality index");
  if (predecessorOrdinal->second == dependentOrdinal->second)
    return true;
  std::vector<bool> visited(model.reverseEdges_.size(), false);
  std::vector<std::uint32_t> worklist{dependentOrdinal->second};
  visited[dependentOrdinal->second] = true;
  for (std::size_t cursor = 0; cursor != worklist.size(); ++cursor)
    for (std::uint32_t parent : model.reverseEdges_[worklist[cursor]]) {
      if (parent == predecessorOrdinal->second)
        return true;
      if (!visited[parent]) {
        visited[parent] = true;
        worklist.push_back(parent);
      }
    }
  return false;
}

llvm::Expected<bool> mappingCompletionFrontierIsAdmissible(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> mappedRoots,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> completedBefore,
    ::dataflow::RootThreadLaunchRef completing,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> activeAfter) {
  if (mappedRoots.empty() || !llvm::is_contained(mappedRoots, completing))
    return false;
  const auto hasDuplicate = [](auto values) {
    for (std::size_t index = 0; index != values.size(); ++index)
      if (llvm::is_contained(values.take_front(index), values[index]))
        return true;
    return false;
  };
  if (hasDuplicate(mappedRoots) || hasDuplicate(completedBefore) ||
      hasDuplicate(activeAfter))
    return false;
  for (const auto root : completedBefore)
    if (!llvm::is_contained(mappedRoots, root) || root == completing ||
        llvm::is_contained(activeAfter, root))
      return false;
  for (const auto root : activeAfter)
    if (!llvm::is_contained(mappedRoots, root) || root == completing)
      return false;

  mlir::ModuleOp module = dataflow.rootThreadLaunches()
                              .front()
                              .op->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return invalid("root thread launch has no canonical module owner");
  bool hasStoredProgramWait = false;
  module.walk([&](::dataflow::ThreadWaitOp) { hasStoredProgramWait = true; });
  if (hasStoredProgramWait)
    return false;

  std::vector<::dataflow::RootThreadLaunchRef> canonicalRoots;
  std::map<mlir::Operation *, ::dataflow::RootThreadLaunchRef> rootByLaunch;
  canonicalRoots.reserve(dataflow.rootThreadLaunches().size());
  std::vector<::dataflow::EventFamilyKey> boundaryEvents;
  boundaryEvents.reserve(dataflow.rootThreadLaunches().size() * 2);
  for (const auto &root : dataflow.rootThreadLaunches()) {
    canonicalRoots.push_back(root.ref);
    rootByLaunch.emplace(root.op, root.ref);
    boundaryEvents.push_back(::dataflow::rootThreadStartEventFamily(root.ref));
    boundaryEvents.push_back(
        ::dataflow::rootThreadCompletionEventFamily(root.ref));
  }
  for (const auto root : mappedRoots) {
    if (!llvm::is_contained(canonicalRoots, root))
      return false;
  }
  auto model = freezeMappingProgressModel(dataflow, boundaryEvents);
  if (!model)
    return model.takeError();

  const ::dataflow::EventFamilyKey trigger =
      ::dataflow::rootThreadCompletionEventFamily(completing);
  const auto causallyReady =
      [&](::dataflow::RootThreadLaunchRef root,
          llvm::ArrayRef<::dataflow::RootThreadLaunchRef> satisfied)
      -> llvm::Expected<bool> {
    const ::dataflow::EventFamilyKey start =
        ::dataflow::rootThreadStartEventFamily(root);
    for (const auto predecessor : canonicalRoots) {
      if (predecessor == root)
        continue;
      auto completionPrecedes = mappingEventPrecedes(
          *model, ::dataflow::rootThreadCompletionEventFamily(predecessor),
          start);
      if (!completionPrecedes)
        return completionPrecedes.takeError();
      auto startPrecedes = mappingEventPrecedes(
          *model, ::dataflow::rootThreadStartEventFamily(predecessor), start);
      if (!startPrecedes)
        return startPrecedes.takeError();
      if ((*completionPrecedes || *startPrecedes) &&
          !llvm::is_contained(satisfied, predecessor))
        return false;
    }
    return true;
  };

  for (const auto completed : completedBefore) {
    auto ready = causallyReady(completed, completedBefore);
    if (!ready)
      return ready.takeError();
    if (!*ready)
      return false;
    auto orderedAfterTrigger = mappingEventPrecedes(
        *model, trigger,
        ::dataflow::rootThreadCompletionEventFamily(completed));
    if (!orderedAfterTrigger)
      return orderedAfterTrigger.takeError();
    if (*orderedAfterTrigger)
      return false;
  }

  auto completingReady = causallyReady(completing, completedBefore);
  if (!completingReady)
    return completingReady.takeError();
  if (!*completingReady)
    return false;
  auto completingStarted = mappingEventPrecedes(
      *model, ::dataflow::rootThreadStartEventFamily(completing), trigger);
  if (!completingStarted)
    return completingStarted.takeError();
  if (!*completingStarted)
    return false;

  std::vector<::dataflow::RootThreadLaunchRef> completedAfter(
      completedBefore.begin(), completedBefore.end());
  completedAfter.push_back(completing);
  for (const auto active : activeAfter) {
    const auto activeRoot = llvm::find_if(
        dataflow.rootThreadLaunches(),
        [&](const auto &candidate) { return candidate.ref == active; });
    if (activeRoot == dataflow.rootThreadLaunches().end())
      return false;
    auto launch = mlir::dyn_cast<::dataflow::ThreadLaunchOp>(activeRoot->op);
    if (!launch)
      return false;
    for (mlir::Value dependency : launch.getAsyncDependencies()) {
      auto producer = dependency.getDefiningOp<::dataflow::ThreadLaunchOp>();
      if (!producer || !rootByLaunch.count(producer.getOperation()))
        return false;
    }
    auto alreadyStarted = mappingEventPrecedes(
        *model, ::dataflow::rootThreadStartEventFamily(active), trigger);
    if (!alreadyStarted)
      return alreadyStarted.takeError();
    if (*alreadyStarted)
      return false;
    auto ready = causallyReady(active, completedAfter);
    if (!ready)
      return ready.takeError();
    if (!*ready)
      return false;
  }
  return true;
}

llvm::Expected<MappingProgressProjection> projectSystemMappingProgress(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemMappingClosureProjection &closure) {
  MappingProgressProjection result;
  result.basis = closure.progressBasis;
  result.routeObligations = closure.routeObligations;
  result.capacityCells.reserve(closure.capacityCells.size());
  for (const SystemCapacityCellProjection &cell : closure.capacityCells)
    result.capacityCells.push_back({cell.capacity, cell.baselineOccupancy});
  result.resourceActivations.reserve(closure.resourceActivations.size());
  for (const SystemResourceActivationProjection &activation :
       closure.resourceActivations) {
    auto physical = fabric.resolvePhysicalOwner(activation.physicalOwner);
    if (!physical)
      return physical.takeError();
    const ::fabric::ResourceContract *contract =
        physical->artifact.resourceContract(physical->localOwner);
    if (!contract ||
        activation.usePatternOrdinal >= contract->usePatternCount())
      return invalid("System activation UsePattern is absent from Fabric");
    const ::fabric::UsePattern pattern = contract->usePattern(
        ::fabric::UsePatternKey(activation.usePatternOrdinal));
    MappingResourceGrantPolicyKind policy =
        MappingResourceGrantPolicyKind::None;
    if (const auto grant = contract->grantPolicy())
      policy = std::holds_alternative<::fabric::FixedPriorityView>(*grant)
                   ? MappingResourceGrantPolicyKind::FixedPriority
                   : MappingResourceGrantPolicyKind::RoundRobin;
    const auto ownerBytes =
        ::loom::fabric::canonicalFabricBytes(activation.physicalOwner);
    const std::string ownerKey(
        reinterpret_cast<const char *>(ownerBytes.data()), ownerBytes.size());
    if (activation.triggerAlternatives.empty())
      return invalid("System activation has no relation-root event");
    auto relationRoot =
        dataflow.eventRootThreadLaunch(activation.triggerAlternatives.front());
    if (!relationRoot)
      return relationRoot.takeError();
    for (const auto &trigger : activation.triggerAlternatives) {
      auto root = dataflow.eventRootThreadLaunch(trigger);
      if (!root)
        return root.takeError();
      if (*root != *relationRoot)
        return invalid("System activation crosses relation-root spaces");
    }
    MappingProgressActivationProjection projected{
        activation.context,
        *relationRoot,
        activation.relationDomain,
        activation.triggerAlternatives,
        {},
        {},
        MappingResourceProgressUse{ownerKey, pattern.requester.ordinal(),
                                   policy}};
    projected.capacityClaims.reserve(activation.capacityClaims.size());
    for (const SystemCapacityClaimProjection &claim : activation.capacityClaims)
      projected.capacityClaims.push_back(
          {claim.capacityCellOrdinal, claim.amount});
    projected.causalRelease.reserve(activation.causalRelease.size());
    for (const SystemCausalReleasePointProjection &release :
         activation.causalRelease)
      projected.causalRelease.push_back({release.alternatives});
    result.resourceActivations.push_back(std::move(projected));
  }
  return result;
}

llvm::Expected<MappingProgressClosure> deriveSystemMappingProgressClosure(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemMappingClosureProjection &closure) {
  auto projection = projectSystemMappingProgress(dataflow, fabric, closure);
  if (!projection)
    return projection.takeError();
  std::vector<::dataflow::EventFamilyKey> events;
  for (const MappingProgressActivationProjection &activation :
       projection->resourceActivations) {
    events.insert(events.end(), activation.triggerAlternatives.begin(),
                  activation.triggerAlternatives.end());
    for (const MappingProgressCausalReleaseProjection &release :
         activation.causalRelease)
      events.insert(events.end(), release.alternatives.begin(),
                    release.alternatives.end());
  }
  auto model = freezeMappingProgressModel(dataflow, events);
  if (!model)
    return model.takeError();
  return deriveMappingProgressClosure(*model, *projection);
}

llvm::Expected<MappingProgressClosure> qualifySystemMappingResourceTimeProgress(
    const FinalizedSystemMapping &mapping,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric) {
  // Resource-time qualification consumes the same strict closure projection
  // as System Mapping verification. Initialized feedback is not rejected by
  // category; the exact capacity/arbitration/causal-release kernel decides
  // whether its finite recurrence is closed.
  return deriveSystemMappingProgressClosure(dataflow, fabric,
                                            mapping.verifiedClosure());
}

llvm::Expected<std::vector<SystemTransferRouteProgressDependency>>
deriveSystemTransferRouteProgressDependencies(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<CanonicalServiceLegKey> transferLegs) {
  SystemEventDependencyGraph graph(dataflow);
  struct LegEvents final {
    CanonicalServiceLegKey leg;
    std::vector<std::uint32_t> sinkEvents;
  };
  std::vector<LegEvents> projected;
  projected.reserve(transferLegs.size());
  std::set<std::string> seenLegs;
  for (const CanonicalServiceLegKey &leg : transferLegs) {
    auto legBytes = encodeCanonicalServiceLegKey(dataflow.identity(), leg);
    if (!legBytes)
      return legBytes.takeError();
    if (!seenLegs
             .emplace(reinterpret_cast<const char *>(legBytes->data()),
                      legBytes->size())
             .second)
      continue;
    const auto *producer =
        std::get_if<TransferObligationFamilyKey>(&leg.obligation);
    if (!producer)
      continue;
    std::vector<::dataflow::CanonicalSinkTerminalRef> sinks;
    if (llvm::Error error = dataflow.pairedSinks(
            *producer, [&](const auto &sink) { sinks.push_back(sink); }))
      return std::move(error);
    LegEvents record{leg, {}};
    record.sinkEvents.reserve(sinks.size());
    for (const auto &sink : sinks) {
      const ::dataflow::EventFamilyKey event(::dataflow::StaticTransferEventRef(
          ::dataflow::ConsumedTransferEventRef{sink}));
      auto ordinal = graph.intern(event);
      if (!ordinal)
        return ordinal.takeError();
      record.sinkEvents.push_back(*ordinal);
    }
    projected.push_back(std::move(record));
  }
  if (llvm::Error error = buildSystemEventDependencies(dataflow, graph))
    return std::move(error);

  std::vector<SystemTransferRouteProgressDependency> result;
  for (const LegEvents &record : projected)
    for (std::uint64_t dependentSink = 0;
         dependentSink < record.sinkEvents.size(); ++dependentSink) {
      const std::vector<bool> ancestors =
          graph.ancestors(record.sinkEvents[dependentSink]);
      for (std::uint64_t prerequisiteSink = 0;
           prerequisiteSink < record.sinkEvents.size(); ++prerequisiteSink) {
        if (prerequisiteSink == dependentSink ||
            record.sinkEvents[prerequisiteSink] ==
                record.sinkEvents[dependentSink] ||
            record.sinkEvents[prerequisiteSink] >= ancestors.size() ||
            !ancestors[record.sinkEvents[prerequisiteSink]])
          continue;
        result.push_back({record.leg, prerequisiteSink, dependentSink});
      }
    }
  return result;
}

llvm::Expected<std::vector<MappingRouteProgressObligationProjection>>
projectSystemTransferRouteProgress(
    llvm::ArrayRef<SystemTransferLegView> transferLegs,
    llvm::ArrayRef<SystemTransferRouteProgressDependency> dependencies) {
  std::vector<MappingRouteProgressObligationProjection> result;
  for (const SystemTransferLegView &route : transferLegs) {
    std::map<std::uint64_t, std::vector<std::uint64_t>> routeNodesBySink;
    for (const SystemTransferRouteSinkView &sink : route.sinks) {
      const auto *terminal =
          std::get_if<SystemTransferSinkTerminalKey>(&sink.terminal);
      if (!terminal)
        return invalid("System route progress sink uses a source terminal");
      if (terminal->leg != route.leg)
        return invalid("System route progress sink belongs to another leg");
      routeNodesBySink[terminal->sinkOrdinal].push_back(sink.nodeOrdinal);
    }
    for (auto &[sinkOrdinal, nodes] : routeNodesBySink) {
      (void)sinkOrdinal;
      llvm::sort(nodes);
      nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    }
    for (const SystemTransferRouteProgressDependency &dependency :
         dependencies) {
      if (dependency.leg != route.leg)
        continue;
      const auto prerequisite =
          routeNodesBySink.find(dependency.prerequisiteSinkOrdinal);
      const auto dependent =
          routeNodesBySink.find(dependency.dependentSinkOrdinal);
      if (prerequisite == routeNodesBySink.end() ||
          dependent == routeNodesBySink.end())
        continue;
      for (std::uint64_t prerequisiteNode : prerequisite->second)
        for (std::uint64_t dependentNode : dependent->second) {
          auto durable = systemDependentBranchHasDurableBoundary(
              route, prerequisiteNode, dependentNode);
          if (!durable)
            return durable.takeError();
          result.push_back({MappingRouteProgressObligationKind::
                                DurableBoundaryAfterDivergence,
                            *durable});
        }
    }
  }
  return result;
}

llvm::Expected<std::vector<MappingRouteProgressObligationProjection>>
projectSystemTransferRouteProgress(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<SystemTransferLegView> transferLegs) {
  std::vector<CanonicalServiceLegKey> legs;
  legs.reserve(transferLegs.size());
  for (const SystemTransferLegView &route : transferLegs)
    legs.push_back(route.leg);
  auto dependencies =
      deriveSystemTransferRouteProgressDependencies(dataflow, legs);
  if (!dependencies)
    return dependencies.takeError();
  return projectSystemTransferRouteProgress(transferLegs, *dependencies);
}

} // namespace loom::mapping
