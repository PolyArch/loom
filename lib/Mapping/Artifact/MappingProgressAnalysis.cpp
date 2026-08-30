#include "Mapping/Artifact/MappingProgressAnalysis.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"
#include "Mapping/Artifact/SystemMappingClosureProjection.h"

#include "ConfiguredHardwareProjectionInternal.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/FifoResourceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/DenseMap.h"
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
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_progress_analysis_invalid: " + message);
}

void appendU32(std::string &bytes, std::uint32_t value) {
  for (unsigned shift = 24;; shift -= 8) {
    bytes.push_back(static_cast<char>(value >> shift));
    if (shift == 0)
      break;
  }
}

void appendU64(std::string &bytes, std::uint64_t value) {
  for (unsigned shift = 56;; shift -= 8) {
    bytes.push_back(static_cast<char>(value >> shift));
    if (shift == 0)
      break;
  }
}

void appendI64(std::string &bytes, std::int64_t value) {
  appendU64(bytes, static_cast<std::uint64_t>(value));
}

void appendSized(std::string &bytes, llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.append(reinterpret_cast<const char *>(value.data()), value.size());
}

/// Canonical byte key of one static wait-for node, used for interning and for
/// the deterministic witness order. The key is a pure function of the typed
/// identity; it is never persisted.
std::string staticWaitNodeKey(const MappingStaticWaitNode &node) {
  std::string key;
  key.push_back(static_cast<char>(node.index()));
  if (const auto *storage =
          std::get_if<MappingStorageQueueProgressNode>(&node)) {
    const std::vector<std::uint8_t> owner =
        ::loom::fabric::canonicalFabricBytes(storage->owner);
    key.append(reinterpret_cast<const char *>(owner.data()), owner.size());
    key.push_back(
        static_cast<char>(storage->queueClass.kind ==
                                  MappingStaticQueueClassKind::PhysicalTag
                              ? 1
                              : 0));
    appendU32(key, storage->queueClass.tagValue.getBitWidth());
    for (unsigned word = 0; word != storage->queueClass.tagValue.getNumWords();
         ++word)
      appendU64(key, storage->queueClass.tagValue.getRawData()[word]);
  } else if (const auto *actor = std::get_if<::dataflow::ActorRef>(&node)) {
    for (std::uint8_t byte : actor->artifact.bytes())
      key.push_back(static_cast<char>(byte));
    appendU64(key, actor->entity.value());
  } else {
    const auto &operand = std::get<MappingOperandQueueProgressNode>(node);
    const std::vector<std::uint8_t> context =
        ::loom::fabric::canonicalFabricBytes(operand.queue.context);
    key.append(reinterpret_cast<const char *>(context.data()), context.size());
    appendU64(key, operand.queue.fuOccurrence);
    appendU64(key, operand.queue.fuInput);
    const std::vector<std::uint8_t> fu =
        ::loom::fabric::canonicalFabricBytes(operand.fu);
    key.append(reinterpret_cast<const char *>(fu.data()), fu.size());
  }
  return key;
}

llvm::Expected<std::string> eventKey(const ArtifactIdentity &dataflowIdentity,
                                     const ::dataflow::EventFamilyKey &event) {
  auto encoded = ::dataflow::encodeDataflowReference(dataflowIdentity, event);
  if (!encoded)
    return encoded.takeError();
  return std::string(reinterpret_cast<const char *>(encoded->data()),
                     encoded->size());
}

llvm::Expected<std::string>
eventKey(const ::dataflow::CanonicalDataflowProgramView &dataflow,
         const ::dataflow::EventFamilyKey &event) {
  return eventKey(dataflow.identity(), event);
}

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
    auto key = eventKey(dataflow_, event);
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

void appendCellKey(std::string &bytes, const SystemPresburgerCell &cell) {
  appendU32(bytes, cell.dimensionCount);
  appendU32(bytes, cell.symbolCount);
  appendU32(bytes, cell.localCount);
  const auto appendRows = [&](const auto &rows) {
    appendU64(bytes, rows.size());
    for (const auto &row : rows) {
      appendU64(bytes, row.size());
      for (std::int64_t value : row)
        appendI64(bytes, value);
    }
  };
  appendRows(cell.equalities);
  appendRows(cell.inequalities);
}

llvm::Expected<std::string>
atomicActivationKey(const ArtifactIdentity &dataflowIdentity,
                    const MappingProgressActivationProjection &activation) {
  auto context = encodeExecutionContextKey(activation.context);
  if (!context)
    return context.takeError();
  if (activation.relationRoot.artifact != dataflowIdentity)
    return invalid("resource activation has a foreign relation root");
  std::string result;
  appendSized(result, *context);
  appendU64(result, activation.relationRoot.entity.value());
  std::vector<std::string> cells;
  cells.reserve(activation.relationDomain.size());
  for (const SystemPresburgerCell &cell : activation.relationDomain) {
    std::string key;
    appendCellKey(key, cell);
    cells.push_back(std::move(key));
  }
  llvm::sort(cells);
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  appendU64(result, cells.size());
  for (const std::string &cell : cells) {
    appendU64(result, cell.size());
    result.append(cell);
  }
  std::vector<std::string> triggers;
  triggers.reserve(activation.triggerAlternatives.size());
  for (const auto &event : activation.triggerAlternatives) {
    auto key = eventKey(dataflowIdentity, event);
    if (!key)
      return key.takeError();
    triggers.push_back(std::move(*key));
  }
  llvm::sort(triggers);
  triggers.erase(std::unique(triggers.begin(), triggers.end()), triggers.end());
  appendU64(result, triggers.size());
  for (const std::string &trigger : triggers) {
    appendU64(result, trigger.size());
    result.append(trigger);
  }
  return result;
}

struct ProgressActivationGroup final {
  std::vector<std::uint64_t> activationOrdinals;
  std::optional<::dataflow::RootThreadLaunchRef> relationRoot;
  llvm::ArrayRef<SystemPresburgerCell> relationDomain;
  std::vector<std::uint32_t> triggers;
  std::vector<std::vector<std::uint32_t>> releases;
  std::map<std::uint64_t, std::uint64_t> claims;
};

llvm::Error checkedAdd(std::uint64_t amount, std::uint64_t &value,
                       llvm::StringRef subject) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - value)
    return invalid(subject + " overflows u64");
  value += amount;
  return llvm::Error::success();
}

bool capacityBlocks(
    const ProgressActivationGroup &pending,
    const ProgressActivationGroup &holder,
    llvm::ArrayRef<MappingProgressCapacityCellProjection> cells) {
  for (const auto &[cell, pendingAmount] : pending.claims) {
    const auto held = holder.claims.find(cell);
    if (held == holder.claims.end() || cell >= cells.size())
      continue;
    const auto &capacity = cells[cell];
    const unsigned __int128 demand =
        static_cast<unsigned __int128>(capacity.baselineOccupancy) +
        pendingAmount + held->second;
    if (demand > capacity.capacity)
      return true;
  }
  return false;
}

llvm::Expected<bool>
relationDomainsIntersect(const ProgressActivationGroup &lhs,
                         const ProgressActivationGroup &rhs) {
  if (lhs.relationDomain.empty() || rhs.relationDomain.empty())
    return invalid("resource activation relation domain is empty");
  if (!lhs.relationRoot || !rhs.relationRoot)
    return invalid("resource activation relation root is absent");
  if (lhs.relationRoot != rhs.relationRoot)
    return true;
  for (const SystemPresburgerCell &left : lhs.relationDomain)
    for (const SystemPresburgerCell &right : rhs.relationDomain) {
      // Frozen relation domains are canonical. Identical cells are a common
      // single-partition case and are necessarily non-empty here.
      if (left == right)
        return true;
      auto intersects = systemPresburgerCellsIntersect(left, right);
      if (!intersects)
        return intersects.takeError();
      if (*intersects)
        return true;
    }
  return false;
}

std::vector<std::uint32_t>
findDirectedCycle(llvm::ArrayRef<std::vector<std::uint32_t>> edges) {
  constexpr std::uint32_t absent = std::numeric_limits<std::uint32_t>::max();
  std::vector<std::uint8_t> state(edges.size(), 0);
  std::vector<std::uint32_t> stack;
  std::vector<std::uint32_t> stackPositions(edges.size(), absent);
  std::vector<std::uint32_t> cycle;
  std::function<bool(std::uint32_t)> visit = [&](std::uint32_t node) {
    state[node] = 1;
    stackPositions[node] = stack.size();
    stack.push_back(node);
    for (std::uint32_t sink : edges[node]) {
      if (sink >= edges.size())
        continue;
      if (state[sink] == 0) {
        if (visit(sink))
          return true;
        continue;
      }
      if (state[sink] != 1)
        continue;
      cycle.assign(stack.begin() + stackPositions[sink], stack.end());
      cycle.push_back(sink);
      return true;
    }
    stack.pop_back();
    stackPositions[node] = absent;
    state[node] = 2;
    return false;
  };
  for (std::uint32_t node = 0; node != edges.size(); ++node)
    if (state[node] == 0 && visit(node))
      break;
  return cycle;
}

llvm::Expected<std::vector<std::uint32_t>>
initializedFeedbackInputOrdinals(const ::dataflow::CanonicalActorView &actor) {
  auto projected = ::dataflow::semantics::projectActorInitializedFeedbackInputs(
      ::dataflow::requireOperationSchema(actor.op), actor.op->getNumOperands(),
      actor.op->getNumResults());
  if (!projected)
    return projected.takeError();
  std::vector<std::uint32_t> result;
  result.reserve(projected->size());
  for (const auto &input : *projected)
    result.push_back(input.inputOrdinal);
  llvm::sort(result);
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

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

struct ActorDependencyGraph final {
  struct InitializedFeedbackEdge final {
    ::dataflow::ActorTokenResultRef producer;
    ::dataflow::ActorTokenOperandRef consumer;
    std::uint32_t producerActor = 0;
    std::uint32_t consumerActor = 0;
    bool closesCycle = false;
  };

  llvm::DenseMap<std::uint64_t, std::uint32_t> actorOrdinals;
  /// The actor DAG after initialized-feedback edges have been removed.
  std::vector<std::size_t> offsets;
  std::vector<std::uint32_t> destinations;
  std::vector<::dataflow::ActorRef> actorRefs;
  std::vector<InitializedFeedbackEdge> initializedFeedbackEdges;
  std::vector<::dataflow::ActorRef> postInitializationCycle;
  bool acyclic = false;
  bool postInitializationAcyclic = false;
};

llvm::Expected<bool>
closeActorEdges(std::size_t actorCount,
                std::vector<std::pair<std::uint32_t, std::uint32_t>> edges,
                std::vector<std::size_t> *closedOffsets = nullptr,
                std::vector<std::uint32_t> *closedDestinations = nullptr) {
  llvm::sort(edges);
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  std::vector<std::size_t> offsets(actorCount + 1, 0);
  std::vector<std::uint32_t> indegrees(actorCount, 0);
  for (const auto &[source, sink] : edges) {
    if (source >= actorCount || sink >= actorCount)
      return invalid("actor dependency edge is out of range");
    ++offsets[static_cast<std::size_t>(source) + 1];
    if (indegrees[sink] == std::numeric_limits<std::uint32_t>::max())
      return invalid("actor dependency indegree overflows");
    ++indegrees[sink];
  }
  for (std::size_t index = 1; index < offsets.size(); ++index)
    offsets[index] += offsets[index - 1];
  std::vector<std::uint32_t> destinations(edges.size());
  std::vector<std::size_t> cursors = offsets;
  cursors.pop_back();
  for (const auto &[source, sink] : edges)
    destinations[cursors[source]++] = sink;

  std::vector<std::uint32_t> ready;
  ready.reserve(actorCount);
  for (std::uint32_t actor = 0; actor != actorCount; ++actor)
    if (indegrees[actor] == 0)
      ready.push_back(actor);
  std::size_t visited = 0;
  for (std::size_t cursor = 0; cursor != ready.size(); ++cursor) {
    const std::uint32_t source = ready[cursor];
    ++visited;
    for (std::size_t edge = offsets[source]; edge != offsets[source + 1];
         ++edge) {
      const std::uint32_t sink = destinations[edge];
      if (--indegrees[sink] == 0)
        ready.push_back(sink);
    }
  }
  if (closedOffsets)
    *closedOffsets = std::move(offsets);
  if (closedDestinations)
    *closedDestinations = std::move(destinations);
  return visited == actorCount;
}

bool actorReachable(const ActorDependencyGraph &graph, std::uint32_t source,
                    std::uint32_t target) {
  if (source == target)
    return true;
  std::vector<bool> visited(graph.actorOrdinals.size(), false);
  std::vector<std::uint32_t> worklist{source};
  visited[source] = true;
  for (std::size_t cursor = 0; cursor != worklist.size(); ++cursor)
    for (std::size_t edge = graph.offsets[worklist[cursor]];
         edge != graph.offsets[worklist[cursor] + 1]; ++edge) {
      const std::uint32_t sink = graph.destinations[edge];
      if (sink == target)
        return true;
      if (!visited[sink]) {
        visited[sink] = true;
        worklist.push_back(sink);
      }
    }
  return false;
}

llvm::Expected<ActorDependencyGraph> buildActorDependencyGraph(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::GraphRef> coveredGraphs) {
  llvm::DenseMap<std::uint64_t, bool> covered;
  covered.reserve(coveredGraphs.size());
  for (const ::dataflow::GraphRef graph : coveredGraphs) {
    if (!covered.try_emplace(graph.entity.value(), true).second)
      return invalid("covered graph inventory contains a duplicate");
    auto resolved = dataflow.resolve(graph);
    if (!resolved)
      return resolved.takeError();
  }

  std::vector<::dataflow::CanonicalActorView> actors;
  for (const ::dataflow::CanonicalActorView &actor : dataflow.actors())
    if (covered.count(actor.graph.entity.value()) != 0)
      actors.push_back(actor);
  if (actors.size() > std::numeric_limits<std::uint32_t>::max())
    return invalid("actor inventory exceeds the native analysis domain");

  ActorDependencyGraph result;
  result.actorOrdinals.reserve(actors.size());
  result.actorRefs.reserve(actors.size());
  std::vector<std::vector<std::uint32_t>> initializedFeedbackInputs;
  initializedFeedbackInputs.reserve(actors.size());
  for (const auto [ordinal, actor] : llvm::enumerate(actors))
    if (!result.actorOrdinals
             .try_emplace(actor.ref.entity.value(),
                          static_cast<std::uint32_t>(ordinal))
             .second) {
      return invalid("canonical actor inventory contains a duplicate");
    } else {
      result.actorRefs.push_back(actor.ref);
      auto inputs = initializedFeedbackInputOrdinals(actor);
      if (!inputs)
        return inputs.takeError();
      initializedFeedbackInputs.push_back(std::move(*inputs));
    }

  std::vector<std::pair<std::uint32_t, std::uint32_t>> allEdges;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> nonFeedbackEdges;
  if (llvm::Error error = dataflow.forEachGraphEdge(
          [&](const ::dataflow::CanonicalGraphProducerEndpointRef &producer,
              const ::dataflow::CanonicalGraphConsumerEndpointRef &consumer)
              -> llvm::Error {
            const auto *source =
                std::get_if<::dataflow::ActorTokenResultRef>(&producer);
            const auto *sink =
                std::get_if<::dataflow::ActorTokenOperandRef>(&consumer);
            if (!source || !sink)
              return llvm::Error::success();
            const auto sourceOrdinal =
                result.actorOrdinals.find(source->actor.entity.value());
            const auto sinkOrdinal =
                result.actorOrdinals.find(sink->actor.entity.value());
            if (sourceOrdinal == result.actorOrdinals.end() &&
                sinkOrdinal == result.actorOrdinals.end())
              return llvm::Error::success();
            if (sourceOrdinal == result.actorOrdinals.end() ||
                sinkOrdinal == result.actorOrdinals.end())
              return invalid("covered graph edge crosses the actor catalog");
            allEdges.emplace_back(sourceOrdinal->second, sinkOrdinal->second);
            const bool initializedFeedback = llvm::is_contained(
                initializedFeedbackInputs[sinkOrdinal->second], sink->ordinal);
            if (initializedFeedback) {
              result.initializedFeedbackEdges.push_back(
                  {*source, *sink, sourceOrdinal->second, sinkOrdinal->second,
                   false});
            } else {
              nonFeedbackEdges.emplace_back(sourceOrdinal->second,
                                            sinkOrdinal->second);
            }
            return llvm::Error::success();
          }))
    return std::move(error);

  auto acyclic = closeActorEdges(actors.size(), std::move(allEdges));
  if (!acyclic)
    return acyclic.takeError();
  result.acyclic = *acyclic;
  auto postInitializationAcyclic =
      closeActorEdges(actors.size(), std::move(nonFeedbackEdges),
                      &result.offsets, &result.destinations);
  if (!postInitializationAcyclic)
    return postInitializationAcyclic.takeError();
  result.postInitializationAcyclic = *postInitializationAcyclic;
  if (!result.postInitializationAcyclic) {
    std::vector<std::vector<std::uint32_t>> edges(actors.size());
    for (std::uint32_t source = 0; source != actors.size(); ++source)
      edges[source].assign(result.destinations.begin() + result.offsets[source],
                           result.destinations.begin() +
                               result.offsets[source + 1]);
    std::vector<std::uint32_t> cycle = findDirectedCycle(edges);
    if (!cycle.empty() && cycle.front() == cycle.back())
      cycle.pop_back();
    result.postInitializationCycle.reserve(cycle.size());
    for (std::uint32_t ordinal : cycle)
      result.postInitializationCycle.push_back(result.actorRefs[ordinal]);
  }
  if (result.postInitializationAcyclic)
    for (ActorDependencyGraph::InitializedFeedbackEdge &edge :
         result.initializedFeedbackEdges)
      edge.closesCycle =
          actorReachable(result, edge.consumerActor, edge.producerActor);
  return result;
}

std::optional<std::uint32_t>
actorOrdinal(const ActorDependencyGraph &graph,
             const ::dataflow::CanonicalGraphConsumerEndpointRef &endpoint) {
  const auto *operand =
      std::get_if<::dataflow::ActorTokenOperandRef>(&endpoint);
  if (!operand)
    return std::nullopt;
  const auto found = graph.actorOrdinals.find(operand->actor.entity.value());
  if (found == graph.actorOrdinals.end())
    return std::nullopt;
  return found->second;
}

bool isBuffered(const std::optional<::loom::fabric::FabricPhysicalTraversalRef>
                    &traversal) {
  if (!traversal)
    return false;
  const auto *fifo = std::get_if<::loom::fabric::FabricFifoTraversalPayload>(
      &traversal->payload);
  return fifo &&
         fifo->mode == ::loom::fabric::FabricFifoTraversalMode::Buffered;
}

llvm::Expected<bool> dependentBranchHasDurableBoundary(
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> computeBindings,
    const SpatialRouteTreeView &route,
    const SpatialRouteProgressPrerequisite &prerequisite,
    std::uint64_t dependentSink) {
  const auto *external =
      std::get_if<SpatialRouteExternalSinkPrerequisite>(&prerequisite);
  if ((external && external->sinkOrdinal >= route.sinks.size()) ||
      dependentSink >= route.sinks.size())
    return invalid("route progress dependency names an absent sink");
  const SpatialRouteSinkView &dependent = route.sinks[dependentSink];
  if (dependent.nodeOrdinal >= route.nodes.size())
    return invalid("route progress dependency names an absent node");
  auto localBoundary = deriveSpatialSinkDurableProgressBoundary(
      techMapping, fabric, computeBindings, route, dependent);
  if (!localBoundary)
    return localBoundary.takeError();
  if (*localBoundary)
    return true;

  if (!external) {
    std::uint64_t node = dependent.nodeOrdinal;
    for (std::size_t visited = 0;; ++visited) {
      if (visited >= route.nodes.size())
        return invalid("route dependent ancestry is cyclic");
      if (isBuffered(route.nodes[node].incomingTraversal))
        return true;
      const auto parent = route.nodes[node].parentOrdinal;
      if (!parent)
        return false;
      if (*parent >= route.nodes.size())
        return invalid("route dependent parent is out of range");
      node = *parent;
    }
  }

  const SpatialRouteSinkView &prerequisiteSink =
      route.sinks[external->sinkOrdinal];
  if (prerequisiteSink.nodeOrdinal >= route.nodes.size())
    return invalid("route progress prerequisite names an absent node");

  std::vector<bool> prerequisiteAncestors(route.nodes.size(), false);
  std::uint64_t node = prerequisiteSink.nodeOrdinal;
  for (std::size_t visited = 0;; ++visited) {
    if (visited >= route.nodes.size() || prerequisiteAncestors[node])
      return invalid("route prerequisite ancestry is cyclic");
    prerequisiteAncestors[node] = true;
    const auto parent = route.nodes[node].parentOrdinal;
    if (!parent)
      break;
    if (*parent >= route.nodes.size())
      return invalid("route prerequisite parent is out of range");
    node = *parent;
  }

  node = dependent.nodeOrdinal;
  for (std::size_t visited = 0; !prerequisiteAncestors[node]; ++visited) {
    if (visited >= route.nodes.size())
      return invalid("route dependent ancestry is cyclic");
    if (isBuffered(route.nodes[node].incomingTraversal))
      return true;
    const auto parent = route.nodes[node].parentOrdinal;
    if (!parent || *parent >= route.nodes.size())
      return invalid("route sink branches have no common ancestor");
    node = *parent;
  }
  return false;
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

llvm::Expected<MappingDataflowProgressBasis> deriveMappingDataflowProgressBasis(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::GraphRef> coveredGraphs) {
  auto graph = buildActorDependencyGraph(dataflow, coveredGraphs);
  if (!graph)
    return graph.takeError();
  MappingDataflowProgressBasisKind kind =
      MappingDataflowProgressBasisKind::Cyclic;
  if (graph->acyclic)
    kind = MappingDataflowProgressBasisKind::Acyclic;
  else if (graph->postInitializationAcyclic)
    kind = MappingDataflowProgressBasisKind::InitializedFeedback;
  return MappingDataflowProgressBasis{
      kind, graph->actorRefs.size(), graph->initializedFeedbackEdges.size(),
      std::move(graph->postInitializationCycle)};
}

void emitMappingDataflowProgressBasisDiagnostic(
    const MappingDataflowProgressBasis &basis,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    mapping_debug::Stage stage) {
  if (basis.kind != MappingDataflowProgressBasisKind::Cyclic)
    return;
  mapping_debug::emit(
      mapping_debug::Level::Summary, stage,
      mapping_debug::Event::MappingFailure, [&](llvm::json::Object &fields) {
        fields["failure_scope"] = "dataflow_progress_basis";
        fields["closure_status"] = "proof_not_established";
        fields["covered_actor_count"] = basis.coveredActorCount;
        fields["initialized_feedback_edge_count"] =
            basis.initializedFeedbackEdgeCount;
        llvm::json::Array cycle;
        for (const ::dataflow::ActorRef actor : basis.residualCycle) {
          llvm::json::Object member;
          member["actor"] = actor.entity.value();
          auto resolved = dataflow.resolve(actor);
          if (resolved) {
            member["operation"] = resolved->op->getName().getStringRef().str();
          } else {
            llvm::consumeError(resolved.takeError());
            member["operation"] = "<unresolved>";
          }
          cycle.push_back(std::move(member));
        }
        fields["residual_cycle"] = std::move(cycle);
      });
}

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

llvm::Expected<MappingProgressClosure>
deriveMappingProgressClosure(const FrozenMappingProgressModel &model,
                             const MappingProgressProjection &projection) {
  return deriveMappingProgressClosure(
      model, MappingProgressProjectionView{
                 projection.basis, projection.routeObligations,
                 projection.capacityCells, projection.resourceActivations,
                 projection.bufferDependencyEdges,
                 projection.reconvergentCapacityObligations});
}

llvm::Expected<MappingProgressClosure>
deriveMappingProgressClosure(const FrozenMappingProgressModel &model,
                             MappingProgressProjectionView projection) {
  if (projection.basis.kind == MappingDataflowProgressBasisKind::Cyclic)
    return MappingProgressClosure{
        MappingProgressClosureKind::ProofNotEstablished,
        MappingProgressClosureReason::CyclicDataflowBasis,
        {},
        {}};
  if (llvm::any_of(projection.routeObligations, [](const auto &obligation) {
        return obligation.kind == MappingRouteProgressObligationKind::
                                      DurableBoundaryAfterDivergence &&
               !obligation.established;
      }))
    return MappingProgressClosure{
        MappingProgressClosureKind::ProvenClosedWaitSet,
        MappingProgressClosureReason::MissingDurableBoundary,
        {},
        {}};
  if (llvm::any_of(projection.routeObligations, [](const auto &obligation) {
        return obligation.kind ==
                   MappingRouteProgressObligationKind::FiniteBufferRecurrence &&
               !obligation.established;
      }))
    return MappingProgressClosure{
        MappingProgressClosureKind::ProofNotEstablished,
        MappingProgressClosureReason::FiniteBufferRecurrenceNotEstablished,
        {},
        {}};

  // The static buffer-dependency graph quotes only mandatory conjunctive wait
  // facts. A strongly connected component is a certificate only when it is
  // closed: every member's waits stay inside the component. A closed component
  // of pure order/join edges is a proven closed wait; one carrying a capacity
  // edge additionally requires the capacity proof, so it remains unestablished
  // here. An indeterminate construction is unestablished, never a proven
  // cycle.
  if (!projection.bufferDependencyEdges)
    return MappingProgressClosure{
        MappingProgressClosureKind::ProofNotEstablished,
        MappingProgressClosureReason::BufferDependencyNotEstablished,
        {},
        {}};
  std::vector<MappingStaticWaitNode> bufferNodes;
  std::vector<std::vector<std::uint32_t>> capacityComponents;
  if (!projection.bufferDependencyEdges->empty()) {
    const auto &edges = *projection.bufferDependencyEdges;
    std::map<std::string, std::uint32_t> nodeOrdinals;
    std::vector<MappingStaticWaitNode> nodes;
    const auto intern = [&](const MappingStaticWaitNode &node) {
      std::string key = staticWaitNodeKey(node);
      auto [position, inserted] =
          nodeOrdinals.try_emplace(std::move(key), nodes.size());
      if (inserted)
        nodes.push_back(node);
      return position->second;
    };
    for (const MappingBufferDependencyEdge &edge : edges) {
      intern(edge.from);
      intern(edge.to);
    }
    // The std::map key order is the canonical node order, so ordinals and the
    // witness order are deterministic.
    std::vector<std::vector<std::uint32_t>> successors(nodes.size());
    std::vector<std::vector<std::uint32_t>> successorEdges(nodes.size());
    std::vector<bool> selfLoop(nodes.size(), false);
    for (const auto [edgeOrdinal, edge] : llvm::enumerate(edges)) {
      const std::uint32_t from = nodeOrdinals[staticWaitNodeKey(edge.from)];
      const std::uint32_t to = nodeOrdinals[staticWaitNodeKey(edge.to)];
      if (from == to)
        selfLoop[from] = true;
      successors[from].push_back(to);
      successorEdges[from].push_back(static_cast<std::uint32_t>(edgeOrdinal));
    }
    for (std::uint32_t node = 0; node != nodes.size(); ++node) {
      std::vector<std::uint32_t> order(successors[node].size());
      std::iota(order.begin(), order.end(), 0);
      llvm::sort(order, [&](std::uint32_t lhs, std::uint32_t rhs) {
        return successors[node][lhs] < successors[node][rhs];
      });
      std::vector<std::uint32_t> sortedSuccessors;
      std::vector<std::uint32_t> sortedEdges;
      for (std::uint32_t position : order) {
        sortedSuccessors.push_back(successors[node][position]);
        sortedEdges.push_back(successorEdges[node][position]);
      }
      successors[node] = std::move(sortedSuccessors);
      successorEdges[node] = std::move(sortedEdges);
    }

    constexpr std::uint32_t absent = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> index(nodes.size(), absent);
    std::vector<std::uint32_t> lowlink(nodes.size(), 0);
    std::vector<bool> onStack(nodes.size(), false);
    std::vector<std::uint32_t> stack;
    std::uint32_t nextIndex = 0;
    std::vector<std::vector<std::uint32_t>> components;
    std::function<void(std::uint32_t)> connect = [&](std::uint32_t node) {
      index[node] = lowlink[node] = nextIndex++;
      stack.push_back(node);
      onStack[node] = true;
      for (std::uint32_t successor : successors[node]) {
        if (index[successor] == absent) {
          connect(successor);
          lowlink[node] = std::min(lowlink[node], lowlink[successor]);
        } else if (onStack[successor]) {
          lowlink[node] = std::min(lowlink[node], index[successor]);
        }
      }
      if (lowlink[node] != index[node])
        return;
      std::vector<std::uint32_t> component;
      std::uint32_t member;
      do {
        member = stack.back();
        stack.pop_back();
        onStack[member] = false;
        component.push_back(member);
      } while (member != node);
      llvm::sort(component);
      components.push_back(std::move(component));
    };
    for (std::uint32_t node = 0; node != nodes.size(); ++node)
      if (index[node] == absent)
        connect(node);

    std::optional<std::size_t> provenComponent;
    for (const auto [componentOrdinal, component] :
         llvm::enumerate(components)) {
      if (component.size() < 2 && !selfLoop[component.front()])
        continue;
      std::vector<bool> member(nodes.size(), false);
      for (std::uint32_t node : component)
        member[node] = true;
      bool closed = true;
      bool carriesCapacity = false;
      for (std::uint32_t node : component) {
        for (const auto [position, successor] :
             llvm::enumerate(successors[node])) {
          if (!member[successor]) {
            closed = false;
            continue;
          }
          if (edges[successorEdges[node][position]].kind ==
              MappingBufferDependencyEdgeKind::DownstreamCapacity)
            carriesCapacity = true;
        }
      }
      if (!closed)
        continue;
      if (carriesCapacity)
        capacityComponents.push_back(component);
      else if (!provenComponent)
        provenComponent = componentOrdinal;
    }
    if (provenComponent) {
      std::vector<MappingStaticWaitNode> witness;
      for (std::uint32_t node : components[*provenComponent])
        witness.push_back(nodes[node]);
      return MappingProgressClosure{
          MappingProgressClosureKind::ProvenClosedWaitSet,
          MappingProgressClosureReason::ClosedBufferDependencyCycle,
          {}, std::move(witness)};
    }
    bufferNodes = std::move(nodes);
  }

  // The reconvergent capacity proof: a queue class whose proven minimum legal
  // depth exceeds its selected pool is a proven closed wait by itself — a full
  // queue at cycle start admits no same-cycle replacement. A closed component
  // carrying a capacity edge is resolved exactly when every member class has a
  // proven obligation within its pool; otherwise it stays unestablished.
  std::set<std::string> capacityOwners;
  for (const MappingReconvergentCapacityObligation &obligation :
       projection.reconvergentCapacityObligations) {
    const std::vector<std::uint8_t> ownerBytes =
        ::loom::fabric::canonicalFabricBytes(obligation.owner);
    if (!capacityOwners
             .insert(std::string(
                 reinterpret_cast<const char *>(ownerBytes.data()),
                 ownerBytes.size()))
             .second)
      return invalid("reconvergent capacity repeats a shared FIFO owner");
    if (obligation.queueClasses.empty())
      return invalid("reconvergent capacity owner has no queue class");
    const bool proven = obligation.kind ==
                        MappingReconvergentCapacityProofKind::Proven;
    if (proven != obligation.minimumLegalCapacity.has_value())
      return invalid("reconvergent capacity proof state and minimum differ");
    std::optional<std::string> previousClass;
    bool global = false;
    for (const MappingStaticQueueClass &queueClass :
         obligation.queueClasses) {
      global |= queueClass.kind == MappingStaticQueueClassKind::Global;
      const std::string key = staticWaitNodeKey(
          MappingStorageQueueProgressNode{obligation.owner, queueClass});
      if (previousClass && *previousClass >= key)
        return invalid("reconvergent capacity queue classes are not canonical");
      previousClass = key;
    }
    if (global && obligation.queueClasses.size() != 1)
      return invalid("global FIFO class was combined with another class");
    std::optional<std::vector<std::uint8_t>> previousAnchor;
    for (const auto &anchor : obligation.routeAnchors) {
      std::vector<std::uint8_t> key =
          ::loom::fabric::canonicalFabricBytes(anchor);
      if (previousAnchor && *previousAnchor >= key)
        return invalid("reconvergent capacity route anchors are not canonical");
      previousAnchor = std::move(key);
    }
  }
  for (const MappingReconvergentCapacityObligation &obligation :
       projection.reconvergentCapacityObligations) {
    if (obligation.kind == MappingReconvergentCapacityProofKind::Proven &&
        obligation.minimumLegalCapacity &&
        *obligation.minimumLegalCapacity > obligation.selectedCapacity)
      return MappingProgressClosure{
          MappingProgressClosureKind::ProvenClosedWaitSet,
          MappingProgressClosureReason::ReconvergentCapacityShortfall,
          {},
          {}};
  }
  if (llvm::any_of(
          projection.reconvergentCapacityObligations,
          [](const MappingReconvergentCapacityObligation &obligation) {
            return obligation.kind == MappingReconvergentCapacityProofKind::
                                          ProofNotEstablished;
          }))
    return MappingProgressClosure{
        MappingProgressClosureKind::ProofNotEstablished,
        MappingProgressClosureReason::ReconvergentCapacityNotEstablished,
        {},
        {}};
  for (const std::vector<std::uint32_t> &component : capacityComponents) {
    for (std::uint32_t nodeOrdinal : component) {
      const auto *storage = std::get_if<MappingStorageQueueProgressNode>(
          &bufferNodes[nodeOrdinal]);
      if (!storage)
        continue;
      const auto obligation =
          llvm::find_if(projection.reconvergentCapacityObligations,
                        [&](const MappingReconvergentCapacityObligation
                                &candidate) {
                        return candidate.owner == storage->owner;
                      });
      if (obligation == projection.reconvergentCapacityObligations.end() ||
          obligation->kind !=
              MappingReconvergentCapacityProofKind::Proven)
        return MappingProgressClosure{
            MappingProgressClosureKind::ProofNotEstablished,
            MappingProgressClosureReason::ReconvergentCapacityNotEstablished,
            {},
            {}};
    }
  }

  const auto eventOrdinal = [&](const ::dataflow::EventFamilyKey &event)
      -> llvm::Expected<std::uint32_t> {
    auto key = eventKey(model.dataflowIdentity_, event);
    if (!key)
      return key.takeError();
    const auto found = model.eventOrdinals_.find(*key);
    if (found == model.eventOrdinals_.end())
      return invalid("System activation event is absent from the frozen "
                     "Dataflow progress model");
    return found->second;
  };
  const std::vector<bool> noAncestors;
  std::vector<std::optional<std::vector<bool>>> ancestorCache(
      model.reverseEdges_.size());
  const auto ancestors = [&](std::uint32_t event) -> const std::vector<bool> & {
    if (event >= model.reverseEdges_.size())
      return noAncestors;
    if (ancestorCache[event])
      return *ancestorCache[event];
    std::vector<bool> result(model.reverseEdges_.size(), false);
    std::vector<std::uint32_t> worklist{event};
    result[event] = true;
    for (std::size_t cursor = 0; cursor != worklist.size(); ++cursor)
      for (std::uint32_t predecessor : model.reverseEdges_[worklist[cursor]])
        if (!result[predecessor]) {
          result[predecessor] = true;
          worklist.push_back(predecessor);
        }
    ancestorCache[event] = std::move(result);
    return *ancestorCache[event];
  };

  std::map<std::string, std::size_t> groupOrdinals;
  std::vector<ProgressActivationGroup> groups;
  struct OwnerUse final {
    MappingResourceGrantPolicyKind policy =
        MappingResourceGrantPolicyKind::None;
    std::set<std::uint32_t> requesters;
  };
  std::map<std::string, OwnerUse> owners;

  for (const auto &[activationOrdinal, activation] :
       llvm::enumerate(projection.resourceActivations)) {
    if (activation.triggerAlternatives.empty())
      return invalid("System resource activation has no trigger alternative");
    auto key = atomicActivationKey(model.dataflowIdentity_, activation);
    if (!key)
      return key.takeError();
    auto [position, inserted] = groupOrdinals.try_emplace(*key, groups.size());
    if (inserted)
      groups.emplace_back();
    ProgressActivationGroup &group = groups[position->second];
    group.activationOrdinals.push_back(activationOrdinal);
    if (inserted) {
      group.relationRoot = activation.relationRoot;
      group.relationDomain = activation.relationDomain;
    }
    for (const auto &trigger : activation.triggerAlternatives) {
      auto ordinal = eventOrdinal(trigger);
      if (!ordinal)
        return ordinal.takeError();
      group.triggers.push_back(*ordinal);
    }
    for (const MappingProgressCausalReleaseProjection &point :
         activation.causalRelease) {
      if (point.alternatives.empty())
        return invalid("System causal release point has no alternative");
      std::vector<std::uint32_t> alternatives;
      alternatives.reserve(point.alternatives.size());
      for (const auto &release : point.alternatives) {
        auto ordinal = eventOrdinal(release);
        if (!ordinal)
          return ordinal.takeError();
        alternatives.push_back(*ordinal);
      }
      llvm::sort(alternatives);
      alternatives.erase(std::unique(alternatives.begin(), alternatives.end()),
                         alternatives.end());
      group.releases.push_back(std::move(alternatives));
    }
    for (const MappingProgressCapacityClaimProjection &claim :
         activation.capacityClaims) {
      if (claim.capacityCellOrdinal >= projection.capacityCells.size())
        return invalid("System activation claim names a foreign capacity cell");
      if (llvm::Error error =
              checkedAdd(claim.amount, group.claims[claim.capacityCellOrdinal],
                         "atomic activation capacity claim"))
        return std::move(error);
    }
    const MappingResourceProgressUse &use = activation.arbitration;
    if (use.physicalOwnerKey.empty())
      return invalid("System activation has an empty physical owner key");
    auto [owner, ownerInserted] =
        owners.try_emplace(use.physicalOwnerKey, OwnerUse{use.grantPolicy, {}});
    if (!ownerInserted && owner->second.policy != use.grantPolicy)
      return invalid("one physical owner has inconsistent grant policies");
    owner->second.requesters.insert(use.requester);
  }

  for (ProgressActivationGroup &group : groups) {
    llvm::sort(group.triggers);
    group.triggers.erase(
        std::unique(group.triggers.begin(), group.triggers.end()),
        group.triggers.end());
    llvm::sort(group.releases);
    group.releases.erase(
        std::unique(group.releases.begin(), group.releases.end()),
        group.releases.end());
    for (const auto &[cell, amount] : group.claims) {
      const auto &capacity = projection.capacityCells[cell];
      const unsigned __int128 usage =
          static_cast<unsigned __int128>(capacity.baselineOccupancy) + amount;
      if (usage > capacity.capacity)
        return MappingProgressClosure{
            MappingProgressClosureKind::ProvenClosedWaitSet,
            MappingProgressClosureReason::ActivationCapacityExceeded,
            {},
            {}};
    }
  }

  for (const auto &[key, owner] : owners) {
    (void)key;
    if (owner.policy == MappingResourceGrantPolicyKind::FixedPriority &&
        owner.requesters.size() > 1)
      return MappingProgressClosure{
          MappingProgressClosureKind::ProofNotEstablished,
          MappingProgressClosureReason::FixedPriorityStarvation,
          {},
          {}};
  }

  // Active and pending are separate nodes. This prevents ordinary contention
  // between two pending atomic acquisitions from masquerading as hold-and-wait.
  const auto activeNode = [](std::size_t group) {
    return static_cast<std::uint32_t>(2 * group);
  };
  const auto pendingNode = [](std::size_t group) {
    return static_cast<std::uint32_t>(2 * group + 1);
  };
  std::vector<std::vector<std::uint32_t>> waitFor(groups.size() * 2);

  for (std::size_t holder = 0; holder != groups.size(); ++holder) {
    for (std::size_t pending = 0; pending != groups.size(); ++pending) {
      if (pending == holder)
        continue;
      bool causallyRequired = false;
      for (const auto &releasePoint : groups[holder].releases) {
        bool required = false;
        for (std::uint32_t pendingTrigger : groups[pending].triggers) {
          for (std::uint32_t holderTrigger : groups[holder].triggers) {
            if (holderTrigger == pendingTrigger ||
                llvm::is_contained(releasePoint, holderTrigger))
              continue;
            const std::vector<bool> &pendingAncestors =
                ancestors(pendingTrigger);
            if (holderTrigger >= pendingAncestors.size() ||
                !pendingAncestors[holderTrigger])
              continue;
            const bool strictlyPrecedesRelease =
                llvm::any_of(releasePoint, [&](std::uint32_t release) {
                  const std::vector<bool> &releaseAncestors =
                      ancestors(release);
                  return pendingTrigger != release &&
                         pendingTrigger < releaseAncestors.size() &&
                         releaseAncestors[pendingTrigger];
                });
            if (strictlyPrecedesRelease) {
              required = true;
              break;
            }
          }
          if (required)
            break;
        }
        if (required) {
          causallyRequired = true;
          break;
        }
      }
      if (!causallyRequired)
        continue;
      auto domainsOverlap =
          relationDomainsIntersect(groups[holder], groups[pending]);
      if (!domainsOverlap)
        return domainsOverlap.takeError();
      if (*domainsOverlap)
        waitFor[activeNode(holder)].push_back(pendingNode(pending));
    }
  }
  for (std::size_t pending = 0; pending != groups.size(); ++pending)
    for (std::size_t holder = 0; holder != groups.size(); ++holder) {
      if (pending == holder || !capacityBlocks(groups[pending], groups[holder],
                                               projection.capacityCells))
        continue;
      waitFor[pendingNode(pending)].push_back(activeNode(holder));
    }
  for (auto &successors : waitFor) {
    llvm::sort(successors);
    successors.erase(std::unique(successors.begin(), successors.end()),
                     successors.end());
  }
  const std::vector<std::uint32_t> cycle = findDirectedCycle(waitFor);
  if (!cycle.empty()) {
    std::vector<MappingProgressWaitCycleNode> witness;
    witness.reserve(cycle.size());
    for (std::uint32_t node : cycle) {
      const std::uint64_t groupOrdinal = node / 2;
      if (groupOrdinal >= groups.size())
        return invalid("possible wait cycle names a foreign activation group");
      const ProgressActivationGroup &group = groups[groupOrdinal];
      std::vector<std::uint64_t> capacityCells;
      capacityCells.reserve(group.claims.size());
      for (const auto &[cell, amount] : group.claims) {
        (void)amount;
        capacityCells.push_back(cell);
      }
      std::vector<std::uint32_t> releases;
      for (const auto &point : group.releases)
        releases.insert(releases.end(), point.begin(), point.end());
      llvm::sort(releases);
      releases.erase(std::unique(releases.begin(), releases.end()),
                     releases.end());
      witness.push_back({
          groupOrdinal,
          (node & 1) == 0 ? MappingProgressWaitNodeKind::Active
                          : MappingProgressWaitNodeKind::Pending,
          group.activationOrdinals,
          std::move(capacityCells),
          group.triggers,
          std::move(releases),
      });
    }
    return MappingProgressClosure{
        MappingProgressClosureKind::ProofNotEstablished,
        MappingProgressClosureReason::PossibleWaitCycle, std::move(witness),
        {}};
  }
  return MappingProgressClosure{
      MappingProgressClosureKind::ProvenNoClosedWaitSet,
      MappingProgressClosureReason::None,
      {},
      {}};
}

llvm::StringRef
mappingProgressClosureReasonSpelling(MappingProgressClosureReason reason) {
  switch (reason) {
  case MappingProgressClosureReason::None:
    return "none";
  case MappingProgressClosureReason::CyclicDataflowBasis:
    return "cyclic_dataflow_basis";
  case MappingProgressClosureReason::MissingDurableBoundary:
    return "missing_durable_boundary";
  case MappingProgressClosureReason::ActivationCapacityExceeded:
    return "activation_capacity_exceeded";
  case MappingProgressClosureReason::FixedPriorityStarvation:
    return "fixed_priority_starvation";
  case MappingProgressClosureReason::PossibleWaitCycle:
    return "possible_wait_cycle";
  case MappingProgressClosureReason::FiniteBufferRecurrenceNotEstablished:
    return "finite_buffer_recurrence_not_established";
  case MappingProgressClosureReason::ClosedBufferDependencyCycle:
    return "closed_buffer_dependency_cycle";
  case MappingProgressClosureReason::BufferDependencyNotEstablished:
    return "buffer_dependency_not_established";
  case MappingProgressClosureReason::ReconvergentCapacityShortfall:
    return "reconvergent_capacity_shortfall";
  case MappingProgressClosureReason::ReconvergentCapacityNotEstablished:
    return "reconvergent_capacity_not_established";
  }
  llvm_unreachable("unknown Mapping progress closure reason");
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

llvm::Expected<std::vector<SpatialRouteProgressDependency>>
deriveSpatialRouteProgressDependencies(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping) {
  auto graph = buildActorDependencyGraph(dataflow, techMapping.covers());
  if (!graph)
    return graph.takeError();

  std::vector<SpatialRouteProgressDependency> result;
  std::vector<std::uint64_t> reachabilityMarks(graph->actorOrdinals.size(), 0);
  std::vector<std::uint32_t> worklist;
  worklist.reserve(graph->actorOrdinals.size());
  std::uint64_t generation = 0;
  for (const auto [netOrdinal, net] :
       llvm::enumerate(techMapping.residualLogicalNets())) {
    struct PrerequisiteCandidate final {
      ::dataflow::CanonicalGraphConsumerEndpointRef consumer;
      SpatialRouteProgressPrerequisite prerequisite;
    };
    std::vector<PrerequisiteCandidate> candidates;
    candidates.reserve(net.sinks.size());
    for (const auto [sinkOrdinal, sink] : llvm::enumerate(net.sinks))
      candidates.push_back(
          {sink, SpatialRouteExternalSinkPrerequisite{sinkOrdinal}});
    for (const auto [realizationOrdinal, realization] :
         llvm::enumerate(techMapping.memoryRealizations()))
      for (const auto &edgeRecord :
           llvm::enumerate(realization.internalEdges)) {
        const std::size_t edgeOrdinal = edgeRecord.index();
        const TechMemoryInternalEdgeView &edge = edgeRecord.value();
        if (edge.producer == net.producer &&
            !llvm::any_of(candidates, [&](const auto &candidate) {
              return candidate.consumer == edge.consumer;
            }))
          candidates.push_back(
              {edge.consumer, SpatialRouteInternalMemoryConnectionPrerequisite{
                                  realizationOrdinal, edgeOrdinal}});
      }
    std::vector<std::optional<std::uint32_t>> sinkActors;
    sinkActors.reserve(candidates.size());
    for (const PrerequisiteCandidate &candidate : candidates)
      sinkActors.push_back(actorOrdinal(*graph, candidate.consumer));
    for (std::size_t prerequisite = 0; prerequisite != candidates.size();
         ++prerequisite) {
      if (!sinkActors[prerequisite])
        continue;
      ++generation;
      if (generation == 0) {
        std::fill(reachabilityMarks.begin(), reachabilityMarks.end(), 0);
        generation = 1;
      }
      worklist.clear();
      reachabilityMarks[*sinkActors[prerequisite]] = generation;
      worklist.push_back(*sinkActors[prerequisite]);
      for (std::size_t cursor = 0; cursor != worklist.size(); ++cursor)
        for (std::size_t edge = graph->offsets[worklist[cursor]];
             edge != graph->offsets[worklist[cursor] + 1]; ++edge) {
          const std::uint32_t successor = graph->destinations[edge];
          if (reachabilityMarks[successor] == generation)
            continue;
          reachabilityMarks[successor] = generation;
          worklist.push_back(successor);
        }
      for (std::size_t dependent = 0; dependent != net.sinks.size();
           ++dependent) {
        if (dependent == prerequisite || !sinkActors[dependent] ||
            *sinkActors[dependent] == *sinkActors[prerequisite] ||
            reachabilityMarks[*sinkActors[dependent]] != generation)
          continue;
        result.push_back(
            {netOrdinal, candidates[prerequisite].prerequisite, dependent});
      }
    }
  }
  for (const ActorDependencyGraph::InitializedFeedbackEdge &feedback :
       graph->initializedFeedbackEdges) {
    if (!feedback.closesCycle)
      continue;
    bool found = false;
    for (const auto [netOrdinal, net] :
         llvm::enumerate(techMapping.residualLogicalNets())) {
      if (net.producer !=
          ::dataflow::CanonicalGraphProducerEndpointRef(feedback.producer))
        continue;
      for (const auto [sinkOrdinal, sink] : llvm::enumerate(net.sinks)) {
        if (sink !=
            ::dataflow::CanonicalGraphConsumerEndpointRef(feedback.consumer))
          continue;
        if (found)
          return invalid("initialized feedback edge has multiple residual "
                         "physical dispositions");
        result.push_back({netOrdinal,
                          SpatialRouteInitializedFeedbackPrerequisite{},
                          sinkOrdinal});
        found = true;
      }
    }
    if (!found)
      return invalid("initialized feedback edge has no residual physical "
                     "disposition");
  }
  const auto prerequisiteKey = [](const auto &prerequisite) {
    return std::visit(
        [](const auto &typed) {
          using T = std::decay_t<decltype(typed)>;
          if constexpr (std::is_same_v<T, SpatialRouteExternalSinkPrerequisite>)
            return std::tuple<std::uint8_t, std::uint64_t, std::uint64_t>{
                0, typed.sinkOrdinal, 0};
          else if constexpr (
              std::is_same_v<T,
                             SpatialRouteInternalMemoryConnectionPrerequisite>)
            return std::tuple<std::uint8_t, std::uint64_t, std::uint64_t>{
                1, typed.memoryRealizationOrdinal, typed.internalEdgeOrdinal};
          else
            return std::tuple<std::uint8_t, std::uint64_t, std::uint64_t>{2, 0,
                                                                          0};
        },
        prerequisite);
  };
  llvm::sort(result, [&](const auto &lhs, const auto &rhs) {
    const auto lhsPrefix =
        std::tie(lhs.logicalNetOrdinal, lhs.dependentSinkOrdinal);
    const auto rhsPrefix =
        std::tie(rhs.logicalNetOrdinal, rhs.dependentSinkOrdinal);
    if (lhsPrefix != rhsPrefix)
      return lhsPrefix < rhsPrefix;
    return prerequisiteKey(lhs.prerequisite) <
           prerequisiteKey(rhs.prerequisite);
  });
  result.erase(std::unique(result.begin(), result.end(),
                           [&](const auto &lhs, const auto &rhs) {
                             return lhs.logicalNetOrdinal ==
                                        rhs.logicalNetOrdinal &&
                                    lhs.dependentSinkOrdinal ==
                                        rhs.dependentSinkOrdinal &&
                                    prerequisiteKey(lhs.prerequisite) ==
                                        prerequisiteKey(rhs.prerequisite);
                           }),
               result.end());
  return result;
}

std::uint64_t countSpatialSharedFiniteBuffers(
    llvm::ArrayRef<SpatialFiniteBufferSelection> selections) {
  struct SelectedFifo final {
    ::loom::fabric::FabricFifoOccurrenceRef fifo;
    std::uint64_t logicalNetOrdinal = 0;
    bool shared = false;
  };
  std::vector<SelectedFifo> selectedFifos;
  std::uint64_t count = 0;
  for (const SpatialFiniteBufferSelection &selection : selections) {
    auto selected = llvm::find_if(selectedFifos, [&](const auto &use) {
      return use.fifo == selection.fifo;
    });
    if (selected == selectedFifos.end()) {
      selectedFifos.push_back(
          {selection.fifo, selection.logicalNetOrdinal, false});
      continue;
    }
    if (selected->logicalNetOrdinal == selection.logicalNetOrdinal ||
        selected->shared)
      continue;
    ++count;
    selected->shared = true;
  }
  return count;
}

MappingRouteProgressObligationProjection projectSpatialFiniteBufferRecurrence(
    llvm::ArrayRef<SpatialRouteTreeView> routes) {
  std::vector<::dataflow::CanonicalGraphProducerEndpointRef> logicalNets;
  std::vector<SpatialFiniteBufferSelection> selections;
  for (const SpatialRouteTreeView &route : routes) {
    auto logicalNet = llvm::find(logicalNets, route.logicalNet);
    if (logicalNet == logicalNets.end()) {
      logicalNets.push_back(route.logicalNet);
      logicalNet = std::prev(logicalNets.end());
    }
    const std::uint64_t logicalNetOrdinal =
        std::distance(logicalNets.begin(), logicalNet);
    const auto append =
        [&](const std::optional<::loom::fabric::FabricPhysicalTraversalRef>
                &traversal) {
          if (!traversal)
            return;
          const auto *fifo =
              std::get_if<::loom::fabric::FabricFifoTraversalPayload>(
                  &traversal->payload);
          if (fifo &&
              fifo->mode == ::loom::fabric::FabricFifoTraversalMode::Buffered)
            selections.push_back({logicalNetOrdinal, fifo->owner});
        };
    append(route.localTraversal);
    for (const SpatialRouteNodeView &node : route.nodes)
      append(node.incomingTraversal);
    for (const SpatialRouteSinkView &sink : route.sinks)
      append(sink.localTraversal);
  }
  return {MappingRouteProgressObligationKind::FiniteBufferRecurrence,
          countSpatialSharedFiniteBuffers(selections) == 0};
}

namespace {

/// One selected Buffered FIFO occurrence and its queue class along one
/// producer-to-sink channel path, in token flow order.
struct SpatialChannelStorageStop final {
  MappingStorageQueueProgressNode node;
  ::loom::fabric::FabricPhysicalTraversalRef traversal;
};

/// The buffered storage sequence of one residual channel path.
struct SpatialChannelStorageChain final {
  std::optional<::dataflow::ActorRef> producer;
  std::optional<::dataflow::ActorRef> consumer;
  std::optional<::dataflow::CanonicalGraphProducerEndpointRef> producerEndpoint;
  std::optional<::dataflow::CanonicalGraphConsumerEndpointRef> consumerEndpoint;
  std::vector<SpatialChannelStorageStop> stops;
  std::uint64_t netOrdinal = 0;
  bool primed = false;
};

/// The complete per-channel storage inventory of the selected graphs, shared
/// by the buffer-dependency obligation and the reconvergent capacity proof.
struct SpatialChannelStorageInventory final {
  std::vector<SpatialChannelStorageChain> channels;
  std::vector<::dataflow::CanonicalGraphProducerEndpointRef> logicalNets;
};

} // namespace

/// Rebuilds the per-channel buffered-storage inventory of the selected graphs.
/// Channels primed by initialized feedback are marked but retained: their
/// initial tokens occupy the queue classes even though their wait facts never
/// participate in a closed wait. Returns a disengaged optional when a queue
/// class, tag residency, or relation domain is indeterminate.
llvm::Expected<std::optional<SpatialChannelStorageInventory>>
projectSpatialChannelStorageInventory(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments,
    llvm::ArrayRef<::dataflow::GraphRef> selectedGraphs) {
  std::set<std::uint64_t> selectedGraphEntities;
  for (const ::dataflow::GraphRef graph : selectedGraphs) {
    if (!llvm::is_contained(techMapping.covers(), graph))
      return invalid("selected progress graph is absent from TechMapping");
    if (!selectedGraphEntities.insert(graph.entity.value()).second)
      return invalid("selected progress graph inventory contains a duplicate");
  }

  // Channels primed by initialized feedback carry an initial token, so their
  // queue order and acceptance facts cannot participate in a closed wait.
  auto actorGraph = buildActorDependencyGraph(dataflow, techMapping.covers());
  if (!actorGraph)
    return actorGraph.takeError();
  std::set<std::pair<std::string, std::string>> primedChannels;
  for (const ActorDependencyGraph::InitializedFeedbackEdge &feedback :
       actorGraph->initializedFeedbackEdges) {
    auto producerKey = ::dataflow::encodeDataflowReference(
        dataflow.identity(),
        ::dataflow::CanonicalGraphProducerEndpointRef(feedback.producer));
    if (!producerKey)
      return producerKey.takeError();
    auto consumerKey = ::dataflow::encodeDataflowReference(
        dataflow.identity(),
        ::dataflow::CanonicalGraphConsumerEndpointRef(feedback.consumer));
    if (!consumerKey)
      return consumerKey.takeError();
    primedChannels.emplace(
        std::string(reinterpret_cast<const char *>(producerKey->data()),
                    producerKey->size()),
        std::string(reinterpret_cast<const char *>(consumerKey->data()),
                    consumerKey->size()));
  }

  SpatialChannelStorageInventory inventory;
  const auto producerActorOf =
      [](const ::dataflow::CanonicalGraphProducerEndpointRef &endpoint)
      -> std::optional<::dataflow::ActorRef> {
    const auto *token = std::get_if<::dataflow::ActorTokenResultRef>(&endpoint);
    return token ? std::optional<::dataflow::ActorRef>(token->actor)
                 : std::nullopt;
  };
  const auto consumerActorOf =
      [](const ::dataflow::CanonicalGraphConsumerEndpointRef &endpoint)
      -> std::optional<::dataflow::ActorRef> {
    const auto *token = std::get_if<::dataflow::ActorTokenOperandRef>(&endpoint);
    return token ? std::optional<::dataflow::ActorRef>(token->actor)
                 : std::nullopt;
  };

  for (const auto [routeOrdinal, route] : llvm::enumerate(routes)) {
    auto graph = dataflow.graphOf(route.logicalNet);
    if (!graph)
      return graph.takeError();
    if (selectedGraphEntities.count(graph->entity.value()) == 0)
      continue;
    const auto logicalNet =
        llvm::find(inventory.logicalNets, route.logicalNet);
    if (logicalNet == inventory.logicalNets.end())
      inventory.logicalNets.push_back(route.logicalNet);
    const std::uint64_t netOrdinal = static_cast<std::uint64_t>(
        std::distance(inventory.logicalNets.begin(), logicalNet));
    const std::optional<::dataflow::ActorRef> producerActor =
        producerActorOf(route.logicalNet);

    for (const SpatialRouteSinkView &sink : route.sinks) {
      const std::optional<::dataflow::ActorRef> consumerActor =
          consumerActorOf(sink.sink);
      // The buffered storage sequence along this producer-to-sink path, in
      // token flow order: the root arc, the ancestor node arcs, then the sink
      // arc. Each stop names the traversal position whose node carries the
      // Physical Tag assignment for its queue class.
      std::vector<
          std::pair<std::optional<::loom::fabric::FabricPhysicalTraversalRef>,
                    std::uint64_t>>
          arcs;
      arcs.push_back({route.localTraversal, 0});
      std::vector<
          std::pair<std::optional<::loom::fabric::FabricPhysicalTraversalRef>,
                    std::uint64_t>>
          nodeArcs;
      std::optional<std::uint64_t> cursor = sink.nodeOrdinal;
      while (cursor) {
        if (*cursor >= route.nodes.size())
          return invalid("RouteTree sink names an absent node");
        const SpatialRouteNodeView &node = route.nodes[*cursor];
        if (node.incomingTraversal)
          nodeArcs.push_back({node.incomingTraversal, node.ordinal});
        cursor = node.parentOrdinal;
      }
      arcs.insert(arcs.end(), nodeArcs.rbegin(), nodeArcs.rend());
      arcs.push_back({sink.localTraversal, sink.nodeOrdinal});

      SpatialChannelStorageChain channel;
      channel.producer = producerActor;
      channel.consumer = consumerActor;
      channel.producerEndpoint = route.logicalNet;
      channel.consumerEndpoint = sink.sink;
      channel.netOrdinal = netOrdinal;
      for (const auto &[traversal, tagNodeOrdinal] : arcs) {
        if (!traversal)
          continue;
        const auto *fifo =
            std::get_if<::loom::fabric::FabricFifoTraversalPayload>(
                &traversal->payload);
        if (!fifo || fifo->mode != ::loom::fabric::FabricFifoTraversalMode::
                                       Buffered)
          continue;
        const ::fabric::FifoQueueDiscipline discipline =
            fabric.fifoQueueDiscipline(fifo->owner).value_or(
                ::fabric::FifoQueueDiscipline::StrictFifo);
        MappingStaticQueueClass queueClass{MappingStaticQueueClassKind::Global,
                                           llvm::APInt(1, 0)};
        if (discipline == ::fabric::FifoQueueDiscipline::PerTagVirtualChannel) {
          // A virtual-channel class is keyed by the exact resident tag bit
          // value; an untagged path or a missing tag assignment makes the
          // class indeterminate and the whole construction unestablished.
          if (route.nodes.empty())
            return std::optional<SpatialChannelStorageInventory>{};
          const auto &endpoint = route.nodes[tagNodeOrdinal].endpoint;
          const auto path = fabric.transportEndpointDataPath(endpoint);
          if (!path || path->tagWidthBits == 0)
            return std::optional<SpatialChannelStorageInventory>{};
          auto tag = detail::resolveConfiguredHardwarePhysicalTag(
              fabric, routes, resourceUses, physicalTagSegments, routeOrdinal,
              tagNodeOrdinal);
          if (!tag)
            return std::optional<SpatialChannelStorageInventory>{};
          queueClass = MappingStaticQueueClass{MappingStaticQueueClassKind::
                                                   PhysicalTag,
                                               *tag};
        }
        channel.stops.push_back(
            SpatialChannelStorageStop{
                MappingStorageQueueProgressNode{fifo->owner, queueClass},
                *traversal});
      }

      // A channel primed by initialized feedback carries its initial token;
      // none of its wait facts can participate in a closed wait.
      if (producerActor && consumerActor) {
        auto producerKey = ::dataflow::encodeDataflowReference(
            dataflow.identity(), route.logicalNet);
        if (!producerKey)
          return producerKey.takeError();
        auto consumerKey =
            ::dataflow::encodeDataflowReference(dataflow.identity(), sink.sink);
        if (!consumerKey)
          return consumerKey.takeError();
        channel.primed =
            primedChannels.count(
                {std::string(reinterpret_cast<const char *>(producerKey->data()),
                             producerKey->size()),
                 std::string(reinterpret_cast<const char *>(consumerKey->data()),
                             consumerKey->size())}) != 0;
      }
      inventory.channels.push_back(std::move(channel));
    }
  }
  return std::optional<SpatialChannelStorageInventory>(std::move(inventory));
}

llvm::Expected<std::optional<std::vector<MappingBufferDependencyEdge>>>
projectSpatialBufferDependencyEdges(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments,
    llvm::ArrayRef<SpatialPeOperandQueueMatchGroupView> operandQueueGroups,
    llvm::ArrayRef<::dataflow::GraphRef> selectedGraphs) {
  auto inventory = projectSpatialChannelStorageInventory(
      dataflow, techMapping, fabric, routes, resourceUses, physicalTagSegments,
      selectedGraphs);
  if (!inventory)
    return inventory.takeError();
  if (!*inventory)
    return std::optional<std::vector<MappingBufferDependencyEdge>>{};
  std::set<std::uint64_t> selectedGraphEntities;
  for (const ::dataflow::GraphRef graph : selectedGraphs)
    selectedGraphEntities.insert(graph.entity.value());

  std::vector<MappingBufferDependencyEdge> edges;
  std::set<std::string> edgeKeys;
  const auto emit = [&](MappingStaticWaitNode from, MappingStaticWaitNode to,
                        MappingBufferDependencyEdgeKind kind,
                        std::uint64_t logicalNetOrdinal,
                        std::optional<::loom::fabric::FabricPhysicalTraversalRef>
                            routeAnchor) -> llvm::Error {
    std::string key = staticWaitNodeKey(from) + staticWaitNodeKey(to);
    key.push_back(static_cast<char>(kind));
    appendU64(key, logicalNetOrdinal);
    if (routeAnchor) {
      const std::vector<std::uint8_t> anchor =
          ::loom::fabric::canonicalFabricBytes(*routeAnchor);
      key.append(reinterpret_cast<const char *>(anchor.data()), anchor.size());
    }
    if (!edgeKeys.insert(std::move(key)).second)
      return llvm::Error::success();
    edges.push_back(MappingBufferDependencyEdge{std::move(from), std::move(to),
                                                kind, logicalNetOrdinal,
                                                std::move(routeAnchor)});
    return llvm::Error::success();
  };
  const auto consumerActorOf =
      [](const ::dataflow::CanonicalGraphConsumerEndpointRef &endpoint)
      -> std::optional<::dataflow::ActorRef> {
    const auto *token = std::get_if<::dataflow::ActorTokenOperandRef>(&endpoint);
    return token ? std::optional<::dataflow::ActorRef>(token->actor)
                 : std::nullopt;
  };

  // A queue class couples the nets resident in it. The terminal head-release
  // wait (queue class to consumer) exists only when the class couples at
  // least two nets: with one net the head always belongs to that net and the
  // delivery is the same handshake as the consumer's input join, never an
  // independent wait.
  std::map<std::string, std::set<std::uint64_t>> classNets;
  for (const SpatialChannelStorageChain &channel : (*inventory)->channels) {
    if (channel.primed)
      continue;
    for (const SpatialChannelStorageStop &stop : channel.stops)
      classNets[staticWaitNodeKey(stop.node)].insert(channel.netOrdinal);
  }

  for (const SpatialChannelStorageChain &channel : (*inventory)->channels) {
    if (channel.primed)
      continue;
    if (channel.stops.empty()) {
      if (channel.producer && channel.consumer) {
        if (llvm::Error error =
                emit(*channel.consumer, *channel.producer,
                     MappingBufferDependencyEdgeKind::ActorInputJoin,
                     channel.netOrdinal, std::nullopt))
          return std::move(error);
      }
      continue;
    }
    if (channel.producer) {
      if (llvm::Error error =
              emit(*channel.producer, channel.stops.front().node,
                   MappingBufferDependencyEdgeKind::OutputCausalRelease,
                   channel.netOrdinal, channel.stops.front().traversal))
        return std::move(error);
    }
    for (std::size_t stop = 1; stop != channel.stops.size(); ++stop)
      if (llvm::Error error =
              emit(channel.stops[stop - 1].node, channel.stops[stop].node,
                   MappingBufferDependencyEdgeKind::DownstreamCapacity,
                   channel.netOrdinal, channel.stops[stop].traversal))
        return std::move(error);
    if (channel.consumer) {
      const SpatialChannelStorageStop &terminal = channel.stops.back();
      if (llvm::Error error =
              emit(*channel.consumer, terminal.node,
                   MappingBufferDependencyEdgeKind::ActorInputJoin,
                   channel.netOrdinal, terminal.traversal))
        return std::move(error);
      const auto coupled = classNets.find(staticWaitNodeKey(terminal.node));
      if (coupled != classNets.end() && coupled->second.size() >= 2) {
        if (llvm::Error error =
                emit(terminal.node, *channel.consumer,
                     MappingBufferDependencyEdgeKind::ActorInputJoin,
                     channel.netOrdinal, terminal.traversal))
          return std::move(error);
      }
    }
  }

  for (const SpatialPeOperandQueueMatchGroupView &group : operandQueueGroups) {
    auto graph = dataflow.graphOf(group.logicalNet);
    if (!graph)
      return graph.takeError();
    if (selectedGraphEntities.count(graph->entity.value()) == 0)
      continue;
    auto logicalNet =
        llvm::find((*inventory)->logicalNets, group.logicalNet);
    if (logicalNet == (*inventory)->logicalNets.end())
      (*inventory)->logicalNets.push_back(group.logicalNet);
    const std::uint64_t netOrdinal = static_cast<std::uint64_t>(
        std::distance((*inventory)->logicalNets.begin(), logicalNet));
    for (const SpatialPeOperandQueueMatchView &match : group.matches) {
      const MappingOperandQueueProgressNode queueNode{match.queue, match.fu};
      for (const ::dataflow::CanonicalGraphConsumerEndpointRef &consumer :
           match.consumers) {
        const std::optional<::dataflow::ActorRef> consumerActor =
            consumerActorOf(consumer);
        if (!consumerActor)
          continue;
        if (llvm::Error error =
                emit(*consumerActor, queueNode,
                     MappingBufferDependencyEdgeKind::OperandQueueOwner,
                     netOrdinal, std::nullopt))
          return std::move(error);
      }
    }
  }
  return std::optional<std::vector<MappingBufferDependencyEdge>>(
      std::move(edges));
}

llvm::Expected<std::vector<MappingReconvergentCapacityObligation>>
deriveMappingReconvergentCapacityProof(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments,
    llvm::ArrayRef<::dataflow::GraphRef> selectedGraphs) {
  auto inventory = projectSpatialChannelStorageInventory(
      dataflow, techMapping, fabric, routes, resourceUses, physicalTagSegments,
      selectedGraphs);
  if (!inventory)
    return inventory.takeError();

  struct OwnerInventory final {
    ::loom::fabric::FabricFifoOccurrenceRef owner;
    std::map<std::string, MappingStaticQueueClass> queueClasses;
    std::map<std::string, ::loom::fabric::FabricPhysicalTraversalRef>
        routeAnchors;
    std::uint64_t selectedChannelCount = 0;
    bool simpleTopology = true;
  };
  const auto ownerKey = [](const ::loom::fabric::FabricFifoOccurrenceRef owner) {
    const std::vector<std::uint8_t> bytes =
        ::loom::fabric::canonicalFabricBytes(owner);
    return std::string(reinterpret_cast<const char *>(bytes.data()),
                       bytes.size());
  };
  std::map<std::string, OwnerInventory> ownerByKey;
  if (*inventory)
    for (const SpatialChannelStorageChain &channel : (*inventory)->channels) {
      std::set<std::string> channelOwners;
      for (const SpatialChannelStorageStop &stop : channel.stops) {
        const std::string key = ownerKey(stop.node.owner);
        auto [position, inserted] = ownerByKey.try_emplace(
            key, OwnerInventory{stop.node.owner, {}, {}, 0, true});
        if (!inserted && position->second.owner != stop.node.owner)
          return invalid("FIFO capacity owner key collision");
        if (!channelOwners.insert(key).second)
          position->second.simpleTopology = false;
        else if (position->second.selectedChannelCount ==
                 std::numeric_limits<std::uint64_t>::max())
          return invalid("FIFO selected channel count exceeds u64");
        else
          ++position->second.selectedChannelCount;
        position->second.queueClasses.emplace(staticWaitNodeKey(stop.node),
                                              stop.node.queueClass);
        const std::vector<std::uint8_t> anchorBytes =
            ::loom::fabric::canonicalFabricBytes(stop.traversal);
        position->second.routeAnchors.emplace(
            std::string(reinterpret_cast<const char *>(anchorBytes.data()),
                        anchorBytes.size()),
            stop.traversal);
      }
    }
  std::vector<OwnerInventory> owners;
  std::map<std::string, std::size_t> ownerOrdinals;
  owners.reserve(ownerByKey.size());
  for (auto &[key, owner] : ownerByKey) {
    ownerOrdinals.emplace(key, owners.size());
    owners.push_back(std::move(owner));
  }
  const std::size_t ownerCount = owners.size();

  auto actorGraph = buildActorDependencyGraph(dataflow, techMapping.covers());
  if (!actorGraph)
    return actorGraph.takeError();
  std::vector<bool> established(ownerCount, actorGraph->postInitializationAcyclic);
  for (std::size_t ordinal = 0; ordinal != ownerCount; ++ordinal)
    established[ordinal] =
        established[ordinal] && owners[ordinal].simpleTopology &&
        owners[ordinal].selectedChannelCount == 1 &&
        owners[ordinal].queueClasses.size() == 1;

  // Initialized feedback needs the full marked relation: its initial tokens,
  // actor firings, fork/join skew, and route places determine the live shared
  // capacity. Until that relation is built, every touched owner remains typed
  // ProofNotEstablished rather than inheriting a guessed distance-plus-one.
  for (const ActorDependencyGraph::InitializedFeedbackEdge &feedback :
       actorGraph->initializedFeedbackEdges) {
    const SpatialChannelStorageChain *channel = nullptr;
    if (*inventory)
      for (const SpatialChannelStorageChain &candidate :
           (*inventory)->channels)
        if (candidate.producerEndpoint &&
            *candidate.producerEndpoint ==
                ::dataflow::CanonicalGraphProducerEndpointRef(
                    feedback.producer) &&
            candidate.consumerEndpoint &&
            *candidate.consumerEndpoint ==
                ::dataflow::CanonicalGraphConsumerEndpointRef(
                    feedback.consumer)) {
          if (channel)
            return invalid("initialized feedback edge has multiple residual "
                           "route channels");
          channel = &candidate;
        }
    if (!channel)
      continue;
    std::set<std::size_t> pathOwners;
    for (const SpatialChannelStorageStop &stop : channel->stops) {
      const auto owner = ownerOrdinals.find(ownerKey(stop.node.owner));
      if (owner == ownerOrdinals.end())
        return invalid("feedback route lost its FIFO capacity owner");
      pathOwners.insert(owner->second);
    }
    for (std::size_t ordinal : pathOwners)
      established[ordinal] = false;
  }

  std::vector<MappingReconvergentCapacityObligation> obligations;
  obligations.reserve(ownerCount);
  for (std::size_t ordinal = 0; ordinal != ownerCount; ++ordinal) {
    const OwnerInventory &owner = owners[ordinal];
    // Read the shared queue-slot pool through the FIFO contract's typed state
    // and capacity-domain owners. Key-ordered storage is an implementation
    // detail of ResourceContract, not this proof's semantic selector.
    const ::fabric::ResourceContract *contract =
        fabric.resourceContract(
            ::loom::fabric::FabricInventoryOwnerRef::of(owner.owner));
    std::optional<std::uint64_t> pool;
    const ::fabric::StateKey bufferedQueue = ::fabric::fifoResourceState(
        ::fabric::FifoResourceState::BufferedQueue);
    const ::fabric::CapacityDimensionKey queueSlot =
        ::fabric::fifoBufferedCapacity(
            ::fabric::FifoBufferedCapacity::QueueSlot);
    if (contract && contract->stateCount() > bufferedQueue.ordinal()) {
      const auto dimensions = contract->capacityDimensions(bufferedQueue);
      if (dimensions.size() > queueSlot.ordinal())
        pool = dimensions[queueSlot.ordinal()].capacity.value();
    }
    const bool proven = established[ordinal] && pool.has_value();
    const std::optional<std::uint64_t> minimum =
        proven ? std::optional<std::uint64_t>(1) : std::nullopt;
    std::vector<MappingStaticQueueClass> queueClasses;
    queueClasses.reserve(owner.queueClasses.size());
    for (const auto &[key, queueClass] : owner.queueClasses)
      queueClasses.push_back(queueClass);
    std::vector<::loom::fabric::FabricPhysicalTraversalRef> routeAnchors;
    routeAnchors.reserve(owner.routeAnchors.size());
    for (const auto &[key, anchor] : owner.routeAnchors)
      routeAnchors.push_back(anchor);
    MappingReconvergentCapacityObligation obligation{
        owner.owner,
        std::move(queueClasses),
        std::move(routeAnchors),
        pool.value_or(0),
        minimum,
        proven ? MappingReconvergentCapacityProofKind::Proven
               : MappingReconvergentCapacityProofKind::ProofNotEstablished};
    obligations.push_back(std::move(obligation));
  }
  return obligations;
}

llvm::Expected<MappingProgressProjection> projectSpatialMappingProgress(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> computeBindings,
    llvm::ArrayRef<SpatialRegisterFifoTransferView> registerFifoTransfers,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments,
    llvm::ArrayRef<SpatialPeOperandQueueMatchGroupView> operandQueueGroups,
    llvm::ArrayRef<::dataflow::GraphRef> selectedGraphs) {
  std::set<std::uint64_t> selectedGraphEntities;
  for (const ::dataflow::GraphRef graph : selectedGraphs) {
    if (!llvm::is_contained(techMapping.covers(), graph))
      return invalid("selected progress graph is absent from TechMapping");
    if (!selectedGraphEntities.insert(graph.entity.value()).second)
      return invalid("selected progress graph inventory contains a duplicate");
  }
  auto basis = deriveMappingDataflowProgressBasis(dataflow, selectedGraphs);
  if (!basis)
    return basis.takeError();
  MappingProgressProjection result;
  result.basis = *basis;
  result.routeObligations.push_back(
      projectSpatialFiniteBufferRecurrence(routes));
  auto bufferDependencyEdges = projectSpatialBufferDependencyEdges(
      dataflow, techMapping, fabric, routes, resourceUses, physicalTagSegments,
      operandQueueGroups, selectedGraphs);
  if (!bufferDependencyEdges)
    return bufferDependencyEdges.takeError();
  result.bufferDependencyEdges = std::move(*bufferDependencyEdges);
  auto capacityObligations = deriveMappingReconvergentCapacityProof(
      dataflow, techMapping, fabric, routes, resourceUses, physicalTagSegments,
      selectedGraphs);
  if (!capacityObligations)
    return capacityObligations.takeError();
  result.reconvergentCapacityObligations = std::move(*capacityObligations);
  auto temporalDispatchDomains =
      deriveSpatialTemporalPeDispatchDomains(fabric, computeBindings);
  if (!temporalDispatchDomains)
    return temporalDispatchDomains.takeError();
  std::set<std::uint64_t> dispatchedRealizations;
  for (const SpatialTemporalPeDispatchDomainView &domain :
       *temporalDispatchDomains) {
    if (domain.candidates.empty() ||
        domain.resetPosition >= domain.candidates.size())
      return invalid("temporal dispatch progress domain is malformed");
    for (const SpatialTemporalPeDispatchCandidateView &candidate :
         domain.candidates)
      if (!dispatchedRealizations.insert(candidate.realization).second)
        return invalid("temporal compute realization belongs to multiple fair "
                       "dispatch domains");
  }
  auto dependencies =
      deriveSpatialRouteProgressDependencies(dataflow, techMapping);
  if (!dependencies)
    return dependencies.takeError();
  const auto nets = techMapping.residualLogicalNets();
  for (const SpatialRouteProgressDependency &dependency : *dependencies) {
    if (dependency.logicalNetOrdinal >= nets.size())
      return invalid("route progress dependency names an absent logical net");
    const TechResidualLogicalNetView &net = nets[dependency.logicalNetOrdinal];
    auto graph = dataflow.graphOf(net.producer);
    if (!graph)
      return graph.takeError();
    if (selectedGraphEntities.count(graph->entity.value()) == 0)
      continue;
    const auto *external = std::get_if<SpatialRouteExternalSinkPrerequisite>(
        &dependency.prerequisite);
    if ((external && external->sinkOrdinal >= net.sinks.size()) ||
        dependency.dependentSinkOrdinal >= net.sinks.size())
      return invalid("route progress dependency names an absent logical sink");
    const auto localTransfer =
        llvm::find_if(registerFifoTransfers, [&](const auto &transfer) {
          return transfer.logicalNet == net.producer &&
                 transfer.sink == net.sinks[dependency.dependentSinkOrdinal];
        });
    if (localTransfer != registerFifoTransfers.end()) {
      result.routeObligations.push_back(
          {MappingRouteProgressObligationKind::DurableBoundaryAfterDivergence,
           true});
      continue;
    }
    const auto route = llvm::find_if(routes, [&](const auto &candidate) {
      return candidate.logicalNet == net.producer;
    });
    if (route == routes.end())
      return invalid("route progress dependency has no selected route");
    if (const auto *internal =
            std::get_if<SpatialRouteInternalMemoryConnectionPrerequisite>(
                &dependency.prerequisite)) {
      const auto realizations = techMapping.memoryRealizations();
      if (internal->memoryRealizationOrdinal >= realizations.size() ||
          internal->internalEdgeOrdinal >=
              realizations[internal->memoryRealizationOrdinal]
                  .internalEdges.size() ||
          realizations[internal->memoryRealizationOrdinal]
                  .internalEdges[internal->internalEdgeOrdinal]
                  .producer != net.producer)
        return invalid(
            "route progress dependency names an absent internal connection");
    }
    const auto prerequisite =
        external ? llvm::find_if(route->sinks,
                                 [&](const auto &sink) {
                                   return sink.sink ==
                                          net.sinks[external->sinkOrdinal];
                                 })
                 : route->sinks.end();
    const auto dependent = llvm::find_if(route->sinks, [&](const auto &sink) {
      return sink.sink == net.sinks[dependency.dependentSinkOrdinal];
    });
    if ((external && prerequisite == route->sinks.end()) ||
        dependent == route->sinks.end())
      return invalid("selected route omits a progress dependency sink");
    SpatialRouteProgressPrerequisite routedPrerequisite =
        dependency.prerequisite;
    if (external)
      routedPrerequisite =
          SpatialRouteExternalSinkPrerequisite{static_cast<std::uint64_t>(
              std::distance(route->sinks.begin(), prerequisite))};
    auto durable = dependentBranchHasDurableBoundary(
        techMapping, fabric, computeBindings, *route, routedPrerequisite,
        std::distance(route->sinks.begin(), dependent));
    if (!durable)
      return durable.takeError();
    result.routeObligations.push_back(
        {MappingRouteProgressObligationKind::DurableBoundaryAfterDivergence,
         *durable});
  }
  return result;
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

llvm::Expected<MappingProgressClosure> deriveSpatialMappingProgressClosure(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> computeBindings,
    llvm::ArrayRef<SpatialRegisterFifoTransferView> registerFifoTransfers,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments,
    llvm::ArrayRef<SpatialPeOperandQueueMatchGroupView> operandQueueGroups) {
  if (!operandQueueGroups.empty()) {
    auto feedback = deriveSpatialPeOperandProgressFeedback(
        dataflow, techMapping, operandQueueGroups);
    if (!feedback)
      return feedback.takeError();
    mapping_debug::emit(
        mapping_debug::Level::Decision, mapping_debug::Stage::SpatialPnr,
        mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
          fields["context_kind"] = "temporal_operand_queue_progress";
          fields["group_count"] = feedback->groupCount;
          fields["potentially_blocking_group_count"] =
              feedback->potentiallyBlockingGroupCount;
          fields["pairing_opportunity_count"] =
              feedback->pairingOpportunityCount;
          fields["distinct_ingress_count"] = feedback->distinctIngressCount;
          fields["shared_ingress_count"] = feedback->sharedIngressCount;
          fields["shared_ingress_pressure"] = feedback->sharedIngressPressure;
          fields["pairing_key_count"] = feedback->pairingKeyCount;
          fields["distinct_pairing_key_count"] =
              feedback->distinctPairingKeyCount;
          fields["status"] = static_cast<std::uint64_t>(feedback->status);
          fields["support"] = static_cast<std::uint64_t>(feedback->support);
        });
  }
  // The queue/pairing projection is a ranking and runtime-witness input at
  // this boundary. Analytic risk is deliberately not a Mapping gate; only an
  // exact queue-level closed-wait witness may change closure status.
  auto projection = projectSpatialMappingProgress(
      dataflow, techMapping, fabric, computeBindings, registerFifoTransfers,
      routes, resourceUses, physicalTagSegments, operandQueueGroups,
      techMapping.covers());
  if (!projection)
    return projection.takeError();
  auto model = freezeMappingProgressModel(dataflow, {});
  if (!model)
    return model.takeError();
  return deriveMappingProgressClosure(*model, *projection);
}

} // namespace loom::mapping
