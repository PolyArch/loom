#include "MappingProgressInternal.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/OperationSchema.h"
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

namespace loom::mapping::progress_detail {

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
    key.push_back(static_cast<char>(
        storage->queueClass.kind == MappingStaticQueueClassKind::PhysicalTag
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

bool isBuffered(const std::optional<::loom::fabric::FabricPhysicalTraversalRef>
                    &traversal) {
  if (!traversal)
    return false;
  const auto *fifo = std::get_if<::loom::fabric::FabricFifoTraversalPayload>(
      &traversal->payload);
  return fifo &&
         fifo->mode == ::loom::fabric::FabricFifoTraversalMode::Buffered;
}

} // namespace loom::mapping::progress_detail

namespace loom::mapping {
using namespace progress_detail;
namespace {

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

} // namespace

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
  std::vector<std::uint64_t> capacityComponentRouteAnchorCounts;
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
      std::set<std::string> routeAnchors;
      for (std::uint32_t node : component)
        for (const auto [position, successor] :
             llvm::enumerate(successors[node])) {
          if (!member[successor])
            continue;
          const auto &anchor =
              edges[successorEdges[node][position]].routeAnchor;
          if (!anchor)
            continue;
          const std::vector<std::uint8_t> bytes =
              ::loom::fabric::canonicalFabricBytes(*anchor);
          routeAnchors.insert(std::string(
              reinterpret_cast<const char *>(bytes.data()), bytes.size()));
        }
      if (carriesCapacity) {
        capacityComponents.push_back(component);
        capacityComponentRouteAnchorCounts.push_back(routeAnchors.size());
      } else if (!provenComponent) {
        provenComponent = componentOrdinal;
      }
      if (!carriesCapacity && provenComponent == componentOrdinal) {
        std::vector<MappingStaticWaitNode> witness;
        for (std::uint32_t node : component)
          witness.push_back(nodes[node]);
        return MappingProgressClosure{
            MappingProgressClosureKind::ProvenClosedWaitSet,
            MappingProgressClosureReason::ClosedBufferDependencyCycle,
            {},
            std::move(witness),
            0,
            routeAnchors.size()};
      }
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
             .insert(
                 std::string(reinterpret_cast<const char *>(ownerBytes.data()),
                             ownerBytes.size()))
             .second)
      return invalid("reconvergent capacity repeats a shared FIFO owner");
    if (obligation.queueClasses.empty())
      return invalid("reconvergent capacity owner has no queue class");
    const bool proven =
        obligation.kind == MappingReconvergentCapacityProofKind::Proven;
    if (proven != obligation.minimumLegalCapacity.has_value())
      return invalid("reconvergent capacity proof state and minimum differ");
    std::optional<std::string> previousClass;
    bool global = false;
    for (const MappingStaticQueueClass &queueClass : obligation.queueClasses) {
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
          {},
          *obligation.minimumLegalCapacity - obligation.selectedCapacity,
          obligation.routeAnchors.size()};
  }
  const auto unproven = llvm::find_if(
      projection.reconvergentCapacityObligations,
      [](const MappingReconvergentCapacityObligation &obligation) {
        return obligation.kind ==
               MappingReconvergentCapacityProofKind::ProofNotEstablished;
      });
  if (unproven != projection.reconvergentCapacityObligations.end())
    return MappingProgressClosure{
        MappingProgressClosureKind::ProofNotEstablished,
        MappingProgressClosureReason::ReconvergentCapacityNotEstablished,
        {},
        {},
        0,
        unproven->routeAnchors.size()};
  for (const auto [componentOrdinal, component] :
       llvm::enumerate(capacityComponents)) {
    for (std::uint32_t nodeOrdinal : component) {
      const auto *storage = std::get_if<MappingStorageQueueProgressNode>(
          &bufferNodes[nodeOrdinal]);
      if (!storage)
        continue;
      const auto obligation = llvm::find_if(
          projection.reconvergentCapacityObligations,
          [&](const MappingReconvergentCapacityObligation &candidate) {
            return candidate.owner == storage->owner;
          });
      if (obligation == projection.reconvergentCapacityObligations.end() ||
          obligation->kind != MappingReconvergentCapacityProofKind::Proven)
        return MappingProgressClosure{
            MappingProgressClosureKind::ProofNotEstablished,
            MappingProgressClosureReason::ReconvergentCapacityNotEstablished,
            {},
            {},
            0,
            capacityComponentRouteAnchorCounts[componentOrdinal]};
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
        MappingProgressClosureReason::PossibleWaitCycle,
        std::move(witness),
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

MappingProgressObjectiveProjection
projectMappingProgressObjective(const MappingProgressClosure &closure) {
  MappingProgressObjectiveProjection result;
  result.hardViolationCount =
      closure.kind == MappingProgressClosureKind::ProvenClosedWaitSet ? 1 : 0;
  result.proofDebtWitnessCount =
      closure.kind == MappingProgressClosureKind::ProofNotEstablished ? 1 : 0;
  result.capacityShortfall = closure.capacityShortfall;
  result.routeAnchorCount = closure.routeAnchorCount;
  return result;
}

} // namespace loom::mapping
