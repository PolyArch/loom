#include "CGRATransportPlan.h"

#include "Fabric/IR/FifoResourceContract.h"
#include "Fabric/IR/TemporalPeResourceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include <limits>
#include <map>
#include <system_error>
#include <utility>
#include <variant>

namespace loom::sim::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

template <typename Size>
llvm::Expected<std::uint32_t> checkedU32(Size value, llvm::StringRef label) {
  if (value > std::numeric_limits<std::uint32_t>::max())
    return llvm::createStringError(
        std::make_error_code(std::errc::value_too_large),
        label + " exceeds u32");
  return static_cast<std::uint32_t>(value);
}

llvm::Expected<::dataflow::GraphRef>
resolveGraphOf(const ::dataflow::CanonicalDataflowProgramView &dataflow,
               const ::dataflow::CanonicalGraphProducerEndpointRef &producer) {
  if (const auto *result =
          std::get_if<::dataflow::ActorTokenResultRef>(&producer)) {
    auto actor = dataflow.resolve(result->actor);
    if (!actor)
      return actor.takeError();
    return actor->graph;
  }
  const auto &ingress = std::get<::dataflow::GraphIngressTokenRef>(producer);
  return std::visit([](const auto &typed) { return typed.graph; }, ingress);
}

using RefBytes = std::vector<std::uint8_t>;

RefBytes bytes(const ::loom::fabric::FabricPhysicalTraversalRef &reference) {
  return ::loom::fabric::canonicalFabricBytes(reference);
}

void collect(
    const std::optional<::loom::fabric::FabricPhysicalTraversalRef> &reference,
    std::map<RefBytes, ::loom::fabric::FabricPhysicalTraversalRef> &selected) {
  if (reference)
    selected.try_emplace(bytes(*reference), *reference);
}

llvm::Expected<std::pair<CgraTraversalStorageKind, std::uint32_t>>
storageContract(const ::loom::fabric::FabricArtifactView &fabric,
                const ::loom::fabric::FabricPhysicalTraversalRef &reference) {
  if (const auto *fifo =
          std::get_if<::loom::fabric::FabricFifoTraversalPayload>(
              &reference.payload)) {
    if (fifo->mode == ::loom::fabric::FabricFifoTraversalMode::Bypass)
      return std::make_pair(CgraTraversalStorageKind::None, 0U);
    const auto owner = ::loom::fabric::FabricInventoryOwnerRef::of(fifo->owner);
    const ::fabric::ResourceContract *contract = fabric.resourceContract(owner);
    if (!contract || static_cast<std::uint32_t>(
                         ::fabric::FifoResourceState::BufferedQueue) >=
                         contract->stateCount())
      return invalid("selected buffered FIFO has no ResourceContract");
    const auto dimensions = contract->capacityDimensions(
        ::fabric::StateKey(static_cast<std::uint32_t>(
            ::fabric::FifoResourceState::BufferedQueue)));
    const auto queue =
        static_cast<std::uint32_t>(::fabric::FifoBufferedCapacity::QueueSlot);
    if (queue >= dimensions.size() || dimensions[queue].capacity.value() == 0)
      return invalid("selected buffered FIFO has no positive queue capacity");
    return std::make_pair(CgraTraversalStorageKind::BufferedFifo,
                          dimensions[queue].capacity.value());
  }
  if (const auto *registerFifo =
          std::get_if<::loom::fabric::FabricPeRegisterFifoPayload>(
              &reference.payload)) {
    const auto owner =
        ::loom::fabric::FabricInventoryOwnerRef::of(registerFifo->owner);
    const ::fabric::ResourceContract *contract = fabric.resourceContract(owner);
    const std::uint64_t count = fabric.inventorySize(
        owner, ::loom::fabric::FabricInventoryKind::RegisterFifo);
    if (!contract || count > std::numeric_limits<std::uint32_t>::max() ||
        registerFifo->registerFifo >= count)
      return invalid("selected register FIFO has no ResourceContract");
    auto state = ::fabric::resolveTemporalPeRegisterFifoState(
        *contract, static_cast<std::uint32_t>(count),
        static_cast<std::uint32_t>(registerFifo->registerFifo));
    if (!state)
      return state.takeError();
    const auto dimensions = contract->capacityDimensions(*state);
    if (dimensions.empty() || dimensions.front().capacity.value() == 0)
      return invalid("selected register FIFO has no positive queue capacity");
    const auto kind =
        registerFifo->role == ::loom::fabric::FabricRegisterFifoPathRole::Write
            ? CgraTraversalStorageKind::RegisterFifoWrite
            : CgraTraversalStorageKind::RegisterFifoRead;
    return std::make_pair(kind, dimensions.front().capacity.value());
  }
  return std::make_pair(CgraTraversalStorageKind::None, 0U);
}

} // namespace

llvm::Expected<CgraTransportPlan> freezeCgraTransportPlan(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::mapping::SpatialMappingView &spatial) {
  std::map<RefBytes, ::loom::fabric::FabricPhysicalTraversalRef> selected;
  for (const auto &route : spatial.routeTrees()) {
    collect(route.localTraversal, selected);
    for (const auto &node : route.nodes)
      collect(node.incomingTraversal, selected);
    for (const auto &sink : route.sinks)
      collect(sink.localTraversal, selected);
  }

  std::map<RefBytes, const ::loom::fabric::FabricPhysicalTraversalView *>
      physical;
  for (const auto &traversal : fabric.physicalTraversals())
    if (!physical.try_emplace(bytes(traversal.reference), &traversal).second)
      return invalid("Fabric contains duplicate physical traversal references");

  CgraTransportPlan result;
  std::map<RefBytes, std::uint64_t> selectedOrdinals;
  result.traversals.reserve(selected.size());
  for (const auto &[key, reference] : selected) {
    auto found = physical.find(key);
    if (found == physical.end())
      return invalid("selected RouteTree traversal is absent from Fabric");
    auto storage = storageContract(fabric, reference);
    if (!storage)
      return storage.takeError();
    const std::uint64_t useOffset = result.traversalUses.size();
    if (found->second->impliedUses.size() >
        std::numeric_limits<std::uint32_t>::max())
      return invalid("selected traversal implied-use count exceeds u32");
    for (const auto &use : found->second->impliedUses)
      result.traversalUses.push_back({use.pattern, use.activationGroup});
    const std::uint64_t ordinal = result.traversals.size();
    selectedOrdinals.emplace(key, ordinal);
    result.traversals.push_back(
        {reference, reference.kind(), storage->first, storage->second,
         useOffset,
         static_cast<std::uint32_t>(found->second->impliedUses.size())});
  }

  const auto ordinalOf =
      [&](const std::optional<::loom::fabric::FabricPhysicalTraversalRef>
              &reference) -> llvm::Expected<std::uint64_t> {
    if (!reference)
      return invalidCgraTransportOrdinal;
    auto found = selectedOrdinals.find(bytes(*reference));
    if (found == selectedOrdinals.end())
      return invalid("RouteTree traversal is absent from selected catalog");
    return found->second;
  };

  result.routes.reserve(spatial.routeTrees().size());
  for (const auto &route : spatial.routeTrees()) {
    auto graph = resolveGraphOf(dataflow, route.logicalNet);
    if (!graph)
      return graph.takeError();
    auto sourceTraversal = ordinalOf(route.localTraversal);
    if (!sourceTraversal)
      return sourceTraversal.takeError();
    auto nodeCount = checkedU32(route.nodes.size(), "CGRA route-node count");
    if (!nodeCount)
      return nodeCount.takeError();
    auto sinkCount = checkedU32(route.sinks.size(), "CGRA route-sink count");
    if (!sinkCount)
      return sinkCount.takeError();
    const std::uint64_t nodeOffset = result.routeNodes.size();
    const std::uint64_t sinkOffset = result.routeSinks.size();
    for (const auto &node : route.nodes) {
      auto traversal = ordinalOf(node.incomingTraversal);
      if (!traversal)
        return traversal.takeError();
      std::uint32_t parent = std::numeric_limits<std::uint32_t>::max();
      if (node.parentOrdinal) {
        auto checkedParent =
            checkedU32(*node.parentOrdinal, "CGRA route parent ordinal");
        if (!checkedParent)
          return checkedParent.takeError();
        parent = *checkedParent;
      }
      result.routeNodes.push_back({parent, *traversal});
    }
    for (const auto &sink : route.sinks) {
      auto node = checkedU32(sink.nodeOrdinal, "CGRA route sink node ordinal");
      if (!node)
        return node.takeError();
      auto traversal = ordinalOf(sink.localTraversal);
      if (!traversal)
        return traversal.takeError();
      result.routeSinks.push_back({sink.sink, *node, *traversal});
    }
    result.routes.push_back({route.logicalNet, *graph, *sourceTraversal,
                             nodeOffset, *nodeCount, sinkOffset, *sinkCount});
  }
  return result;
}

} // namespace loom::sim::detail
