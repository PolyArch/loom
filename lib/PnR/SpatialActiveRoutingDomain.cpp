#include "SpatialActiveRoutingDomain.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <system_error>
#include <vector>

namespace loom::pnr {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_active_routing_domain_invalid: " + message);
}

void incrementWork(std::uint64_t &work, std::uint64_t amount = 1) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - work)
    work = std::numeric_limits<std::uint64_t>::max();
  else
    work += amount;
}

llvm::Error appendBindingEndpoints(const FrozenSpatialTerminalBinding &binding,
                                   const FrozenSpatialPortIndex &ports,
                                   std::vector<PnrIndex> &endpoints,
                                   std::uint64_t &work) {
  const auto appendOptions = [&](PnrIndex offset,
                                 PnrIndex count) -> llvm::Error {
    const auto options = ports.attachmentOptions();
    if (offset > options.size() || count > options.size() - offset)
      return invalid("terminal attachment slice is out of range");
    for (const FrozenSpatialAttachmentOption &option :
         options.slice(offset, count)) {
      endpoints.push_back(option.endpoint);
      incrementWork(work);
    }
    return llvm::Error::success();
  };

  switch (binding.kind) {
  case FrozenSpatialTerminalBindingKind::PortDemand: {
    const auto demands = ports.portDemands();
    const auto domains = ports.placementDomains();
    if (binding.index >= demands.size())
      return invalid("terminal names an unknown PortDemand");
    const FrozenSpatialPortDemand &demand = demands[binding.index];
    if (demand.placementDomainOffset > domains.size() ||
        demand.placementDomainCount >
            domains.size() - demand.placementDomainOffset)
      return invalid("PortDemand placement slice is out of range");
    for (const FrozenSpatialPortPlacementDomain &domain : domains.slice(
             demand.placementDomainOffset, demand.placementDomainCount)) {
      incrementWork(work);
      if (llvm::Error error = appendOptions(domain.attachmentOptionOffset,
                                            domain.attachmentOptionCount))
        return error;
    }
    break;
  }
  case FrozenSpatialTerminalBindingKind::GraphBoundary: {
    const auto boundaries = ports.graphBoundaries();
    if (binding.index >= boundaries.size())
      return invalid("terminal names an unknown graph boundary");
    const FrozenSpatialGraphBoundary &boundary = boundaries[binding.index];
    if (llvm::Error error = appendOptions(boundary.attachmentOptionOffset,
                                          boundary.attachmentOptionCount))
      return error;
    break;
  }
  }
  llvm::sort(endpoints);
  endpoints.erase(std::unique(endpoints.begin(), endpoints.end()),
                  endpoints.end());
  return llvm::Error::success();
}

} // namespace

llvm::Expected<FrozenSpatialActiveRoutingDomain>
buildFrozenSpatialActiveRoutingDomain(
    const FrozenSpatialTransferIndex &transfers,
    const FrozenSpatialLocalTransferIndex &localTransfers,
    const FrozenSpatialPortIndex &ports,
    const FrozenSpatialRoutingGraph &routing) {
  FrozenSpatialActiveRoutingDomain result;
  const auto endpoints = routing.routingEndpoints();
  const auto traversals = routing.traversals();
  const auto arcs = routing.routingArcs();
  const auto arcSources = routing.arcSources();
  const auto adjacency = routing.adjacencyOffsets();
  const auto reverseAdjacency = routing.reverseAdjacencyOffsets();
  const auto reverseArcs = routing.reverseArcOrdinals();
  if (arcSources.size() != arcs.size() ||
      adjacency.size() != endpoints.size() + 1 ||
      reverseAdjacency.size() != endpoints.size() + 1 ||
      reverseArcs.size() != arcs.size())
    return invalid("routing graph shape is inconsistent");

  result.activeEndpoints_.assign(endpoints.size(), 0);
  result.activeTraversals_.assign(traversals.size(), 0);
  result.activeArcs_.assign(arcs.size(), 0);
  result.activeTraversalBits_.assign((traversals.size() + 63) / 64, 0);

  const auto markEndpoint = [&](PnrIndex endpoint) -> llvm::Error {
    if (endpoint >= result.activeEndpoints_.size())
      return invalid("active endpoint is out of range");
    if (!result.activeEndpoints_[endpoint]) {
      result.activeEndpoints_[endpoint] = 1;
      ++result.activeEndpointCount_;
    }
    return llvm::Error::success();
  };
  const auto markTraversal = [&](PnrIndex traversal) -> llvm::Error {
    if (traversal >= result.activeTraversals_.size())
      return invalid("active traversal is out of range");
    if (!result.activeTraversals_[traversal]) {
      result.activeTraversals_[traversal] = 1;
      result.activeTraversalBits_[traversal / 64] |= std::uint64_t{1}
                                                     << (traversal % 64);
      ++result.activeTraversalCount_;
    }
    return llvm::Error::success();
  };
  const auto markArc = [&](PnrIndex arc) -> llvm::Error {
    if (arc >= result.activeArcs_.size() ||
        arcSources[arc] >= endpoints.size() ||
        arcs[arc].target >= endpoints.size())
      return invalid("active routing arc is out of range");
    if (llvm::Error error = markEndpoint(arcSources[arc]))
      return error;
    if (llvm::Error error = markEndpoint(arcs[arc].target))
      return error;
    if (llvm::Error error = markTraversal(arcs[arc].traversal))
      return error;
    if (!result.activeArcs_[arc]) {
      result.activeArcs_[arc] = 1;
      ++result.activeArcCount_;
    }
    return llvm::Error::success();
  };
  const auto markExplicitTraversal = [&](PnrIndex traversal) -> llvm::Error {
    if (llvm::Error error = markTraversal(traversal))
      return error;
    const auto offsets = routing.traversalArcOffsets();
    const auto traversalArcs = routing.traversalArcs();
    if (traversal + 1 >= offsets.size() ||
        offsets[traversal] > offsets[traversal + 1] ||
        offsets[traversal + 1] > traversalArcs.size())
      return invalid("explicit traversal arc slice is out of range");
    for (PnrIndex offset = offsets[traversal]; offset < offsets[traversal + 1];
         ++offset) {
      incrementWork(result.deterministicWork_);
      if (llvm::Error error = markArc(traversalArcs[offset]))
        return error;
    }
    return llvm::Error::success();
  };

  for (const FrozenSpatialAttachmentOption &option :
       ports.attachmentOptions()) {
    incrementWork(result.deterministicWork_);
    if (llvm::Error error = markEndpoint(option.endpoint))
      return std::move(error);
    if (option.localTraversal)
      if (llvm::Error error = markExplicitTraversal(*option.localTraversal))
        return std::move(error);
  }
  for (const FrozenSpatialRegisterFifoTransferOption &option :
       localTransfers.options()) {
    incrementWork(result.deterministicWork_);
    if (llvm::Error error = markExplicitTraversal(option.writeTraversal))
      return std::move(error);
    if (llvm::Error error = markExplicitTraversal(option.readTraversal))
      return std::move(error);
  }

  const auto nets = transfers.logicalNets();
  const auto sourceBindings = transfers.logicalNetSourceBindings();
  const auto sinkBindings = transfers.logicalNetSinkBindings();
  if (sourceBindings.size() != nets.size())
    return invalid("logical-net source binding table is incomplete");
  std::vector<std::uint8_t> reverseReachable(endpoints.size(), 0);
  std::vector<std::uint8_t> forwardReachable(endpoints.size(), 0);
  std::vector<PnrIndex> worklist;
  worklist.reserve(endpoints.size());
  std::vector<PnrIndex> sources;
  std::vector<PnrIndex> targets;

  for (auto [logicalNet, net] : llvm::enumerate(nets)) {
    incrementWork(result.deterministicWork_);
    if (net.sinkOffset > sinkBindings.size() ||
        net.sinkCount > sinkBindings.size() - net.sinkOffset)
      return invalid("logical-net sink binding slice is out of range");
    sources.clear();
    if (llvm::Error error =
            appendBindingEndpoints(sourceBindings[logicalNet], ports, sources,
                                   result.deterministicWork_))
      return std::move(error);
    targets.clear();
    for (const FrozenSpatialTerminalBinding &binding :
         sinkBindings.slice(net.sinkOffset, net.sinkCount))
      if (llvm::Error error = appendBindingEndpoints(binding, ports, targets,
                                                     result.deterministicWork_))
        return std::move(error);
    llvm::sort(targets);
    targets.erase(std::unique(targets.begin(), targets.end()), targets.end());

    std::fill(reverseReachable.begin(), reverseReachable.end(), 0);
    worklist.clear();
    for (PnrIndex endpoint : targets) {
      if (endpoint >= endpoints.size())
        return invalid("logical-net sink endpoint is out of range");
      if (llvm::Error error = markEndpoint(endpoint))
        return std::move(error);
      if (!reverseReachable[endpoint]) {
        reverseReachable[endpoint] = 1;
        worklist.push_back(endpoint);
      }
    }
    for (std::size_t cursor = 0; cursor < worklist.size(); ++cursor) {
      const PnrIndex endpoint = worklist[cursor];
      for (PnrIndex offset = reverseAdjacency[endpoint];
           offset < reverseAdjacency[endpoint + 1]; ++offset) {
        incrementWork(result.deterministicWork_);
        const PnrIndex arc = reverseArcs[offset];
        if (arc >= arcs.size() || arcSources[arc] >= endpoints.size())
          return invalid("reverse routing incidence is out of range");
        if (arcs[arc].payloadCapacityBits < net.payloadWidthBits ||
            reverseReachable[arcSources[arc]])
          continue;
        reverseReachable[arcSources[arc]] = 1;
        worklist.push_back(arcSources[arc]);
      }
    }

    std::fill(forwardReachable.begin(), forwardReachable.end(), 0);
    worklist.clear();
    for (PnrIndex endpoint : sources) {
      if (endpoint >= endpoints.size())
        return invalid("logical-net source endpoint is out of range");
      if (llvm::Error error = markEndpoint(endpoint))
        return std::move(error);
      if (reverseReachable[endpoint] && !forwardReachable[endpoint]) {
        forwardReachable[endpoint] = 1;
        worklist.push_back(endpoint);
      }
    }
    for (std::size_t cursor = 0; cursor < worklist.size(); ++cursor) {
      const PnrIndex endpoint = worklist[cursor];
      for (PnrIndex arc = adjacency[endpoint]; arc < adjacency[endpoint + 1];
           ++arc) {
        incrementWork(result.deterministicWork_);
        if (arc >= arcs.size() || arcs[arc].target >= endpoints.size())
          return invalid("forward routing incidence is out of range");
        if (arcs[arc].payloadCapacityBits < net.payloadWidthBits ||
            !reverseReachable[arcs[arc].target])
          continue;
        if (llvm::Error error = markArc(arc))
          return std::move(error);
        if (!forwardReachable[arcs[arc].target]) {
          forwardReachable[arcs[arc].target] = 1;
          worklist.push_back(arcs[arc].target);
        }
      }
    }
  }
  return result;
}

} // namespace loom::pnr
