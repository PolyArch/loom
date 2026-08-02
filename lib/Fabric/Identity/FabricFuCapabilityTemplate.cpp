#include "Fabric/Identity/FabricFuCapabilityTemplate.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace loom::fabric {
namespace {

llvm::Error malformed(const llvm::Twine &message) {
  return makeFabricRefError(FabricRefErrorKind::MalformedSyntax, message);
}

std::vector<std::uint8_t>
endpointBytes(const FabricFuCapabilityTemplateEndpointRef &endpoint) {
  FabricByteWriter writer;
  writer.tag(static_cast<std::uint32_t>(endpoint.kind()));
  std::visit([&](const auto &value) { encodeFabricRef(writer, value); },
             endpoint.payload);
  return writer.take();
}

std::vector<std::uint8_t>
edgeBytes(const FabricFuCapabilityTemplateEdge &edge) {
  std::vector<std::uint8_t> bytes = endpointBytes(edge.source);
  std::vector<std::uint8_t> destination = endpointBytes(edge.destination);
  bytes.insert(bytes.end(), destination.begin(), destination.end());
  return bytes;
}

template <typename Value, typename Key>
bool canonicalLess(const std::pair<Key, Value> &lhs,
                   const std::pair<Key, Value> &rhs) {
  return lhs.first < rhs.first;
}

FabricFuTemplateRef
endpointOwner(const FabricFuCapabilityTemplateEndpointRef &endpoint) {
  if (endpoint.kind() == FabricFuCapabilityTemplateEndpointKind::BoundaryPort)
    return std::get<FabricFuTemplatePortRef>(endpoint.payload).fu;
  return std::get<FabricFuNodePortRef>(endpoint.payload).node.fu;
}

FabricPortDirection
endpointDirection(const FabricFuCapabilityTemplateEndpointRef &endpoint) {
  if (endpoint.kind() == FabricFuCapabilityTemplateEndpointKind::BoundaryPort)
    return std::get<FabricFuTemplatePortRef>(endpoint.payload).direction;
  return std::get<FabricFuNodePortRef>(endpoint.payload).direction;
}

const FabricFuTemplateNodeRef *
endpointNode(const FabricFuCapabilityTemplateEndpointRef &endpoint) {
  if (endpoint.kind() == FabricFuCapabilityTemplateEndpointKind::BoundaryPort)
    return nullptr;
  return &std::get<FabricFuNodePortRef>(endpoint.payload).node;
}

bool isValidSource(const FabricFuCapabilityTemplateEndpointRef &endpoint) {
  if (endpoint.kind() == FabricFuCapabilityTemplateEndpointKind::BoundaryPort)
    return endpointDirection(endpoint) == FabricPortDirection::Input;
  return endpointDirection(endpoint) == FabricPortDirection::Output;
}

bool isValidDestination(const FabricFuCapabilityTemplateEndpointRef &endpoint) {
  if (endpoint.kind() == FabricFuCapabilityTemplateEndpointKind::BoundaryPort)
    return endpointDirection(endpoint) == FabricPortDirection::Output;
  return endpointDirection(endpoint) == FabricPortDirection::Input;
}

bool containsNode(llvm::ArrayRef<FabricFuTemplateNodeRef> nodes,
                  const FabricFuTemplateNodeRef &node) {
  return std::find(nodes.begin(), nodes.end(), node) != nodes.end();
}

void encodeEndpoint(FabricByteWriter &writer,
                    const FabricFuCapabilityTemplateEndpointRef &endpoint) {
  writer.tag(static_cast<std::uint32_t>(endpoint.kind()));
  std::visit([&](const auto &value) { encodeFabricRef(writer, value); },
             endpoint.payload);
}

std::vector<std::uint8_t>
encodeNormalizedRecord(const FabricFuCapabilityTemplateRecord &record) {
  FabricByteWriter writer;
  writer.field(record.activeNodes.size());
  for (const FabricFuTemplateNodeRef &node : record.activeNodes)
    encodeFabricRef(writer, node);
  writer.field(record.activeEdges.size());
  for (const FabricFuCapabilityTemplateEdge &edge : record.activeEdges) {
    encodeEndpoint(writer, edge.source);
    encodeEndpoint(writer, edge.destination);
  }
  return writer.take();
}

llvm::Expected<FabricFuCapabilityTemplateEndpointRef>
decodeEndpoint(FabricByteReader &reader) {
  llvm::Expected<std::uint32_t> tag =
      readFabricClosedTag(reader, 2, "FU capability-template endpoint kind");
  if (!tag)
    return tag.takeError();
  if (*tag == static_cast<std::uint32_t>(
                  FabricFuCapabilityTemplateEndpointKind::BoundaryPort)) {
    FabricFuTemplatePortRef port;
    if (llvm::Error error = decodeFabricRefInto(reader, port))
      return std::move(error);
    return FabricFuCapabilityTemplateEndpointRef::boundaryPort(port);
  }

  FabricFuNodePortRef port;
  if (llvm::Error error = decodeFabricRefInto(reader, port))
    return std::move(error);
  return FabricFuCapabilityTemplateEndpointRef::nodePort(port);
}

} // namespace

llvm::Expected<FabricFuCapabilityTemplateRecord>
normalizeFabricFuCapabilityTemplateRecord(
    FabricFuCapabilityTemplateRecord record) {
  if (record.activeNodes.empty())
    return malformed("an FU capability template requires an active node");

  std::vector<std::pair<std::vector<std::uint8_t>, FabricFuTemplateNodeRef>>
      nodes;
  nodes.reserve(record.activeNodes.size());
  for (const FabricFuTemplateNodeRef &node : record.activeNodes)
    nodes.emplace_back(canonicalFabricBytes(node), node);
  std::sort(nodes.begin(), nodes.end(),
            canonicalLess<FabricFuTemplateNodeRef, std::vector<std::uint8_t>>);

  const FabricFuTemplateRef owner = nodes.front().second.fu;
  record.activeNodes.clear();
  record.activeNodes.reserve(nodes.size());
  for (std::size_t index = 0; index < nodes.size(); ++index) {
    if (nodes[index].second.fu != owner)
      return makeFabricRefError(
          FabricRefErrorKind::WrongOwner,
          "active nodes of one FU capability template have different owners");
    if (index != 0 && nodes[index - 1].first == nodes[index].first)
      return malformed("duplicate active node in FU capability template");
    record.activeNodes.push_back(nodes[index].second);
  }

  std::vector<
      std::pair<std::vector<std::uint8_t>, FabricFuCapabilityTemplateEdge>>
      edges;
  edges.reserve(record.activeEdges.size());
  for (const FabricFuCapabilityTemplateEdge &edge : record.activeEdges) {
    if (endpointOwner(edge.source) != owner ||
        endpointOwner(edge.destination) != owner)
      return makeFabricRefError(
          FabricRefErrorKind::WrongOwner,
          "an FU capability-template edge names a different FU definition");
    if (!isValidSource(edge.source) || !isValidDestination(edge.destination))
      return malformed(
          "an FU capability-template edge has an invalid directed endpoint");
    if (const FabricFuTemplateNodeRef *node = endpointNode(edge.source);
        node && !containsNode(record.activeNodes, *node))
      return malformed("an edge source names an inactive FU node");
    if (const FabricFuTemplateNodeRef *node = endpointNode(edge.destination);
        node && !containsNode(record.activeNodes, *node))
      return malformed("an edge destination names an inactive FU node");
    edges.emplace_back(edgeBytes(edge), edge);
  }
  std::sort(
      edges.begin(), edges.end(),
      canonicalLess<FabricFuCapabilityTemplateEdge, std::vector<std::uint8_t>>);

  record.activeEdges.clear();
  record.activeEdges.reserve(edges.size());
  for (std::size_t index = 0; index < edges.size(); ++index) {
    if (index != 0 && edges[index - 1].first == edges[index].first)
      return malformed("duplicate active edge in FU capability template");
    record.activeEdges.push_back(std::move(edges[index].second));
  }
  return record;
}

llvm::Expected<std::vector<std::uint8_t>>
canonicalFabricFuCapabilityTemplateBytes(
    const FabricFuCapabilityTemplateRecord &record) {
  llvm::Expected<FabricFuCapabilityTemplateRecord> normalized =
      normalizeFabricFuCapabilityTemplateRecord(record);
  if (!normalized)
    return normalized.takeError();
  return encodeNormalizedRecord(*normalized);
}

llvm::Expected<FabricFuCapabilityTemplateRecord>
decodeFabricFuCapabilityTemplateRecord(llvm::ArrayRef<std::uint8_t> bytes) {
  FabricByteReader reader(bytes);
  llvm::Expected<std::uint64_t> nodeCount = reader.field();
  if (!nodeCount)
    return nodeCount.takeError();

  FabricFuCapabilityTemplateRecord decoded;
  for (std::uint64_t index = 0; index < *nodeCount; ++index) {
    FabricFuTemplateNodeRef node;
    if (llvm::Error error = decodeFabricRefInto(reader, node))
      return std::move(error);
    decoded.activeNodes.push_back(node);
  }

  llvm::Expected<std::uint64_t> edgeCount = reader.field();
  if (!edgeCount)
    return edgeCount.takeError();
  for (std::uint64_t index = 0; index < *edgeCount; ++index) {
    llvm::Expected<FabricFuCapabilityTemplateEndpointRef> source =
        decodeEndpoint(reader);
    if (!source)
      return source.takeError();
    llvm::Expected<FabricFuCapabilityTemplateEndpointRef> destination =
        decodeEndpoint(reader);
    if (!destination)
      return destination.takeError();
    decoded.activeEdges.push_back(FabricFuCapabilityTemplateEdge{
        std::move(*source), std::move(*destination)});
  }
  if (!reader.empty())
    return malformed("trailing FU capability-template bytes");

  llvm::Expected<FabricFuCapabilityTemplateRecord> normalized =
      normalizeFabricFuCapabilityTemplateRecord(decoded);
  if (!normalized)
    return normalized.takeError();
  if (*normalized != decoded)
    return malformed("noncanonical FU capability-template record order");
  return decoded;
}

llvm::Expected<std::vector<FabricFuCapabilityTemplateRecord>>
normalizeFabricFuCapabilityTemplateInventory(
    llvm::ArrayRef<FabricFuCapabilityTemplateRecord> records) {
  using KeyedRecord =
      std::pair<std::vector<std::uint8_t>, FabricFuCapabilityTemplateRecord>;
  std::vector<KeyedRecord> normalized;
  normalized.reserve(records.size());

  for (const FabricFuCapabilityTemplateRecord &record : records) {
    llvm::Expected<FabricFuCapabilityTemplateRecord> value =
        normalizeFabricFuCapabilityTemplateRecord(record);
    if (!value)
      return value.takeError();
    std::vector<std::uint8_t> bytes = encodeNormalizedRecord(*value);
    normalized.emplace_back(std::move(bytes), std::move(*value));
  }

  std::sort(normalized.begin(), normalized.end(),
            canonicalLess<FabricFuCapabilityTemplateRecord,
                          std::vector<std::uint8_t>>);
  std::vector<FabricFuCapabilityTemplateRecord> result;
  result.reserve(normalized.size());
  for (std::size_t index = 0; index < normalized.size(); ++index) {
    if (index != 0 && normalized[index - 1].first == normalized[index].first)
      return malformed("duplicate FU capability-template record");
    if (!result.empty() && result.front().activeNodes.front().fu !=
                               normalized[index].second.activeNodes.front().fu)
      return makeFabricRefError(
          FabricRefErrorKind::WrongOwner,
          "one capability-template inventory has multiple FU owners");
    result.push_back(std::move(normalized[index].second));
  }
  return result;
}

llvm::Expected<std::vector<FabricFuCapabilityTemplateEdge>>
projectFabricFuCapabilityTemplateTerminalEdges(
    const FabricFuCapabilityTemplateRecord &record) {
  auto normalized = normalizeFabricFuCapabilityTemplateRecord(record);
  if (!normalized)
    return normalized.takeError();
  if (*normalized != record)
    return malformed(
        "FU capability-template terminal projection requires canonical input");

  using Endpoint = FabricFuCapabilityTemplateEndpointRef;
  using Edge = FabricFuCapabilityTemplateEdge;
  std::vector<Edge> adjacency = record.activeEdges;

  for (const FabricFuTemplateNodeRef &node : record.activeNodes) {
    if (node.node == FabricFuNodeKind::Op)
      continue;
    std::vector<Endpoint> inputs;
    std::vector<Endpoint> outputs;
    for (const Edge &edge : record.activeEdges) {
      if (const auto *destination =
              std::get_if<FabricFuNodePortRef>(&edge.destination.payload);
          destination && destination->node == node)
        inputs.push_back(edge.destination);
      if (const auto *source =
              std::get_if<FabricFuNodePortRef>(&edge.source.payload);
          source && source->node == node)
        outputs.push_back(edge.source);
    }
    if (inputs.empty() || outputs.empty())
      return malformed("active structural FU node has an incomplete route");
    for (const Endpoint &input : inputs)
      for (const Endpoint &output : outputs)
        adjacency.push_back(Edge{input, output});
  }

  const auto isSourceTerminal = [](const Endpoint &endpoint) {
    if (const auto *boundary =
            std::get_if<FabricFuTemplatePortRef>(&endpoint.payload))
      return boundary->direction == FabricPortDirection::Input;
    const auto &port = std::get<FabricFuNodePortRef>(endpoint.payload);
    return port.node.node == FabricFuNodeKind::Op &&
           port.direction == FabricPortDirection::Output;
  };
  const auto isSinkTerminal = [](const Endpoint &endpoint) {
    if (const auto *boundary =
            std::get_if<FabricFuTemplatePortRef>(&endpoint.payload))
      return boundary->direction == FabricPortDirection::Output;
    const auto &port = std::get<FabricFuNodePortRef>(endpoint.payload);
    return port.node.node == FabricFuNodeKind::Op &&
           port.direction == FabricPortDirection::Input;
  };

  std::vector<Endpoint> sources;
  for (const Edge &edge : adjacency)
    if (isSourceTerminal(edge.source) &&
        !llvm::is_contained(sources, edge.source))
      sources.push_back(edge.source);

  std::vector<std::pair<std::vector<std::uint8_t>, Edge>> projected;
  for (const Endpoint &source : sources) {
    std::vector<Endpoint> visited{source};
    for (std::size_t cursor = 0; cursor < visited.size(); ++cursor) {
      const Endpoint current = visited[cursor];
      for (const Edge &edge : adjacency) {
        if (edge.source != current)
          continue;
        if (isSinkTerminal(edge.destination)) {
          Edge terminal{source, edge.destination};
          projected.emplace_back(edgeBytes(terminal), std::move(terminal));
          continue;
        }
        if (!llvm::is_contained(visited, edge.destination))
          visited.push_back(edge.destination);
      }
    }
  }

  std::sort(projected.begin(), projected.end(),
            canonicalLess<Edge, std::vector<std::uint8_t>>);
  std::vector<Edge> result;
  result.reserve(projected.size());
  for (std::size_t index = 0; index < projected.size(); ++index) {
    if (index != 0 && projected[index - 1].first == projected[index].first)
      continue;
    result.push_back(std::move(projected[index].second));
  }
  return result;
}

llvm::Error validateFabricFuCapabilityTemplateRef(
    llvm::ArrayRef<FabricFuCapabilityTemplateRecord> inventory,
    const FabricFuCapabilityTemplateRef &ref) {
  if (ref.ordinal >= inventory.size())
    return makeFabricRefError(FabricRefErrorKind::OrdinalOutOfRange,
                              llvm::Twine("FU capability-template ordinal ") +
                                  llvm::Twine(ref.ordinal) +
                                  " is outside [0, " +
                                  llvm::Twine(inventory.size()) + ")");
  if (inventory[ref.ordinal].activeNodes.empty())
    return malformed("an FU capability template requires an active node");
  if (inventory[ref.ordinal].activeNodes.front().fu != ref.fu)
    return makeFabricRefError(
        FabricRefErrorKind::WrongOwner,
        "the FU capability-template reference names a different FU definition");
  return llvm::Error::success();
}

} // namespace loom::fabric
