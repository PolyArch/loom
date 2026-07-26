#include "FabricVisualizationInternal.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::fabric::visualization {
namespace {

constexpr double horizontalMargin = 72.0;
constexpr double verticalMargin = 56.0;
constexpr double layerGapBase = 112.0;
constexpr double nodeGap = 38.0;
constexpr double routeTrackGap = 8.0;

struct Layering final {
  std::vector<std::size_t> layerByNode;
  std::vector<std::vector<std::size_t>> nodesByLayer;
};

Layering computeLayers(const Graph &graph) {
  const std::size_t count = graph.nodes.size();
  std::vector<std::vector<std::size_t>> adjacency(count);
  for (const Edge &edge : graph.edges)
    if (edge.source < count && edge.destination < count)
      adjacency[edge.source].push_back(edge.destination);
  for (auto &neighbors : adjacency) {
    llvm::sort(neighbors);
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                    neighbors.end());
  }

  std::vector<int> discovery(count, -1);
  std::vector<int> low(count, -1);
  std::vector<bool> onStack(count, false);
  std::vector<std::size_t> stack;
  std::vector<std::size_t> componentByNode(count, 0);
  std::vector<std::vector<std::size_t>> components;
  int nextDiscovery = 0;
  std::function<void(std::size_t)> visit = [&](std::size_t node) {
    discovery[node] = low[node] = nextDiscovery++;
    stack.push_back(node);
    onStack[node] = true;
    for (std::size_t neighbor : adjacency[node]) {
      if (discovery[neighbor] == -1) {
        visit(neighbor);
        low[node] = std::min(low[node], low[neighbor]);
      } else if (onStack[neighbor]) {
        low[node] = std::min(low[node], discovery[neighbor]);
      }
    }
    if (low[node] != discovery[node])
      return;
    const std::size_t component = components.size();
    components.emplace_back();
    while (true) {
      const std::size_t member = stack.back();
      stack.pop_back();
      onStack[member] = false;
      componentByNode[member] = component;
      components.back().push_back(member);
      if (member == node)
        break;
    }
    llvm::sort(components.back());
  };
  for (std::size_t node = 0; node < count; ++node)
    if (discovery[node] == -1)
      visit(node);

  std::vector<std::set<std::size_t>> componentEdges(components.size());
  std::vector<std::size_t> indegree(components.size(), 0);
  for (std::size_t source = 0; source < count; ++source)
    for (std::size_t destination : adjacency[source]) {
      const std::size_t sourceComponent = componentByNode[source];
      const std::size_t destinationComponent = componentByNode[destination];
      if (sourceComponent == destinationComponent)
        continue;
      if (componentEdges[sourceComponent].insert(destinationComponent).second)
        ++indegree[destinationComponent];
    }

  std::set<std::size_t> ready;
  for (std::size_t component = 0; component < components.size(); ++component)
    if (indegree[component] == 0)
      ready.insert(component);
  std::vector<std::size_t> componentLayer(components.size(), 0);
  while (!ready.empty()) {
    const std::size_t component = *ready.begin();
    ready.erase(ready.begin());
    for (std::size_t destination : componentEdges[component]) {
      componentLayer[destination] =
          std::max(componentLayer[destination], componentLayer[component] + 1);
      if (--indegree[destination] == 0)
        ready.insert(destination);
    }
  }

  Layering result;
  result.layerByNode.resize(count, 0);
  std::size_t layerCount = 1;
  for (std::size_t node = 0; node < count; ++node) {
    result.layerByNode[node] = componentLayer[componentByNode[node]];
    layerCount = std::max(layerCount, result.layerByNode[node] + 1);
  }
  result.nodesByLayer.resize(layerCount);
  for (std::size_t node = 0; node < count; ++node)
    result.nodesByLayer[result.layerByNode[node]].push_back(node);
  for (auto &layer : result.nodesByLayer)
    llvm::sort(layer, [&](std::size_t left, std::size_t right) {
      return graph.nodes[left].id < graph.nodes[right].id;
    });
  return result;
}

void expandCrowdedLayers(Layering &layering, std::size_t nodeCount) {
  const std::size_t rowLimit = std::max<std::size_t>(
      4, static_cast<std::size_t>(
             std::ceil(std::sqrt(static_cast<double>(nodeCount)) * 1.5)));

  std::vector<std::vector<std::size_t>> expanded;
  expanded.reserve(layering.nodesByLayer.size());
  for (const auto &layer : layering.nodesByLayer) {
    for (std::size_t offset = 0; offset < layer.size(); offset += rowLimit) {
      const std::size_t end = std::min(layer.size(), offset + rowLimit);
      const std::size_t newLayer = expanded.size();
      expanded.emplace_back(layer.begin() + offset, layer.begin() + end);
      for (std::size_t node : expanded.back())
        layering.layerByNode[node] = newLayer;
    }
  }
  layering.nodesByLayer = std::move(expanded);
}

void reduceCrossings(const Graph &graph, Layering &layering) {
  const std::size_t count = graph.nodes.size();
  std::vector<std::vector<std::size_t>> incoming(count), outgoing(count);
  for (const Edge &edge : graph.edges) {
    if (edge.source >= count || edge.destination >= count)
      continue;
    outgoing[edge.source].push_back(edge.destination);
    incoming[edge.destination].push_back(edge.source);
  }

  auto sweep = [&](bool forward) {
    std::vector<std::size_t> position(count, 0);
    for (const auto &layer : layering.nodesByLayer)
      for (auto [ordinal, node] : llvm::enumerate(layer))
        position[node] = ordinal;
    const std::size_t layerCount = layering.nodesByLayer.size();
    for (std::size_t step = 1; step < layerCount; ++step) {
      const std::size_t layerIndex = forward ? step : layerCount - 1 - step;
      auto &layer = layering.nodesByLayer[layerIndex];
      llvm::stable_sort(layer, [&](std::size_t left, std::size_t right) {
        auto barycenter = [&](std::size_t node) {
          const auto &neighbors = forward ? incoming[node] : outgoing[node];
          double total = 0.0;
          std::size_t used = 0;
          for (std::size_t neighbor : neighbors) {
            const std::size_t neighborLayer = layering.layerByNode[neighbor];
            if ((forward && neighborLayer >= layerIndex) ||
                (!forward && neighborLayer <= layerIndex))
              continue;
            total += static_cast<double>(position[neighbor]);
            ++used;
          }
          return used == 0 ? static_cast<double>(position[node])
                           : total / static_cast<double>(used);
        };
        const double leftCenter = barycenter(left);
        const double rightCenter = barycenter(right);
        if (leftCenter != rightCenter)
          return leftCenter < rightCenter;
        return graph.nodes[left].id < graph.nodes[right].id;
      });
      for (auto [ordinal, node] : llvm::enumerate(layer))
        position[node] = ordinal;
    }
  };
  for (unsigned iteration = 0; iteration < 4; ++iteration) {
    sweep(true);
    sweep(false);
  }
}

void removeCollinear(std::vector<Point> &route) {
  if (route.size() < 3)
    return;
  std::vector<Point> compact;
  compact.reserve(route.size());
  compact.push_back(route.front());
  for (std::size_t index = 1; index + 1 < route.size(); ++index) {
    const Point &before = compact.back();
    const Point &current = route[index];
    const Point &after = route[index + 1];
    const bool vertical = before.x == current.x && current.x == after.x;
    const bool horizontal = before.y == current.y && current.y == after.y;
    if (!vertical && !horizontal)
      compact.push_back(current);
  }
  compact.push_back(route.back());
  route = std::move(compact);
}

} // namespace

void computeLayeredLayout(Graph &graph) {
  if (graph.nodes.empty()) {
    graph.width = 640.0;
    graph.height = 360.0;
    return;
  }

  for (Node &node : graph.nodes) {
    const std::size_t longest = std::max(node.label.size(), node.detail.size());
    node.width =
        std::clamp(112.0 + static_cast<double>(longest) * 3.6, 164.0, 292.0);
    node.height = node.kind == "fabric.acc_core_occurrence" ? 88.0 : 72.0;
  }

  Layering layering = computeLayers(graph);
  expandCrowdedLayers(layering, graph.nodes.size());
  reduceCrossings(graph, layering);

  std::map<std::pair<std::size_t, std::size_t>, std::size_t> adjacentCounts;
  std::size_t topRoutes = 0;
  std::size_t bottomRoutes = 0;
  for (const Edge &edge : graph.edges) {
    if (edge.source >= graph.nodes.size() ||
        edge.destination >= graph.nodes.size())
      continue;
    const std::size_t sourceLayer = layering.layerByNode[edge.source];
    const std::size_t destinationLayer = layering.layerByNode[edge.destination];
    if (destinationLayer == sourceLayer + 1)
      ++adjacentCounts[{sourceLayer, destinationLayer}];
    else if (destinationLayer > sourceLayer + 1)
      ++topRoutes;
    else
      ++bottomRoutes;
  }

  const double topReserve = topRoutes == 0 ? 0.0 : 30.0 + topRoutes * 10.0;
  const double bottomReserve =
      bottomRoutes == 0 ? 0.0 : 30.0 + bottomRoutes * 10.0;
  std::vector<double> layerWidths(layering.nodesByLayer.size(), 0.0);
  std::vector<double> layerHeights(layering.nodesByLayer.size(), 0.0);
  for (auto [layerIndex, layer] : llvm::enumerate(layering.nodesByLayer)) {
    for (std::size_t node : layer) {
      layerWidths[layerIndex] =
          std::max(layerWidths[layerIndex], graph.nodes[node].width);
      layerHeights[layerIndex] += graph.nodes[node].height;
    }
    if (!layer.empty())
      layerHeights[layerIndex] += (layer.size() - 1) * nodeGap;
  }
  const double contentHeight =
      *std::max_element(layerHeights.begin(), layerHeights.end());

  std::vector<double> layerX(layering.nodesByLayer.size(), horizontalMargin);
  for (std::size_t layer = 1; layer < layerX.size(); ++layer) {
    const std::size_t tracks = adjacentCounts[{layer - 1, layer}];
    const double gap =
        layerGapBase + static_cast<double>(tracks) * routeTrackGap;
    layerX[layer] = layerX[layer - 1] + layerWidths[layer - 1] + gap;
  }
  const double contentTop = verticalMargin + topReserve;
  for (auto [layerIndex, layer] : llvm::enumerate(layering.nodesByLayer)) {
    double y = contentTop + (contentHeight - layerHeights[layerIndex]) / 2.0;
    for (std::size_t nodeIndex : layer) {
      Node &node = graph.nodes[nodeIndex];
      node.x =
          layerX[layerIndex] + (layerWidths[layerIndex] - node.width) / 2.0;
      node.y = y;
      y += node.height + nodeGap;
    }
  }

  graph.width = layerX.back() + layerWidths.back() + horizontalMargin;
  graph.height = contentTop + contentHeight + bottomReserve + verticalMargin;

  std::map<std::pair<std::size_t, std::size_t>, std::size_t> nextTrack;
  std::size_t nextTop = 0;
  std::size_t nextBottom = 0;
  for (Edge &edge : graph.edges) {
    if (edge.source >= graph.nodes.size() ||
        edge.destination >= graph.nodes.size())
      continue;
    const Node &source = graph.nodes[edge.source];
    const Node &destination = graph.nodes[edge.destination];
    const std::size_t sourceLayer = layering.layerByNode[edge.source];
    const std::size_t destinationLayer = layering.layerByNode[edge.destination];
    const double sourceX = source.x + source.width;
    const double sourceY = source.y + source.height / 2.0;
    const double destinationX = destination.x;
    const double destinationY = destination.y + destination.height / 2.0;

    if (destinationLayer == sourceLayer + 1) {
      const auto pair = std::make_pair(sourceLayer, destinationLayer);
      const std::size_t track = nextTrack[pair]++;
      const double channelX = layerX[destinationLayer] - layerGapBase / 2.0 -
                              static_cast<double>(track) * routeTrackGap;
      edge.route = {{sourceX, sourceY},
                    {channelX, sourceY},
                    {channelX, destinationY},
                    {destinationX, destinationY}};
    } else if (destinationLayer > sourceLayer + 1) {
      const double highwayY = verticalMargin + nextTop++ * 10.0;
      const double sourceChannel = sourceX + 18.0;
      const double destinationChannel = destinationX - 18.0;
      edge.route = {{sourceX, sourceY},
                    {sourceChannel, sourceY},
                    {sourceChannel, highwayY},
                    {destinationChannel, highwayY},
                    {destinationChannel, destinationY},
                    {destinationX, destinationY}};
    } else if (edge.source == edge.destination) {
      const double loopX = source.x + source.width + 26.0;
      const double loopY = source.y + source.height + 20.0;
      edge.route = {{sourceX, sourceY},
                    {loopX, sourceY},
                    {loopX, loopY},
                    {source.x + source.width / 2.0, loopY},
                    {source.x + source.width / 2.0, source.y + source.height}};
    } else {
      const double highwayY =
          contentTop + contentHeight + 24.0 + nextBottom++ * 10.0;
      const double sourceChannel = sourceX + 18.0;
      const double destinationChannel = destinationX - 18.0;
      edge.route = {{sourceX, sourceY},
                    {sourceChannel, sourceY},
                    {sourceChannel, highwayY},
                    {destinationChannel, highwayY},
                    {destinationChannel, destinationY},
                    {destinationX, destinationY}};
    }
    removeCollinear(edge.route);
  }
}

} // namespace loom::fabric::visualization
