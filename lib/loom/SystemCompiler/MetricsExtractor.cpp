#include "loom/SystemCompiler/L2CoreCompiler.h"
#include "loom/Mapper/ADGFlattener.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/Mapper.h"
#include "loom/Mapper/MapperTiming.h"
#include "loom/Mapper/MappingState.h"

namespace loom {

namespace {

/// Count the number of PE hardware nodes that have at least one mapped
/// software node.
double computePEUtilization(const MappingState &state, const Graph &adg,
                            const ADGFlattener &flattener) {
  const auto &peContainment = flattener.getPEContainment();
  if (peContainment.empty())
    return 0.0;

  unsigned usedPEs = 0;
  unsigned totalPEs = static_cast<unsigned>(peContainment.size());

  for (const auto &pe : peContainment) {
    bool peUsed = false;
    for (IdIndex fuId : pe.fuNodeIds) {
      if (fuId < state.hwNodeToSwNodes.size() &&
          !state.hwNodeToSwNodes[fuId].empty()) {
        peUsed = true;
        break;
      }
    }
    if (peUsed)
      ++usedPEs;
  }

  return totalPEs > 0 ? static_cast<double>(usedPEs) / totalPEs : 0.0;
}

/// Count the number of functional unit hardware nodes that have at least one
/// mapped software node.
double computeFUUtilization(const MappingState &state, const Graph &adg) {
  unsigned usedFUs = 0;
  unsigned totalFUs = 0;

  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
       ++hwId) {
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      continue;
    if (getNodeAttrStr(hwNode, "resource_class") != "functional")
      continue;
    ++totalFUs;
    if (hwId < state.hwNodeToSwNodes.size() &&
        !state.hwNodeToSwNodes[hwId].empty()) {
      ++usedFUs;
    }
  }

  return totalFUs > 0 ? static_cast<double>(usedFUs) / totalFUs : 0.0;
}

/// Compute switch utilization as the fraction of hardware edges that carry
/// at least one routed software edge.
double computeSwitchUtilization(const MappingState &state, const Graph &adg) {
  unsigned usedHwEdges = 0;
  unsigned totalHwEdges = 0;

  for (IdIndex hwEdgeId = 0;
       hwEdgeId < static_cast<IdIndex>(adg.edges.size()); ++hwEdgeId) {
    const Edge *hwEdge = adg.getEdge(hwEdgeId);
    if (!hwEdge)
      continue;
    ++totalHwEdges;
    if (hwEdgeId < state.hwEdgeToSwEdges.size() &&
        !state.hwEdgeToSwEdges[hwEdgeId].empty()) {
      ++usedHwEdges;
    }
  }

  return totalHwEdges > 0 ? static_cast<double>(usedHwEdges) / totalHwEdges
                          : 0.0;
}

/// Estimate SPM usage from memory nodes that are mapped.
uint64_t computeSPMUsage(const MappingState &state, const Graph &dfg,
                         const Graph &adg) {
  uint64_t totalBytes = 0;

  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
       ++hwId) {
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      continue;
    if (getNodeAttrStr(hwNode, "resource_class") != "memory")
      continue;
    if (hwId >= state.hwNodeToSwNodes.size() ||
        state.hwNodeToSwNodes[hwId].empty())
      continue;
    // If this memory node is used, count its capacity as consumed.
    int64_t capacity = getNodeAttrInt(hwNode, "mem_size_bytes", 0);
    if (capacity > 0)
      totalBytes += static_cast<uint64_t>(capacity);
  }

  return totalBytes;
}

/// Compute achieved stream rate: elements produced per cycle.
/// Approximated as (number of output edges) / II.
double computeStreamRate(const Mapper::Result &result, const Graph &dfg) {
  unsigned ii = result.timingSummary.estimatedInitiationInterval;
  if (ii == 0)
    ii = 1;

  // Count output boundary ports as a proxy for elements per invocation.
  unsigned outputElements = 0;
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(dfg.nodes.size());
       ++nodeId) {
    const Node *node = dfg.getNode(nodeId);
    if (!node)
      continue;
    if (node->kind == Node::ModuleOutputNode)
      outputElements += static_cast<unsigned>(node->inputPorts.size());
  }

  return static_cast<double>(outputElements) / static_cast<double>(ii);
}

} // namespace

KernelMetrics extractMetrics(const Mapper::Result &result, const Graph &dfg,
                             const Graph &adg, const ADGFlattener &flattener,
                             const std::string &kernelName) {
  KernelMetrics metrics;
  metrics.kernelName = kernelName;
  metrics.achievedII = result.timingSummary.estimatedInitiationInterval;
  metrics.peUtilization = computePEUtilization(result.state, adg, flattener);
  metrics.fuUtilization = computeFUUtilization(result.state, adg);
  metrics.switchUtilization = computeSwitchUtilization(result.state, adg);
  metrics.spmBytesUsed = computeSPMUsage(result.state, dfg, adg);
  metrics.achievedStreamRate = computeStreamRate(result, dfg);
  return metrics;
}

} // namespace loom
