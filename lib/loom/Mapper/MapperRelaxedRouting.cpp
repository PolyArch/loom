#include "loom/Mapper/MapperRelaxedRouting.h"

#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MapperOptions.h"
#include "loom/Mapper/MappingState.h"
#include "loom/Mapper/TypeCompat.h"
#include "MapperRoutingInternal.h"

#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <cmath>

namespace loom {

namespace {

bool isTrackedNonTaggedRoutingOutput(IdIndex outPortId, const Graph &adg) {
  if (!routing_detail::isRoutingCrossbarOutputPort(outPortId, adg))
    return false;
  const Port *port = adg.getPort(outPortId);
  if (!port)
    return false;
  auto typeInfo = detail::getPortTypeInfo(port->type);
  return !(typeInfo && typeInfo->isTagged);
}

template <typename T>
void eraseValue(llvm::SmallVectorImpl<T> &values, const T &value) {
  values.erase(std::remove(values.begin(), values.end(), value), values.end());
}

} // namespace

void RelaxedRoutingState::init(const Graph &adg) {
  size_t numPorts = adg.ports.size();
  trackedOutputs.assign(numPorts, 0);
  outputSourceUses.assign(numPorts, {});
  outputUsingEdges.assign(numPorts, {});
  for (IdIndex portId = 0; portId < static_cast<IdIndex>(numPorts); ++portId) {
    if (isTrackedNonTaggedRoutingOutput(portId, adg))
      trackedOutputs[portId] = 1;
  }
}

void RelaxedRoutingState::clear() {
  for (auto &uses : outputSourceUses)
    uses.clear();
  for (auto &edges : outputUsingEdges)
    edges.clear();
}

RelaxedRoutingState::Snapshot RelaxedRoutingState::save() const {
  Snapshot snapshot;
  snapshot.trackedOutputs = trackedOutputs;
  snapshot.outputSourceUses = outputSourceUses;
  snapshot.outputUsingEdges = outputUsingEdges;
  return snapshot;
}

void RelaxedRoutingState::restore(const Snapshot &snapshot) {
  trackedOutputs = snapshot.trackedOutputs;
  outputSourceUses = snapshot.outputSourceUses;
  outputUsingEdges = snapshot.outputUsingEdges;
}

bool RelaxedRoutingState::isTrackedOutput(IdIndex outPortId) const {
  return outPortId >= 0 && outPortId < static_cast<IdIndex>(trackedOutputs.size()) &&
         trackedOutputs[outPortId] != 0;
}

unsigned RelaxedRoutingState::distinctSources(IdIndex outPortId) const {
  if (!isTrackedOutput(outPortId))
    return 0;
  unsigned count = 0;
  for (const auto &use : outputSourceUses[outPortId]) {
    if (use.count > 0)
      ++count;
  }
  return count;
}

bool RelaxedRoutingState::wouldConflict(IdIndex outPortId,
                                        IdIndex logicalSourcePort) const {
  if (!isTrackedOutput(outPortId))
    return false;
  const auto &uses = outputSourceUses[outPortId];
  if (uses.empty())
    return false;
  for (const auto &use : uses) {
    if (use.logicalSourcePort == logicalSourcePort && use.count > 0)
      return false;
  }
  return distinctSources(outPortId) > 0;
}

double RelaxedRoutingState::softConflictPenalty(
    IdIndex outPortId, IdIndex logicalSourcePort,
    const MapperRelaxedRoutingOptions &opts) const {
  if (!wouldConflict(outPortId, logicalSourcePort))
    return 0.0;
  unsigned conflictingSources = 0;
  for (const auto &use : outputSourceUses[outPortId]) {
    if (use.count == 0 || use.logicalSourcePort == logicalSourcePort)
      continue;
    ++conflictingSources;
  }
  if (conflictingSources == 0)
    return 0.0;
  return opts.baseOverusePenalty *
         std::pow(opts.repeatedOveruseScale,
                  static_cast<double>(conflictingSources - 1));
}

void RelaxedRoutingState::commitRoute(IdIndex swEdgeId, IdIndex logicalSourcePort,
                                      llvm::ArrayRef<IdIndex> path,
                                      const Graph &adg) {
  updateRoute(swEdgeId, logicalSourcePort, path, adg, true);
}

void RelaxedRoutingState::uncommitRoute(IdIndex swEdgeId,
                                        IdIndex logicalSourcePort,
                                        llvm::ArrayRef<IdIndex> path,
                                        const Graph &adg) {
  updateRoute(swEdgeId, logicalSourcePort, path, adg, false);
}

llvm::SmallVector<IdIndex, 8> RelaxedRoutingState::collectOverusedOutputs() const {
  llvm::SmallVector<IdIndex, 8> outputs;
  for (IdIndex portId = 0;
       portId < static_cast<IdIndex>(trackedOutputs.size()); ++portId) {
    if (!trackedOutputs[portId])
      continue;
    if (distinctSources(portId) > 1)
      outputs.push_back(portId);
  }
  return outputs;
}

llvm::SmallVector<IdIndex, 16> RelaxedRoutingState::collectEdgesTouchingOutputs(
    llvm::ArrayRef<IdIndex> outputs) const {
  llvm::DenseSet<IdIndex> seen;
  llvm::SmallVector<IdIndex, 16> edges;
  for (IdIndex outPortId : outputs) {
    if (!isTrackedOutput(outPortId))
      continue;
    for (IdIndex edgeId : outputUsingEdges[outPortId]) {
      if (seen.insert(edgeId).second)
        edges.push_back(edgeId);
    }
  }
  return edges;
}

void RelaxedRoutingState::updateRoute(IdIndex swEdgeId, IdIndex logicalSourcePort,
                                      llvm::ArrayRef<IdIndex> path,
                                      const Graph &adg, bool addRoute) {
  llvm::DenseSet<IdIndex> touchedOutputs;
  for (IdIndex portId : path) {
    if (!isTrackedNonTaggedRoutingOutput(portId, adg))
      continue;
    if (!touchedOutputs.insert(portId).second)
      continue;

    auto &edges = outputUsingEdges[portId];
    auto &uses = outputSourceUses[portId];
    if (addRoute) {
      edges.push_back(swEdgeId);
      auto it = llvm::find_if(uses, [&](const SourceUse &use) {
        return use.logicalSourcePort == logicalSourcePort;
      });
      if (it == uses.end())
        uses.push_back(SourceUse{logicalSourcePort, 1});
      else
        ++it->count;
      continue;
    }

    eraseValue(edges, swEdgeId);
    auto it = llvm::find_if(uses, [&](const SourceUse &use) {
      return use.logicalSourcePort == logicalSourcePort;
    });
    if (it == uses.end())
      continue;
    if (it->count > 0)
      --it->count;
    if (it->count == 0)
      uses.erase(it);
  }
}

bool isRelaxableRoutingOutput(IdIndex outPortId, const Graph &adg) {
  return isTrackedNonTaggedRoutingOutput(outPortId, adg);
}

unsigned countDistinctLogicalSourcesForOutput(IdIndex outPortId,
                                              const MappingState &state) {
  if (outPortId < 0 || outPortId >= static_cast<IdIndex>(state.portToUsingEdges.size()))
    return 0;
  llvm::DenseSet<IdIndex> sources;
  for (IdIndex edgeId : state.portToUsingEdges[outPortId]) {
    if (edgeId < 0 || edgeId >= static_cast<IdIndex>(state.swEdgeToHwPaths.size()))
      continue;
    const auto &path = state.swEdgeToHwPaths[edgeId];
    if (path.empty())
      continue;
    sources.insert(path.front());
  }
  return sources.size();
}

llvm::SmallVector<IdIndex, 8>
collectOverusedNonTaggedRoutingOutputs(const MappingState &state,
                                       const Graph &adg) {
  llvm::SmallVector<IdIndex, 8> outputs;
  for (IdIndex outPortId = 0;
       outPortId < static_cast<IdIndex>(adg.ports.size()); ++outPortId) {
    if (!isTrackedNonTaggedRoutingOutput(outPortId, adg))
      continue;
    if (countDistinctLogicalSourcesForOutput(outPortId, state) > 1)
      outputs.push_back(outPortId);
  }
  return outputs;
}

llvm::SmallVector<IdIndex, 16>
collectEdgesTouchingRoutingOutputs(const MappingState &state,
                                   llvm::ArrayRef<IdIndex> outputs) {
  llvm::DenseSet<IdIndex> seen;
  llvm::SmallVector<IdIndex, 16> edges;
  for (IdIndex outPortId : outputs) {
    if (outPortId < 0 ||
        outPortId >= static_cast<IdIndex>(state.portToUsingEdges.size()))
      continue;
    for (IdIndex edgeId : state.portToUsingEdges[outPortId]) {
      if (seen.insert(edgeId).second)
        edges.push_back(edgeId);
    }
  }
  return edges;
}

} // namespace loom
