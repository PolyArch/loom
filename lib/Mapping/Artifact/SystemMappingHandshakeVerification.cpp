#include "SystemMappingHandshakeVerification.h"

#include "Mapping/Artifact/SystemMappingExecutionProjection.h"

#include "Common/ArtifactLocalReference.h"
#include "Fabric/Identity/FabricHandshake.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace loom::mapping::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_mapping_handshake_invalid: " +
                                     message);
}

std::string byteKey(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

std::string signalKey(const ::loom::fabric::HandshakeSignalRef &signal) {
  std::string result(1, static_cast<char>(signal.signal));
  const auto endpoint = ::loom::fabric::canonicalFabricBytes(signal.endpoint);
  result.append(reinterpret_cast<const char *>(endpoint.data()),
                endpoint.size());
  return result;
}

std::string boundarySignalKey(
    const ::loom::fabric::FabricModuleBoundaryEndpointRef &boundary,
    ::loom::fabric::HandshakeSignalKind signal) {
  std::string result(1, static_cast<char>(signal));
  const auto bytes = ::loom::fabric::canonicalFabricBytes(boundary);
  result.append(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  return result;
}

void appendU64(std::string &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<char>(value >> shift));
}

void appendSized(std::string &bytes, llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.append(reinterpret_cast<const char *>(value.data()), value.size());
}

std::string occurrenceMappingKey(::loom::fabric::AccCoreOccurrenceRef core,
                                 const ArtifactRootReference &mapping) {
  std::string result;
  appendSized(result, ::loom::fabric::canonicalFabricBytes(core));
  const auto mappingBytes = encodeArtifactRootReference(mapping);
  appendSized(result, mappingBytes);
  return result;
}

const ::loom::fabric::FabricArtifactView *
resolveOccurrenceModule(const ::loom::fabric::FabricSystemRootView &fabric,
                        ::loom::fabric::AccCoreOccurrenceRef core,
                        const SpatialMappingView &mapping) {
  const auto target = fabric.spatialCoreTarget(core);
  if (!target ||
      target->dependencyOrdinal >= fabric.artifact().importedModules().size())
    return nullptr;
  const auto &module =
      fabric.artifact().importedModules()[target->dependencyOrdinal];
  const auto root = module.moduleRootTemplate();
  if (!root || *root != target->target ||
      module.identity() != mapping.fabricIdentity())
    return nullptr;
  return &module;
}

llvm::Expected<std::map<std::string, ::loom::fabric::HandshakeSignalRef>>
systemBoundarySignals(const ::loom::fabric::FabricSystemRootView &fabric,
                      ::loom::fabric::AccCoreOccurrenceRef core,
                      const ::loom::fabric::FabricArtifactView &module) {
  const auto target = fabric.spatialCoreTarget(core);
  if (!target)
    return invalid("SpatialCore occurrence has no imported Module target");
  const auto moduleRoot = module.moduleRootTemplate();
  if (!moduleRoot || *moduleRoot != target->target)
    return invalid("SpatialCore occurrence and Module root disagree");

  std::map<std::string, ::loom::fabric::HandshakeSignalRef> result;
  const auto spatialCore = ::loom::fabric::SpatialCoreOccurrenceRef{core};
  for (const auto &attachment : fabric.spatialAttachments()) {
    const auto *transport = attachment.spatialEndpoint.transport();
    if (!transport ||
        transport->owner.kind() !=
            ::loom::fabric::FabricTransportEndpointOwnerKind::
                SpatialCoreOccurrence ||
        std::get<::loom::fabric::SpatialCoreOccurrenceRef>(
            transport->owner.payload) != spatialCore ||
        attachment.moduleEndpoint.dependencyOrdinal !=
            target->dependencyOrdinal ||
        attachment.moduleEndpoint.target.module != *moduleRoot)
      continue;
    for (const auto signal : {::loom::fabric::HandshakeSignalKind::Valid,
                              ::loom::fabric::HandshakeSignalKind::Ready}) {
      const auto key =
          boundarySignalKey(attachment.moduleEndpoint.target, signal);
      if (!result
               .emplace(key,
                        ::loom::fabric::HandshakeSignalRef{*transport, signal})
               .second)
        return invalid("Module boundary has duplicate System attachments");
    }
  }
  return result;
}

using BoundaryHandshakeArc = std::pair<std::string, std::string>;

llvm::Expected<std::vector<BoundaryHandshakeArc>> deriveModuleBoundaryArcs(
    const ::loom::fabric::FabricArtifactView &module,
    const SpatialMappingView &mapping,
    const ::loom::fabric::FabricHandshakeContext &handshakeContext,
    ExecutionControlView executionControl) {
  std::vector<::loom::fabric::HandshakeSignalRef> localTerminals;
  std::map<std::string, std::string> boundaryByLocalSignal;
  for (const auto &attachment : module.moduleBoundaryTransportAttachments()) {
    for (const auto signal : {::loom::fabric::HandshakeSignalKind::Valid,
                              ::loom::fabric::HandshakeSignalKind::Ready}) {
      ::loom::fabric::HandshakeSignalRef local{attachment.endpoint, signal};
      const std::string localKey = signalKey(local);
      if (!boundaryByLocalSignal
               .emplace(localKey,
                        boundarySignalKey(attachment.boundary, signal))
               .second)
        return invalid("Module endpoint has duplicate boundary attachments");
      localTerminals.push_back(std::move(local));
    }
  }
  auto localReachability = ::loom::fabric::deriveSelectedHandshakeReachability(
      module, mapping.handshakeSelection(), localTerminals, handshakeContext,
      executionControl);
  if (!localReachability)
    return localReachability.takeError();

  std::vector<BoundaryHandshakeArc> boundaryArcs;
  boundaryArcs.reserve(localReachability->size() +
                       module.moduleBoundaryTransportPassthroughs().size() * 2);
  for (const auto &arc : *localReachability) {
    const auto source = boundaryByLocalSignal.find(signalKey(arc.source));
    const auto destination =
        boundaryByLocalSignal.find(signalKey(arc.destination));
    if (source == boundaryByLocalSignal.end() ||
        destination == boundaryByLocalSignal.end())
      return invalid("selected Module reachability escaped its boundary map");
    boundaryArcs.emplace_back(source->second, destination->second);
  }
  for (const auto &passthrough : module.moduleBoundaryTransportPassthroughs()) {
    boundaryArcs.emplace_back(
        boundarySignalKey(passthrough.input,
                          ::loom::fabric::HandshakeSignalKind::Valid),
        boundarySignalKey(passthrough.output,
                          ::loom::fabric::HandshakeSignalKind::Valid));
    boundaryArcs.emplace_back(
        boundarySignalKey(passthrough.output,
                          ::loom::fabric::HandshakeSignalKind::Ready),
        boundarySignalKey(passthrough.input,
                          ::loom::fabric::HandshakeSignalKind::Ready));
  }
  return boundaryArcs;
}

llvm::Expected<std::vector<::loom::fabric::HandshakeDependencyArc>>
projectOccurrenceBoundaryArcs(
    const ::loom::fabric::FabricSystemRootView &fabric,
    ::loom::fabric::AccCoreOccurrenceRef core,
    const ::loom::fabric::FabricArtifactView &module,
    llvm::ArrayRef<BoundaryHandshakeArc> boundaryArcs) {
  auto systemSignals = systemBoundarySignals(fabric, core, module);
  if (!systemSignals)
    return systemSignals.takeError();

  std::vector<::loom::fabric::HandshakeDependencyArc> result;
  result.reserve(boundaryArcs.size());
  for (const auto &[sourceKey, destinationKey] : boundaryArcs) {
    const auto source = systemSignals->find(sourceKey);
    const auto destination = systemSignals->find(destinationKey);
    if (source == systemSignals->end() || destination == systemSignals->end())
      return invalid("selected Module boundary has no exact System attachment");
    result.push_back({source->second, destination->second});
  }
  return result;
}

struct SystemSwitchRouteSignature final {
  ::loom::fabric::FabricOrdinal input = 0;
  std::vector<::loom::fabric::FabricOrdinal> outputs;
  std::vector<::loom::fabric::FabricPhysicalTraversalRef> traversals;
};

struct SystemSwitchRouteDemand final {
  ::loom::fabric::FabricSwitchOccurrenceRef occurrence;
  std::vector<SystemSwitchRouteSignature> signatures;
};

bool disjointOutputs(llvm::ArrayRef<::loom::fabric::FabricOrdinal> lhs,
                     llvm::ArrayRef<::loom::fabric::FabricOrdinal> rhs) {
  std::size_t left = 0;
  std::size_t right = 0;
  while (left != lhs.size() && right != rhs.size()) {
    if (lhs[left] == rhs[right])
      return false;
    if (lhs[left] < rhs[right])
      ++left;
    else
      ++right;
  }
  return true;
}

bool compatibleSignatures(const SystemSwitchRouteSignature &lhs,
                          const SystemSwitchRouteSignature &rhs) {
  return lhs.input == rhs.input ? lhs.outputs == rhs.outputs
                                : disjointOutputs(lhs.outputs, rhs.outputs);
}

bool compatibleDemand(const SystemSwitchRouteDemand &demand,
                      llvm::ArrayRef<const SystemSwitchRouteDemand *> row) {
  return llvm::all_of(row, [&](const auto *existing) {
    return llvm::all_of(demand.signatures, [&](const auto &candidate) {
      return llvm::all_of(existing->signatures, [&](const auto &resident) {
        return compatibleSignatures(candidate, resident);
      });
    });
  });
}

llvm::Error appendSystemHandshakeSelection(
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SystemServiceRealizationView> services,
    ::loom::fabric::FabricHandshakeSelection &selection) {
  std::map<std::string, std::vector<SystemSwitchRouteDemand>> demands;
  for (const auto &service : services)
    for (const auto &plan : service.plans)
      for (const auto &leg : plan.transferLegs) {
        std::map<std::string, SystemSwitchRouteDemand> routeDemands;
        for (const auto &node : leg.nodes) {
          const auto *crosspoint =
              std::get_if<::loom::fabric::FabricSwitchTraversalPayload>(
                  &node.incomingTraversal.payload);
          if (!crosspoint || fabric.switchSchedule(crosspoint->owner) !=
                                 ::fabric::Schedule::Temporal) {
            selection.traversals.push_back(node.incomingTraversal);
            continue;
          }
          const auto occurrenceBytes =
              ::loom::fabric::canonicalFabricBytes(crosspoint->owner);
          const std::string occurrenceKey(
              reinterpret_cast<const char *>(occurrenceBytes.data()),
              occurrenceBytes.size());
          auto [position, inserted] = routeDemands.try_emplace(
              occurrenceKey, SystemSwitchRouteDemand{crosspoint->owner, {}});
          auto signature = llvm::find_if(
              position->second.signatures, [&](const auto &candidate) {
                return candidate.input == crosspoint->input;
              });
          if (signature == position->second.signatures.end()) {
            position->second.signatures.push_back(
                SystemSwitchRouteSignature{crosspoint->input, {}, {}});
            signature = std::prev(position->second.signatures.end());
          }
          signature->outputs.push_back(crosspoint->output);
          signature->traversals.push_back(node.incomingTraversal);
        }
        for (auto &[occurrenceKey, demand] : routeDemands) {
          for (auto &signature : demand.signatures) {
            llvm::sort(signature.outputs);
            signature.outputs.erase(
                std::unique(signature.outputs.begin(), signature.outputs.end()),
                signature.outputs.end());
            llvm::sort(signature.traversals,
                       [](const auto &lhs, const auto &rhs) {
                         return ::loom::fabric::canonicalFabricBytes(lhs) <
                                ::loom::fabric::canonicalFabricBytes(rhs);
                       });
            signature.traversals.erase(std::unique(signature.traversals.begin(),
                                                   signature.traversals.end()),
                                       signature.traversals.end());
          }
          llvm::sort(demand.signatures, [](const auto &lhs, const auto &rhs) {
            return lhs.input < rhs.input;
          });
          for (std::size_t left = 0; left != demand.signatures.size(); ++left)
            for (std::size_t right = left + 1;
                 right != demand.signatures.size(); ++right)
              if (!compatibleSignatures(demand.signatures[left],
                                        demand.signatures[right]))
                return invalid("one System route requires incompatible "
                               "Temporal switch crosspoints");
          demands[occurrenceKey].push_back(std::move(demand));
        }
      }

  for (auto &[occurrenceKey, occurrenceDemands] : demands) {
    (void)occurrenceKey;
    std::vector<std::vector<const SystemSwitchRouteDemand *>> rows;
    for (const SystemSwitchRouteDemand &demand : occurrenceDemands) {
      auto row = llvm::find_if(rows, [&](const auto &candidate) {
        return compatibleDemand(demand, candidate);
      });
      if (row == rows.end()) {
        rows.emplace_back();
        row = std::prev(rows.end());
      }
      row->push_back(&demand);
    }
    if (!occurrenceDemands.empty() &&
        rows.size() >
            fabric.switchRouteTableSize(occurrenceDemands.front().occurrence))
      return invalid("System Temporal switch packed rows exceed resident "
                     "capacity");
    for (const auto &[rowOrdinal, row] : llvm::enumerate(rows)) {
      std::map<::loom::fabric::FabricOrdinal,
               std::vector<::loom::fabric::FabricPhysicalTraversalRef>>
          traversalsByInput;
      for (const SystemSwitchRouteDemand *demand : row)
        for (const SystemSwitchRouteSignature &signature : demand->signatures)
          traversalsByInput[signature.input].insert(
              traversalsByInput[signature.input].end(),
              signature.traversals.begin(), signature.traversals.end());
      for (auto &[input, traversals] : traversalsByInput) {
        llvm::sort(traversals, [](const auto &lhs, const auto &rhs) {
          return ::loom::fabric::canonicalFabricBytes(lhs) <
                 ::loom::fabric::canonicalFabricBytes(rhs);
        });
        traversals.erase(std::unique(traversals.begin(), traversals.end()),
                         traversals.end());
        selection.switchActivations.push_back(
            {{occurrenceDemands.front().occurrence,
              static_cast<::loom::fabric::FabricOrdinal>(rowOrdinal), input},
             std::move(traversals)});
      }
    }
  }
  llvm::sort(selection.traversals, [](const auto &lhs, const auto &rhs) {
    return ::loom::fabric::canonicalFabricBytes(lhs) <
           ::loom::fabric::canonicalFabricBytes(rhs);
  });
  selection.traversals.erase(
      std::unique(selection.traversals.begin(), selection.traversals.end()),
      selection.traversals.end());
  return llvm::Error::success();
}

llvm::Error
verifyAcyclic(llvm::ArrayRef<::loom::fabric::HandshakeDependencyArc> arcs) {
  std::map<std::string, std::size_t> nodes;
  for (const auto &arc : arcs) {
    nodes.try_emplace(signalKey(arc.source), 0);
    nodes.try_emplace(signalKey(arc.destination), 0);
  }
  std::size_t nextOrdinal = 0;
  for (auto &[key, ordinal] : nodes) {
    (void)key;
    ordinal = nextOrdinal++;
  }

  std::vector<std::vector<std::size_t>> adjacency(nodes.size());
  std::vector<std::size_t> indegree(nodes.size(), 0);
  std::set<std::pair<std::size_t, std::size_t>> unique;
  for (const auto &arc : arcs) {
    const std::size_t source = nodes.at(signalKey(arc.source));
    const std::size_t destination = nodes.at(signalKey(arc.destination));
    if (!unique.emplace(source, destination).second)
      continue;
    adjacency[source].push_back(destination);
    ++indegree[destination];
  }
  std::vector<std::size_t> ready;
  ready.reserve(nodes.size());
  for (std::size_t node = 0; node < nodes.size(); ++node)
    if (indegree[node] == 0)
      ready.push_back(node);
  std::size_t visited = 0;
  while (!ready.empty()) {
    const std::size_t node = ready.back();
    ready.pop_back();
    ++visited;
    for (std::size_t destination : adjacency[node])
      if (--indegree[destination] == 0)
        ready.push_back(destination);
  }
  if (visited != nodes.size())
    return invalid("SelectedCombinationalHandshakeCycle");
  return llvm::Error::success();
}

} // namespace

llvm::Error verifySystemMappingHandshakeClosure(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemExecutionBindingView &execution,
    llvm::ArrayRef<SystemServiceRealizationView> services,
    const SpatialMappingImportContext &spatialMappings,
    ExecutionControlView executionControl) {
  if (executionControl.stopRequested())
    return invalid("System handshake verification was interrupted");
  ::loom::fabric::FabricHandshakeSelection systemSelection;
  if (llvm::Error error = appendSystemHandshakeSelection(
          fabric.artifact(), services, systemSelection))
    return error;

  std::vector<::loom::fabric::HandshakeSignalRef> systemTerminals;
  std::set<std::string> terminalKeys;
  for (const auto &attachment : fabric.spatialAttachments()) {
    const auto *transport = attachment.spatialEndpoint.transport();
    if (!transport)
      continue;
    for (const auto signal : {::loom::fabric::HandshakeSignalKind::Valid,
                              ::loom::fabric::HandshakeSignalKind::Ready}) {
      ::loom::fabric::HandshakeSignalRef terminal{*transport, signal};
      if (terminalKeys.insert(signalKey(terminal)).second)
        systemTerminals.push_back(std::move(terminal));
    }
  }
  auto combined = ::loom::fabric::deriveSelectedHandshakeReachability(
      fabric.artifact(), systemSelection, systemTerminals, executionControl);
  if (!combined)
    return combined.takeError();

  auto contexts = projectSystemExecutionContexts(dataflow, execution);
  if (!contexts)
    return contexts.takeError();
  std::map<std::string, SpatialMappingView> mappings;
  std::map<ArtifactIdentity::Storage,
           std::shared_ptr<const ::loom::fabric::FabricHandshakeContext>>
      handshakeContexts;
  std::map<std::string, std::vector<BoundaryHandshakeArc>>
      boundaryArcsByMapping;
  std::set<std::string> projected;
  for (const auto &context : contexts->spatialDomains) {
    if (executionControl.stopRequested())
      return invalid("System handshake verification was interrupted");
    const std::string mappingKey =
        byteKey(encodeArtifactRootReference(context.spatialMapping));
    auto found = mappings.find(mappingKey);
    if (found == mappings.end()) {
      auto imported =
          resolveSpatialMappingImport(spatialMappings, context.spatialMapping);
      if (!imported)
        return imported.takeError();
      found = mappings.emplace(mappingKey, (*imported)->view()).first;
    }
    const auto core = context.context.accCore;
    if (!projected.insert(occurrenceMappingKey(core, context.spatialMapping))
             .second)
      continue;
    const auto *module = resolveOccurrenceModule(fabric, core, found->second);
    if (!module)
      return invalid("imported SpatialMapping does not match its AccCore");
    auto boundaryArcs = boundaryArcsByMapping.find(mappingKey);
    if (boundaryArcs == boundaryArcsByMapping.end()) {
      auto handshakeContext =
          handshakeContexts.find(module->identity().bytes());
      if (handshakeContext == handshakeContexts.end()) {
        auto acquired = ::loom::fabric::acquireFabricHandshakeContext(*module);
        if (!acquired)
          return acquired.takeError();
        handshakeContext =
            handshakeContexts
                .emplace(module->identity().bytes(), std::move(*acquired))
                .first;
      }
      auto derived =
          deriveModuleBoundaryArcs(*module, found->second,
                                   *handshakeContext->second, executionControl);
      if (!derived)
        return derived.takeError();
      boundaryArcs =
          boundaryArcsByMapping.emplace(mappingKey, std::move(*derived)).first;
    }
    auto occurrence = projectOccurrenceBoundaryArcs(fabric, core, *module,
                                                    boundaryArcs->second);
    if (!occurrence)
      return occurrence.takeError();
    combined->insert(combined->end(),
                     std::make_move_iterator(occurrence->begin()),
                     std::make_move_iterator(occurrence->end()));
  }
  return verifyAcyclic(*combined);
}

} // namespace loom::mapping::detail
