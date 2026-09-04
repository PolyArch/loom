#include "SystemMappingHandshakeVerification.h"

#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "Mapping/Artifact/SystemServiceBindingProjection.h"

#include "Common/ArtifactLocalReference.h"
#include "Fabric/Identity/FabricHandshake.h"
#include "Fabric/Identity/FabricMemoryServiceHandshake.h"
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

using SelectedHandshakeSignalRef =
    std::variant<::loom::fabric::HandshakeSignalRef,
                 ::loom::fabric::MemoryServiceHandshakeSignalRef>;

struct SelectedHandshakeArc final {
  SelectedHandshakeSignalRef source;
  SelectedHandshakeSignalRef destination;
};

std::string signalKey(const SelectedHandshakeSignalRef &signal) {
  return std::visit(
      [](const auto &selected) {
        using Signal = std::decay_t<decltype(selected)>;
        if constexpr (std::is_same_v<
                          Signal, ::loom::fabric::HandshakeSignalRef>)
          return std::string("transport:") + signalKey(selected);
        else {
          std::string result("memory:");
          result.push_back(static_cast<char>(selected.channel));
          result.push_back(static_cast<char>(selected.signal));
          const auto endpoint =
              ::loom::fabric::canonicalFabricBytes(selected.endpoint);
          result.append(reinterpret_cast<const char *>(endpoint.data()),
                        endpoint.size());
          return result;
        }
      },
      signal);
}

std::string boundarySignalKey(
    const ::loom::fabric::ModuleBoundaryHandshakeSignalRef &signal) {
  std::string result;
  result.push_back(signal.memoryChannel
                       ? static_cast<char>(1 + static_cast<std::uint8_t>(
                                                   *signal.memoryChannel))
                       : 0);
  result.push_back(static_cast<char>(signal.signal));
  const auto bytes =
      ::loom::fabric::canonicalFabricBytes(signal.boundary);
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

llvm::Expected<std::map<std::string, SelectedHandshakeSignalRef>>
systemBoundarySignals(const ::loom::fabric::FabricSystemRootView &fabric,
                      ::loom::fabric::AccCoreOccurrenceRef core,
                      const ::loom::fabric::FabricArtifactView &module) {
  const auto target = fabric.spatialCoreTarget(core);
  if (!target)
    return invalid("SpatialCore occurrence has no imported Module target");
  const auto moduleRoot = module.moduleRootTemplate();
  if (!moduleRoot || *moduleRoot != target->target)
    return invalid("SpatialCore occurrence and Module root disagree");

  std::map<std::string, SelectedHandshakeSignalRef> result;
  const auto spatialCore = ::loom::fabric::SpatialCoreOccurrenceRef{core};
  for (const auto &attachment : fabric.spatialAttachments()) {
    if (attachment.moduleEndpoint.dependencyOrdinal !=
            target->dependencyOrdinal ||
        attachment.moduleEndpoint.target.module != *moduleRoot)
      continue;
    if (const auto *transport = attachment.spatialEndpoint.transport()) {
      if (transport->owner.kind() !=
              ::loom::fabric::FabricTransportEndpointOwnerKind::
                  SpatialCoreOccurrence ||
          std::get<::loom::fabric::SpatialCoreOccurrenceRef>(
              transport->owner.payload) != spatialCore)
        continue;
      for (const auto signal : {::loom::fabric::HandshakeSignalKind::Valid,
                                ::loom::fabric::HandshakeSignalKind::Ready}) {
        const ::loom::fabric::ModuleBoundaryHandshakeSignalRef boundary{
            attachment.moduleEndpoint.target, std::nullopt, signal};
        if (!result
                 .emplace(boundarySignalKey(boundary),
                          ::loom::fabric::HandshakeSignalRef{*transport,
                                                             signal})
                 .second)
          return invalid("Module boundary has duplicate System attachments");
      }
      continue;
    }
    const auto *memory = attachment.spatialEndpoint.memory();
    if (!memory ||
        memory->owner.kind() !=
            ::loom::fabric::FabricMemoryEndpointOwnerKind::
                SpatialCoreOccurrence ||
        std::get<::loom::fabric::SpatialCoreOccurrenceRef>(
            memory->owner.payload) != spatialCore ||
        !attachment.serviceEndpoint)
      continue;
    for (const auto channel :
         {::loom::fabric::MemoryServiceHandshakeChannel::Request,
          ::loom::fabric::MemoryServiceHandshakeChannel::Response}) {
      for (const auto signal : {::loom::fabric::HandshakeSignalKind::Valid,
                                ::loom::fabric::HandshakeSignalKind::Ready}) {
        const ::loom::fabric::ModuleBoundaryHandshakeSignalRef boundary{
            attachment.moduleEndpoint.target, channel, signal};
        if (!result
                 .emplace(
                     boundarySignalKey(boundary),
                     ::loom::fabric::MemoryServiceHandshakeSignalRef{
                         *memory, channel, signal})
                 .second)
          return invalid("Module memory boundary has duplicate System "
                         "attachments");
      }
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
  auto localReachability =
      ::loom::fabric::deriveSelectedModuleBoundaryHandshakeReachability(
          module, mapping.handshakeSelection(),
          mapping.memoryServiceHandshakeSelection(), handshakeContext,
          executionControl);
  if (!localReachability)
    return localReachability.takeError();

  std::vector<BoundaryHandshakeArc> boundaryArcs;
  boundaryArcs.reserve(localReachability->size());
  for (const auto &arc : *localReachability) {
    boundaryArcs.emplace_back(boundarySignalKey(arc.source),
                              boundarySignalKey(arc.destination));
  }
  return boundaryArcs;
}

llvm::Expected<std::vector<SelectedHandshakeArc>>
projectOccurrenceBoundaryArcs(
    const ::loom::fabric::FabricSystemRootView &fabric,
    ::loom::fabric::AccCoreOccurrenceRef core,
    const ::loom::fabric::FabricArtifactView &module,
    llvm::ArrayRef<BoundaryHandshakeArc> boundaryArcs) {
  auto systemSignals = systemBoundarySignals(fabric, core, module);
  if (!systemSignals)
    return systemSignals.takeError();

  std::vector<SelectedHandshakeArc> result;
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

SelectedHandshakeSignalRef memorySignal(
    ::loom::fabric::FabricMemoryEndpointRef endpoint,
    ::loom::fabric::MemoryServiceHandshakeChannel channel,
    ::loom::fabric::HandshakeSignalKind signal) {
  return ::loom::fabric::MemoryServiceHandshakeSignalRef{endpoint, channel,
                                                          signal};
}

void appendMemoryConnection(
    ::loom::fabric::FabricMemoryEndpointRef manager,
    ::loom::fabric::FabricMemoryEndpointRef subordinate,
    std::vector<SelectedHandshakeArc> &arcs) {
  using Channel = ::loom::fabric::MemoryServiceHandshakeChannel;
  using Signal = ::loom::fabric::HandshakeSignalKind;
  arcs.push_back({memorySignal(manager, Channel::Request, Signal::Valid),
                  memorySignal(subordinate, Channel::Request, Signal::Valid)});
  arcs.push_back({memorySignal(subordinate, Channel::Request, Signal::Ready),
                  memorySignal(manager, Channel::Request, Signal::Ready)});
  arcs.push_back({memorySignal(subordinate, Channel::Response, Signal::Valid),
                  memorySignal(manager, Channel::Response, Signal::Valid)});
  arcs.push_back({memorySignal(manager, Channel::Response, Signal::Ready),
                  memorySignal(subordinate, Channel::Response,
                               Signal::Ready)});
}

bool sameInterval(const SpatialMemoryIntervalView &left,
                  const SpatialMemoryIntervalView &right) {
  if (left.index() != right.index())
    return false;
  if (std::holds_alternative<SpatialMemoryWholeIntervalView>(left))
    return true;
  const auto &leftRange = std::get<SpatialMemoryByteRangeView>(left);
  const auto &rightRange = std::get<SpatialMemoryByteRangeView>(right);
  return leftRange.offsetBytes == rightRange.offsetBytes &&
         leftRange.sizeBytes == rightRange.sizeBytes;
}

std::vector<std::uint64_t>
selectedPlanOrdinals(const SystemServicePlanSelectionView &selection) {
  std::vector<std::uint64_t> result;
  for (const auto &clause : selection.clauses)
    result.push_back(clause.target);
  if (selection.defaultPlanOrdinal)
    result.push_back(*selection.defaultPlanOrdinal);
  for (const auto &entry : selection.stableKeyEntries)
    result.push_back(entry.target);
  llvm::sort(result);
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

const SystemServicePlanView *
findPlan(llvm::ArrayRef<SystemServicePlanView> plans, std::uint64_t ordinal) {
  const auto found =
      llvm::find_if(plans, [&](const auto &plan) { return plan.ordinal == ordinal; });
  return found == plans.end() ? nullptr : &*found;
}

std::string branchKey(
    const ::loom::fabric::FabricMemoryServiceTargetBranch &branch) {
  std::string result;
  for (const auto transform : branch.transformPath) {
    const auto bytes = ::loom::fabric::canonicalFabricBytes(transform);
    result.append(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  }
  const auto region = ::loom::fabric::canonicalFabricBytes(branch.region);
  result.append(reinterpret_cast<const char *>(region.data()), region.size());
  return result;
}

llvm::Expected<::loom::fabric::FabricMemoryServiceTargetPlan>
selectedTargetPlan(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    ::loom::fabric::SystemServiceEndpointRef endpoint,
    const ::dataflow::LogicalMemoryRootOrViewRef &logicalMemory,
    const SpatialMemoryIntervalView &interval,
    const SystemServicePlanView &plan) {
  ::loom::fabric::FabricMemoryServiceTargetPlan selected;
  for (const SystemMemoryRegionTargetView &target : plan.memoryTargets) {
    if (target.element.logicalMemory != logicalMemory ||
        !sameInterval(target.element.interval, interval))
      continue;
    selected.branches.push_back(
        {target.element.transformPath, target.element.serviceRegion});
  }
  if (selected.branches.empty())
    return invalid("selected System memory plan has no exact target branch");
  llvm::sort(selected.branches, [](const auto &left, const auto &right) {
    return branchKey(left) < branchKey(right);
  });
  selected.branches.erase(
      std::unique(selected.branches.begin(), selected.branches.end()),
      selected.branches.end());
  auto domain = projectSystemMemoryTargetPlans(dataflow, fabric, endpoint,
                                               logicalMemory, interval);
  if (!domain)
    return domain.takeError();
  const auto regionKeys = [](const auto &candidate) {
    std::vector<std::vector<std::uint8_t>> result;
    result.reserve(candidate.branches.size());
    for (const auto &branch : candidate.branches)
      result.push_back(
          ::loom::fabric::canonicalFabricBytes(branch.region));
    llvm::sort(result);
    return result;
  };
  const auto selectedRegions = regionKeys(selected);
  std::vector<const ::loom::fabric::FabricMemoryServiceTargetPlan *>
      regionMatches;
  for (const auto &candidate : *domain)
    if (regionKeys(candidate) == selectedRegions)
      regionMatches.push_back(&candidate);
  if (regionMatches.empty())
    return invalid("selected System memory target is outside its exact "
                   "attachment-bound domain");
  if (regionMatches.size() == 1) {
    if (llvm::any_of(selected.branches, [](const auto &branch) {
          return !branch.transformPath.empty();
        }))
      return invalid("uniquely derived System memory transform path must be "
                     "omitted");
    return *regionMatches.front();
  }
  const auto selectedBranches = [&] {
    std::vector<std::string> result;
    result.reserve(selected.branches.size());
    for (const auto &branch : selected.branches)
      result.push_back(branchKey(branch));
    llvm::sort(result);
    return result;
  }();
  for (const auto *candidate : regionMatches) {
    std::vector<std::string> candidateBranches;
    candidateBranches.reserve(candidate->branches.size());
    for (const auto &branch : candidate->branches)
      candidateBranches.push_back(branchKey(branch));
    llvm::sort(candidateBranches);
    if (candidateBranches == selectedBranches)
      return *candidate;
  }
  return invalid("selected System memory transform path is outside its exact "
                 "attachment-bound domain");
}

llvm::Expected<bool> appendSelectedSystemMemoryBranch(
    const ::loom::fabric::FabricSystemRootView &fabric,
    ::loom::fabric::FabricMemoryEndpointRef current,
    const ::loom::fabric::FabricMemoryServiceTargetBranch &branch,
    std::size_t transformOrdinal, std::set<std::string> visited,
    std::vector<SelectedHandshakeArc> &arcs) {
  using Channel = ::loom::fabric::MemoryServiceHandshakeChannel;
  using Role = ::loom::fabric::FabricMemoryEndpointRole;
  using Signal = ::loom::fabric::HandshakeSignalKind;
  const auto currentBytes = ::loom::fabric::canonicalFabricBytes(current);
  const std::string currentKey(
      reinterpret_cast<const char *>(currentBytes.data()),
      currentBytes.size());
  if (!visited.insert(currentKey).second)
    return false;

  const auto role = fabric.artifact().memoryEndpointRole(current);
  if (!role)
    return invalid("selected System memory path has an untyped endpoint");
  if (*role == Role::Manager) {
    const ::loom::fabric::FabricMemoryServiceConnectionPayload *selected =
        nullptr;
    for (const auto &connection :
         fabric.artifact().memoryServiceConnections()) {
      if (connection.source != current)
        continue;
      if (selected)
        return invalid("selected System memory manager has ambiguous "
                       "connections");
      selected = &connection;
    }
    if (!selected)
      return false;
    std::vector<SelectedHandshakeArc> suffix;
    auto reaches = appendSelectedSystemMemoryBranch(
        fabric, selected->destination, branch, transformOrdinal,
        std::move(visited), suffix);
    if (!reaches)
      return reaches.takeError();
    if (!*reaches)
      return false;
    appendMemoryConnection(selected->source, selected->destination, arcs);
    arcs.insert(arcs.end(), std::make_move_iterator(suffix.begin()),
                std::make_move_iterator(suffix.end()));
    return true;
  }

  const auto *endpoint =
      std::get_if<::loom::fabric::SystemServiceEndpointRef>(
          &current.owner.payload);
  const auto *owner =
      endpoint ? fabric.serviceEndpointOwner(*endpoint) : nullptr;
  if (!owner)
    return invalid("selected System memory endpoint has no canonical owner");
  if (const auto *service =
          std::get_if<::loom::fabric::FabricMemoryServiceRef>(
              &owner->owner().payload)) {
    if (transformOrdinal != branch.transformPath.size() ||
        branch.region.service != *service ||
        !std::holds_alternative<::loom::fabric::SystemMemoryServiceRef>(
            service->payload))
      return false;
    arcs.push_back(
        {memorySignal(current, Channel::Response, Signal::Ready),
         memorySignal(current, Channel::Request, Signal::Ready)});
    return true;
  }

  const auto *transform =
      std::get_if<::loom::fabric::SystemServiceTransformRef>(
          &owner->owner().payload);
  if (!transform || transformOrdinal >= branch.transformPath.size() ||
      branch.transformPath[transformOrdinal] != *transform)
    return false;
  const auto *record = fabric.serviceTransform(*transform);
  if (!record || !llvm::is_contained(record->inputs(), current))
    return invalid("selected System transform path has a foreign input");

  bool matched = false;
  for (const ::loom::fabric::FabricMemoryEndpointRef output :
       record->outputs()) {
    std::vector<SelectedHandshakeArc> suffix;
    auto reaches = appendSelectedSystemMemoryBranch(
        fabric, output, branch, transformOrdinal + 1, visited, suffix);
    if (!reaches)
      return reaches.takeError();
    if (!*reaches)
      continue;
    matched = true;
    arcs.push_back(
        {memorySignal(current, Channel::Request, Signal::Valid),
         memorySignal(output, Channel::Request, Signal::Valid)});
    arcs.push_back(
        {memorySignal(output, Channel::Request, Signal::Ready),
         memorySignal(current, Channel::Request, Signal::Ready)});
    arcs.push_back(
        {memorySignal(output, Channel::Response, Signal::Valid),
         memorySignal(current, Channel::Response, Signal::Valid)});
    arcs.push_back(
        {memorySignal(current, Channel::Response, Signal::Ready),
         memorySignal(output, Channel::Response, Signal::Ready)});
    arcs.insert(arcs.end(), std::make_move_iterator(suffix.begin()),
                std::make_move_iterator(suffix.end()));
  }
  return matched;
}

llvm::Error appendSelectedSystemMemoryPath(
    const ::loom::fabric::FabricSystemRootView &fabric,
    ::loom::fabric::SystemServiceEndpointRef endpoint,
    const ::loom::fabric::FabricMemoryServiceTargetPlan *selectedPlan,
    std::vector<SelectedHandshakeArc> &arcs) {
  const ::loom::fabric::FabricMemoryEndpointRef root{
      ::loom::fabric::FabricMemoryEndpointOwnerRef::of(endpoint), 0};
  if (!selectedPlan) {
    // Fences carry no address-region plan. Their endpoint capability and
    // consistency target were independently checked before handshake closure;
    // the current closed transform catalog has no fence-routing transform.
    const auto role = fabric.artifact().memoryEndpointRole(root);
    ::loom::fabric::FabricMemoryEndpointRef terminal = root;
    if (role == ::loom::fabric::FabricMemoryEndpointRole::Manager) {
      const auto found = llvm::find_if(
          fabric.artifact().memoryServiceConnections(),
          [&](const auto &connection) { return connection.source == root; });
      if (found == fabric.artifact().memoryServiceConnections().end())
        return invalid("selected System fence manager has no connection");
      appendMemoryConnection(found->source, found->destination, arcs);
      terminal = found->destination;
    }
    const auto *terminalEndpoint =
        std::get_if<::loom::fabric::SystemServiceEndpointRef>(
            &terminal.owner.payload);
    const auto *owner = terminalEndpoint
                            ? fabric.serviceEndpointOwner(*terminalEndpoint)
                            : nullptr;
    const auto *service =
        owner ? std::get_if<::loom::fabric::FabricMemoryServiceRef>(
                    &owner->owner().payload)
              : nullptr;
    if (!service ||
        !std::holds_alternative<::loom::fabric::SystemMemoryServiceRef>(
            service->payload))
      return invalid("selected System fence path has no direct terminal "
                     "service");
    arcs.push_back(
        {memorySignal(
             terminal,
             ::loom::fabric::MemoryServiceHandshakeChannel::Response,
             ::loom::fabric::HandshakeSignalKind::Ready),
         memorySignal(
             terminal,
             ::loom::fabric::MemoryServiceHandshakeChannel::Request,
             ::loom::fabric::HandshakeSignalKind::Ready)});
    return llvm::Error::success();
  }

  for (const auto &branch : selectedPlan->branches) {
    std::vector<SelectedHandshakeArc> branchArcs;
    auto reaches = appendSelectedSystemMemoryBranch(
        fabric, root, branch, 0, {}, branchArcs);
    if (!reaches)
      return reaches.takeError();
    if (!*reaches)
      return invalid("selected System memory target path is incomplete or "
                     "ambiguous");
    arcs.insert(arcs.end(), std::make_move_iterator(branchArcs.begin()),
                std::make_move_iterator(branchArcs.end()));
  }
  return llvm::Error::success();
}

llvm::Error appendSystemMemoryHandshakeSelection(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    llvm::ArrayRef<SystemServiceRealizationView> services,
    const SpatialMappingImportContext &spatialMappings,
    std::vector<SelectedHandshakeArc> &arcs) {
  for (const SystemServiceRealizationView &service : services) {
    const auto *operation =
        std::get_if<OperationServiceObligationFamilyKey>(&service.key);
    if (!operation)
      continue;
    const auto *logicalMemory =
        std::get_if<::dataflow::LogicalMemoryRootOrViewRef>(operation);
    const bool fence =
        std::holds_alternative<::dataflow::FenceActorFamilyRef>(*operation);
    for (const SystemServicePlanSelectionView &selection :
         service.selections) {
      const auto *context =
          std::get_if<SpatialExecutionContextKey>(&selection.key.context);
      if (!context)
        return invalid("System memory handshake selection has a non-Spatial "
                       "execution context");
      const ArtifactRootReference mappingReference{
          mappingArtifactSchema.identity.str(), mappingArtifactSchema.version,
          context->spatialMapping};
      auto imported =
          resolveSpatialMappingImport(spatialMappings, mappingReference);
      if (!imported)
        return imported.takeError();
      const auto target = fabric.spatialCoreTarget(context->accCore);
      if (!target ||
          target->dependencyOrdinal >=
              fabric.artifact().importedModules().size() ||
          fabric.artifact().importedModules()[target->dependencyOrdinal]
                  .identity() != (*imported)->view().fabricIdentity())
        return invalid("System memory handshake context has a foreign "
                       "SpatialMapping");
      auto binding = projectSystemSpatialMemoryBinding(
          fabric, (*imported)->view(), target->dependencyOrdinal,
          selection.key.anchor, context->accCore);
      if (!binding)
        return binding.takeError();
      if (binding->endpointPairs.size() != 1)
        return invalid("System memory handshake selection has an incomplete "
                       "or ambiguous endpoint pair");
      const SystemBoundMemoryEndpointPairView &pair =
          binding->endpointPairs.front();
      const ::loom::fabric::FabricMemoryEndpointRef systemEndpoint{
          ::loom::fabric::FabricMemoryEndpointOwnerRef::of(
              pair.systemEndpoint),
          0};
      const auto occurrenceRole =
          fabric.artifact().memoryEndpointRole(pair.occurrenceEndpoint);
      const auto systemRole =
          fabric.artifact().memoryEndpointRole(systemEndpoint);
      if (!occurrenceRole || !systemRole || *occurrenceRole == *systemRole)
        return invalid("System memory attachment has non-complementary roles");
      if (*occurrenceRole ==
          ::loom::fabric::FabricMemoryEndpointRole::Manager)
        appendMemoryConnection(pair.occurrenceEndpoint, systemEndpoint, arcs);
      else
        appendMemoryConnection(systemEndpoint, pair.occurrenceEndpoint, arcs);

      for (std::uint64_t ordinal : selectedPlanOrdinals(selection)) {
        const SystemServicePlanView *plan = findPlan(service.plans, ordinal);
        if (!plan)
          return invalid("System memory handshake selection names an absent "
                         "plan");
        std::optional<::loom::fabric::FabricMemoryServiceTargetPlan>
            selectedPlan;
        if (logicalMemory) {
          if (!binding->interval)
            return invalid("System memory handshake selection has no logical "
                           "interval");
          auto projected = selectedTargetPlan(
              dataflow, fabric, pair.systemEndpoint, *logicalMemory,
              *binding->interval, *plan);
          if (!projected)
            return projected.takeError();
          selectedPlan.emplace(std::move(*projected));
        } else if (!fence) {
          return invalid("System memory handshake obligation has an unknown "
                         "operation family");
        }
        if (llvm::Error error = appendSelectedSystemMemoryPath(
                fabric, pair.systemEndpoint,
                selectedPlan ? &*selectedPlan : nullptr, arcs))
          return error;
      }
    }
  }
  return llvm::Error::success();
}

llvm::Error
verifyAcyclic(llvm::ArrayRef<SelectedHandshakeArc> arcs) {
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
  for (const auto &service : services)
    for (const auto &plan : service.plans)
      for (const auto &leg : plan.transferLegs)
        for (const auto &node : leg.nodes)
          systemSelection.traversals.push_back(node.incomingTraversal);
  llvm::sort(systemSelection.traversals,
             [](const auto &lhs, const auto &rhs) {
               return ::loom::fabric::canonicalFabricBytes(lhs) <
                      ::loom::fabric::canonicalFabricBytes(rhs);
             });
  systemSelection.traversals.erase(
      std::unique(systemSelection.traversals.begin(),
                  systemSelection.traversals.end()),
      systemSelection.traversals.end());

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
  auto transportReachability =
      ::loom::fabric::deriveSelectedHandshakeReachability(
      fabric.artifact(), systemSelection, systemTerminals, executionControl);
  if (!transportReachability)
    return transportReachability.takeError();
  std::vector<SelectedHandshakeArc> combined;
  combined.reserve(transportReachability->size());
  for (const auto &arc : *transportReachability)
    combined.push_back({arc.source, arc.destination});

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
    combined.insert(combined.end(),
                    std::make_move_iterator(occurrence->begin()),
                    std::make_move_iterator(occurrence->end()));
  }
  if (llvm::Error error = appendSystemMemoryHandshakeSelection(
          dataflow, fabric, services, spatialMappings, combined))
    return error;
  return verifyAcyclic(combined);
}

} // namespace loom::mapping::detail
