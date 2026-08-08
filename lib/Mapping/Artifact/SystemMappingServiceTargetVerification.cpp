#include "SystemMappingServiceTargetVerification.h"

#include "Mapping/Artifact/SystemServiceBindingProjection.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace loom::mapping::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_mapping_service_target_invalid: " +
                                     message);
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

struct BranchKey final {
  std::vector<std::uint8_t> region;
  std::vector<std::vector<std::uint8_t>> transforms;

  friend bool operator<(const BranchKey &left, const BranchKey &right) {
    return std::tie(left.region, left.transforms) <
           std::tie(right.region, right.transforms);
  }
  friend bool operator==(const BranchKey &left, const BranchKey &right) {
    return left.region == right.region && left.transforms == right.transforms;
  }
};

BranchKey branchKey(
    ::loom::fabric::FabricMemoryServiceRegionRef region,
    llvm::ArrayRef<::loom::fabric::SystemServiceTransformRef> transforms) {
  BranchKey result{::loom::fabric::canonicalFabricBytes(region), {}};
  result.transforms.reserve(transforms.size());
  for (const auto transform : transforms)
    result.transforms.push_back(
        ::loom::fabric::canonicalFabricBytes(transform));
  return result;
}

std::vector<BranchKey>
exactBranchKeys(const ::loom::fabric::FabricMemoryServiceTargetPlan &plan) {
  std::vector<BranchKey> result;
  result.reserve(plan.branches.size());
  for (const auto &branch : plan.branches)
    result.push_back(branchKey(branch.region, branch.transformPath));
  llvm::sort(result);
  return result;
}

std::vector<std::vector<std::uint8_t>>
regionKeys(const ::loom::fabric::FabricMemoryServiceTargetPlan &plan) {
  std::vector<std::vector<std::uint8_t>> result;
  result.reserve(plan.branches.size());
  for (const auto &branch : plan.branches)
    result.push_back(::loom::fabric::canonicalFabricBytes(branch.region));
  llvm::sort(result);
  return result;
}

std::vector<std::uint64_t>
selectedPlanOrdinals(const SystemServicePlanSelectionView &selection) {
  std::vector<std::uint64_t> result;
  for (const auto &clause : selection.clauses)
    result.push_back(clause.target);
  if (selection.defaultPlanOrdinal)
    result.push_back(*selection.defaultPlanOrdinal);
  llvm::sort(result);
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

const SystemServicePlanView *
findPlan(llvm::ArrayRef<SystemServicePlanView> plans, std::uint64_t ordinal) {
  auto found = llvm::find_if(
      plans, [&](const auto &plan) { return plan.ordinal == ordinal; });
  return found == plans.end() ? nullptr : &*found;
}

struct PlanMarks final {
  std::vector<bool> memoryTargets;
  std::vector<std::vector<bool>> exposures;
  std::vector<bool> consistencyTargets;
};

using MessageOwner = SystemMessageExecutionOwner;

std::vector<std::uint8_t> ownerBytes(const MessageOwner &owner) {
  return std::visit(
      [](const auto value) {
        return ::loom::fabric::canonicalFabricBytes(value);
      },
      owner);
}

struct MessagePairDomain final {
  ::dataflow::StructuralOrdinal sinkOrdinal = 0;
  MessageOwner owner;
  std::vector<SystemPresburgerCell> cells;
};

struct MessagePairKey final {
  ::dataflow::StructuralOrdinal sinkOrdinal = 0;
  std::vector<std::uint8_t> owner;

  friend bool operator<(const MessagePairKey &left,
                        const MessagePairKey &right) {
    return std::tie(left.sinkOrdinal, left.owner) <
           std::tie(right.sinkOrdinal, right.owner);
  }
  friend bool operator==(const MessagePairKey &left,
                         const MessagePairKey &right) {
    return left.sinkOrdinal == right.sinkOrdinal && left.owner == right.owner;
  }
};

::loom::fabric::AccCoreOccurrenceRef
contextAccCore(const ExecutionContextKey &context) {
  return std::visit([](const auto &typed) { return typed.accCore; }, context);
}

llvm::Expected<MessageOwner>
uniqueHostOwner(const ::loom::fabric::FabricSystemRootView &fabric) {
  if (fabric.artifact().hostCoreOccurrences().size() != 1)
    return invalid("message runtime terminal has no unique HostCore owner");
  return MessageOwner{fabric.artifact().hostCoreOccurrences().front()};
}

llvm::Expected<SystemPresburgerCell>
liftImageSymbols(SystemPresburgerCell image, std::uint32_t symbolCount) {
  if (image.symbolCount == symbolCount)
    return image;
  if (image.symbolCount != 0)
    return invalid("channel source_map image has a foreign symbol signature");
  const std::size_t insertion = image.dimensionCount;
  for (auto &row : image.equalities)
    row.insert(row.begin() + insertion, symbolCount, 0);
  for (auto &row : image.inequalities)
    row.insert(row.begin() + insertion, symbolCount, 0);
  image.symbolCount = symbolCount;
  return canonicalizeSystemPresburgerCell(image);
}

::dataflow::RootThreadLaunchRef
channelConsumerRoot(const ::dataflow::ChannelConsumerRef &consumer) {
  return std::visit(
      [](const auto &value) {
        using Consumer = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Consumer,
                                     ::dataflow::GraphStreamInputConsumerRef>)
          return value.launch.rootThreadLaunch;
        else
          return value.launch;
      },
      consumer);
}

llvm::Expected<std::vector<SystemPresburgerCell>>
selectionContextDomain(const SystemServicePlanSelectionView &selection,
                       const SystemServiceObligationProjection &obligation,
                       const SystemExecutionContextProjection &contexts) {
  const auto *producer =
      std::get_if<TransferObligationFamilyKey>(&obligation.key);
  if (!producer)
    return invalid("message selection belongs to a non-transfer obligation");
  if (const auto *instruction =
          std::get_if<InstructionExecutionContextKey>(&selection.key.context)) {
    std::optional<::dataflow::RootThreadLaunchRef> root;
    if (const auto *boundary =
            std::get_if<::dataflow::RootThreadBoundarySourceRef>(producer))
      root = std::visit([](const auto &transfer) { return transfer.launch; },
                        boundary->transfer);
    else if (const auto *channel =
                 std::get_if<::dataflow::ChannelProducerTerminalRef>(producer))
      if (const auto *thread =
              std::get_if<::dataflow::ThreadChannelSendSiteRef>(
                  &channel->producer))
        root = thread->launch;
    if (!root)
      return invalid("message Instruction context has a non-thread producer");
    for (const auto &domain : contexts.instructionDomains)
      if (domain.root == *root && domain.context == *instruction)
        return domain.cells;
    return invalid("message selection has an unreachable Instruction context");
  }

  const auto &spatial =
      std::get<SpatialExecutionContextKey>(selection.key.context);
  std::optional<::dataflow::RootedGraphLaunchRef> graph;
  if (const auto *boundary =
          std::get_if<::dataflow::GraphLaunchBoundarySourceRef>(producer))
    graph = std::visit([](const auto &transfer) { return transfer.launch; },
                       boundary->transfer);
  else if (const auto *channel =
               std::get_if<::dataflow::ChannelProducerTerminalRef>(producer))
    if (const auto *stream =
            std::get_if<::dataflow::GraphStreamOutputProducerRef>(
                &channel->producer))
      graph = stream->launch;
  if (!graph)
    return invalid("message Spatial context has a non-graph producer");
  for (const auto &domain : contexts.spatialDomains)
    if (domain.graph == *graph && domain.context == spatial)
      return domain.cells;
  return invalid("message selection has an unreachable Spatial context");
}

llvm::Expected<std::vector<MessagePairDomain>> deriveMessagePairDomains(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemServiceObligationProjection &obligation,
    const SystemExecutionContextProjection &contexts,
    const SystemServicePlanSelectionView &selection,
    llvm::ArrayRef<SystemPresburgerCell> producerDomain) {
  const auto *producer =
      std::get_if<TransferObligationFamilyKey>(&obligation.key);
  if (!producer)
    return invalid("message applicability belongs to a non-transfer service");
  std::map<MessagePairKey, MessagePairDomain> grouped;
  const auto append = [&](::dataflow::StructuralOrdinal sinkOrdinal,
                          MessageOwner owner,
                          llvm::ArrayRef<SystemPresburgerCell> cells) {
    MessagePairKey key{sinkOrdinal, ownerBytes(owner)};
    auto [entry, inserted] = grouped.try_emplace(
        key, MessagePairDomain{sinkOrdinal, std::move(owner), {}});
    entry->second.cells.insert(entry->second.cells.end(), cells.begin(),
                               cells.end());
  };

  const auto &selectedCore = contextAccCore(selection.key.context);
  const auto *channelProducer =
      std::get_if<::dataflow::ChannelProducerTerminalRef>(producer);
  llvm::ArrayRef<::dataflow::ChannelConsumerBinding> channelConsumers;
  if (channelProducer) {
    auto consumers = dataflow.channelConsumers(channelProducer->producer);
    if (!consumers)
      return consumers.takeError();
    channelConsumers = *consumers;
  }

  for (const auto &[sinkOrdinal, sink] : llvm::enumerate(obligation.sinks)) {
    if (const auto *root =
            std::get_if<::dataflow::RootThreadBoundarySinkRef>(&sink)) {
      const bool completion =
          std::holds_alternative<::dataflow::RootThreadCompletionTransferRef>(
              root->transfer);
      if (completion) {
        auto host = uniqueHostOwner(fabric);
        if (!host)
          return host.takeError();
        append(sinkOrdinal, std::move(*host), producerDomain);
      } else {
        append(sinkOrdinal, MessageOwner{selectedCore}, producerDomain);
      }
      continue;
    }
    if (std::holds_alternative<::dataflow::GraphLaunchBoundarySinkRef>(sink)) {
      append(sinkOrdinal, MessageOwner{selectedCore}, producerDomain);
      continue;
    }

    if (!channelProducer || sinkOrdinal >= channelConsumers.size())
      return invalid("channel message sink has no Dataflow consumer binding");
    const auto &binding = channelConsumers[sinkOrdinal];
    const auto expectedSink = ::dataflow::CanonicalSinkTerminalRef(
        ::dataflow::ChannelConsumerTerminalRef{binding.consumer});
    if (sink != expectedSink)
      return invalid("channel consumer order disagrees with message sinks");
    const auto ownerRoot = channelConsumerRoot(binding.consumer);
    bool foundOwner = false;
    for (const auto &domain : contexts.instructionDomains) {
      if (domain.root != ownerRoot)
        continue;
      foundOwner = true;
      std::vector<SystemPresburgerCell> images;
      if (binding.sourceMap) {
        images.reserve(domain.cells.size());
        for (const auto &cell : domain.cells) {
          auto image = imageSystemPresburgerCell(cell, *binding.sourceMap);
          if (!image)
            return image.takeError();
          auto lifted = liftImageSymbols(std::move(*image),
                                         producerDomain.front().symbolCount);
          if (!lifted)
            return lifted.takeError();
          images.push_back(std::move(*lifted));
        }
      } else {
        if (producerDomain.front().dimensionCount != 0 ||
            llvm::any_of(domain.cells, [](const auto &cell) {
              return cell.dimensionCount != 0;
            }))
          return invalid("ranked direct channel receive has no source_map");
        images.assign(producerDomain.begin(), producerDomain.end());
      }
      append(sinkOrdinal, MessageOwner{domain.context.accCore}, images);
    }
    if (!foundOwner)
      return invalid("channel message sink has no execution-owner domain");
  }

  std::vector<MessagePairDomain> result;
  result.reserve(grouped.size());
  for (auto &[key, domain] : grouped) {
    (void)key;
    result.push_back(std::move(domain));
  }
  return result;
}

struct SelectedMessagePlanDomain final {
  std::uint64_t planOrdinal = 0;
  std::vector<SystemPresburgerCell> cells;
};

llvm::Expected<std::vector<SelectedMessagePlanDomain>>
selectedMessagePlanDomains(const SystemServicePlanSelectionView &selection,
                           llvm::ArrayRef<SystemPresburgerCell> contextDomain) {
  std::map<std::uint64_t, SelectedMessagePlanDomain> grouped;
  std::vector<SystemPresburgerCell> explicitCells;
  for (const auto &clause : selection.clauses) {
    explicitCells.insert(explicitCells.end(), clause.cells.begin(),
                         clause.cells.end());
    auto [entry, inserted] = grouped.try_emplace(
        clause.target, SelectedMessagePlanDomain{clause.target, {}});
    entry->second.cells.insert(entry->second.cells.end(), clause.cells.begin(),
                               clause.cells.end());
  }
  if (selection.defaultPlanOrdinal) {
    auto complement = splitSystemPresburgerSet(contextDomain, explicitCells);
    if (!complement)
      return complement.takeError();
    auto [entry, inserted] = grouped.try_emplace(
        *selection.defaultPlanOrdinal,
        SelectedMessagePlanDomain{*selection.defaultPlanOrdinal, {}});
    entry->second.cells.insert(
        entry->second.cells.end(),
        std::make_move_iterator(complement->outside.begin()),
        std::make_move_iterator(complement->outside.end()));
  }
  std::vector<SelectedMessagePlanDomain> result;
  result.reserve(grouped.size());
  for (auto &[ordinal, domain] : grouped) {
    (void)ordinal;
    result.push_back(std::move(domain));
  }
  return result;
}

llvm::Expected<std::vector<MessagePairKey>>
applicablePairsForCell(const SystemPresburgerCell &cell,
                       llvm::ArrayRef<MessagePairDomain> domains) {
  std::vector<MessagePairKey> result;
  for (const auto &domain : domains) {
    bool intersects = false;
    for (const auto &candidate : domain.cells) {
      auto overlap = systemPresburgerCellsIntersect(cell, candidate);
      if (!overlap)
        return overlap.takeError();
      intersects |= *overlap;
    }
    auto covered = systemPresburgerSetIsSubsetOf({cell}, domain.cells);
    if (!covered)
      return covered.takeError();
    if (intersects != *covered)
      return invalid(
          "applicable sink-owner set changes within one plan-selection cell");
    if (*covered)
      result.push_back({domain.sinkOrdinal, ownerBytes(domain.owner)});
  }
  llvm::sort(result);
  return result;
}

llvm::Expected<MessageOwner>
messageEndpointOwner(const ::loom::fabric::FabricSystemRootView &fabric,
                     ::loom::fabric::FabricTransportEndpointRef endpoint) {
  const auto *service = std::get_if<::loom::fabric::SystemServiceEndpointRef>(
      &endpoint.owner.payload);
  if (!service)
    return invalid("message route terminal is not a service endpoint");
  const auto *owner = fabric.serviceEndpointOwner(*service);
  if (!owner)
    return invalid("message route terminal has no Fabric owner");
  if (const auto *host = std::get_if<::loom::fabric::HostCoreOccurrenceRef>(
          &owner->owner().payload))
    return MessageOwner{*host};
  if (const auto *core = std::get_if<::loom::fabric::AccCoreOccurrenceRef>(
          &owner->owner().payload))
    return MessageOwner{*core};
  return invalid("message route terminal has a nonexecution owner");
}

llvm::Expected<std::map<
    std::uint64_t, std::vector<::loom::fabric::FabricTransportEndpointRef>>>
messageRouteEndpoints(const ::loom::fabric::FabricSystemRootView &fabric,
                      const SystemTransferLegView &route) {
  std::map<std::uint64_t,
           std::vector<::loom::fabric::FabricTransportEndpointRef>>
      result;
  result.emplace(0, std::vector{route.rootEndpoint});
  for (const auto &node : route.nodes) {
    const auto found = llvm::find_if(
        fabric.artifact().physicalTraversals(), [&](const auto &candidate) {
          return candidate.reference == node.incomingTraversal;
        });
    if (found == fabric.artifact().physicalTraversals().end())
      return invalid("message route node names an absent Fabric traversal");
    if (!result.emplace(node.ordinal, found->destinations).second)
      return invalid("message route repeats a node ordinal");
  }
  return result;
}

llvm::Expected<bool> messageEndpointCompatible(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemTransferTerminalKey &terminal, mlir::Type payload,
    const MessageOwner &owner,
    ::loom::fabric::FabricTransportEndpointRef endpoint) {
  auto domains = projectSystemMessageTerminalEndpointDomains(
      dataflow, fabric, terminal, payload, owner);
  if (!domains)
    return domains.takeError();
  const auto found = llvm::find_if(*domains, [&](const auto &domain) {
    return domain.endpoint == endpoint;
  });
  return found != domains->end() && found->payloadCompatible;
}

llvm::Error verifySelectedMessagePlan(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemServiceObligationProjection &obligation,
    const SystemServicePlanView &plan, const ExecutionContextKey &context,
    mlir::Type payload, llvm::ArrayRef<MessagePairKey> expectedPairs) {
  if (!plan.memoryTargets.empty() || !plan.consistencyTargets.empty())
    return invalid("message plan contains a non-message target");
  if (expectedPairs.empty()) {
    if (!plan.transferLegs.empty())
      return invalid(
          "message plan disagrees with its applicable sink-owner set");
    return llvm::Error::success();
  }
  if (plan.transferLegs.size() != obligation.legs.size())
    return invalid("message plan disagrees with its applicable sink-owner set");

  const auto *producer =
      std::get_if<TransferObligationFamilyKey>(&obligation.key);
  if (!producer)
    return invalid("message plan belongs to a non-transfer obligation");
  MessageOwner sourceOwner{contextAccCore(context)};
  if (const auto *root =
          std::get_if<::dataflow::RootThreadBoundarySourceRef>(producer)) {
    const bool completion =
        std::holds_alternative<::dataflow::RootThreadCompletionTransferRef>(
            root->transfer);
    if (!completion) {
      auto host = uniqueHostOwner(fabric);
      if (!host)
        return host.takeError();
      sourceOwner = std::move(*host);
    }
  }

  for (const auto &leg : obligation.legs) {
    const auto route = llvm::find_if(
        plan.transferLegs, [&](const auto &value) { return value.leg == leg; });
    if (route == plan.transferLegs.end())
      return invalid("message plan omits a canonical service leg");
    if (std::count_if(plan.transferLegs.begin(), plan.transferLegs.end(),
                      [&](const auto &value) { return value.leg == leg; }) != 1)
      return invalid("message plan repeats a canonical service leg");
    const SystemTransferTerminalKey sourceTerminal(
        SystemTransferSourceTerminalKey{leg});
    auto sourceCompatible =
        messageEndpointCompatible(dataflow, fabric, sourceTerminal, payload,
                                  sourceOwner, route->rootEndpoint);
    if (!sourceCompatible)
      return sourceCompatible.takeError();
    if (!*sourceCompatible)
      return invalid("message route source is outside its exact owner domain");

    auto endpoints = messageRouteEndpoints(fabric, *route);
    if (!endpoints)
      return endpoints.takeError();
    std::vector<MessagePairKey> actualPairs;
    actualPairs.reserve(route->sinks.size());
    for (const auto &sink : route->sinks) {
      const auto *key =
          std::get_if<SystemTransferSinkTerminalKey>(&sink.terminal);
      auto endpoint = endpoints->find(sink.nodeOrdinal);
      if (!key || key->leg != leg || endpoint == endpoints->end())
        return invalid("message route sink is outside its canonical leg");
      std::vector<MessagePairKey> matchingPairs;
      for (const auto destination : endpoint->second) {
        if (!std::holds_alternative<::loom::fabric::SystemServiceEndpointRef>(
                destination.owner.payload))
          continue;
        auto owner = messageEndpointOwner(fabric, destination);
        if (!owner)
          return owner.takeError();
        auto compatible = messageEndpointCompatible(
            dataflow, fabric, sink.terminal, payload, *owner, destination);
        if (!compatible)
          return compatible.takeError();
        if (*compatible)
          matchingPairs.push_back({key->sinkOrdinal, ownerBytes(*owner)});
      }
      if (matchingPairs.size() != 1)
        return invalid("message route sink is outside its exact owner domain");
      actualPairs.push_back(std::move(matchingPairs.front()));
    }
    llvm::sort(actualPairs);
    if (std::adjacent_find(actualPairs.begin(), actualPairs.end()) !=
            actualPairs.end() ||
        llvm::ArrayRef(actualPairs) != expectedPairs)
      return invalid(
          "message plan disagrees with its applicable sink-owner set");
  }
  return llvm::Error::success();
}

llvm::Error verifyMessageTargetClosure(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemServiceObligationProjection &obligation,
    const SystemExecutionContextProjection &contexts,
    llvm::ArrayRef<SystemServicePlanView> plans,
    llvm::ArrayRef<SystemServicePlanSelectionView> selections) {
  const auto *producer =
      std::get_if<TransferObligationFamilyKey>(&obligation.key);
  if (!producer || obligation.members.size() != 1 ||
      !std::holds_alternative<::dataflow::MessageTransferMemberRef>(
          obligation.members.front()))
    return invalid("message service has an invalid canonical obligation");
  auto terminal = dataflow.resolve(*producer);
  if (!terminal)
    return terminal.takeError();

  for (const auto &selection : selections) {
    auto contextDomain =
        selectionContextDomain(selection, obligation, contexts);
    if (!contextDomain)
      return contextDomain.takeError();
    auto pairDomains = deriveMessagePairDomains(
        dataflow, fabric, obligation, contexts, selection, *contextDomain);
    if (!pairDomains)
      return pairDomains.takeError();
    auto selected = selectedMessagePlanDomains(selection, *contextDomain);
    if (!selected)
      return selected.takeError();
    for (const auto &planDomain : *selected) {
      const auto *plan = findPlan(plans, planDomain.planOrdinal);
      if (!plan)
        return invalid("message selection names an absent plan");
      for (const auto &cell : planDomain.cells) {
        auto expected = applicablePairsForCell(cell, *pairDomains);
        if (!expected)
          return expected.takeError();
        if (llvm::Error error = verifySelectedMessagePlan(
                dataflow, fabric, obligation, *plan, selection.key.context,
                terminal->payloadType, *expected))
          return error;
      }
    }
  }
  return llvm::Error::success();
}

llvm::Expected<std::uint64_t>
moduleDependencyOrdinal(const ::loom::fabric::FabricSystemRootView &fabric,
                        ::loom::fabric::AccCoreOccurrenceRef core,
                        const SpatialMappingView &mapping) {
  auto target = fabric.spatialCoreTarget(core);
  if (!target ||
      target->dependencyOrdinal >= fabric.artifact().importedModules().size())
    return invalid("selected AccCore has no exact SpatialCore target");
  if (fabric.artifact()
          .importedModules()[target->dependencyOrdinal]
          .identity() != mapping.fabricIdentity())
    return invalid("selected SpatialMapping does not match the AccCore Module "
                   "target");
  return target->dependencyOrdinal;
}

llvm::Expected<
    std::pair<FinalizedSpatialMapping, SystemSpatialMemoryBindingProjection>>
resolveBinding(const ::loom::fabric::FabricSystemRootView &fabric,
               const ArtifactStore &store,
               const SpatialExecutionContextKey &context,
               const ServicePlanSelectionAnchor &anchor) {
  ArtifactRootReference reference{mappingArtifactSchema.identity.str(),
                                  mappingArtifactSchema.version,
                                  context.spatialMapping};
  auto mapping = importSpatialMapping(reference, store);
  if (!mapping)
    return mapping.takeError();
  auto dependency =
      moduleDependencyOrdinal(fabric, context.accCore, mapping->view());
  if (!dependency)
    return dependency.takeError();
  auto binding = projectSystemSpatialMemoryBinding(
      fabric, mapping->view(), *dependency, anchor, context.accCore);
  if (!binding)
    return binding.takeError();
  if (binding->endpointPairs.size() != 1)
    return invalid("selected execution does not resolve exactly one "
                   "attachment-bound memory endpoint pair");
  return std::make_pair(std::move(*mapping), std::move(*binding));
}

llvm::Expected<std::vector<::loom::fabric::FabricMemoryServiceTargetPlan>>
compatibleMemoryPlans(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                      const ::loom::fabric::FabricSystemRootView &fabric,
                      const SystemServiceObligationProjection &obligation,
                      const ServicePlanSelectionAnchor &anchor,
                      const SystemSpatialMemoryBindingProjection &binding) {
  const auto *operation =
      std::get_if<OperationServiceObligationFamilyKey>(&obligation.key);
  const auto *logicalMemory =
      operation ? std::get_if<::dataflow::LogicalMemoryRootOrViewRef>(operation)
                : nullptr;
  if (!logicalMemory || !binding.interval)
    return invalid("memory target anchor has no logical interval owner");
  const auto endpoint = binding.endpointPairs.front().systemEndpoint;
  auto plans = projectSystemMemoryTargetPlans(
      dataflow, fabric, endpoint, *logicalMemory, *binding.interval);
  if (!plans)
    return plans.takeError();

  std::optional<std::vector<::loom::fabric::FabricMemoryServiceRegionRef>>
      compatibleRegions;
  if (const auto *member =
          std::get_if<ServiceMemberPlanSelectionAnchor>(&anchor)) {
    auto regions = projectSystemOperationTargetRegions(
        dataflow, fabric, endpoint, member->member);
    if (!regions)
      return regions.takeError();
    compatibleRegions.emplace(std::move(*regions));
  }
  std::vector<::loom::fabric::FabricMemoryServiceTargetPlan> result;
  for (const auto &plan : *plans) {
    if (plan.branches.empty())
      return invalid("Fabric closure contains an empty memory target plan");
    if (!compatibleRegions ||
        llvm::all_of(plan.branches, [&](const auto &branch) {
          return llvm::is_contained(*compatibleRegions, branch.region);
        }))
      result.push_back(plan);
  }
  return result;
}

llvm::Error
verifyMemoryTarget(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                   const ::loom::fabric::FabricSystemRootView &fabric,
                   const SystemServiceObligationProjection &obligation,
                   const ServicePlanSelectionAnchor &anchor,
                   const SystemSpatialMemoryBindingProjection &binding,
                   const SystemServicePlanView &plan, PlanMarks &marks) {
  const auto *operation =
      std::get_if<OperationServiceObligationFamilyKey>(&obligation.key);
  const auto *logicalMemory =
      operation ? std::get_if<::dataflow::LogicalMemoryRootOrViewRef>(operation)
                : nullptr;
  if (!logicalMemory || !binding.interval)
    return invalid("memory target anchor has no logical interval owner");
  auto domain =
      compatibleMemoryPlans(dataflow, fabric, obligation, anchor, binding);
  if (!domain)
    return domain.takeError();

  std::vector<std::size_t> targetOrdinals;
  for (const auto &[ordinal, target] : llvm::enumerate(plan.memoryTargets))
    if (target.element.logicalMemory == *logicalMemory &&
        sameInterval(target.element.interval, *binding.interval))
      targetOrdinals.push_back(ordinal);
  if (targetOrdinals.empty())
    return invalid("selected memory plan omits its logical interval target");

  std::vector<std::vector<std::uint8_t>> selectedRegions;
  std::vector<BranchKey> selectedBranches;
  selectedRegions.reserve(targetOrdinals.size());
  selectedBranches.reserve(targetOrdinals.size());
  for (std::size_t ordinal : targetOrdinals) {
    const auto &target = plan.memoryTargets[ordinal];
    selectedRegions.push_back(
        ::loom::fabric::canonicalFabricBytes(target.element.serviceRegion));
    selectedBranches.push_back(
        branchKey(target.element.serviceRegion, target.element.transformPath));
  }
  llvm::sort(selectedRegions);
  llvm::sort(selectedBranches);

  std::vector<const ::loom::fabric::FabricMemoryServiceTargetPlan *>
      regionMatches;
  for (const auto &candidate : *domain)
    if (regionKeys(candidate) == selectedRegions)
      regionMatches.push_back(&candidate);
  if (regionMatches.empty())
    return invalid("selected service target is outside its attachment-bound "
                   "closure");
  if (regionMatches.size() == 1) {
    if (llvm::any_of(targetOrdinals, [&](std::size_t ordinal) {
          return !plan.memoryTargets[ordinal].element.transformPath.empty();
        }))
      return invalid("uniquely derived service transform path must be omitted");
  } else if (!llvm::any_of(regionMatches, [&](const auto *candidate) {
               return exactBranchKeys(*candidate) == selectedBranches;
             })) {
    return invalid("selected service target is outside its attachment-bound "
                   "closure");
  }

  const auto *exposure =
      std::get_if<MemoryExposurePlanSelectionAnchor>(&anchor);
  for (std::size_t ordinal : targetOrdinals) {
    marks.memoryTargets[ordinal] = true;
    if (!exposure)
      continue;
    if (!binding.exposureTerminal)
      return invalid("memory exposure has no Spatial provider terminal");
    std::optional<std::size_t> matched;
    for (const auto &[exposureOrdinal, child] :
         llvm::enumerate(plan.memoryTargets[ordinal].exposures)) {
      if (child.exposure != exposure->exposure)
        continue;
      if (matched || child.terminal != *binding.exposureTerminal)
        return invalid("memory exposure target has a duplicate or wrong "
                       "provider terminal");
      matched = exposureOrdinal;
    }
    if (!matched)
      return invalid("memory exposure target is incomplete");
    marks.exposures[ordinal][*matched] = true;
  }
  return llvm::Error::success();
}

llvm::Error verifyConsistencyTarget(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemServiceObligationProjection &obligation,
    const ServicePlanSelectionAnchor &anchor,
    const SystemSpatialMemoryBindingProjection &binding,
    const SystemServicePlanView &plan, PlanMarks &marks) {
  const auto *member = std::get_if<ServiceMemberPlanSelectionAnchor>(&anchor);
  if (!member ||
      !std::holds_alternative<::dataflow::FenceActorMemberRef>(member->member))
    return invalid("consistency target has a non-fence anchor");
  const auto endpoint = binding.endpointPairs.front().systemEndpoint;
  auto domains = projectSystemFenceTargetDomains(dataflow, fabric, endpoint,
                                                 member->member);
  if (!domains)
    return domains.takeError();
  const auto *operation =
      std::get_if<OperationServiceObligationFamilyKey>(&obligation.key);
  const auto *fence =
      operation ? std::get_if<::dataflow::FenceActorFamilyRef>(operation)
                : nullptr;
  if (!fence)
    return invalid("fence anchor belongs to a non-fence obligation");
  std::optional<std::size_t> matched;
  for (const auto &[ordinal, target] :
       llvm::enumerate(plan.consistencyTargets)) {
    if (target.fence != *fence)
      continue;
    if (matched || !llvm::is_contained(*domains, target.consistencyDomain))
      return invalid("selected consistency target is outside its "
                     "attachment-bound domain");
    matched = ordinal;
  }
  if (!matched)
    return invalid("selected fence plan has no compatible consistency target");
  marks.consistencyTargets[*matched] = true;
  return llvm::Error::success();
}

} // namespace

llvm::Error verifySystemServiceTargetClosure(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ArtifactStore &store,
    const SystemServiceObligationProjection &obligation,
    const SystemExecutionContextProjection &contexts,
    llvm::ArrayRef<SystemServicePlanView> plans,
    llvm::ArrayRef<SystemServicePlanSelectionView> selections) {
  if (std::holds_alternative<TransferObligationFamilyKey>(obligation.key))
    return verifyMessageTargetClosure(dataflow, fabric, obligation, contexts,
                                      plans, selections);

  std::map<std::uint64_t, PlanMarks> marks;
  for (const auto &plan : plans) {
    PlanMarks planMarks;
    planMarks.memoryTargets.resize(plan.memoryTargets.size(), false);
    planMarks.exposures.reserve(plan.memoryTargets.size());
    for (const auto &target : plan.memoryTargets)
      planMarks.exposures.emplace_back(target.exposures.size(), false);
    planMarks.consistencyTargets.resize(plan.consistencyTargets.size(), false);
    marks.emplace(plan.ordinal, std::move(planMarks));
  }

  for (const auto &selection : selections) {
    const auto *context =
        std::get_if<SpatialExecutionContextKey>(&selection.key.context);
    if (!context)
      return invalid("operation service target has a non-Spatial execution "
                     "context");
    const bool reachable =
        llvm::any_of(contexts.spatialDomains, [&](const auto &domain) {
          return domain.context == *context;
        });
    if (!reachable)
      return invalid("operation service target has an unreachable execution "
                     "context");
    auto resolved =
        resolveBinding(fabric, store, *context, selection.key.anchor);
    if (!resolved)
      return resolved.takeError();
    const auto &binding = resolved->second;

    for (std::uint64_t ordinal : selectedPlanOrdinals(selection)) {
      const auto *plan = findPlan(plans, ordinal);
      auto mark = marks.find(ordinal);
      if (!plan || mark == marks.end())
        return invalid("service target selection names an absent plan");
      const auto *member =
          std::get_if<ServiceMemberPlanSelectionAnchor>(&selection.key.anchor);
      const bool fence =
          member && std::holds_alternative<::dataflow::FenceActorMemberRef>(
                        member->member);
      llvm::Error error =
          fence ? verifyConsistencyTarget(dataflow, fabric, obligation,
                                          selection.key.anchor, binding, *plan,
                                          mark->second)
                : verifyMemoryTarget(dataflow, fabric, obligation,
                                     selection.key.anchor, binding, *plan,
                                     mark->second);
      if (error)
        return error;
    }
  }

  for (const auto &plan : plans) {
    const auto &planMarks = marks.at(plan.ordinal);
    if (llvm::is_contained(planMarks.memoryTargets, false) ||
        llvm::is_contained(planMarks.consistencyTargets, false))
      return invalid("service plan contains a foreign target element");
    for (const auto &exposures : planMarks.exposures)
      if (llvm::is_contained(exposures, false))
        return invalid("service plan contains a foreign memory exposure");
  }
  return llvm::Error::success();
}

} // namespace loom::mapping::detail
