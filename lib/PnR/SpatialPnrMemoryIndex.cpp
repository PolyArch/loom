#include "SpatialPnrMemoryIndex.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/IR/MemoryCapabilityDomains.h"
#include "Fabric/IR/MemoryConnectivityContract.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using namespace loom;
using namespace loom::fabric;
using namespace loom::mapping;
using namespace loom::pnr;

namespace {

constexpr llvm::StringLiteral frozenArtifact = "FrozenSpatialPnrProblem";
constexpr PnrCapacityContext bindingIndexContext{
    frozenArtifact, "logical_memory_bindings", "logical_memory_bindings",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext targetIndexContext{
    frozenArtifact, "memory_binding_targets", "memory_binding_targets",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext useOffsetContext{frozenArtifact, "memory_actors",
                                              "rooted_memory_uses",
                                              PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext useCountContext{
    frozenArtifact, "rooted_memory_uses", "rooted_memory_uses",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext serviceGroupContext{
    frozenArtifact, "memory_service_use_groups", "memory_service_use_groups",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext serviceGroupUseOffsetContext{
    frozenArtifact, "memory_service_use_groups", "rooted_memory_uses",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext serviceGroupUseCountContext{
    frozenArtifact, "memory_service_group_uses", "rooted_memory_uses",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext exposureOffsetContext{
    frozenArtifact, "logical_memory_bindings", "memory_exposures",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext exposureCountContext{
    frozenArtifact, "memory_exposures", "memory_exposures",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext exposureProviderContext{
    frozenArtifact, "memory_exposure_providers", "memory_exposure_providers",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext exposureOptionContext{
    frozenArtifact, "memory_exposure_options", "memory_exposure_options",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext domainOffsetContext{
    frozenArtifact, "memory_placements", "memory_dispatch_domains",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext optionOffsetContext{
    frozenArtifact, "memory_dispatch_domains", "memory_dispatch_options",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext optionCountContext{
    frozenArtifact, "memory_dispatch_options", "memory_dispatch_options",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext regionOffsetContext{
    frozenArtifact, "memory_dispatch_options", "service_regions",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext regionCountContext{
    frozenArtifact, "service_regions", "service_regions",
    PnrCapacityMeasure::Count};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::make_error<SpatialPnrFreezeFailure>(
      SpatialPnrFreezeFailureKind::Invalid, message.str());
}

llvm::Error infeasible(const llvm::Twine &message) {
  return llvm::make_error<SpatialPnrFreezeFailure>(
      SpatialPnrFreezeFailureKind::ProvenInfeasible, message.str());
}

llvm::Expected<PnrIndex> checked(PnrCapacityContext context,
                                 std::size_t value) {
  return checkedPnrIndex(context, static_cast<std::uint64_t>(value));
}

template <typename Ref>
llvm::Expected<std::string> dataflowKey(const ArtifactIdentity &owner,
                                        const Ref &reference) {
  auto bytes = ::dataflow::encodeDataflowReference(owner, reference);
  if (!bytes)
    return bytes.takeError();
  return std::string(reinterpret_cast<const char *>(bytes->data()),
                     bytes->size());
}

template <typename Ref> std::string fabricKey(const Ref &reference) {
  const auto bytes = canonicalFabricBytes(reference);
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

struct ActorProjection final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  ::dataflow::semantics::CanonicalService service;
  std::optional<::dataflow::semantics::CanonicalMemoryAccessView> access;
};

struct MemoryEndpointInventory final {
  std::vector<ManagerEndpointRef> managers;
  std::vector<SubordinateEndpointRef> subordinates;
};

llvm::Expected<MemoryEndpointInventory>
buildEndpointInventory(const FabricArtifactView &fabric,
                       FabricMemoryOccurrenceRef occurrence) {
  MemoryEndpointInventory result;
  const FabricMemoryEndpointOwnerRef owner =
      FabricMemoryEndpointOwnerRef::of(occurrence);
  for (std::uint64_t ordinal = 0; ordinal < fabric.memoryEndpointCount(owner);
       ++ordinal) {
    const FabricMemoryEndpointRef endpoint{owner, ordinal};
    const auto role = fabric.memoryEndpointRole(endpoint);
    if (!role)
      return invalid("memory occurrence has an untyped endpoint");
    if (*role == FabricMemoryEndpointRole::Manager)
      result.managers.emplace_back(endpoint);
    else
      result.subordinates.emplace_back(endpoint);
  }
  return result;
}

llvm::Expected<ActorProjection>
projectActor(const ::dataflow::CanonicalDataflowProgramView &dataflow,
             ::dataflow::ActorRef actor) {
  auto resolved = dataflow.resolve(actor);
  if (!resolved)
    return resolved.takeError();
  auto projection =
      ::dataflow::projectRegisteredActorSchemaProjection(resolved->op);
  if (!projection)
    return projection.takeError();
  auto service =
      ::dataflow::semantics::CanonicalService::forActor(resolved->op);
  if (!service)
    return service.takeError();
  std::optional<::dataflow::semantics::CanonicalMemoryAccessView> access;
  if (service->kind() != ::dataflow::semantics::ServiceKind::MemoryFence) {
    auto projected =
        ::dataflow::semantics::getCanonicalMemoryAccessView(resolved->op);
    if (!projected)
      return projected.takeError();
    access.emplace(std::move(*projected));
  }
  return ActorProjection{std::move(*projection), std::move(*service),
                         std::move(access)};
}

struct DispatchDraft final {
  FrozenSpatialMemoryDispatchTarget target;
  std::optional<FabricUsePatternRef> serviceUsePattern;
  std::vector<std::uint64_t> serviceRegions;
  std::string key;
};

std::string
dispatchKey(const FrozenSpatialMemoryDispatchTarget &target,
            const std::optional<FabricUsePatternRef> &serviceUsePattern) {
  std::string result = std::visit(
      [](const auto &typed) {
        using Target = std::decay_t<decltype(typed)>;
        constexpr char tag = std::is_same_v<Target, ManagerEndpointRef> ? 1 : 0;
        std::string key(1, tag);
        key += fabricKey(typed);
        return key;
      },
      target);
  result.push_back(serviceUsePattern ? 1 : 0);
  if (serviceUsePattern)
    result += fabricKey(*serviceUsePattern);
  return result;
}

llvm::Expected<std::vector<DispatchDraft>>
buildDispatchOptions(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                     const FabricArtifactView &fabric,
                     const FrozenSpatialMemoryActorBinding &actor,
                     FabricMemoryOccurrenceRef occurrence,
                     const MemoryEndpointInventory &endpoints) {
  auto projection = projectActor(dataflow, actor.actor);
  if (!projection)
    return projection.takeError();
  const FabricMemoryOperationPortRef port{occurrence,
                                          actor.operationPort.ordinal};
  const auto *operationPort = fabric.memoryOperationPort(port);
  const auto *connectivity = fabric.memoryConnectivity(occurrence);
  if (!operationPort || !connectivity ||
      port.ordinal >= connectivity->operationPorts().size())
    return invalid("memory placement has no exact operation dispatch row");

  auto matches = operationPort->matchingCapabilities(
      projection->actor, projection->service, projection->access);
  if (!matches)
    return matches.takeError();
  if (actor.capability.port != actor.operationPort)
    return invalid("memory actor capability belongs to another operation port");
  const auto selected = llvm::find_if(
      *matches, [&](const ::fabric::MemoryCapabilityMatch &match) {
        return match.alternativeOrdinal == actor.capability.ordinal;
      });
  if (selected == matches->end())
    return invalid("memory actor capability is not admitted by its occurrence");
  const auto &portDispatch = connectivity->operationPorts()[port.ordinal];
  if (actor.capability.ordinal >= portDispatch.capabilityTargetDomains.size())
    return invalid("memory actor capability has no H_dispatch domain");

  std::map<std::string, DispatchDraft> options;
  const auto add = [&](FrozenSpatialMemoryDispatchTarget target,
                       std::optional<FabricUsePatternRef> serviceUsePattern,
                       llvm::ArrayRef<std::uint64_t> regions = {}) {
    const std::string key = dispatchKey(target, serviceUsePattern);
    auto [iterator, inserted] = options.try_emplace(
        key, DispatchDraft{
                 std::move(target), std::move(serviceUsePattern), {}, key});
    iterator->second.serviceRegions.insert(
        iterator->second.serviceRegions.end(), regions.begin(), regions.end());
    llvm::sort(iterator->second.serviceRegions);
    iterator->second.serviceRegions.erase(
        std::unique(iterator->second.serviceRegions.begin(),
                    iterator->second.serviceRegions.end()),
        iterator->second.serviceRegions.end());
    (void)inserted;
  };

  for (const ::fabric::MemoryDispatchTarget &target :
       portDispatch.capabilityTargetDomains[actor.capability.ordinal]) {
    if (const auto *manager =
            std::get_if<::fabric::ManagerMemoryDispatchTarget>(&target)) {
      if (manager->endpointOrdinal >= endpoints.managers.size())
        return invalid("memory dispatch names an absent manager endpoint");
      const ManagerEndpointRef endpoint =
          endpoints.managers[manager->endpointOrdinal];
      if (llvm::Error error = validateFabricRef(fabric, endpoint))
        return std::move(error);
      add(endpoint, std::nullopt);
      continue;
    }

    const auto *service = fabric.localMemoryService(occurrence);
    if (!service)
      continue;
    auto serviceMatches =
        service->matchingCapabilities(projection->actor, projection->access);
    if (!serviceMatches)
      return serviceMatches.takeError();
    if (projection->access) {
      for (std::uint64_t capabilityOrdinal : *serviceMatches) {
        const auto &capability = service->capabilities()[capabilityOrdinal];
        for (::fabric::UsePatternKey pattern :
             capability.admissibleUsePatterns) {
          const FabricUsePatternRef usePattern{
              FabricUsePatternOwnerRef(FabricInventoryOwnerRef::of(
                  FabricMemoryServiceRef::local(occurrence))),
              pattern.ordinal()};
          if (llvm::Error error = validateFabricRef(fabric, usePattern))
            return std::move(error);
          add(LocalMemoryServiceRef(FabricMemoryServiceRef::local(occurrence)),
              usePattern, capability.serviceRegionOrdinals);
        }
      }
      continue;
    }

    for (std::uint64_t capabilityOrdinal : *serviceMatches) {
      const auto &capability = service->capabilities()[capabilityOrdinal];
      if (const auto *domain = std::get_if<MemoryConsistencyDomainRef>(
              &capability.consistencyBinding)) {
        if (llvm::Error error = validateFabricRef(fabric, *domain))
          return std::move(error);
        add(*domain, std::nullopt);
      }
    }
  }

  if (options.empty())
    return infeasible("memory actor has no exact dispatch target");
  std::vector<DispatchDraft> result;
  result.reserve(options.size());
  for (auto &[key, option] : options)
    result.push_back(std::move(option));
  return result;
}

bool rangeFits(PnrIndex offset, PnrIndex count, std::size_t size) {
  const std::size_t begin = static_cast<std::size_t>(offset);
  const std::size_t length = static_cast<std::size_t>(count);
  return begin <= size && length <= size - begin;
}

} // namespace

llvm::Expected<FrozenSpatialMemoryIndex> FrozenSpatialMemoryIndexBuilder::build(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping, const FabricArtifactView &fabric,
    const FrozenSpatialRealizationIndex &realizations) {
  if (techMapping.memoryRealizations().size() !=
      realizations.memoryRealizations().size())
    return invalid("memory realization projection is incomplete");

  FrozenSpatialMemoryIndex result;

  std::map<std::string, MemoryEndpointInventory> endpointInventories;
  for (FabricMemoryOccurrenceRef occurrence : fabric.memoryOccurrences()) {
    auto endpoints = buildEndpointInventory(fabric, occurrence);
    if (!endpoints)
      return endpoints.takeError();
    endpointInventories.emplace(fabricKey(occurrence), std::move(*endpoints));
  }

  std::map<std::string, std::pair<FabricMemoryServiceRegionRef, std::uint64_t>>
      localTargets;
  for (FabricMemoryOccurrenceRef occurrence : fabric.memoryOccurrences()) {
    const auto *service = fabric.localMemoryService(occurrence);
    if (!service)
      continue;
    for (auto [ordinal, region] : llvm::enumerate(service->regions())) {
      const FabricMemoryServiceRegionRef reference{
          FabricMemoryServiceRef::local(occurrence), ordinal};
      const std::string key = fabricKey(reference);
      auto [iterator, inserted] =
          localTargets.try_emplace(key, reference, region.sizeBytes);
      if (!inserted && iterator->second.second != region.sizeBytes)
        return invalid("one local service region has inconsistent capacity");
    }
  }
  if (llvm::Error error = preflightPnrIndexCapacity(targetIndexContext,
                                                    localTargets.size() + 1))
    return std::move(error);
  result.bindingTargets_.reserve(localTargets.size() + 1);
  for (const auto &[key, target] : localTargets)
    result.bindingTargets_.push_back(
        {FrozenSpatialMemoryBindingTarget(target.first), target.second});
  result.bindingTargets_.push_back(
      {FrozenSpatialMemoryBindingTarget(FrozenSpatialMemoryBoundaryProxy{}),
       0});

  std::map<std::string, std::vector<::dataflow::RootedGraphLaunchRef>>
      launchesByGraph;
  std::optional<std::string> launchError;
  dataflow.forEachRootedGraphLaunch([&](auto launch) {
    if (launchError)
      return;
    auto graph = dataflow.resolve(launch);
    if (!graph) {
      launchError = llvm::toString(graph.takeError());
      return;
    }
    auto graphKey = dataflowKey(dataflow.identity(), *graph);
    if (!graphKey) {
      launchError = llvm::toString(graphKey.takeError());
      return;
    }
    launchesByGraph[*graphKey].push_back(launch);
  });
  if (launchError)
    return invalid("cannot enumerate rooted graph launches: " + *launchError);
  for (auto &[graph, launches] : launchesByGraph) {
    std::vector<std::pair<std::string, ::dataflow::RootedGraphLaunchRef>> keyed;
    keyed.reserve(launches.size());
    for (const auto &launch : launches) {
      auto key = dataflowKey(dataflow.identity(), launch);
      if (!key)
        return key.takeError();
      keyed.emplace_back(std::move(*key), launch);
    }
    llvm::sort(keyed, [](const auto &left, const auto &right) {
      return left.first < right.first;
    });
    launches.clear();
    for (auto &entry : keyed)
      launches.push_back(std::move(entry.second));
  }

  struct BindingDraft final {
    ::dataflow::LogicalMemoryRootOrViewRef logicalMemory;
    std::optional<std::uint64_t> extent;
  };
  struct UseDraft final {
    ::dataflow::RootedGraphLaunchRef launch;
    PnrIndex actor = 0;
    std::optional<std::string> bindingKey;
  };
  struct ExposureDraft final {
    ::dataflow::MemoryExposureRef exposure;
    std::string bindingKey;
  };
  std::map<std::string, BindingDraft> bindingDrafts;
  std::map<std::string, ExposureDraft> exposureDrafts;
  std::vector<std::vector<UseDraft>> usesByActor(
      realizations.memoryActors().size());

  for (auto [actorOrdinalValue, actorBinding] :
       llvm::enumerate(realizations.memoryActors())) {
    auto actorOrdinal = checked(useCountContext, actorOrdinalValue);
    if (!actorOrdinal)
      return actorOrdinal.takeError();
    auto actor = dataflow.resolve(actorBinding.actor);
    if (!actor)
      return actor.takeError();
    auto graphKey = dataflowKey(dataflow.identity(), actor->graph);
    if (!graphKey)
      return graphKey.takeError();
    const auto foundLaunches = launchesByGraph.find(*graphKey);
    if (foundLaunches == launchesByGraph.end() || foundLaunches->second.empty())
      return infeasible("memory actor has no rooted launch use");
    auto projected = projectActor(dataflow, actorBinding.actor);
    if (!projected)
      return projected.takeError();
    for (const auto &launch : foundLaunches->second) {
      UseDraft use{launch, *actorOrdinal, std::nullopt};
      if (projected->access) {
        auto logical = dataflow.resolveAddressedMemory(
            ::dataflow::ContextualActorRef{launch, actorBinding.actor});
        if (!logical)
          return logical.takeError();
        auto key = dataflowKey(dataflow.identity(), *logical);
        if (!key)
          return key.takeError();
        auto extent = dataflow.staticMemoryByteExtent(*logical);
        if (!extent)
          return extent.takeError();
        auto [iterator, inserted] =
            bindingDrafts.try_emplace(*key, BindingDraft{*logical, *extent});
        if (!inserted && (iterator->second.logicalMemory != *logical ||
                          iterator->second.extent != *extent))
          return invalid("one logical memory has inconsistent projections");
        use.bindingKey = std::move(*key);
      }
      usesByActor[*actorOrdinal].push_back(std::move(use));
    }
  }

  std::optional<std::string> exposureError;
  dataflow.forEachMemoryExposure([&](auto exposure) {
    if (exposureError)
      return;
    auto logical = dataflow.resolveExposure(exposure);
    if (!logical) {
      exposureError = llvm::toString(logical.takeError());
      return;
    }
    auto logicalKey = dataflowKey(dataflow.identity(), *logical);
    if (!logicalKey) {
      exposureError = llvm::toString(logicalKey.takeError());
      return;
    }
    auto extent = dataflow.staticMemoryByteExtent(*logical);
    if (!extent) {
      exposureError = llvm::toString(extent.takeError());
      return;
    }
    auto [binding, bindingInserted] =
        bindingDrafts.try_emplace(*logicalKey, BindingDraft{*logical, *extent});
    if (!bindingInserted && (binding->second.logicalMemory != *logical ||
                             binding->second.extent != *extent)) {
      exposureError = "one exposed logical memory has inconsistent projections";
      return;
    }
    auto exposureKey = dataflowKey(dataflow.identity(), exposure);
    if (!exposureKey) {
      exposureError = llvm::toString(exposureKey.takeError());
      return;
    }
    auto [entry, inserted] = exposureDrafts.try_emplace(
        *exposureKey, ExposureDraft{exposure, *logicalKey});
    if (!inserted && (entry->second.exposure != exposure ||
                      entry->second.bindingKey != *logicalKey)) {
      exposureError = "one memory exposure has inconsistent projections";
      return;
    }
  });
  if (exposureError)
    return invalid("cannot enumerate memory exposures: " + *exposureError);

  if (llvm::Error error =
          preflightPnrIndexCapacity(bindingIndexContext, bindingDrafts.size()))
    return std::move(error);
  std::map<std::string, PnrIndex> bindingByKey;
  result.logicalBindings_.reserve(bindingDrafts.size());
  for (auto &[key, binding] : bindingDrafts) {
    auto ordinal = checked(bindingIndexContext, result.logicalBindings_.size());
    if (!ordinal)
      return ordinal.takeError();
    bindingByKey.emplace(key, *ordinal);
    result.logicalBindings_.push_back(
        {std::move(binding.logicalMemory), binding.extent});
  }

  if (llvm::Error error = preflightPnrIndexCapacity(exposureCountContext,
                                                    exposureDrafts.size()))
    return std::move(error);
  result.exposures_.reserve(exposureDrafts.size());
  for (auto &[key, exposure] : exposureDrafts) {
    const auto binding = bindingByKey.find(exposure.bindingKey);
    if (binding == bindingByKey.end())
      return invalid("memory exposure has no logical binding");
    result.exposures_.push_back(
        {std::move(exposure.exposure), binding->second});
  }

  result.bindingExposureOffsets_.assign(result.logicalBindings_.size() + 1, 0);
  for (const auto &exposure : result.exposures_)
    ++result.bindingExposureOffsets_[exposure.logicalBinding + 1];
  for (std::size_t index = 1; index < result.bindingExposureOffsets_.size();
       ++index) {
    auto prefix = checkedPnrIndexAdd(exposureOffsetContext,
                                     result.bindingExposureOffsets_[index - 1],
                                     result.bindingExposureOffsets_[index]);
    if (!prefix)
      return prefix.takeError();
    result.bindingExposureOffsets_[index] = *prefix;
  }
  result.bindingExposures_.resize(result.bindingExposureOffsets_.back());
  std::vector<PnrIndex> exposureCursors = result.bindingExposureOffsets_;
  for (auto [ordinalValue, exposure] : llvm::enumerate(result.exposures_)) {
    auto ordinal = checked(exposureCountContext, ordinalValue);
    if (!ordinal)
      return ordinal.takeError();
    result.bindingExposures_[exposureCursors[exposure.logicalBinding]++] =
        *ordinal;
  }

  if (!result.exposures_.empty()) {
    for (FabricMemoryOccurrenceRef occurrence : fabric.memoryOccurrences()) {
      const auto endpointInventory =
          endpointInventories.find(fabricKey(occurrence));
      if (endpointInventory == endpointInventories.end())
        return invalid("memory occurrence has no endpoint inventory");
      const auto *connectivity = fabric.memoryConnectivity(occurrence);
      if (!connectivity || connectivity->subordinateEndpoints().size() !=
                               endpointInventory->second.subordinates.size())
        return invalid("memory occurrence has no exact subordinate dispatch "
                       "inventory");
      for (auto [rowOrdinal, row] :
           llvm::enumerate(connectivity->subordinateEndpoints())) {
        auto provider =
            checked(exposureProviderContext, result.exposureProviders_.size());
        if (!provider)
          return provider.takeError();
        result.exposureProviders_.push_back(
            {endpointInventory->second.subordinates[rowOrdinal],
             row.maxExposedBindings});
        for (const ::fabric::MemoryDispatchTarget &target : row.targetDomain) {
          if (std::holds_alternative<::fabric::LocalMemoryDispatchTarget>(
                  target)) {
            if (!fabric.localMemoryService(occurrence))
              return invalid("subordinate H_dispatch names an absent local "
                             "memory service");
            result.exposureOptions_.push_back(
                {*provider,
                 FrozenSpatialMemoryExposureDispatchTarget(
                     std::in_place_type<LocalMemoryServiceRef>,
                     LocalMemoryServiceRef(
                         FabricMemoryServiceRef::local(occurrence)))});
            continue;
          }
          const auto manager =
              std::get<::fabric::ManagerMemoryDispatchTarget>(target);
          if (manager.endpointOrdinal >=
              endpointInventory->second.managers.size())
            return invalid("subordinate H_dispatch names an absent manager "
                           "endpoint");
          result.exposureOptions_.push_back(
              {*provider, FrozenSpatialMemoryExposureDispatchTarget(
                              std::in_place_type<ManagerEndpointRef>,
                              endpointInventory->second
                                  .managers[manager.endpointOrdinal])});
        }
      }
    }
    if (result.exposureProviders_.empty() || result.exposureOptions_.empty())
      return infeasible("memory exposure has no Fabric provider dispatch");
    if (llvm::Error error = preflightPnrIndexCapacity(
            exposureProviderContext, result.exposureProviders_.size()))
      return std::move(error);
    if (llvm::Error error = preflightPnrIndexCapacity(
            exposureOptionContext, result.exposureOptions_.size()))
      return std::move(error);
  }

  result.actorUseOffsets_.reserve(usesByActor.size() + 1);
  result.actorUseOffsets_.push_back(0);
  for (auto &actorUses : usesByActor) {
    for (UseDraft &use : actorUses) {
      std::optional<PnrIndex> binding;
      if (use.bindingKey) {
        auto found = bindingByKey.find(*use.bindingKey);
        if (found == bindingByKey.end())
          return invalid("rooted memory use has no logical binding");
        binding = found->second;
      }
      result.rootedUses_.push_back({std::move(use.launch), use.actor, binding});
    }
    auto offset = checked(useOffsetContext, result.rootedUses_.size());
    if (!offset)
      return offset.takeError();
    result.actorUseOffsets_.push_back(*offset);
  }

  result.bindingUseOffsets_.assign(result.logicalBindings_.size() + 1, 0);
  for (const auto &use : result.rootedUses())
    if (use.logicalBinding)
      ++result.bindingUseOffsets_[*use.logicalBinding + 1];
  for (std::size_t index = 1; index < result.bindingUseOffsets_.size();
       ++index) {
    auto prefix = checkedPnrIndexAdd(useOffsetContext,
                                     result.bindingUseOffsets_[index - 1],
                                     result.bindingUseOffsets_[index]);
    if (!prefix)
      return prefix.takeError();
    result.bindingUseOffsets_[index] = *prefix;
  }
  result.bindingUses_.resize(result.bindingUseOffsets_.back());
  std::vector<PnrIndex> bindingCursors = result.bindingUseOffsets_;
  for (auto [useOrdinalValue, use] : llvm::enumerate(result.rootedUses())) {
    if (!use.logicalBinding)
      continue;
    auto useOrdinal = checked(useCountContext, useOrdinalValue);
    if (!useOrdinal)
      return useOrdinal.takeError();
    result.bindingUses_[bindingCursors[*use.logicalBinding]++] = *useOrdinal;
  }

  std::map<std::pair<PnrIndex, PnrIndex>, std::vector<PnrIndex>> serviceGroups;
  for (auto [useOrdinalValue, use] : llvm::enumerate(result.rootedUses())) {
    if (!use.logicalBinding)
      continue;
    auto useOrdinal = checked(useCountContext, useOrdinalValue);
    if (!useOrdinal)
      return useOrdinal.takeError();
    serviceGroups[{use.actor, *use.logicalBinding}].push_back(*useOrdinal);
  }
  if (llvm::Error error =
          preflightPnrIndexCapacity(serviceGroupContext, serviceGroups.size()))
    return std::move(error);
  result.rootedUseServiceGroups_.assign(result.rootedUses().size(),
                                        getInvalidPnrIndex());
  result.serviceUseGroups_.reserve(serviceGroups.size());
  for (auto &[key, uses] : serviceGroups) {
    auto group = checked(serviceGroupContext, result.serviceUseGroups_.size());
    if (!group)
      return group.takeError();
    auto offset =
        checked(serviceGroupUseOffsetContext, result.serviceGroupUses_.size());
    if (!offset)
      return offset.takeError();
    auto count = checked(serviceGroupUseCountContext, uses.size());
    if (!count)
      return count.takeError();
    result.serviceUseGroups_.push_back(
        {key.first, key.second, *offset, *count});
    for (PnrIndex use : uses) {
      if (use >= result.rootedUseServiceGroups_.size() ||
          result.rootedUseServiceGroups_[use] != getInvalidPnrIndex())
        return invalid("rooted memory use has duplicate service-use owners");
      result.rootedUseServiceGroups_[use] = *group;
      result.serviceGroupUses_.push_back(use);
    }
  }

  result.memoryPlacementDomainOffsets_.reserve(
      realizations.memoryPlacements().size() + 1);
  result.memoryPlacementDomainOffsets_.push_back(0);
  for (auto [placementOrdinalValue, placement] :
       llvm::enumerate(realizations.memoryPlacements())) {
    auto placementOrdinal = checked(domainOffsetContext, placementOrdinalValue);
    if (!placementOrdinal)
      return placementOrdinal.takeError();
    if (placement.realization >= realizations.memoryRealizations().size())
      return invalid("memory placement has a foreign realization");
    const auto &realization =
        realizations.memoryRealizations()[placement.realization];
    for (PnrIndex localActor = 0; localActor < realization.actorCount;
         ++localActor) {
      const PnrIndex actorOrdinal = realization.actorOffset + localActor;
      const auto endpoints =
          endpointInventories.find(fabricKey(placement.memory));
      if (endpoints == endpointInventories.end())
        return invalid("memory placement has no endpoint inventory");
      auto options = buildDispatchOptions(
          dataflow, fabric, realizations.memoryActors()[actorOrdinal],
          placement.memory, endpoints->second);
      if (!options)
        return options.takeError();
      auto optionOffset =
          checked(optionOffsetContext, result.dispatchOptions_.size());
      if (!optionOffset)
        return optionOffset.takeError();
      for (DispatchDraft &option : *options) {
        auto regionOffset = checked(
            regionOffsetContext, result.dispatchServiceRegionOrdinals_.size());
        if (!regionOffset)
          return regionOffset.takeError();
        auto regionCount =
            checked(regionCountContext, option.serviceRegions.size());
        if (!regionCount)
          return regionCount.takeError();
        result.dispatchServiceRegionOrdinals_.insert(
            result.dispatchServiceRegionOrdinals_.end(),
            option.serviceRegions.begin(), option.serviceRegions.end());
        result.dispatchOptions_.push_back({std::move(option.target),
                                           std::move(option.serviceUsePattern),
                                           *regionOffset, *regionCount});
      }
      auto optionCount = checked(optionCountContext, options->size());
      if (!optionCount)
        return optionCount.takeError();
      result.dispatchDomains_.push_back(
          {*placementOrdinal, actorOrdinal, *optionOffset, *optionCount});
    }
    auto domainOffset =
        checked(domainOffsetContext, result.dispatchDomains_.size());
    if (!domainOffset)
      return domainOffset.takeError();
    result.memoryPlacementDomainOffsets_.push_back(*domainOffset);
  }

  if (llvm::Error error = verify(result, realizations))
    return std::move(error);
  return result;
}

llvm::Error FrozenSpatialMemoryIndexBuilder::verify(
    const FrozenSpatialMemoryIndex &memory,
    const FrozenSpatialRealizationIndex &realizations) {
  if (memory.actorUseOffsets().size() !=
          realizations.memoryActors().size() + 1 ||
      memory.bindingUseOffsets().size() !=
          memory.logicalBindings().size() + 1 ||
      memory.rootedUseServiceGroups().size() != memory.rootedUses().size() ||
      memory.bindingExposureOffsets().size() !=
          memory.logicalBindings().size() + 1 ||
      memory.memoryPlacementDomainOffsets().size() !=
          realizations.memoryPlacements().size() + 1)
    return invalid("memory CSR dimensions are incomplete");
  if (memory.actorUseOffsets().empty() ||
      memory.actorUseOffsets().front() != 0 ||
      memory.actorUseOffsets().back() != memory.rootedUses().size() ||
      memory.bindingUseOffsets().empty() ||
      memory.bindingUseOffsets().front() != 0 ||
      memory.bindingUseOffsets().back() != memory.bindingUses().size() ||
      memory.bindingExposureOffsets().empty() ||
      memory.bindingExposureOffsets().front() != 0 ||
      memory.bindingExposureOffsets().back() !=
          memory.bindingExposures().size() ||
      memory.memoryPlacementDomainOffsets().empty() ||
      memory.memoryPlacementDomainOffsets().front() != 0 ||
      memory.memoryPlacementDomainOffsets().back() !=
          memory.dispatchDomains().size())
    return invalid("memory CSR offsets are incomplete");
  for (PnrIndex actor = 0; actor < realizations.memoryActors().size();
       ++actor) {
    if (memory.actorUseOffsets()[actor] > memory.actorUseOffsets()[actor + 1])
      return invalid("rooted memory-use offsets are not monotonic");
    for (const auto &use :
         memory.rootedUses().slice(memory.actorUseOffsets()[actor],
                                   memory.actorUseOffsets()[actor + 1] -
                                       memory.actorUseOffsets()[actor])) {
      if (use.actor != actor ||
          (use.logicalBinding &&
           *use.logicalBinding >= memory.logicalBindings().size()))
        return invalid("rooted memory-use projection is inconsistent");
    }
  }
  for (PnrIndex binding = 0; binding < memory.logicalBindings().size();
       ++binding) {
    if (memory.bindingUseOffsets()[binding] >
        memory.bindingUseOffsets()[binding + 1])
      return invalid("logical-memory use offsets are not monotonic");
    for (PnrIndex use :
         memory.bindingUses().slice(memory.bindingUseOffsets()[binding],
                                    memory.bindingUseOffsets()[binding + 1] -
                                        memory.bindingUseOffsets()[binding]))
      if (use >= memory.rootedUses().size() ||
          memory.rootedUses()[use].logicalBinding != binding)
        return invalid("logical-memory reverse-use projection is inconsistent");
    if (memory.bindingExposureOffsets()[binding] >
        memory.bindingExposureOffsets()[binding + 1])
      return invalid("logical-memory exposure offsets are not monotonic");
    for (PnrIndex exposure : memory.bindingExposures().slice(
             memory.bindingExposureOffsets()[binding],
             memory.bindingExposureOffsets()[binding + 1] -
                 memory.bindingExposureOffsets()[binding]))
      if (exposure >= memory.exposures().size() ||
          memory.exposures()[exposure].logicalBinding != binding)
        return invalid(
            "logical-memory reverse-exposure projection is inconsistent");
  }
  for (auto [groupOrdinal, group] :
       llvm::enumerate(memory.serviceUseGroups())) {
    if (group.actor >= realizations.memoryActors().size() ||
        group.logicalBinding >= memory.logicalBindings().size() ||
        !rangeFits(group.useOffset, group.useCount,
                   memory.serviceGroupUses().size()) ||
        group.useCount == 0)
      return invalid("memory service-use group is inconsistent");
    for (PnrIndex use :
         memory.serviceGroupUses().slice(group.useOffset, group.useCount)) {
      if (use >= memory.rootedUses().size() ||
          memory.rootedUses()[use].actor != group.actor ||
          memory.rootedUses()[use].logicalBinding != group.logicalBinding ||
          memory.rootedUseServiceGroups()[use] != groupOrdinal)
        return invalid("memory service-use group reverse projection diverges");
    }
  }
  for (auto [useOrdinal, use] : llvm::enumerate(memory.rootedUses())) {
    const PnrIndex group = memory.rootedUseServiceGroups()[useOrdinal];
    if (use.logicalBinding) {
      if (group >= memory.serviceUseGroups().size())
        return invalid("addressed memory use has no service-use group");
    } else if (group != getInvalidPnrIndex()) {
      return invalid("fence memory use has a service-use group");
    }
  }
  for (const auto &exposure : memory.exposures())
    if (exposure.logicalBinding >= memory.logicalBindings().size())
      return invalid("memory exposure has a foreign logical binding");
  for (const auto &provider : memory.exposureProviders())
    if (provider.maxExposedBindings == 0)
      return invalid("memory exposure provider has zero binding capacity");
  for (const auto &option : memory.exposureOptions())
    if (option.provider >= memory.exposureProviders().size())
      return invalid("memory exposure option has a foreign provider");
  if (!memory.exposures().empty() && memory.exposureOptions().empty())
    return invalid("memory exposure option inventory is incomplete");
  for (PnrIndex placement = 0;
       placement < realizations.memoryPlacements().size(); ++placement) {
    if (memory.memoryPlacementDomainOffsets()[placement] >
        memory.memoryPlacementDomainOffsets()[placement + 1])
      return invalid("memory dispatch-domain offsets are not monotonic");
    for (const auto &domain : memory.dispatchDomains().slice(
             memory.memoryPlacementDomainOffsets()[placement],
             memory.memoryPlacementDomainOffsets()[placement + 1] -
                 memory.memoryPlacementDomainOffsets()[placement])) {
      if (domain.placement != placement || domain.optionCount == 0 ||
          !rangeFits(domain.optionOffset, domain.optionCount,
                     memory.dispatchOptions().size()))
        return invalid("memory dispatch domain is inconsistent");
    }
  }
  for (const auto &option : memory.dispatchOptions()) {
    if (!rangeFits(option.serviceRegionOffset, option.serviceRegionCount,
                   memory.dispatchServiceRegionOrdinals().size()))
      return invalid("memory dispatch service-region slice is inconsistent");
    const bool local =
        std::holds_alternative<LocalMemoryServiceRef>(option.target);
    if (local != option.serviceUsePattern.has_value())
      return invalid(
          "local memory dispatch and service UsePattern are inconsistent");
  }
  return llvm::Error::success();
}
