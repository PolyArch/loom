#include "SpatialMappingMemoryImport.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/IR/MemoryPortTransaction.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace mlir;

llvm::Expected<loom::mapping::SpatialActorTransitionEventRef>
loom::mapping::deriveSpatialMemoryIssueEvent(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::ActorRef actor) {
  auto resolved = dataflow.resolve(actor);
  if (!resolved)
    return resolved.takeError();
  auto projection =
      ::dataflow::projectRegisteredActorSchemaProjection(resolved->op);
  if (!projection)
    return projection.takeError();
  auto transitions = ::dataflow::semantics::projectActorHandshakeCases(
      projection->schema, resolved->op->getNumOperands(),
      resolved->op->getNumResults());
  if (!transitions)
    return transitions.takeError();
  if (transitions->size() != 1 || transitions->front().ordinal != 0)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "mapping_artifact_invalid: memory actor has no unique issue "
        "transition");
  return SpatialActorTransitionEventRef{actor, transitions->front().ordinal};
}

namespace loom::mapping::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "mapping_artifact_invalid: " + message);
}

std::vector<std::uint8_t> unsignedBytes(DenseI8ArrayAttr record) {
  std::vector<std::uint8_t> result;
  result.reserve(record.size());
  for (std::int8_t byte : record.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

std::string byteKey(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

template <typename Ref, typename Attr>
llvm::Expected<Ref> decodeDataflow(Attr attribute,
                                   const ArtifactIdentity &owner) {
  return ::dataflow::decodeDataflowReference<Ref>(
      unsignedBytes(attribute.getRecord()), owner);
}

template <typename Ref, typename Attr>
llvm::Expected<Ref> decodeFabric(Attr attribute) {
  return ::loom::fabric::decodeFabricRef<Ref>(
      unsignedBytes(attribute.getRecord()));
}

const TechMemoryRealizationView *
findRealization(const TechMappingView &techMapping, std::uint64_t entity) {
  auto found = llvm::find_if(
      techMapping.memoryRealizations(),
      [&](const auto &candidate) { return candidate.entityId == entity; });
  return found == techMapping.memoryRealizations().end() ? nullptr : &*found;
}

const TechMemoryActorView *findActor(const TechMemoryRealizationView &owner,
                                     ::dataflow::ActorRef actor) {
  auto found = llvm::find_if(owner.actors, [&](const auto &candidate) {
    return candidate.actor == actor;
  });
  return found == owner.actors.end() ? nullptr : &*found;
}

struct MemoryActorProjection final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  ::dataflow::semantics::CanonicalService service;
  std::optional<::dataflow::semantics::CanonicalMemoryAccessView> access;
  SpatialActorTransitionEventRef trigger;
};

llvm::Expected<MemoryActorProjection>
projectMemoryActor(const ::dataflow::CanonicalDataflowProgramView &dataflow,
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
  auto trigger = deriveSpatialMemoryIssueEvent(dataflow, actor);
  if (!trigger)
    return trigger.takeError();
  return MemoryActorProjection{std::move(*projection), std::move(*service),
                               std::move(access), std::move(*trigger)};
}

::loom::fabric::FabricMemoryOperationPortRef
operationPortOf(const SpatialMemoryOperationPlacementView &placement) {
  return std::visit(
      [](const auto &selected) {
        using Placement = std::decay_t<decltype(selected)>;
        if constexpr (std::is_same_v<
                          Placement,
                          ::loom::fabric::FabricMemoryOperationPortRef>)
          return selected;
        else
          return selected.port;
      },
      placement);
}

void canonicalizePatterns(
    std::vector<::loom::fabric::FabricUsePatternRef> &patterns) {
  llvm::sort(patterns, [](const auto &left, const auto &right) {
    return ::loom::fabric::canonicalFabricBytes(left) <
           ::loom::fabric::canonicalFabricBytes(right);
  });
  patterns.erase(std::unique(patterns.begin(), patterns.end()), patterns.end());
}

llvm::Expected<std::vector<::loom::fabric::FabricUsePatternRef>>
operationUsePatterns(const ::loom::fabric::FabricArtifactView &fabric,
                     const TechMemoryActorView &actor,
                     const SpatialMemoryOperationPlacementView &placement,
                     const MemoryActorProjection &projection) {
  const auto port = operationPortOf(placement);
  if (actor.capability.port != actor.operationPort ||
      port.ordinal != actor.operationPort.ordinal)
    return invalid("memory operation placement disagrees with Tech capability");
  const auto *record = fabric.memoryOperationPort(port);
  const auto *capability = fabric.memoryCapabilityAlternative(
      ::loom::fabric::FabricMemoryCapabilityAlternativeRef{
          port, actor.capability.ordinal});
  if (!record || !capability)
    return invalid("memory operation placement has no exact capability");

  std::vector<::fabric::MemoryPortTransactionProjection> projections;
  projections.reserve(record->operationPatterns().size());
  for (const auto &pattern : record->operationPatterns())
    projections.push_back(pattern.transactionProjection);
  auto resource = ::fabric::MemoryOperationPortResourceView::create(
      port, record->resourceContract(), projections);
  if (!resource)
    return resource.takeError();

  std::vector<::loom::fabric::FabricUsePatternRef> result;
  result.reserve(capability->admissibleUsePatterns.size());
  for (::fabric::UsePatternKey pattern : capability->admissibleUsePatterns) {
    const ::loom::fabric::FabricUsePatternRef reference{
        ::loom::fabric::FabricUsePatternOwnerRef(
            ::loom::fabric::FabricInventoryOwnerRef::of(port)),
        pattern.ordinal()};
    auto selected = resource->operationPattern(reference);
    if (!selected)
      return selected.takeError();
    auto plan = ::fabric::deriveMemoryPortTransactionPlan(
        *selected, projection.actor, projection.service, projection.access);
    if (!plan)
      return plan.takeError();
    result.push_back(reference);
  }
  canonicalizePatterns(result);
  if (result.empty())
    return invalid("memory operation capability has no admissible UsePattern");
  return result;
}

llvm::Expected<std::vector<::loom::fabric::FabricUsePatternRef>>
localServiceUsePatterns(const ::loom::fabric::FabricArtifactView &fabric,
                        ::loom::fabric::FabricMemoryOccurrenceRef occurrence,
                        std::uint64_t serviceRegionOrdinal,
                        const MemoryActorProjection &projection) {
  const auto *service = fabric.localMemoryService(occurrence);
  if (!service)
    return invalid("local memory dispatch has no service contract");
  auto matches =
      service->matchingCapabilities(projection.actor, projection.access);
  if (!matches)
    return matches.takeError();
  std::vector<::loom::fabric::FabricUsePatternRef> result;
  for (std::uint64_t capabilityOrdinal : *matches) {
    const auto &capability = service->capabilities()[capabilityOrdinal];
    if (!llvm::is_contained(capability.serviceRegionOrdinals,
                            serviceRegionOrdinal))
      continue;
    for (::fabric::UsePatternKey pattern : capability.admissibleUsePatterns)
      result.push_back(
          {::loom::fabric::FabricUsePatternOwnerRef(
               ::loom::fabric::FabricInventoryOwnerRef::of(
                   ::loom::fabric::FabricMemoryServiceRef::local(occurrence))),
           pattern.ordinal()});
  }
  canonicalizePatterns(result);
  if (result.empty())
    return invalid(
        "local memory dispatch has no exact service UsePattern domain");
  for (const auto &pattern : result)
    if (llvm::Error error = ::loom::fabric::validateFabricRef(fabric, pattern))
      return std::move(error);
  return result;
}

llvm::Expected<std::string>
requirementKey(const SpatialMemoryResourceOwnerRef &owner,
               const SpatialActivityEventRef &trigger,
               const ArtifactIdentity &dataflowIdentity) {
  std::string result;
  std::visit(
      [&](const auto &selected) {
        using Owner = std::decay_t<decltype(selected)>;
        std::uint64_t value = 0;
        if constexpr (std::is_same_v<Owner,
                                     SpatialMemoryEngineResourceOwnerRef>) {
          result.push_back(0);
          value = selected.realization;
        } else {
          result.push_back(1);
          value = selected.binding;
        }
        for (unsigned byte = 0; byte < 8; ++byte) {
          result.push_back(static_cast<char>(value >> (8 * (7 - byte))));
        }
      },
      owner);
  auto encoded = encodeSpatialActivityEventKey(dataflowIdentity, trigger);
  if (!encoded)
    return encoded.takeError();
  result.append(reinterpret_cast<const char *>(encoded->data()),
                encoded->size());
  return result;
}

llvm::Error appendRequirement(
    std::vector<SpatialMemoryResourceUseRequirement> &requirements,
    std::map<std::string, std::size_t> &requirementByKey,
    SpatialMemoryResourceOwnerRef owner, SpatialActivityEventRef trigger,
    std::vector<::loom::fabric::FabricUsePatternRef> patterns,
    const ArtifactIdentity &dataflowIdentity) {
  auto key = requirementKey(owner, trigger, dataflowIdentity);
  if (!key)
    return key.takeError();
  auto [found, inserted] =
      requirementByKey.try_emplace(*key, requirements.size());
  if (inserted) {
    requirements.push_back(
        {std::move(owner), std::move(trigger), std::move(patterns)});
    return llvm::Error::success();
  }
  auto &existing = requirements[found->second].admissiblePatterns;
  existing.insert(existing.end(), patterns.begin(), patterns.end());
  canonicalizePatterns(existing);
  return llvm::Error::success();
}

::dataflow::LogicalMemoryRootRef
rootOf(const ::dataflow::LogicalMemoryRootOrViewRef &memory) {
  return std::visit(
      [](const auto &reference) -> ::dataflow::LogicalMemoryRootRef {
        using Ref = std::decay_t<decltype(reference)>;
        if constexpr (std::is_same_v<Ref, ::dataflow::LogicalMemoryRootRef>)
          return reference;
        else
          return reference.root;
      },
      memory);
}

llvm::Expected<std::pair<std::uint64_t, std::uint64_t>>
finiteInterval(const SpatialMemoryBindingView &binding,
               const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  auto extent = dataflow.staticMemoryByteExtent(binding.logicalMemory);
  if (!extent)
    return extent.takeError();
  if (const auto *range =
          std::get_if<SpatialMemoryByteRangeView>(&binding.interval)) {
    if (!*extent)
      return invalid("ByteRange requires a finite logical memory extent");
    if (range->sizeBytes == 0 || range->offsetBytes > **extent ||
        range->sizeBytes > **extent - range->offsetBytes)
      return invalid("MemoryBinding ByteRange exceeds its logical memory");
    return std::make_pair(range->offsetBytes, range->sizeBytes);
  }
  if (!*extent)
    return invalid("local Whole MemoryBinding requires a finite extent");
  return std::make_pair(UINT64_C(0), **extent);
}

llvm::Expected<SpatialMemoryIntervalView> importInterval(Attribute attribute) {
  if (isa<::mapping::MemoryWholeIntervalAttr>(attribute))
    return SpatialMemoryIntervalView(
        std::in_place_type<SpatialMemoryWholeIntervalView>);
  auto range = dyn_cast<::mapping::MemoryByteRangeAttr>(attribute);
  if (!range || range.getSizeBytes() == 0)
    return invalid("MemoryBinding has an invalid logical interval");
  return SpatialMemoryIntervalView(
      std::in_place_type<SpatialMemoryByteRangeView>,
      SpatialMemoryByteRangeView{range.getOffsetBytes(), range.getSizeBytes()});
}

llvm::Expected<SpatialMemoryBindingTargetView>
importBindingTarget(Attribute attribute,
                    const ::loom::fabric::FabricArtifactView &fabric) {
  if (isa<::mapping::MemoryBoundaryProxyAttr>(attribute))
    return SpatialMemoryBindingTargetView(
        std::in_place_type<SpatialMemoryBoundaryProxyView>);
  auto local = dyn_cast<::mapping::MemoryLocalRegionAttr>(attribute);
  if (!local)
    return invalid("MemoryBinding has an unknown target variant");
  auto region = decodeFabric<::loom::fabric::FabricMemoryServiceRegionRef>(
      local.getServiceRegion());
  if (!region)
    return region.takeError();
  if (llvm::Error error = ::loom::fabric::validateFabricRef(fabric, *region))
    return std::move(error);
  if (region->service.kind() != ::loom::fabric::FabricMemoryServiceKind::Local)
    return invalid("LocalRegion names a non-local memory service");
  return SpatialMemoryBindingTargetView(
      std::in_place_type<SpatialMemoryLocalRegionView>,
      SpatialMemoryLocalRegionView{*region, local.getPhysicalOffsetBytes()});
}

llvm::Expected<SpatialMemoryDispatchTargetView>
importDispatch(Attribute attribute,
               const ::loom::fabric::FabricArtifactView &fabric) {
  if (auto local = dyn_cast<::mapping::LocalMemoryServiceRefAttr>(attribute)) {
    auto reference = decodeFabric<::loom::fabric::LocalMemoryServiceRef>(local);
    if (!reference)
      return reference.takeError();
    if (llvm::Error error =
            ::loom::fabric::validateFabricRef(fabric, *reference))
      return std::move(error);
    return SpatialMemoryDispatchTargetView(
        std::in_place_type<::loom::fabric::LocalMemoryServiceRef>, *reference);
  }
  auto manager = dyn_cast<::mapping::ManagerEndpointRefAttr>(attribute);
  if (!manager)
    return invalid("memory use has an unknown dispatch target");
  auto reference = decodeFabric<::loom::fabric::ManagerEndpointRef>(manager);
  if (!reference)
    return reference.takeError();
  if (llvm::Error error = ::loom::fabric::validateFabricRef(fabric, *reference))
    return std::move(error);
  return SpatialMemoryDispatchTargetView(
      std::in_place_type<::loom::fabric::ManagerEndpointRef>, *reference);
}

llvm::Expected<SpatialMemoryConsistencyTargetView>
importConsistency(Attribute attribute,
                  const ::loom::fabric::FabricArtifactView &fabric) {
  if (auto domain =
          dyn_cast<::mapping::MemoryConsistencyDomainRefAttr>(attribute)) {
    auto reference =
        decodeFabric<::loom::fabric::MemoryConsistencyDomainRef>(domain);
    if (!reference)
      return reference.takeError();
    if (llvm::Error error =
            ::loom::fabric::validateFabricRef(fabric, *reference))
      return std::move(error);
    return SpatialMemoryConsistencyTargetView(
        std::in_place_type<::loom::fabric::MemoryConsistencyDomainRef>,
        *reference);
  }
  auto manager = dyn_cast<::mapping::ManagerEndpointRefAttr>(attribute);
  if (!manager)
    return invalid("fence use has an unknown consistency target");
  auto reference = decodeFabric<::loom::fabric::ManagerEndpointRef>(manager);
  if (!reference)
    return reference.takeError();
  if (llvm::Error error = ::loom::fabric::validateFabricRef(fabric, *reference))
    return std::move(error);
  return SpatialMemoryConsistencyTargetView(
      std::in_place_type<::loom::fabric::ManagerEndpointRef>, *reference);
}

llvm::Expected<SpatialMemoryOperationPlacementView>
importPlacement(Attribute attribute,
                const ::loom::fabric::FabricMemoryOccurrenceRef &occurrence,
                const TechMemoryActorView &actor,
                const ::loom::fabric::FabricArtifactView &fabric) {
  const auto expectedPort = ::loom::fabric::FabricMemoryOperationPortRef{
      occurrence, actor.operationPort.ordinal};
  if (auto spatial =
          dyn_cast<::mapping::FabricMemoryOperationPortRefAttr>(attribute)) {
    auto port =
        decodeFabric<::loom::fabric::FabricMemoryOperationPortRef>(spatial);
    if (!port)
      return port.takeError();
    if (llvm::Error error = ::loom::fabric::validateFabricRef(fabric, *port))
      return std::move(error);
    if (*port != expectedPort)
      return invalid("memory actor selects the wrong occurrence port");
    auto schedule = fabric.memorySchedule(occurrence);
    if (!schedule || *schedule != ::fabric::Schedule::Spatial)
      return invalid("Spatial memory placement selects a non-Spatial engine");
    return SpatialMemoryOperationPlacementView(
        std::in_place_type<::loom::fabric::FabricMemoryOperationPortRef>,
        *port);
  }
  auto temporal =
      dyn_cast<::mapping::FabricMemoryOperationContextRefAttr>(attribute);
  if (!temporal)
    return invalid("memory actor has an unknown placement variant");
  auto context =
      decodeFabric<::loom::fabric::FabricMemoryOperationContextRef>(temporal);
  if (!context)
    return context.takeError();
  if (llvm::Error error = ::loom::fabric::validateFabricRef(fabric, *context))
    return std::move(error);
  auto schedule = fabric.memorySchedule(occurrence);
  if (context->port != expectedPort || !schedule ||
      *schedule != ::fabric::Schedule::Temporal ||
      context->ordinal >= fabric.memoryResidentContextCount(occurrence))
    return invalid("Temporal memory placement is incompatible with its engine");
  return SpatialMemoryOperationPlacementView(
      std::in_place_type<::loom::fabric::FabricMemoryOperationContextRef>,
      *context);
}

bool bindingAndDispatchAgree(const SpatialMemoryBindingView &binding,
                             const SpatialMemoryDispatchTargetView &dispatch) {
  if (const auto *local =
          std::get_if<SpatialMemoryLocalRegionView>(&binding.target)) {
    const auto *selected =
        std::get_if<::loom::fabric::LocalMemoryServiceRef>(&dispatch);
    return selected && selected->underlying() == local->serviceRegion.service;
  }
  return std::holds_alternative<::loom::fabric::ManagerEndpointRef>(dispatch);
}

bool dispatchBelongsToOccurrence(
    const SpatialMemoryDispatchTargetView &dispatch,
    ::loom::fabric::FabricMemoryOccurrenceRef occurrence) {
  if (const auto *local =
          std::get_if<::loom::fabric::LocalMemoryServiceRef>(&dispatch))
    return local->underlying() ==
           ::loom::fabric::FabricMemoryServiceRef::local(occurrence);
  const auto &endpoint =
      std::get<::loom::fabric::ManagerEndpointRef>(dispatch).underlying();
  return endpoint.owner ==
         ::loom::fabric::FabricMemoryEndpointOwnerRef::of(occurrence);
}

bool consistencyBelongsToOccurrence(
    const SpatialMemoryConsistencyTargetView &target,
    ::loom::fabric::FabricMemoryOccurrenceRef occurrence) {
  if (const auto *manager =
          std::get_if<::loom::fabric::ManagerEndpointRef>(&target))
    return manager->underlying().owner ==
           ::loom::fabric::FabricMemoryEndpointOwnerRef::of(occurrence);
  return true;
}

struct MemoryEndpointInventory final {
  std::vector<::loom::fabric::ManagerEndpointRef> managers;
  std::vector<::loom::fabric::SubordinateEndpointRef> subordinates;
};

llvm::Expected<MemoryEndpointInventory>
memoryEndpointInventory(const ::loom::fabric::FabricArtifactView &fabric,
                        ::loom::fabric::FabricMemoryOccurrenceRef occurrence) {
  MemoryEndpointInventory result;
  const auto owner =
      ::loom::fabric::FabricMemoryEndpointOwnerRef::of(occurrence);
  for (std::uint64_t ordinal = 0; ordinal < fabric.memoryEndpointCount(owner);
       ++ordinal) {
    const ::loom::fabric::FabricMemoryEndpointRef endpoint{owner, ordinal};
    const auto role = fabric.memoryEndpointRole(endpoint);
    if (!role)
      return invalid("memory occurrence has an untyped endpoint");
    if (*role == ::loom::fabric::FabricMemoryEndpointRole::Manager)
      result.managers.emplace_back(endpoint);
    else
      result.subordinates.emplace_back(endpoint);
  }
  return result;
}

bool targetAdmitted(llvm::ArrayRef<::fabric::MemoryDispatchTarget> domain,
                    const SpatialMemoryDispatchTargetView &target,
                    const MemoryEndpointInventory &endpoints) {
  return llvm::any_of(domain, [&](const auto &candidate) {
    if (std::holds_alternative<::loom::fabric::LocalMemoryServiceRef>(target))
      return std::holds_alternative<::fabric::LocalMemoryDispatchTarget>(
          candidate);
    const auto &manager =
        std::get<::loom::fabric::ManagerEndpointRef>(target).underlying();
    const auto *declared =
        std::get_if<::fabric::ManagerMemoryDispatchTarget>(&candidate);
    return declared && declared->endpointOrdinal < endpoints.managers.size() &&
           endpoints.managers[declared->endpointOrdinal].underlying() ==
               manager;
  });
}

llvm::Expected<std::uint64_t>
validateExposureDispatch(const ::loom::fabric::SubordinateEndpointRef &terminal,
                         const SpatialMemoryDispatchTargetView &dispatch,
                         const ::loom::fabric::FabricArtifactView &fabric) {
  const auto &endpoint = terminal.underlying();
  if (endpoint.owner.kind() !=
      ::loom::fabric::FabricMemoryEndpointOwnerKind::FabricMemoryOccurrence)
    return invalid("ExposureEntry terminal is not memory-occurrence-owned");
  const auto occurrence = std::get<::loom::fabric::FabricMemoryOccurrenceRef>(
      endpoint.owner.payload);
  auto endpoints = memoryEndpointInventory(fabric, occurrence);
  if (!endpoints)
    return endpoints.takeError();
  const auto found = llvm::find(endpoints->subordinates, terminal);
  if (found == endpoints->subordinates.end())
    return invalid("ExposureEntry terminal is absent from the subordinate "
                   "inventory");
  const std::size_t row = std::distance(endpoints->subordinates.begin(), found);
  const auto *connectivity = fabric.memoryConnectivity(occurrence);
  if (!connectivity || row >= connectivity->subordinateEndpoints().size())
    return invalid("ExposureEntry terminal has no subordinate H_dispatch row");
  if (!dispatchBelongsToOccurrence(dispatch, occurrence) ||
      !targetAdmitted(connectivity->subordinateEndpoints()[row].targetDomain,
                      dispatch, *endpoints))
    return invalid("ExposureEntry dispatch is outside Fabric H_dispatch");
  return connectivity->subordinateEndpoints()[row].maxExposedBindings;
}

using RootedLaunchInventory =
    std::map<std::string, std::vector<::dataflow::RootedGraphLaunchRef>>;

llvm::Expected<RootedLaunchInventory> buildRootedLaunchInventory(
    const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  RootedLaunchInventory result;
  std::optional<std::string> failure;
  dataflow.forEachRootedGraphLaunch([&](auto launch) {
    if (failure)
      return;
    auto graph = dataflow.resolve(launch);
    if (!graph) {
      failure = llvm::toString(graph.takeError());
      return;
    }
    auto key = ::dataflow::encodeDataflowReference(dataflow.identity(), *graph);
    if (!key) {
      failure = llvm::toString(key.takeError());
      return;
    }
    result[byteKey(*key)].push_back(launch);
  });
  if (failure)
    return invalid("cannot enumerate rooted graph launches: " + *failure);
  for (auto &[graph, launches] : result) {
    std::vector<std::pair<std::string, ::dataflow::RootedGraphLaunchRef>> keyed;
    keyed.reserve(launches.size());
    for (const auto &launch : launches) {
      auto key =
          ::dataflow::encodeDataflowReference(dataflow.identity(), launch);
      if (!key)
        return key.takeError();
      keyed.emplace_back(byteKey(*key), launch);
    }
    llvm::sort(keyed, [](const auto &left, const auto &right) {
      return left.first < right.first;
    });
    launches.clear();
    launches.reserve(keyed.size());
    for (auto &entry : keyed)
      launches.push_back(std::move(entry.second));
  }
  return result;
}

llvm::Expected<llvm::ArrayRef<::dataflow::RootedGraphLaunchRef>>
rootedUsesForActor(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                   const RootedLaunchInventory &inventory,
                   ::dataflow::ActorRef actor) {
  auto actorView = dataflow.resolve(actor);
  if (!actorView)
    return actorView.takeError();
  auto key = ::dataflow::encodeDataflowReference(dataflow.identity(),
                                                 actorView->graph);
  if (!key)
    return key.takeError();
  auto found = inventory.find(byteKey(*key));
  if (found == inventory.end() || found->second.empty())
    return invalid("memory actor has no rooted launch use");
  return llvm::ArrayRef<::dataflow::RootedGraphLaunchRef>(found->second);
}

llvm::Error
requireExactUses(llvm::ArrayRef<::dataflow::RootedGraphLaunchRef> expected,
                 llvm::ArrayRef<::dataflow::RootedGraphLaunchRef> actual) {
  if (expected.size() != actual.size())
    return invalid("memory actor rooted-use inventory is incomplete");
  for (auto [left, right] : llvm::zip_equal(expected, actual))
    if (left != right)
      return invalid("memory actor rooted-use inventory has a wrong launch");
  return llvm::Error::success();
}

llvm::Expected<SpatialMemoryBindingView>
importMemoryBinding(::mapping::MemoryBindingOp record,
                    const ::dataflow::CanonicalDataflowProgramView &dataflow,
                    const ::loom::fabric::FabricArtifactView &fabric) {
  auto logical = decodeDataflow<::dataflow::LogicalMemoryRootOrViewRef>(
      record.getLogicalMemory(), dataflow.identity());
  if (!logical)
    return logical.takeError();
  auto type = dataflow.memoryType(*logical);
  if (!type)
    return type.takeError();
  auto interval = importInterval(record.getInterval());
  if (!interval)
    return interval.takeError();
  auto target = importBindingTarget(record.getTarget(), fabric);
  if (!target)
    return target.takeError();
  SpatialMemoryBindingView result{record.getEntityId(),
                                  *logical,
                                  std::move(*interval),
                                  std::move(*target),
                                  {}};

  auto extent = dataflow.staticMemoryByteExtent(result.logicalMemory);
  if (!extent)
    return extent.takeError();
  if (const auto *range =
          std::get_if<SpatialMemoryByteRangeView>(&result.interval)) {
    if (!*extent || range->offsetBytes > **extent ||
        range->sizeBytes > **extent - range->offsetBytes)
      return invalid("MemoryBinding ByteRange exceeds its logical memory");
  }

  if (const auto *local =
          std::get_if<SpatialMemoryLocalRegionView>(&result.target)) {
    auto logicalInterval = finiteInterval(result, dataflow);
    if (!logicalInterval)
      return logicalInterval.takeError();
    const auto &service = local->serviceRegion.service;
    const auto memory =
        std::get<::loom::fabric::FabricMemoryOccurrenceRef>(service.payload);
    const auto *contract = fabric.localMemoryService(memory);
    if (!contract || local->serviceRegion.ordinal >= contract->regions().size())
      return invalid("LocalRegion names an absent service region");
    const auto &region = contract->regions()[local->serviceRegion.ordinal];
    if (local->physicalOffsetBytes > region.sizeBytes ||
        logicalInterval->second > region.sizeBytes - local->physicalOffsetBytes)
      return invalid(
          "LocalRegion physical interval exceeds its service region");
  }

  result.exposures.reserve(std::distance(
      record.getBody().front().getOps<::mapping::ExposureEntryOp>().begin(),
      record.getBody().front().getOps<::mapping::ExposureEntryOp>().end()));
  for (auto exposure :
       record.getBody().front().getOps<::mapping::ExposureEntryOp>()) {
    auto source = decodeDataflow<::dataflow::MemoryExposureRef>(
        exposure.getExposure(), dataflow.identity());
    if (!source)
      return source.takeError();
    auto resolved = dataflow.resolveExposure(*source);
    if (!resolved)
      return resolved.takeError();
    if (*resolved != result.logicalMemory)
      return invalid("ExposureEntry resolves to a different logical memory");
    auto terminal = decodeFabric<::loom::fabric::SubordinateEndpointRef>(
        exposure.getTerminal());
    if (!terminal)
      return terminal.takeError();
    if (llvm::Error error =
            ::loom::fabric::validateFabricRef(fabric, *terminal))
      return std::move(error);
    auto dispatch = importDispatch(exposure.getDispatchTarget(), fabric);
    if (!dispatch)
      return dispatch.takeError();
    if (!bindingAndDispatchAgree(result, *dispatch))
      return invalid("ExposureEntry dispatch disagrees with its MemoryBinding");
    result.exposures.push_back(
        SpatialExposureEntryView{*source, *terminal, std::move(*dispatch)});
  }
  return result;
}

struct ExpectedExposure final {
  ::dataflow::MemoryExposureRef reference;
  ::dataflow::LogicalMemoryRootOrViewRef logicalMemory;
};

llvm::Expected<std::map<std::string, ExpectedExposure>> deriveExpectedExposures(
    const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  std::map<std::string, ExpectedExposure> result;
  std::optional<std::string> failure;
  dataflow.forEachMemoryExposure([&](auto reference) {
    if (failure)
      return;
    auto logical = dataflow.resolveExposure(reference);
    if (!logical) {
      failure = llvm::toString(logical.takeError());
      return;
    }
    auto bytes =
        ::dataflow::encodeDataflowReference(dataflow.identity(), reference);
    if (!bytes) {
      failure = llvm::toString(bytes.takeError());
      return;
    }
    auto [entry, inserted] = result.try_emplace(
        byteKey(*bytes), ExpectedExposure{reference, *logical});
    if (!inserted && (entry->second.reference != reference ||
                      entry->second.logicalMemory != *logical)) {
      failure = "memory exposure inventory is not a function";
      return;
    }
  });
  if (failure)
    return invalid("cannot derive memory exposure inventory: " + *failure);
  return result;
}

llvm::Error verifyExposureInventory(
    llvm::ArrayRef<SpatialMemoryBindingView> bindings,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric) {
  auto expected = deriveExpectedExposures(dataflow);
  if (!expected)
    return expected.takeError();

  struct ProviderUse final {
    std::uint64_t maxBindings = 0;
    std::set<std::uint64_t> bindings;
  };
  std::set<std::string> actual;
  std::map<std::string, ProviderUse> providerUses;
  for (const SpatialMemoryBindingView &binding : bindings) {
    for (const SpatialExposureEntryView &exposure : binding.exposures) {
      auto bytes = ::dataflow::encodeDataflowReference(dataflow.identity(),
                                                       exposure.exposure);
      if (!bytes)
        return bytes.takeError();
      const std::string key = byteKey(*bytes);
      if (!actual.insert(key).second)
        return invalid("SpatialMapping duplicates a MemoryExposureRef");
      const auto expectedExposure = expected->find(key);
      if (expectedExposure == expected->end())
        return invalid("SpatialMapping contains an extra memory exposure");
      if (expectedExposure->second.logicalMemory != binding.logicalMemory)
        return invalid("memory exposure belongs to another MemoryBinding");

      auto maxBindings = validateExposureDispatch(exposure.terminal,
                                                  exposure.dispatch, fabric);
      if (!maxBindings)
        return maxBindings.takeError();
      const std::string terminalKey =
          byteKey(::loom::fabric::canonicalFabricBytes(exposure.terminal));
      auto [provider, inserted] =
          providerUses.try_emplace(terminalKey, ProviderUse{*maxBindings, {}});
      if (!inserted && provider->second.maxBindings != *maxBindings)
        return invalid("one subordinate terminal has inconsistent capacity");
      provider->second.bindings.insert(binding.entityId);
    }
  }
  if (actual.size() != expected->size())
    return invalid("SpatialMapping omits a required memory exposure");
  for (const auto &[terminal, use] : providerUses)
    if (use.bindings.size() > use.maxBindings)
      return invalid("subordinate terminal exceeds max_exposed_bindings");
  return llvm::Error::success();
}

llvm::Error verifyNoBindingOverlap(
    llvm::ArrayRef<SpatialMemoryBindingView> bindings,
    const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  struct Interval final {
    ::dataflow::LogicalMemoryRootRef root;
    std::uint64_t begin = 0;
    std::optional<std::uint64_t> end;
  };
  std::vector<Interval> intervals;
  intervals.reserve(bindings.size());
  for (const auto &binding : bindings) {
    Interval interval{rootOf(binding.logicalMemory), 0, std::nullopt};
    if (const auto *range =
            std::get_if<SpatialMemoryByteRangeView>(&binding.interval)) {
      interval.begin = range->offsetBytes;
      interval.end = range->offsetBytes + range->sizeBytes;
    } else {
      auto extent = dataflow.staticMemoryByteExtent(binding.logicalMemory);
      if (!extent)
        return extent.takeError();
      interval.end = *extent;
    }
    intervals.push_back(interval);
  }
  llvm::sort(intervals, [](const Interval &left, const Interval &right) {
    if (left.root.entity != right.root.entity)
      return left.root.entity.value() < right.root.entity.value();
    return left.begin < right.begin;
  });
  for (std::size_t index = 1; index < intervals.size(); ++index) {
    const Interval &previous = intervals[index - 1];
    const Interval &current = intervals[index];
    if (previous.root == current.root &&
        (!previous.end || current.begin < *previous.end))
      return invalid("overlapping MemoryBinding intervals require an explicit "
                     "Fabric composite service");
  }

  struct PhysicalInterval final {
    std::string region;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
  };
  std::vector<PhysicalInterval> physical;
  physical.reserve(bindings.size());
  for (const auto &binding : bindings) {
    const auto *local =
        std::get_if<SpatialMemoryLocalRegionView>(&binding.target);
    if (!local)
      continue;
    auto logical = finiteInterval(binding, dataflow);
    if (!logical)
      return logical.takeError();
    physical.push_back(
        {byteKey(::loom::fabric::canonicalFabricBytes(local->serviceRegion)),
         local->physicalOffsetBytes,
         local->physicalOffsetBytes + logical->second});
  }
  llvm::sort(physical, [](const auto &left, const auto &right) {
    return std::tie(left.region, left.begin, left.end) <
           std::tie(right.region, right.begin, right.end);
  });
  for (std::size_t index = 1; index < physical.size(); ++index)
    if (physical[index - 1].region == physical[index].region &&
        physical[index].begin < physical[index - 1].end)
      return invalid("MemoryBindings overlap in one local service region");
  return llvm::Error::success();
}

llvm::Expected<SpatialMemoryEngineBindingView>
importEngineBinding(::mapping::MemoryEngineBindingOp record,
                    llvm::ArrayRef<SpatialMemoryBindingView> bindings,
                    const RootedLaunchInventory &rootedLaunches,
                    const ::dataflow::CanonicalDataflowProgramView &dataflow,
                    const TechMappingView &techMapping,
                    const ::loom::fabric::FabricArtifactView &fabric) {
  const std::uint64_t entity = record.getRealization().getEntity();
  const TechMemoryRealizationView *realization =
      findRealization(techMapping, entity);
  if (!realization)
    return invalid("MemoryEngineBinding references an absent realization");
  auto occurrence = decodeFabric<::loom::fabric::FabricMemoryOccurrenceRef>(
      record.getOccurrence());
  if (!occurrence)
    return occurrence.takeError();
  if (llvm::Error error =
          ::loom::fabric::validateFabricRef(fabric, *occurrence))
    return std::move(error);
  auto engine = fabric.memoryEngineTemplateOf(*occurrence);
  if (!engine || *engine != realization->engine)
    return invalid("MemoryEngineBinding occurrence has the wrong template");
  const auto *connectivity = fabric.memoryConnectivity(*occurrence);
  if (!connectivity)
    return invalid(
        "MemoryEngineBinding occurrence has no connectivity contract");
  auto endpoints = memoryEndpointInventory(fabric, *occurrence);
  if (!endpoints)
    return endpoints.takeError();

  std::map<std::uint64_t, const SpatialMemoryBindingView *> bindingById;
  for (const auto &binding : bindings)
    bindingById.emplace(binding.entityId, &binding);

  SpatialMemoryEngineBindingView result{entity, *occurrence, {}};
  std::set<std::string> coveredActors;
  for (Operation &operation : record.getBody().front()) {
    const bool addressed =
        isa<::mapping::AddressedMemoryOperationOp>(operation);
    const bool fence = isa<::mapping::FenceMemoryOperationOp>(operation);
    if (!addressed && !fence)
      return invalid("MemoryEngineBinding has an unknown operation entry");
    Attribute actorAttribute =
        addressed
            ? cast<::mapping::AddressedMemoryOperationOp>(operation).getActor()
            : cast<::mapping::FenceMemoryOperationOp>(operation).getActor();
    auto actor = decodeDataflow<::dataflow::ActorRef>(
        cast<::mapping::ActorRefAttr>(actorAttribute), dataflow.identity());
    if (!actor)
      return actor.takeError();
    auto actorBytes =
        ::dataflow::encodeDataflowReference(dataflow.identity(), *actor);
    if (!actorBytes)
      return actorBytes.takeError();
    if (!coveredActors.insert(byteKey(*actorBytes)).second)
      return invalid("MemoryEngineBinding duplicates an actor entry");
    const TechMemoryActorView *techActor = findActor(*realization, *actor);
    if (!techActor)
      return invalid("memory operation is absent from its Tech realization");
    auto expectedUses = rootedUsesForActor(dataflow, rootedLaunches, *actor);
    if (!expectedUses)
      return expectedUses.takeError();

    if (addressed) {
      auto entry = cast<::mapping::AddressedMemoryOperationOp>(operation);
      auto placement = importPlacement(entry.getPlacement(), *occurrence,
                                       *techActor, fabric);
      if (!placement)
        return placement.takeError();
      SpatialAddressedMemoryOperationView imported{
          *actor, std::move(*placement), {}};
      for (auto use :
           entry.getBody().front().getOps<::mapping::AddressedMemoryUseOp>()) {
        auto launch = decodeDataflow<::dataflow::RootedGraphLaunchRef>(
            use.getLaunch(), dataflow.identity());
        if (!launch)
          return launch.takeError();
        auto found = bindingById.find(use.getBinding().getEntity());
        if (found == bindingById.end())
          return invalid("addressed use references an absent MemoryBinding");
        auto resolved = dataflow.resolveAddressedMemory(
            ::dataflow::ContextualActorRef{*launch, *actor});
        if (!resolved)
          return resolved.takeError();
        if (*resolved != found->second->logicalMemory)
          return invalid("addressed use binds the wrong logical memory");
        auto dispatch = importDispatch(use.getDispatchTarget(), fabric);
        if (!dispatch)
          return dispatch.takeError();
        if (!bindingAndDispatchAgree(*found->second, *dispatch) ||
            !dispatchBelongsToOccurrence(*dispatch, *occurrence))
          return invalid("addressed dispatch disagrees with its binding or "
                         "memory occurrence");
        const auto portOrdinal = techActor->operationPort.ordinal;
        const auto alternativeOrdinal = techActor->capability.ordinal;
        if (portOrdinal >= connectivity->operationPorts().size() ||
            alternativeOrdinal >= connectivity->operationPorts()[portOrdinal]
                                      .capabilityTargetDomains.size() ||
            !targetAdmitted(connectivity->operationPorts()[portOrdinal]
                                .capabilityTargetDomains[alternativeOrdinal],
                            *dispatch, *endpoints))
          return invalid("addressed dispatch is outside Fabric H_dispatch");
        imported.uses.push_back(SpatialAddressedMemoryUseView{
            *launch, use.getBinding().getEntity(), std::move(*dispatch)});
      }
      std::vector<::dataflow::RootedGraphLaunchRef> actual;
      actual.reserve(imported.uses.size());
      for (const auto &use : imported.uses)
        actual.push_back(use.launch);
      if (llvm::Error error = requireExactUses(*expectedUses, actual))
        return std::move(error);
      result.operations.emplace_back(std::move(imported));
      continue;
    }

    auto entry = cast<::mapping::FenceMemoryOperationOp>(operation);
    auto placement =
        importPlacement(entry.getPlacement(), *occurrence, *techActor, fabric);
    if (!placement)
      return placement.takeError();
    auto fenceFamily = dataflow.asFenceFamily(*actor);
    if (!fenceFamily)
      return fenceFamily.takeError();
    SpatialFenceMemoryOperationView imported{*actor, std::move(*placement), {}};
    for (auto use :
         entry.getBody().front().getOps<::mapping::FenceMemoryUseOp>()) {
      auto launch = decodeDataflow<::dataflow::RootedGraphLaunchRef>(
          use.getLaunch(), dataflow.identity());
      if (!launch)
        return launch.takeError();
      if (llvm::Error error = dataflow.validate(
              ::dataflow::ContextualActorRef{*launch, *actor}))
        return std::move(error);
      auto consistency = importConsistency(use.getConsistencyTarget(), fabric);
      if (!consistency)
        return consistency.takeError();
      if (!consistencyBelongsToOccurrence(*consistency, *occurrence))
        return invalid("fence manager target belongs to another occurrence");
      imported.uses.push_back(
          SpatialFenceMemoryUseView{*launch, std::move(*consistency)});
    }
    std::vector<::dataflow::RootedGraphLaunchRef> actual;
    actual.reserve(imported.uses.size());
    for (const auto &use : imported.uses)
      actual.push_back(use.launch);
    if (llvm::Error error = requireExactUses(*expectedUses, actual))
      return std::move(error);
    result.operations.emplace_back(std::move(imported));
  }
  if (coveredActors.size() != realization->actors.size())
    return invalid("MemoryEngineBinding omits a Tech memory actor");
  return result;
}

llvm::Expected<std::vector<SpatialMemoryResourceUseRequirement>>
deriveMemoryResourceUseRequirements(
    llvm::ArrayRef<SpatialMemoryEngineBindingView> engines,
    llvm::ArrayRef<SpatialMemoryBindingView> bindings,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric) {
  std::map<std::uint64_t, const SpatialMemoryBindingView *> bindingById;
  for (const auto &binding : bindings)
    bindingById.emplace(binding.entityId, &binding);

  std::vector<SpatialMemoryResourceUseRequirement> result;
  std::map<std::string, std::size_t> requirementByKey;
  for (const auto &engine : engines) {
    const TechMemoryRealizationView *realization =
        findRealization(techMapping, engine.realization);
    if (!realization)
      return invalid("memory ResourceUse owner has no Tech realization");
    for (const auto &operation : engine.operations) {
      const auto actor = std::visit(
          [](const auto &selected) { return selected.actor; }, operation);
      const auto &placement = std::visit(
          [](const auto &selected)
              -> const SpatialMemoryOperationPlacementView & {
            return selected.placement;
          },
          operation);
      const TechMemoryActorView *techActor = findActor(*realization, actor);
      if (!techActor)
        return invalid("memory ResourceUse actor has no Tech realization");
      auto projection = projectMemoryActor(dataflow, actor);
      if (!projection)
        return projection.takeError();
      auto operationPatterns =
          operationUsePatterns(fabric, *techActor, placement, *projection);
      if (!operationPatterns)
        return operationPatterns.takeError();
      if (llvm::Error error = appendRequirement(
              result, requirementByKey,
              SpatialMemoryEngineResourceOwnerRef{engine.realization},
              projection->trigger, std::move(*operationPatterns),
              dataflow.identity()))
        return std::move(error);

      const auto *addressed =
          std::get_if<SpatialAddressedMemoryOperationView>(&operation);
      if (!addressed)
        continue;
      for (const auto &use : addressed->uses) {
        if (!std::holds_alternative<::loom::fabric::LocalMemoryServiceRef>(
                use.dispatch))
          continue;
        auto foundBinding = bindingById.find(use.binding);
        if (foundBinding == bindingById.end())
          return invalid("local service ResourceUse has no MemoryBinding");
        const auto *target = std::get_if<SpatialMemoryLocalRegionView>(
            &foundBinding->second->target);
        if (!target)
          return invalid(
              "local service ResourceUse has a BoundaryProxy binding");
        const auto service = target->serviceRegion.service;
        if (service.kind() != ::loom::fabric::FabricMemoryServiceKind::Local)
          return invalid("local service ResourceUse has a non-local owner");
        const auto occurrence =
            std::get<::loom::fabric::FabricMemoryOccurrenceRef>(
                service.payload);
        if (occurrence != engine.occurrence)
          return invalid("local service ResourceUse belongs to another engine");
        auto servicePatterns = localServiceUsePatterns(
            fabric, occurrence, target->serviceRegion.ordinal, *projection);
        if (!servicePatterns)
          return servicePatterns.takeError();
        if (llvm::Error error = appendRequirement(
                result, requirementByKey,
                SpatialMemoryBindingResourceOwnerRef{use.binding},
                projection->trigger, std::move(*servicePatterns),
                dataflow.identity()))
          return std::move(error);
      }
    }
  }
  return result;
}

} // namespace

llvm::Expected<ImportedSpatialMemoryView> importSpatialMemoryView(
    ::mapping::SpatialOp root,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric) {
  ImportedSpatialMemoryView result;
  auto rootedLaunches = buildRootedLaunchInventory(dataflow);
  if (!rootedLaunches)
    return rootedLaunches.takeError();
  std::set<std::uint64_t> bindingIds;
  for (auto record :
       root.getBody().front().getOps<::mapping::MemoryBindingOp>()) {
    if (!bindingIds.insert(record.getEntityId()).second)
      return invalid("duplicate MemoryBinding EntityId");
    auto binding = importMemoryBinding(record, dataflow, fabric);
    if (!binding)
      return binding.takeError();
    result.memoryBindings.push_back(std::move(*binding));
  }
  if (llvm::Error error =
          verifyNoBindingOverlap(result.memoryBindings, dataflow))
    return std::move(error);
  if (llvm::Error error =
          verifyExposureInventory(result.memoryBindings, dataflow, fabric))
    return std::move(error);

  std::set<std::uint64_t> engineIds;
  for (auto record :
       root.getBody().front().getOps<::mapping::MemoryEngineBindingOp>()) {
    const std::uint64_t entity = record.getRealization().getEntity();
    if (!engineIds.insert(entity).second)
      return invalid("duplicate MemoryEngineBinding realization");
    auto binding =
        importEngineBinding(record, result.memoryBindings, *rootedLaunches,
                            dataflow, techMapping, fabric);
    if (!binding)
      return binding.takeError();
    result.engineBindings.push_back(std::move(*binding));
  }
  if (result.engineBindings.size() != techMapping.memoryRealizations().size())
    return invalid("SpatialMapping omits a Tech memory realization");
  auto requiredUses = deriveMemoryResourceUseRequirements(
      result.engineBindings, result.memoryBindings, dataflow, techMapping,
      fabric);
  if (!requiredUses)
    return requiredUses.takeError();
  result.requiredResourceUses = std::move(*requiredUses);
  return result;
}

} // namespace loom::mapping::detail
