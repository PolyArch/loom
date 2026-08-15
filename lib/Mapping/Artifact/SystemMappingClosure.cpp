#include "SystemMappingClosure.h"

#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "MappingResourceUseImport.h"
#include "SystemMappingServiceTargetVerification.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <variant>

namespace loom::mapping::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_mapping_invalid: " + message);
}

std::vector<std::uint8_t> unsignedBytes(mlir::DenseI8ArrayAttr record) {
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

void appendU64(std::string &bytes, std::uint64_t value) {
  for (unsigned shift = 56;; shift -= 8) {
    bytes.push_back(static_cast<char>(value >> shift));
    if (shift == 0)
      break;
  }
}

void appendSized(std::string &bytes, llvm::StringRef value) {
  appendU64(bytes, value.size());
  bytes.append(value.data(), value.size());
}

template <typename Ref> std::string fabricKey(const Ref &reference) {
  return byteKey(::loom::fabric::canonicalFabricBytes(reference));
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

llvm::Expected<SystemPresburgerCell>
decodeCell(::mapping::SystemPresburgerCellAttr attribute) {
  SystemPresburgerCell cell;
  cell.dimensionCount = attribute.getDimensionCount();
  cell.symbolCount = attribute.getSymbolCount();
  cell.localCount = attribute.getLocalCount();
  const auto appendRows = [](mlir::ArrayAttr attributes,
                             std::vector<std::vector<std::int64_t>> &rows) {
    rows.reserve(attributes.size());
    for (mlir::Attribute raw : attributes) {
      auto values = mlir::cast<mlir::DenseI64ArrayAttr>(raw).asArrayRef();
      rows.emplace_back(values.begin(), values.end());
    }
  };
  appendRows(attribute.getEqualities(), cell.equalities);
  appendRows(attribute.getInequalities(), cell.inequalities);
  return canonicalizeSystemPresburgerCell(cell);
}

std::string attributeKey(mlir::Attribute attribute) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  attribute.print(stream);
  stream.flush();
  return result;
}

llvm::Expected<SpatialMemoryIntervalView>
importInterval(mlir::Attribute interval) {
  if (mlir::isa<::mapping::MemoryWholeIntervalAttr>(interval))
    return SpatialMemoryIntervalView(SpatialMemoryWholeIntervalView{});
  auto range = mlir::dyn_cast<::mapping::MemoryByteRangeAttr>(interval);
  if (!range)
    return invalid("memory target has an unknown logical interval");
  return SpatialMemoryIntervalView(
      SpatialMemoryByteRangeView{range.getOffsetBytes(), range.getSizeBytes()});
}

llvm::Expected<SystemServicePlanElementView>
importElement(mlir::Attribute element,
              const ::dataflow::CanonicalDataflowProgramView &dataflow,
              const ::loom::fabric::FabricSystemRootView &fabric) {
  if (auto leg =
          mlir::dyn_cast<::mapping::TransferLegElementKeyAttr>(element)) {
    auto key = decodeCanonicalServiceLegKey(
        unsignedBytes(leg.getLeg().getRecord()), dataflow.identity());
    if (!key)
      return key.takeError();
    return SystemServicePlanElementView(std::move(*key));
  }
  if (auto memory =
          mlir::dyn_cast<::mapping::MemoryRegionElementKeyAttr>(element)) {
    auto logical = decodeDataflow<::dataflow::LogicalMemoryRootOrViewRef>(
        memory.getLogicalMemory(), dataflow.identity());
    auto interval = importInterval(memory.getInterval());
    auto region = decodeFabric<::loom::fabric::FabricMemoryServiceRegionRef>(
        memory.getServiceRegion());
    if (!logical)
      return logical.takeError();
    if (!interval)
      return interval.takeError();
    if (!region)
      return region.takeError();
    if (llvm::Error error =
            ::loom::fabric::validateFabricRef(fabric.artifact(), *region))
      return std::move(error);
    std::vector<::loom::fabric::SystemServiceTransformRef> transforms;
    for (mlir::Attribute raw : memory.getTransformPath()) {
      auto transform = decodeFabric<::loom::fabric::SystemServiceTransformRef>(
          mlir::cast<::mapping::SystemServiceTransformRefAttr>(raw));
      if (!transform)
        return transform.takeError();
      if (llvm::Error error =
              ::loom::fabric::validateFabricRef(fabric.artifact(), *transform))
        return std::move(error);
      transforms.push_back(*transform);
    }
    return SystemServicePlanElementView(SystemMemoryRegionElementView{
        *logical, std::move(*interval), *region, std::move(transforms)});
  }
  auto consistency =
      mlir::dyn_cast<::mapping::ConsistencyElementKeyAttr>(element);
  if (!consistency)
    return invalid("ResourceUse owner has an unknown ServicePlan element");
  auto fence = decodeDataflow<::dataflow::FenceActorFamilyRef>(
      consistency.getFence(), dataflow.identity());
  auto domain = decodeFabric<::loom::fabric::MemoryConsistencyDomainRef>(
      consistency.getConsistencyDomain());
  if (!fence)
    return fence.takeError();
  if (!domain)
    return domain.takeError();
  if (llvm::Error error =
          ::loom::fabric::validateFabricRef(fabric.artifact(), *domain))
    return std::move(error);
  return SystemServicePlanElementView(
      SystemConsistencyElementView{*fence, *domain});
}

llvm::Expected<SystemEventPointView>
importEventPoint(::mapping::SystemEventPointAttr point,
                 const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  auto event = decodeDataflow<::dataflow::EventFamilyKey>(point.getEvent(),
                                                          dataflow.identity());
  if (!event)
    return event.takeError();
  if (llvm::Error error = dataflow.validate(*event))
    return std::move(error);
  if (point.getGuaranteedOffset())
    return invalid("guaranteed event offset requires its Fabric timing codec");
  return SystemEventPointView{std::move(*event), std::nullopt};
}

llvm::Expected<SystemRelativeActivationView>
importActivation(::mapping::SystemRelativeActivationAttr activation,
                 const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  auto trigger = importEventPoint(activation.getTrigger(), dataflow);
  if (!trigger)
    return trigger.takeError();
  std::vector<SystemEventPointView> release;
  release.reserve(activation.getRelease().size());
  for (mlir::Attribute attribute : activation.getRelease()) {
    auto imported = importEventPoint(
        mlir::cast<::mapping::SystemEventPointAttr>(attribute), dataflow);
    if (!imported)
      return imported.takeError();
    release.push_back(std::move(*imported));
  }
  return SystemRelativeActivationView{std::move(*trigger), std::move(release)};
}

llvm::Expected<std::string>
eventKey(const ::dataflow::CanonicalDataflowProgramView &dataflow,
         const ::dataflow::EventFamilyKey &event) {
  auto encoded =
      ::dataflow::encodeDataflowReference(dataflow.identity(), event);
  if (!encoded)
    return encoded.takeError();
  return byteKey(*encoded);
}

llvm::Expected<std::string>
resourceKey(mlir::Attribute owner,
            const ::dataflow::CanonicalDataflowProgramView &dataflow,
            const ::dataflow::EventFamilyKey &trigger) {
  auto event = eventKey(dataflow, trigger);
  if (!event)
    return event.takeError();
  return attributeKey(owner) + *event;
}

llvm::Expected<std::string>
capacityActivationKey(llvm::StringRef ownerEvent,
                      const ::loom::fabric::FabricArtifactView &fabric,
                      const ::loom::fabric::FabricUsePatternRef &patternRef,
                      llvm::ArrayRef<::fabric::UsePatternValue> parameters) {
  const ::fabric::ResourceContract *contract =
      fabric.resourceContract(patternRef.owner.catalog());
  if (!contract || patternRef.ordinal >= contract->usePatternCount())
    return invalid("ResourceUse does not resolve a Fabric pattern");
  const ::fabric::UsePattern pattern =
      contract->usePattern(::fabric::UsePatternKey(patternRef.ordinal));
  if (pattern.parameters.size() != parameters.size())
    return invalid("ResourceUse parameter count disagrees with its pattern");

  std::string result;
  appendSized(result, ownerEvent);
  appendU64(result, parameters.size());
  for (const auto &[schema, parameter] :
       llvm::zip_equal(pattern.parameters, parameters)) {
    auto encoded = ::fabric::encodeUsePatternValue(schema, parameter);
    if (!encoded)
      return encoded.takeError();
    appendU64(result, encoded->size());
    result.append(reinterpret_cast<const char *>(encoded->data()),
                  encoded->size());
  }
  return result;
}

::dataflow::EventFamilyKey
rootStartEvent(::dataflow::RootThreadLaunchRef root) {
  const ::dataflow::RootThreadBoundaryTransferRef transfer(
      ::dataflow::RootThreadStartTransferRef{root});
  return ::dataflow::EventFamilyKey(::dataflow::StaticTransferEventRef(
      ::dataflow::ConsumedTransferEventRef{::dataflow::CanonicalSinkTerminalRef(
          ::dataflow::RootThreadBoundarySinkRef{transfer})}));
}

::dataflow::EventFamilyKey
rootCompletionEvent(::dataflow::RootThreadLaunchRef root) {
  const ::dataflow::RootThreadBoundaryTransferRef transfer(
      ::dataflow::RootThreadCompletionTransferRef{root});
  return ::dataflow::EventFamilyKey(
      ::dataflow::StaticTransferEventRef(::dataflow::ProducedTransferEventRef{
          ::dataflow::CanonicalProducerTerminalRef(
              ::dataflow::RootThreadBoundarySourceRef{transfer})}));
}

llvm::Expected<mlir::Attribute> planElementKey(mlir::Operation &operation) {
  if (auto route =
          mlir::dyn_cast<::mapping::TransferLegRealizationOp>(operation))
    return mlir::Attribute(::mapping::TransferLegElementKeyAttr::get(
        route.getContext(), route.getLeg()));
  if (auto target = mlir::dyn_cast<::mapping::MemoryRegionTargetOp>(operation))
    return mlir::Attribute(::mapping::MemoryRegionElementKeyAttr::get(
        target.getContext(), target.getLogicalMemory(), target.getInterval(),
        target.getServiceRegion(), target.getTransformPath()));
  if (auto target = mlir::dyn_cast<::mapping::ConsistencyTargetOp>(operation))
    return mlir::Attribute(::mapping::ConsistencyElementKeyAttr::get(
        target.getContext(), target.getFence(), target.getConsistencyDomain()));
  return invalid("ServicePlan has an unknown element kind");
}

struct PlanRecord final {
  ::mapping::ServicePlanOp operation;
  std::map<std::string, mlir::Attribute> elements;
  SystemServicePlanView view;
};

using PlanMap = std::map<std::uint64_t, PlanRecord>;

llvm::Error verifyRouteContinuity(
    const SystemTransferLegView &route,
    const std::map<std::string,
                   const ::loom::fabric::FabricPhysicalTraversalView *>
        &traversals) {
  std::map<std::uint64_t, const SystemTransferRouteNodeView *> nodes;
  for (const auto &node : route.nodes)
    nodes.emplace(node.ordinal, &node);
  std::map<std::uint64_t,
           std::vector<::loom::fabric::FabricTransportEndpointRef>>
      endpoints;
  endpoints.emplace(0, std::vector{route.rootEndpoint});
  std::set<std::uint64_t> parents;
  std::set<std::uint64_t> sinkNodes;
  std::set<std::uint64_t> visiting;
  std::function<llvm::Error(std::uint64_t)> resolve =
      [&](std::uint64_t ordinal) -> llvm::Error {
    if (endpoints.count(ordinal) != 0)
      return llvm::Error::success();
    auto node = nodes.find(ordinal);
    if (node == nodes.end())
      return invalid("service route references an absent node");
    if (!visiting.insert(ordinal).second)
      return invalid("service route contains a node cycle");
    if (llvm::Error error = resolve(node->second->parentOrdinal))
      return error;
    auto parent = endpoints.find(node->second->parentOrdinal);
    auto traversal =
        traversals.find(fabricKey(node->second->incomingTraversal));
    if (traversal == traversals.end())
      return invalid("service route names an absent Fabric traversal");
    if (!llvm::any_of(parent->second, [&](const auto &endpoint) {
          return llvm::is_contained(traversal->second->sources, endpoint);
        }))
      return invalid("service route traversal is discontinuous");
    endpoints.emplace(ordinal, traversal->second->destinations);
    visiting.erase(ordinal);
    return llvm::Error::success();
  };
  for (const auto &node : route.nodes) {
    parents.insert(node.parentOrdinal);
    if (llvm::Error error = resolve(node.ordinal))
      return error;
  }
  for (const auto &sink : route.sinks) {
    if (endpoints.count(sink.nodeOrdinal) == 0)
      return invalid("service route sink references an absent node");
    sinkNodes.insert(sink.nodeOrdinal);
  }
  for (const auto &node : route.nodes)
    if (parents.count(node.ordinal) == 0 && sinkNodes.count(node.ordinal) == 0)
      return invalid("service route contains a non-sink leaf");
  return llvm::Error::success();
}

llvm::Expected<PlanMap>
importPlans(::mapping::ServiceRealizationOp service,
            const ::dataflow::CanonicalDataflowProgramView &dataflow,
            const ::loom::fabric::FabricSystemRootView &fabric) {
  PlanMap result;
  std::map<std::string, const ::loom::fabric::FabricPhysicalTraversalView *>
      traversals;
  for (const auto &traversal : fabric.artifact().physicalTraversals())
    traversals.emplace(fabricKey(traversal.reference), &traversal);
  for (auto plan :
       service.getBody().front().getOps<::mapping::ServicePlanOp>()) {
    PlanRecord imported{
        plan, {}, SystemServicePlanView{plan.getPlanOrdinal(), {}, {}, {}}};
    for (mlir::Operation &child : plan.getBody().front()) {
      auto element = planElementKey(child);
      if (!element)
        return element.takeError();
      auto importedElement = importElement(*element, dataflow, fabric);
      if (!importedElement)
        return importedElement.takeError();
      imported.elements.emplace(attributeKey(*element), *element);
      if (auto route =
              mlir::dyn_cast<::mapping::TransferLegRealizationOp>(child)) {
        auto leg = decodeCanonicalServiceLegKey(
            unsignedBytes(route.getLeg().getRecord()), dataflow.identity());
        auto rootEndpoint =
            decodeFabric<::loom::fabric::FabricTransportEndpointRef>(
                route.getRootEndpoint());
        if (!leg)
          return leg.takeError();
        if (!rootEndpoint)
          return rootEndpoint.takeError();
        if (llvm::Error error = ::loom::fabric::validateFabricRef(
                fabric.artifact(), *rootEndpoint))
          return std::move(error);
        SystemTransferLegView routeView{*leg, *rootEndpoint, {}, {}};
        for (mlir::Operation &routeChild : route.getBody().front()) {
          if (auto node =
                  mlir::dyn_cast<::mapping::SystemRouteNodeOp>(routeChild)) {
            auto traversal =
                decodeFabric<::loom::fabric::FabricPhysicalTraversalRef>(
                    node.getIncomingTraversal());
            if (!traversal)
              return traversal.takeError();
            if (llvm::Error error = ::loom::fabric::validateFabricRef(
                    fabric.artifact(), *traversal))
              return std::move(error);
            routeView.nodes.push_back(SystemTransferRouteNodeView{
                node.getNodeOrdinal(), node.getParentNodeOrdinal(),
                *traversal});
            continue;
          }
          auto sink = mlir::cast<::mapping::SystemRouteSinkOp>(routeChild);
          auto terminal = decodeSystemTransferTerminalKey(
              unsignedBytes(sink.getTerminal().getRecord()),
              dataflow.identity());
          if (!terminal)
            return terminal.takeError();
          routeView.sinks.push_back(SystemTransferRouteSinkView{
              std::move(*terminal), sink.getNodeOrdinal()});
        }
        if (llvm::Error error = verifyRouteContinuity(routeView, traversals))
          return std::move(error);
        imported.view.transferLegs.push_back(std::move(routeView));
        continue;
      }
      if (auto target =
              mlir::dyn_cast<::mapping::MemoryRegionTargetOp>(child)) {
        SystemMemoryRegionTargetView targetView{
            std::get<SystemMemoryRegionElementView>(*importedElement), {}};
        for (auto exposure : target.getBody()
                                 .front()
                                 .getOps<::mapping::SystemMemoryExposureOp>()) {
          auto exposureRef = decodeDataflow<::dataflow::MemoryExposureRef>(
              exposure.getExposure(), dataflow.identity());
          auto terminal = decodeFabric<::loom::fabric::SubordinateEndpointRef>(
              exposure.getTerminal());
          if (!exposureRef)
            return exposureRef.takeError();
          if (!terminal)
            return terminal.takeError();
          if (llvm::Error error = ::loom::fabric::validateFabricRef(
                  fabric.artifact(), *terminal))
            return std::move(error);
          targetView.exposures.push_back(
              SystemMemoryExposureView{*exposureRef, *terminal});
        }
        imported.view.memoryTargets.push_back(std::move(targetView));
        continue;
      }
      imported.view.consistencyTargets.push_back(
          std::get<SystemConsistencyElementView>(*importedElement));
    }
    if (!result.emplace(plan.getPlanOrdinal(), std::move(imported)).second)
      return invalid("ServiceRealization has duplicate plan ordinals");
  }
  return result;
}

bool anchorBelongsTo(const ServicePlanSelectionAnchor &anchor,
                     const SystemServiceObligationProjection &projection) {
  if (const auto *member =
          std::get_if<ServiceMemberPlanSelectionAnchor>(&anchor))
    return llvm::is_contained(projection.members, member->member);
  return llvm::is_contained(
      projection.exposures,
      std::get<MemoryExposurePlanSelectionAnchor>(anchor).exposure);
}

using SelectionOwner = std::variant<::dataflow::RootThreadLaunchRef,
                                    ::dataflow::RootedGraphLaunchRef>;

llvm::Expected<SelectionOwner>
messageProducerOwner(const TransferObligationFamilyKey &producer) {
  return std::visit(
      [](const auto &typed) -> llvm::Expected<SelectionOwner> {
        using Producer = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Producer,
                                     ::dataflow::RootThreadBoundarySourceRef>) {
          return SelectionOwner(
              std::visit([](const auto &transfer) { return transfer.launch; },
                         typed.transfer));
        } else if constexpr (std::is_same_v<
                                 Producer,
                                 ::dataflow::GraphLaunchBoundarySourceRef>) {
          return SelectionOwner(
              std::visit([](const auto &transfer) { return transfer.launch; },
                         typed.transfer));
        } else {
          return std::visit(
              [](const auto &channel) -> SelectionOwner {
                using Channel = std::decay_t<decltype(channel)>;
                if constexpr (std::is_same_v<
                                  Channel,
                                  ::dataflow::GraphStreamOutputProducerRef>)
                  return channel.launch;
                else
                  return channel.launch;
              },
              typed.producer);
        }
      },
      producer);
}

llvm::Expected<SelectionOwner>
selectionOwner(const SystemServiceObligationProjection &projection,
               const ServicePlanSelectionAnchor &anchor) {
  if (const auto *member =
          std::get_if<ServiceMemberPlanSelectionAnchor>(&anchor)) {
    if (std::holds_alternative<::dataflow::MessageTransferMemberRef>(
            member->member)) {
      const auto *producer =
          std::get_if<TransferObligationFamilyKey>(&projection.key);
      if (!producer)
        return invalid("message selection belongs to a non-message service");
      return messageProducerOwner(*producer);
    }
    if (const auto *addressed =
            std::get_if<::dataflow::AddressedMemoryActorMemberRef>(
                &member->member))
      return SelectionOwner(addressed->actor.launch);
    if (const auto *fence =
            std::get_if<::dataflow::FenceActorMemberRef>(&member->member))
      return SelectionOwner(fence->actor.launch);
    return invalid("selection anchor has an unknown service member");
  }
  return SelectionOwner(
      std::get<MemoryExposurePlanSelectionAnchor>(anchor).exposure.launch);
}

struct SelectionRequirement final {
  ServicePlanSelectionKey key;
  std::vector<SystemPresburgerCell> domain;
};

llvm::Expected<std::map<std::string, SelectionRequirement>>
selectionRequirements(const SystemServiceObligationProjection &projection,
                      const SystemExecutionContextProjection &contexts,
                      const ArtifactIdentity &dataflowIdentity) {
  std::vector<ServicePlanSelectionAnchor> anchors;
  anchors.reserve(projection.members.size() + projection.exposures.size());
  for (const auto &member : projection.members)
    anchors.emplace_back(ServiceMemberPlanSelectionAnchor{member});
  for (const auto &exposure : projection.exposures)
    anchors.emplace_back(MemoryExposurePlanSelectionAnchor{exposure});

  std::map<std::string, SelectionRequirement> result;
  for (const auto &anchor : anchors) {
    auto owner = selectionOwner(projection, anchor);
    if (!owner)
      return owner.takeError();
    bool found = false;
    if (const auto *root =
            std::get_if<::dataflow::RootThreadLaunchRef>(&*owner)) {
      for (const auto &domain : contexts.instructionDomains) {
        if (domain.root != *root)
          continue;
        found = true;
        ServicePlanSelectionKey key{anchor, domain.context};
        auto encoded = encodeServicePlanSelectionKey(dataflowIdentity, key);
        if (!encoded)
          return encoded.takeError();
        if (!result
                 .emplace(byteKey(*encoded),
                          SelectionRequirement{std::move(key), domain.cells})
                 .second)
          return invalid("duplicate derived Instruction selection context");
      }
    } else {
      const auto graph = std::get<::dataflow::RootedGraphLaunchRef>(*owner);
      for (const auto &domain : contexts.spatialDomains) {
        if (domain.graph != graph)
          continue;
        found = true;
        ServicePlanSelectionKey key{anchor, domain.context};
        auto encoded = encodeServicePlanSelectionKey(dataflowIdentity, key);
        if (!encoded)
          return encoded.takeError();
        if (!result
                 .emplace(byteKey(*encoded),
                          SelectionRequirement{std::move(key), domain.cells})
                 .second)
          return invalid("duplicate derived Spatial selection context");
      }
    }
    if (!found)
      return invalid("selection anchor has no reachable execution context");
  }
  return result;
}

llvm::Error
verifySelectionRelation(const SystemServicePlanSelectionView &selection,
                        llvm::ArrayRef<SystemPresburgerCell> contextDomain,
                        const std::set<std::uint64_t> &planOrdinals) {
  std::vector<SystemPresburgerCell> relationCells;
  for (const auto &clause : selection.clauses) {
    if (planOrdinals.count(clause.target) == 0)
      return invalid("ServicePlanSelection names an absent plan");
    if (clause.cells.empty())
      return invalid("ServicePlanSelection has an empty relation clause");
    relationCells.insert(relationCells.end(), clause.cells.begin(),
                         clause.cells.end());
  }
  if (selection.defaultPlanOrdinal &&
      planOrdinals.count(*selection.defaultPlanOrdinal) == 0)
    return invalid("ServicePlanSelection default names an absent plan");
  if (relationCells.empty() && !selection.defaultPlanOrdinal)
    return invalid("ServicePlanSelection relation is empty");

  auto within = systemPresburgerSetIsSubsetOf(relationCells, contextDomain);
  if (!within)
    return within.takeError();
  if (!*within)
    return invalid(
        "ServicePlanSelection extends beyond its execution context domain");
  for (std::size_t left = 0; left < relationCells.size(); ++left)
    for (std::size_t right = left + 1; right < relationCells.size(); ++right) {
      auto overlap = systemPresburgerCellsIntersect(relationCells[left],
                                                    relationCells[right]);
      if (!overlap)
        return overlap.takeError();
      if (*overlap)
        return invalid("ServicePlanSelection relation cells overlap");
    }
  auto covered = systemPresburgerSetIsSubsetOf(contextDomain, relationCells);
  if (!covered)
    return covered.takeError();
  if (selection.defaultPlanOrdinal.has_value() == *covered)
    return invalid(selection.defaultPlanOrdinal
                       ? "ServicePlanSelection default has an empty complement"
                       : "ServicePlanSelection does not cover its context");
  return llvm::Error::success();
}

llvm::Error verifySelectionClosure(
    const SystemServiceObligationProjection &projection,
    const SystemExecutionContextProjection &contexts,
    const ArtifactIdentity &dataflowIdentity, const PlanMap &plans,
    llvm::ArrayRef<SystemServicePlanSelectionView> selections) {
  auto required = selectionRequirements(projection, contexts, dataflowIdentity);
  if (!required)
    return required.takeError();
  std::set<std::uint64_t> planOrdinals;
  for (const auto &[ordinal, plan] : plans) {
    (void)plan;
    planOrdinals.insert(ordinal);
  }
  std::set<std::string> seen;
  std::set<std::uint64_t> selected;
  for (const auto &selection : selections) {
    auto encoded =
        encodeServicePlanSelectionKey(dataflowIdentity, selection.key);
    if (!encoded)
      return encoded.takeError();
    const std::string key = byteKey(*encoded);
    auto expected = required->find(key);
    if (expected == required->end() || !seen.insert(key).second)
      return invalid(
          "ServicePlanSelection closure is incomplete or unreachable");
    if (llvm::Error error = verifySelectionRelation(
            selection, expected->second.domain, planOrdinals))
      return error;
    if (selection.defaultPlanOrdinal)
      selected.insert(*selection.defaultPlanOrdinal);
    for (const auto &clause : selection.clauses)
      selected.insert(clause.target);
  }
  if (seen.size() != required->size())
    return invalid("ServicePlanSelection closure is incomplete");
  if (selected != planOrdinals)
    return invalid("ServiceRealization contains an unselected plan");
  return llvm::Error::success();
}

std::set<std::uint64_t>
selectedPlanOrdinals(::mapping::ServicePlanSelectionOp selection) {
  std::set<std::uint64_t> result;
  if (auto value = selection.getDefaultPlanOrdinal())
    result.insert(*value);
  for (auto clause : selection.getBody()
                         .front()
                         .getOps<::mapping::ServicePlanPresburgerClauseOp>())
    result.insert(clause.getTargetPlanOrdinal());
  return result;
}

llvm::Expected<::mapping::ServicePlanElementRefAttr>
serviceOwnerAttr(::mapping::ServiceRealizationOp service,
                 std::uint64_t planOrdinal, mlir::Attribute element) {
  return ::mapping::ServicePlanElementRefAttr::get(
      service.getContext(), service.getKey(), planOrdinal, element);
}

llvm::Expected<SystemResourceOwnerView>
importResourceOwner(mlir::Attribute owner,
                    const ::dataflow::CanonicalDataflowProgramView &dataflow,
                    const ::loom::fabric::FabricSystemRootView &fabric) {
  if (auto instruction =
          mlir::dyn_cast<::mapping::InstructionExecutionResourceOwnerRefAttr>(
              owner)) {
    auto root = decodeDataflow<::dataflow::RootThreadLaunchRef>(
        instruction.getRoot(), dataflow.identity());
    auto context = decodeFabric<::loom::fabric::InstructionCoreContextRef>(
        instruction.getInstructionContext());
    if (!root)
      return root.takeError();
    if (!context)
      return context.takeError();
    if (llvm::Error error =
            ::loom::fabric::validateFabricRef(fabric.artifact(), *context))
      return std::move(error);
    return SystemResourceOwnerView(
        SystemInstructionResourceOwnerView{*root, *context});
  }
  auto service = mlir::dyn_cast<::mapping::ServicePlanElementRefAttr>(owner);
  if (!service)
    return invalid("ResourceUse has an unknown System owner");
  auto obligation = decodeSystemServiceObligationKey(
      unsignedBytes(service.getService().getRecord()), dataflow.identity());
  if (!obligation)
    return obligation.takeError();
  auto element = importElement(service.getElement(), dataflow, fabric);
  if (!element)
    return element.takeError();
  return SystemResourceOwnerView(SystemServicePlanResourceOwnerView{
      std::move(*obligation), service.getPlanOrdinal(), std::move(*element)});
}

llvm::Error
verifyPatternOwner(const SystemResourceOwnerView &owner,
                   const ::loom::fabric::FabricUsePatternRef &pattern) {
  const auto catalog = pattern.owner.catalog();
  return std::visit(
      [&](const auto &typed) -> llvm::Error {
        using Owner = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Owner,
                                     SystemInstructionResourceOwnerView>) {
          if (catalog != ::loom::fabric::FabricInventoryOwnerRef::of(
                             typed.instructionContext))
            return invalid("InstructionCore ResourceUse has a foreign pattern");
          return llvm::Error::success();
        } else {
          return std::visit(
              [&](const auto &element) -> llvm::Error {
                using Element = std::decay_t<decltype(element)>;
                if constexpr (std::is_same_v<Element,
                                             SystemMemoryRegionElementView>) {
                  const auto targetOwner =
                      ::loom::fabric::FabricInventoryOwnerRef::of(
                          element.serviceRegion.service);
                  if (catalog != targetOwner)
                    return invalid(
                        "memory ResourceUse has a foreign target pattern");
                  return llvm::Error::success();
                } else if constexpr (std::is_same_v<
                                         Element,
                                         SystemConsistencyElementView>) {
                  if (catalog != ::loom::fabric::FabricInventoryOwnerRef::of(
                                     element.consistencyDomain.underlying()))
                    return invalid(
                        "consistency ResourceUse has a foreign target pattern");
                  return llvm::Error::success();
                } else {
                  return invalid("transfer legs cannot own System ResourceUse");
                }
              },
              typed.element);
        }
      },
      owner);
}

} // namespace

llvm::Expected<ImportedSystemClosure> importSystemMappingClosure(
    ::mapping::SystemOp root,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemExecutionBindingView &execution,
    const SpatialMappingImportContext &spatialMappings) {
  auto projected =
      projectSystemServiceObligations(dataflow, execution.rootThreadLaunches());
  if (!projected)
    return projected.takeError();
  auto contexts = projectSystemExecutionContexts(dataflow, execution);
  if (!contexts)
    return contexts.takeError();
  std::map<std::string, const SystemServiceObligationProjection *> expected;
  for (const auto &projection : *projected) {
    auto key =
        encodeSystemServiceObligationKey(dataflow.identity(), projection.key);
    if (!key)
      return key.takeError();
    expected.emplace(byteKey(*key), &projection);
  }

  std::vector<SystemServiceRealizationView> services;
  std::set<std::string> seenServices;
  std::set<std::string> expectedResourceUses;
  for (const auto &thread : execution.threadBindings()) {
    std::vector<::loom::fabric::AccCoreOccurrenceRef> cores;
    for (const auto &clause : thread.clauses)
      cores.push_back(clause.target);
    if (thread.defaultTarget)
      cores.push_back(*thread.defaultTarget);
    llvm::sort(cores, [](const auto &lhs, const auto &rhs) {
      return ::loom::fabric::canonicalFabricBytes(lhs) <
             ::loom::fabric::canonicalFabricBytes(rhs);
    });
    cores.erase(std::unique(cores.begin(), cores.end()), cores.end());
    for (const auto &core : cores) {
      auto rootBytes =
          ::dataflow::encodeDataflowReference(dataflow.identity(), thread.key);
      if (!rootBytes)
        return rootBytes.takeError();
      auto rootAttr = ::mapping::RootThreadLaunchRefAttr::get(
          root.getContext(),
          mlir::DenseI8ArrayAttr::get(
              root.getContext(),
              llvm::ArrayRef<std::int8_t>(
                  reinterpret_cast<const std::int8_t *>(rootBytes->data()),
                  rootBytes->size())));
      const auto context = ::loom::fabric::InstructionCoreContextRef{core};
      auto contextBytes = ::loom::fabric::canonicalFabricBytes(context);
      auto contextAttr = ::mapping::InstructionCoreContextRefAttr::get(
          root.getContext(),
          mlir::DenseI8ArrayAttr::get(
              root.getContext(),
              llvm::ArrayRef<std::int8_t>(
                  reinterpret_cast<const std::int8_t *>(contextBytes.data()),
                  contextBytes.size())));
      auto owner = ::mapping::InstructionExecutionResourceOwnerRefAttr::get(
          root.getContext(), rootAttr, contextAttr);
      auto key = resourceKey(owner, dataflow, rootStartEvent(thread.key));
      if (!key)
        return key.takeError();
      expectedResourceUses.insert(std::move(*key));
    }
  }

  for (auto service :
       root.getBody().front().getOps<::mapping::ServiceRealizationOp>()) {
    const std::vector<std::uint8_t> rawKey =
        unsignedBytes(service.getKey().getRecord());
    auto obligation =
        decodeSystemServiceObligationKey(rawKey, dataflow.identity());
    if (!obligation)
      return obligation.takeError();
    const std::string lookup = byteKey(rawKey);
    auto projection = expected.find(lookup);
    if (projection == expected.end() || !seenServices.insert(lookup).second)
      return invalid("ServiceRealization has a foreign or duplicate key");
    auto plans = importPlans(service, dataflow, fabric);
    if (!plans)
      return plans.takeError();
    std::vector<SystemServicePlanSelectionView> selectionViews;
    for (auto selection : service.getBody()
                              .front()
                              .getOps<::mapping::ServicePlanSelectionOp>()) {
      auto key = decodeServicePlanSelectionKey(
          unsignedBytes(selection.getKey().getRecord()), dataflow.identity());
      if (!key)
        return key.takeError();
      if (!anchorBelongsTo(key->anchor, *projection->second))
        return invalid("ServicePlanSelection anchor is foreign to its service");
      SystemServicePlanSelectionView selectionView{
          *key, {}, selection.getDefaultPlanOrdinal()};
      for (auto clause :
           selection.getBody()
               .front()
               .getOps<::mapping::ServicePlanPresburgerClauseOp>()) {
        SystemPresburgerClauseView<std::uint64_t> clauseView{
            {}, clause.getTargetPlanOrdinal()};
        for (mlir::Attribute raw : clause.getCells()) {
          auto cell =
              decodeCell(mlir::cast<::mapping::SystemPresburgerCellAttr>(raw));
          if (!cell)
            return cell.takeError();
          clauseView.cells.push_back(std::move(*cell));
        }
        selectionView.clauses.push_back(std::move(clauseView));
      }
      selectionViews.push_back(std::move(selectionView));
      const auto *member =
          std::get_if<ServiceMemberPlanSelectionAnchor>(&key->anchor);
      if (!member ||
          std::holds_alternative<::dataflow::MessageTransferMemberRef>(
              member->member))
        continue;
      const ::dataflow::ContextualActorRef *actor = nullptr;
      bool addressed = false;
      if (const auto *memory =
              std::get_if<::dataflow::AddressedMemoryActorMemberRef>(
                  &member->member)) {
        actor = &memory->actor;
        addressed = true;
      } else if (const auto *fence =
                     std::get_if<::dataflow::FenceActorMemberRef>(
                         &member->member)) {
        actor = &fence->actor;
      }
      if (!actor)
        return invalid("service member has no contextual actor owner");
      auto issue = deriveSpatialMemoryIssueEvent(dataflow, actor->actor);
      if (!issue)
        return issue.takeError();
      const ::dataflow::EventFamilyKey trigger(
          ::dataflow::ContextualActorTransitionEventRef{*actor,
                                                        issue->transition});
      for (std::uint64_t ordinal : selectedPlanOrdinals(selection)) {
        auto plan = plans->find(ordinal);
        if (plan == plans->end())
          return invalid("ServicePlanSelection names an absent plan");
        for (const auto &[unused, element] : plan->second.elements) {
          (void)unused;
          const bool memoryElement =
              mlir::isa<::mapping::MemoryRegionElementKeyAttr>(element);
          const bool consistencyElement =
              mlir::isa<::mapping::ConsistencyElementKeyAttr>(element);
          if ((addressed && !memoryElement) ||
              (!addressed && !consistencyElement))
            continue;
          auto owner = serviceOwnerAttr(service, ordinal, element);
          if (!owner)
            return owner.takeError();
          auto expectedUse = resourceKey(*owner, dataflow, trigger);
          if (!expectedUse)
            return expectedUse.takeError();
          expectedResourceUses.insert(std::move(*expectedUse));
        }
      }
    }
    if (llvm::Error error =
            verifySelectionClosure(*projection->second, *contexts,
                                   dataflow.identity(), *plans, selectionViews))
      return std::move(error);
    std::vector<SystemServicePlanView> planViews;
    planViews.reserve(plans->size());
    for (const auto &[ordinal, plan] : *plans) {
      (void)ordinal;
      planViews.push_back(plan.view);
    }
    if (llvm::Error error = verifySystemServiceTargetClosure(
            dataflow, fabric, spatialMappings, *projection->second, *contexts,
            planViews, selectionViews))
      return std::move(error);
    planViews.clear();
    for (auto &[ordinal, plan] : *plans) {
      (void)ordinal;
      planViews.push_back(std::move(plan.view));
    }
    services.push_back(SystemServiceRealizationView{std::move(*obligation),
                                                    std::move(planViews),
                                                    std::move(selectionViews)});
  }
  if (seenServices.size() != expected.size())
    return invalid("ServiceRealization closure is incomplete");

  std::vector<SystemResourceUseView> resourceUses;
  std::vector<std::string> resourceUseActivationKeys;
  std::set<std::string> seenResourceUses;
  for (auto record :
       root.getBody().front().getOps<::mapping::ResourceUseOp>()) {
    auto owner = importResourceOwner(record.getOwner(), dataflow, fabric);
    auto pattern =
        decodeFabric<::loom::fabric::FabricUsePatternRef>(record.getUseSite());
    auto activation =
        importActivation(mlir::cast<::mapping::SystemRelativeActivationAttr>(
                             record.getActivation()),
                         dataflow);
    if (!owner)
      return owner.takeError();
    if (!pattern)
      return pattern.takeError();
    if (!activation)
      return activation.takeError();
    if (llvm::Error error = verifyPatternOwner(*owner, *pattern))
      return std::move(error);
    auto values =
        importResourceUsePatternValues(record, fabric.artifact(), *pattern);
    if (!values)
      return values.takeError();
    auto key =
        resourceKey(record.getOwner(), dataflow, activation->trigger.event);
    if (!key)
      return key.takeError();
    if (!seenResourceUses.insert(*key).second)
      return invalid("ResourceUse closure contains a duplicate owner event");
    auto activationKey = capacityActivationKey(*key, fabric.artifact(),
                                               *pattern, values->parameters);
    if (!activationKey)
      return activationKey.takeError();

    if (const auto *instruction =
            std::get_if<SystemInstructionResourceOwnerView>(&*owner)) {
      if (activation->release.size() != 1 ||
          activation->trigger.event != rootStartEvent(instruction->root) ||
          activation->release.front().event !=
              rootCompletionEvent(instruction->root))
        return invalid("InstructionCore ResourceUse has the wrong activation");
    } else {
      if (!activation->release.empty())
        return invalid("service ResourceUse has a causal release");
    }
    resourceUses.push_back(SystemResourceUseView{
        std::move(*owner), *pattern, std::move(*activation),
        std::move(values->parameters), std::move(values->sharingAssignments)});
    resourceUseActivationKeys.push_back(std::move(*activationKey));
  }
  if (seenResourceUses != expectedResourceUses)
    return invalid(
        "ResourceUse closure is incomplete or contains foreign uses");
  return ImportedSystemClosure{std::move(services), std::move(resourceUses),
                               std::move(resourceUseActivationKeys)};
}

} // namespace loom::mapping::detail
