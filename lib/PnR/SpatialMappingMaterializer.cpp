#include "PnR/SpatialMappingMaterializer.h"

#include "PnR/MappingObjective.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/IR/PhysicalTag.h"
#include "Fabric/IR/UsePatternValue.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/IR/MappingActivationKey.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/IR/MappingOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace loom::pnr {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "spatial_mapping_materialization_invalid: " +
                                     message);
}

mlir::DenseI8ArrayAttr denseBytes(mlir::MLIRContext *context,
                                  llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return mlir::DenseI8ArrayAttr::get(context, signedBytes);
}

template <typename Attr, typename Ref>
Attr fabricAttr(mlir::MLIRContext *context, const Ref &reference) {
  return Attr::get(
      context,
      denseBytes(context, ::loom::fabric::canonicalFabricBytes(reference)));
}

template <typename Attr, typename Ref>
llvm::Expected<Attr> dataflowAttr(mlir::MLIRContext *context,
                                  const ArtifactIdentity &owner,
                                  const Ref &reference) {
  auto bytes = ::dataflow::encodeDataflowReference(owner, reference);
  if (!bytes)
    return bytes.takeError();
  return Attr::get(context, denseBytes(context, *bytes));
}

::mapping::ArtifactIdentityAttr identityAttr(mlir::MLIRContext *context,
                                             const ArtifactIdentity &identity) {
  return ::mapping::ArtifactIdentityAttr::get(
      context, denseBytes(context, identity.bytes()));
}

llvm::Expected<std::vector<::loom::mapping::SpatialComputeBindingView>>
materializeComputeBindings(mlir::OpBuilder &builder, mlir::Location location,
                           mlir::Block &body,
                           const SpatialCandidateState &candidate) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const auto realizations = problem.realizations().computeRealizations();
  const auto placements = problem.realizations().computePlacements();
  const auto contexts = problem.realizations().computeInstructionContexts();
  std::vector<::loom::mapping::SpatialComputeBindingView> bindings;
  bindings.reserve(realizations.size());
  for (PnrIndex ordinal = 0; ordinal < realizations.size(); ++ordinal) {
    const auto &realization = realizations[ordinal];
    const auto &selection = candidate.computeBinding(ordinal);
    if (selection.placement >= placements.size() ||
        selection.instructionContext >= contexts.size())
      return invalid("compute selection exceeds its frozen reverse table");
    const auto &placement = placements[selection.placement];
    if (placement.realization != ordinal ||
        selection.placement < realization.placementOffset ||
        selection.placement >=
            realization.placementOffset + realization.placementCount ||
        selection.instructionContext < placement.contextOffset ||
        selection.instructionContext >=
            placement.contextOffset + placement.contextCount)
      return invalid("compute selection is outside its realization domain");

    builder.setInsertionPointToEnd(&body);
    ::mapping::ComputeBindingOp::create(
        builder, location,
        ::mapping::ComputeRealizationRefAttr::get(builder.getContext(),
                                                  realization.reference.entity),
        fabricAttr<::mapping::FabricFuOccurrenceRefAttr>(builder.getContext(),
                                                         placement.fu),
        fabricAttr<::mapping::InstructionContextRefAttr>(
            builder.getContext(), contexts[selection.instructionContext]),
        builder.getArrayAttr({}));
    bindings.push_back(::loom::mapping::SpatialComputeBindingView{
        realization.reference.entity,
        placement.fu,
        contexts[selection.instructionContext],
        {}});
  }
  return bindings;
}

llvm::Expected<std::vector<::loom::mapping::SpatialRegisterFifoTransferView>>
materializeRegisterFifoTransfers(mlir::OpBuilder &builder,
                                 mlir::Location location, mlir::Block &body,
                                 const SpatialCandidateState &candidate,
                                 const ArtifactIdentity &dataflowIdentity) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const auto logicalNets = problem.transfers().logicalNets();
  const auto sinks = problem.transfers().logicalNetSinks();
  const auto options = problem.localTransfers().options();
  const auto traversals = problem.routing().traversals();
  std::vector<::loom::mapping::SpatialRegisterFifoTransferView> result;
  for (PnrIndex logicalNet = 0; logicalNet < logicalNets.size(); ++logicalNet) {
    const PnrIndex selected = candidate.registerFifoTransfer(logicalNet);
    if (selected == getInvalidPnrIndex())
      continue;
    if (selected >= options.size())
      return invalid("register-FIFO transfer option is out of range");
    const auto &option = options[selected];
    const auto &net = logicalNets[logicalNet];
    if (option.logicalNet != logicalNet || net.sinkCount != 1 ||
        net.sinkOffset >= sinks.size() ||
        option.writeTraversal >= traversals.size() ||
        option.readTraversal >= traversals.size())
      return invalid("register-FIFO transfer frozen projection is invalid");
    auto producer = dataflowAttr<::mapping::GraphProducerEndpointRefAttr>(
        builder.getContext(), dataflowIdentity, net.producer);
    if (!producer)
      return producer.takeError();
    auto sink = dataflowAttr<::mapping::GraphConsumerEndpointRefAttr>(
        builder.getContext(), dataflowIdentity, sinks[net.sinkOffset]);
    if (!sink)
      return sink.takeError();
    builder.setInsertionPointToEnd(&body);
    ::mapping::RegisterFifoTransferOp::create(
        builder, location, *producer, *sink,
        fabricAttr<::mapping::FabricPhysicalTraversalRefAttr>(
            builder.getContext(), traversals[option.writeTraversal].reference),
        fabricAttr<::mapping::FabricPhysicalTraversalRefAttr>(
            builder.getContext(), traversals[option.readTraversal].reference));
    result.push_back(::loom::mapping::SpatialRegisterFifoTransferView{
        net.producer, sinks[net.sinkOffset], option.pe, option.registerFifo,
        traversals[option.writeTraversal].reference,
        traversals[option.readTraversal].reference, option.tag});
  }
  return result;
}

llvm::Expected<mlir::Attribute> materializeMemoryBindingTarget(
    mlir::MLIRContext *context,
    const FrozenSpatialMemoryBindingTargetOption &target,
    const SpatialLogicalMemoryBindingSelection &selection) {
  if (const auto *region =
          std::get_if<::loom::fabric::FabricMemoryServiceRegionRef>(
              &target.target))
    return mlir::Attribute(::mapping::MemoryLocalRegionAttr::get(
        context,
        fabricAttr<::mapping::FabricMemoryServiceRegionRefAttr>(context,
                                                                *region),
        selection.physicalOffsetBytes));
  if (selection.physicalOffsetBytes != 0)
    return invalid("BoundaryProxy carries a physical byte offset");
  return mlir::Attribute(::mapping::MemoryBoundaryProxyAttr::get(context));
}

llvm::Expected<mlir::Attribute> materializeExposureDispatch(
    mlir::MLIRContext *context,
    const FrozenSpatialMemoryExposureDispatchTarget &target) {
  if (const auto *local =
          std::get_if<::loom::fabric::LocalMemoryServiceRef>(&target))
    return mlir::Attribute(
        fabricAttr<::mapping::LocalMemoryServiceRefAttr>(context, *local));
  return mlir::Attribute(fabricAttr<::mapping::ManagerEndpointRefAttr>(
      context, std::get<::loom::fabric::ManagerEndpointRef>(target)));
}

llvm::Error
materializeMemoryBindings(mlir::OpBuilder &builder, mlir::Location location,
                          mlir::Block &body,
                          const SpatialCandidateState &candidate,
                          const ArtifactIdentity &dataflowIdentity) {
  const auto &memory = candidate.problem().memory();
  for (PnrIndex bindingOrdinal = 0;
       bindingOrdinal < memory.logicalBindings().size(); ++bindingOrdinal) {
    const auto &logical = memory.logicalBindings()[bindingOrdinal];
    const auto &selection = candidate.logicalMemoryBinding(bindingOrdinal);
    if (selection.target >= memory.bindingTargets().size())
      return invalid("logical memory binding target is out of range");
    auto logicalAttr = dataflowAttr<::mapping::LogicalMemoryRootOrViewRefAttr>(
        builder.getContext(), dataflowIdentity, logical.logicalMemory);
    if (!logicalAttr)
      return logicalAttr.takeError();
    auto target = materializeMemoryBindingTarget(
        builder.getContext(), memory.bindingTargets()[selection.target],
        selection);
    if (!target)
      return target.takeError();

    builder.setInsertionPointToEnd(&body);
    auto record = ::mapping::MemoryBindingOp::create(
        builder, location, static_cast<std::uint64_t>(bindingOrdinal),
        *logicalAttr,
        ::mapping::MemoryWholeIntervalAttr::get(builder.getContext()), *target);
    record.getBody().push_back(new mlir::Block());
    builder.setInsertionPointToEnd(&record.getBody().front());
    for (PnrIndex exposureOrdinal : memory.bindingExposures().slice(
             memory.bindingExposureOffsets()[bindingOrdinal],
             memory.bindingExposureOffsets()[bindingOrdinal + 1] -
                 memory.bindingExposureOffsets()[bindingOrdinal])) {
      if (exposureOrdinal >= memory.exposures().size())
        return invalid("logical memory binding has a foreign exposure");
      const PnrIndex optionOrdinal =
          candidate.memoryExposureSelection(exposureOrdinal);
      if (optionOrdinal >= memory.exposureOptions().size())
        return invalid("memory exposure option is out of range");
      const auto &exposure = memory.exposures()[exposureOrdinal];
      const auto &option = memory.exposureOptions()[optionOrdinal];
      if (option.provider >= memory.exposureProviders().size())
        return invalid("memory exposure provider is out of range");
      auto exposureAttr = dataflowAttr<::mapping::MemoryExposureRefAttr>(
          builder.getContext(), dataflowIdentity, exposure.exposure);
      if (!exposureAttr)
        return exposureAttr.takeError();
      auto dispatch =
          materializeExposureDispatch(builder.getContext(), option.target);
      if (!dispatch)
        return dispatch.takeError();
      ::mapping::ExposureEntryOp::create(
          builder, location, *exposureAttr,
          fabricAttr<::mapping::SubordinateEndpointRefAttr>(
              builder.getContext(),
              memory.exposureProviders()[option.provider].terminal),
          *dispatch);
    }
  }
  return llvm::Error::success();
}

/// Assigns the exact operation-table row of one Temporal memory operation.
/// Rows belong to the occurrence rather than to an individual operation port,
/// so the ordinal is the actor's position in the canonical realization and
/// actor walk among all realizations placed on that occurrence.
using MemoryResidentContextCursor =
    std::map<std::vector<std::uint8_t>, std::uint64_t>;

llvm::Expected<mlir::Attribute> materializeMemoryPlacement(
    mlir::MLIRContext *context, const FrozenSpatialMemoryPlacement &placement,
    const FrozenSpatialMemoryActorBinding &actor,
    const FrozenSpatialMemoryOperationHandshakePlan &plan,
    MemoryResidentContextCursor &residentContexts) {
  const ::loom::fabric::FabricMemoryOperationPortRef port{
      placement.memory, actor.operationPort.ordinal};
  if (!plan.temporalResident)
    return mlir::Attribute(
        fabricAttr<::mapping::FabricMemoryOperationPortRefAttr>(context, port));
  if (!placement.residentContextCount)
    return invalid("Temporal memory placement has no resident capacity");
  std::uint64_t &next =
      residentContexts[::loom::fabric::canonicalFabricBytes(placement.memory)];
  if (next >= *placement.residentContextCount)
    return invalid("Temporal memory operation table exceeds its capacity");
  return mlir::Attribute(
      fabricAttr<::mapping::FabricMemoryOperationContextRefAttr>(
          context,
          ::loom::fabric::FabricMemoryOperationContextRef{port, next++}));
}

llvm::Expected<mlir::Attribute>
materializeAddressedDispatch(mlir::MLIRContext *context,
                             const FrozenSpatialMemoryDispatchTarget &target) {
  if (const auto *local =
          std::get_if<::loom::fabric::LocalMemoryServiceRef>(&target))
    return mlir::Attribute(
        fabricAttr<::mapping::LocalMemoryServiceRefAttr>(context, *local));
  if (const auto *manager =
          std::get_if<::loom::fabric::ManagerEndpointRef>(&target))
    return mlir::Attribute(
        fabricAttr<::mapping::ManagerEndpointRefAttr>(context, *manager));
  return invalid("addressed memory use selects a consistency-domain target");
}

llvm::Expected<mlir::Attribute>
materializeFenceDispatch(mlir::MLIRContext *context,
                         const FrozenSpatialMemoryDispatchTarget &target) {
  if (const auto *domain =
          std::get_if<::loom::fabric::MemoryConsistencyDomainRef>(&target))
    return mlir::Attribute(
        fabricAttr<::mapping::MemoryConsistencyDomainRefAttr>(context,
                                                              *domain));
  if (const auto *manager =
          std::get_if<::loom::fabric::ManagerEndpointRef>(&target))
    return mlir::Attribute(
        fabricAttr<::mapping::ManagerEndpointRefAttr>(context, *manager));
  return invalid("fence use selects an addressed local-service target");
}

llvm::Error
materializeMemoryEngineBindings(mlir::OpBuilder &builder,
                                mlir::Location location, mlir::Block &body,
                                const SpatialCandidateState &candidate,
                                const ArtifactIdentity &dataflowIdentity) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const auto &realizations = problem.realizations();
  const auto &memory = problem.memory();
  const auto plans = problem.handshake().memoryOperationPlans();
  MemoryResidentContextCursor residentContexts;
  for (PnrIndex realizationOrdinal = 0;
       realizationOrdinal < realizations.memoryRealizations().size();
       ++realizationOrdinal) {
    const auto &realization =
        realizations.memoryRealizations()[realizationOrdinal];
    const PnrIndex placementOrdinal =
        candidate.memoryBinding(realizationOrdinal).placement;
    if (placementOrdinal >= realizations.memoryPlacements().size())
      return invalid("memory realization selects an absent occurrence");
    const auto &placement = realizations.memoryPlacements()[placementOrdinal];
    if (placement.realization != realizationOrdinal)
      return invalid("memory occurrence belongs to another realization");

    builder.setInsertionPointToEnd(&body);
    auto engine = ::mapping::MemoryEngineBindingOp::create(
        builder, location,
        ::mapping::MemoryRealizationRefAttr::get(builder.getContext(),
                                                 realization.reference.entity),
        fabricAttr<::mapping::FabricMemoryOccurrenceRefAttr>(
            builder.getContext(), placement.memory));
    auto *engineBody = new mlir::Block();
    engine.getBody().push_back(engineBody);

    for (PnrIndex localActor = 0; localActor < realization.actorCount;
         ++localActor) {
      const PnrIndex actorOrdinal = realization.actorOffset + localActor;
      if (actorOrdinal >= realizations.memoryActors().size())
        return invalid("memory realization actor slice is out of range");
      const auto &actor = realizations.memoryActors()[actorOrdinal];
      const PnrIndex planOrdinal = candidate.memoryOperationPlan(actorOrdinal);
      if (planOrdinal >= plans.size())
        return invalid("memory actor selects an absent operation plan");
      auto actorAttr = dataflowAttr<::mapping::ActorRefAttr>(
          builder.getContext(), dataflowIdentity, actor.actor);
      if (!actorAttr)
        return actorAttr.takeError();
      auto operationPlacement =
          materializeMemoryPlacement(builder.getContext(), placement, actor,
                                     plans[planOrdinal], residentContexts);
      if (!operationPlacement)
        return operationPlacement.takeError();
      if (actorOrdinal + 1 >= memory.actorUseOffsets().size())
        return invalid("memory actor has no rooted-use slice");
      const PnrIndex useBegin = memory.actorUseOffsets()[actorOrdinal];
      const PnrIndex useEnd = memory.actorUseOffsets()[actorOrdinal + 1];
      if (useBegin >= useEnd || useEnd > memory.rootedUses().size())
        return invalid("memory actor has an incomplete rooted-use inventory");
      const bool addressed =
          memory.rootedUses()[useBegin].logicalBinding.has_value();
      for (PnrIndex useOrdinal = useBegin; useOrdinal < useEnd; ++useOrdinal)
        if (memory.rootedUses()[useOrdinal].logicalBinding.has_value() !=
            addressed)
          return invalid("one memory actor mixes addressed and fence uses");

      builder.setInsertionPointToEnd(engineBody);
      mlir::Block *operationBody = nullptr;
      if (addressed) {
        auto operation = ::mapping::AddressedMemoryOperationOp::create(
            builder, location, *actorAttr, *operationPlacement);
        operationBody = new mlir::Block();
        operation.getBody().push_back(operationBody);
      } else {
        auto operation = ::mapping::FenceMemoryOperationOp::create(
            builder, location, *actorAttr, *operationPlacement);
        operationBody = new mlir::Block();
        operation.getBody().push_back(operationBody);
      }

      builder.setInsertionPointToEnd(operationBody);
      for (PnrIndex useOrdinal = useBegin; useOrdinal < useEnd; ++useOrdinal) {
        const auto &use = memory.rootedUses()[useOrdinal];
        const PnrIndex dispatchOrdinal =
            candidate.memoryUseDispatch(useOrdinal);
        if (dispatchOrdinal >= memory.dispatchOptions().size())
          return invalid("rooted memory use selects an absent dispatch");
        auto launch = dataflowAttr<::mapping::RootedGraphLaunchRefAttr>(
            builder.getContext(), dataflowIdentity, use.launch);
        if (!launch)
          return launch.takeError();
        const auto &dispatch = memory.dispatchOptions()[dispatchOrdinal].target;
        if (addressed) {
          if (!use.logicalBinding)
            return invalid("addressed memory use has no logical binding");
          auto target =
              materializeAddressedDispatch(builder.getContext(), dispatch);
          if (!target)
            return target.takeError();
          ::mapping::AddressedMemoryUseOp::create(
              builder, location, *launch,
              ::mapping::MemoryBindingRefAttr::get(builder.getContext(),
                                                   *use.logicalBinding),
              *target);
        } else {
          auto target =
              materializeFenceDispatch(builder.getContext(), dispatch);
          if (!target)
            return target.takeError();
          ::mapping::FenceMemoryUseOp::create(builder, location, *launch,
                                              *target);
        }
      }
    }
  }
  return llvm::Error::success();
}

llvm::Expected<std::vector<PnrIndex>>
canonicalRouteNodeOrdinals(const RouteTreeState &tree) {
  const auto nodes = tree.nodeStorage();
  std::vector<PnrIndex> ordinals(nodes.size(), getInvalidPnrIndex());
  PnrIndex nextOrdinal = 0;
  for (auto [slot, node] : llvm::enumerate(nodes))
    if (node.isActive())
      ordinals[slot] = nextOrdinal++;
  if (nextOrdinal != tree.activeNodeCount())
    return invalid("RouteTree active-node count diverges from node storage");
  return ordinals;
}

llvm::Error materializeRouteTree(mlir::OpBuilder &builder,
                                 mlir::Location location, mlir::Block &parent,
                                 const SpatialCandidateState &candidate,
                                 PnrIndex netOrdinal,
                                 const ArtifactIdentity &dataflowIdentity) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const auto logicalNets = problem.transfers().logicalNets();
  const auto sinks = problem.transfers().logicalNetSinks();
  const auto endpoints = problem.routing().routingEndpoints();
  const auto arcs = problem.routing().routingArcs();
  const auto arcSources = problem.routing().arcSources();
  const auto traversals = problem.routing().traversals();
  if (netOrdinal >= logicalNets.size())
    return invalid("logical-net ordinal exceeds the frozen transfer index");
  const FrozenSpatialLogicalNet &net = logicalNets[netOrdinal];
  const RouteTreeState &tree = candidate.routeTree(netOrdinal);
  const auto source = tree.sourceEndpoint();
  if (!source || *source >= endpoints.size())
    return invalid("closed RouteTree has no valid source endpoint");
  if (*source != candidate.logicalNetSourceEndpoint(netOrdinal))
    return invalid("RouteTree source diverges from selected terminal binding");

  auto producer = dataflowAttr<::mapping::GraphProducerEndpointRefAttr>(
      builder.getContext(), dataflowIdentity, net.producer);
  if (!producer)
    return producer.takeError();
  builder.setInsertionPointToEnd(&parent);
  auto route = ::mapping::RouteTreeOp::create(
      builder, location, *producer,
      fabricAttr<::mapping::FabricTransportEndpointRefAttr>(
          builder.getContext(), endpoints[*source].reference));
  auto *body = new mlir::Block();
  route.getBody().push_back(body);

  const auto nodes = tree.nodeStorage();
  auto ordinals = canonicalRouteNodeOrdinals(tree);
  if (!ordinals)
    return ordinals.takeError();

  builder.setInsertionPointToEnd(body);
  for (auto [slotValue, node] : llvm::enumerate(nodes)) {
    if (!node.isActive())
      continue;
    const PnrIndex slot = static_cast<PnrIndex>(slotValue);
    mlir::IntegerAttr parentOrdinal;
    ::mapping::FabricPhysicalTraversalRefAttr traversal;
    if (node.parentArc == getInvalidPnrIndex()) {
      if (node.endpoint != *source)
        return invalid("RouteTree has a non-source root node");
    } else {
      if (node.parentArc >= arcs.size() || node.parentArc >= arcSources.size())
        return invalid("RouteTree node names an absent routing arc");
      const auto &arc = arcs[node.parentArc];
      if (arc.target != node.endpoint || arc.traversal >= traversals.size())
        return invalid("RouteTree node disagrees with its incoming arc");
      auto parentSlot = tree.findNode(arcSources[node.parentArc]);
      if (!parentSlot || *parentSlot >= ordinals->size() ||
          (*ordinals)[*parentSlot] == getInvalidPnrIndex())
        return invalid("RouteTree incoming arc has no active parent node");
      parentOrdinal = builder.getI64IntegerAttr((*ordinals)[*parentSlot]);
      traversal = fabricAttr<::mapping::FabricPhysicalTraversalRefAttr>(
          builder.getContext(), traversals[arc.traversal].reference);
    }
    ::mapping::RouteNodeOp::create(builder, location, (*ordinals)[slot],
                                   parentOrdinal, traversal,
                                   builder.getArrayAttr({}));
  }

  for (PnrIndex sinkOrdinal = 0; sinkOrdinal < net.sinkCount; ++sinkOrdinal) {
    const auto endpoint = tree.sinkEndpoint(sinkOrdinal);
    if (!endpoint || *endpoint >= endpoints.size())
      return invalid("RouteTree has an unbound sink endpoint");
    if (*endpoint != candidate.logicalNetSinkEndpoint(netOrdinal, sinkOrdinal))
      return invalid("RouteTree sink diverges from selected terminal binding");
    const auto slot = tree.findNode(*endpoint);
    if (!slot || *slot >= ordinals->size() ||
        (*ordinals)[*slot] == getInvalidPnrIndex())
      return invalid("RouteTree sink endpoint is not present in the tree");
    const PnrIndex sinkIndex = net.sinkOffset + sinkOrdinal;
    if (sinkIndex >= sinks.size())
      return invalid("RouteTree sink exceeds the frozen logical-net slice");
    auto sink = dataflowAttr<::mapping::GraphConsumerEndpointRefAttr>(
        builder.getContext(), dataflowIdentity, sinks[sinkIndex]);
    if (!sink)
      return sink.takeError();
    ::mapping::RouteSinkOp::create(builder, location, *sink,
                                   (*ordinals)[*slot]);
  }
  return llvm::Error::success();
}

llvm::Expected<mlir::Attribute> materializeActivityEvent(
    mlir::MLIRContext *context, const ArtifactIdentity &dataflowIdentity,
    const ::loom::mapping::SpatialActivityEventRef &event) {
  return std::visit(
      [&](const auto &trigger) -> llvm::Expected<mlir::Attribute> {
        using Trigger = std::decay_t<decltype(trigger)>;
        if constexpr (std::is_same_v<
                          Trigger,
                          ::loom::mapping::SpatialActorTransitionEventRef>) {
          auto actor = dataflowAttr<::mapping::ActorRefAttr>(
              context, dataflowIdentity, trigger.actor);
          if (!actor)
            return actor.takeError();
          return ::mapping::ActorTransitionEventAttr::get(context, *actor,
                                                          trigger.transition);
        } else if constexpr (
            std::is_same_v<Trigger,
                           ::dataflow::CanonicalGraphProducerEndpointRef>) {
          auto producer = dataflowAttr<::mapping::GraphProducerEndpointRefAttr>(
              context, dataflowIdentity, trigger);
          if (!producer)
            return producer.takeError();
          return mlir::Attribute(*producer);
        } else {
          auto consumer = dataflowAttr<::mapping::GraphConsumerEndpointRefAttr>(
              context, dataflowIdentity, trigger);
          if (!consumer)
            return consumer.takeError();
          return mlir::Attribute(*consumer);
        }
      },
      event);
}

llvm::Expected<mlir::ArrayAttr> materializePatternValues(
    mlir::MLIRContext *context,
    llvm::ArrayRef<::fabric::UsePatternValueSchema> schemas,
    llvm::ArrayRef<::fabric::UsePatternValue> values, llvm::StringRef field) {
  if (schemas.size() != values.size())
    return invalid("ResourceUse " + field +
                   " count disagrees with its Fabric use pattern schema");
  llvm::SmallVector<mlir::Attribute> attributes;
  attributes.reserve(values.size());
  for (auto [schema, value] : llvm::zip_equal(schemas, values)) {
    auto bytes = ::fabric::encodeUsePatternValue(schema, value);
    if (!bytes)
      return invalid("ResourceUse " + field +
                     " cannot be encoded by its Fabric owner: " +
                     llvm::toString(bytes.takeError()));
    attributes.push_back(::mapping::OwnerTypedValueAttr::get(
        context, denseBytes(context, *bytes)));
  }
  return mlir::ArrayAttr::get(context, attributes);
}

llvm::Error materializeResourceUse(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Block &body,
    mlir::Attribute owner,
    const ::loom::mapping::SpatialActivityEventRef &event,
    llvm::ArrayRef<::loom::mapping::SpatialActivityEventRef> releaseEvents,
    const ::loom::fabric::FabricUsePatternRef &pattern,
    const ArtifactIdentity &dataflowIdentity,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<::fabric::UsePatternValue> parameters = {},
    llvm::ArrayRef<::fabric::UsePatternValue> sharingAssignments = {}) {
  const ::fabric::ResourceContract *contract =
      fabric.resourceContract(pattern.owner.catalog());
  if (!contract || pattern.ordinal >= contract->usePatternCount())
    return invalid("ResourceUse does not resolve an exact Fabric use pattern");
  const ::fabric::UsePattern declaration =
      contract->usePattern(::fabric::UsePatternKey(pattern.ordinal));
  auto parameterAttrs = materializePatternValues(
      builder.getContext(), declaration.parameters, parameters, "parameters");
  if (!parameterAttrs)
    return parameterAttrs.takeError();
  auto sharingAttrs = materializePatternValues(
      builder.getContext(), declaration.sharingAssignments, sharingAssignments,
      "sharing assignments");
  if (!sharingAttrs)
    return sharingAttrs.takeError();
  auto encodedEvent =
      materializeActivityEvent(builder.getContext(), dataflowIdentity, event);
  if (!encodedEvent)
    return encodedEvent.takeError();
  auto trigger = ::mapping::SpatialEventPointAttr::get(
      builder.getContext(), *encodedEvent, ::mapping::OwnerTypedValueAttr());
  llvm::SmallVector<mlir::Attribute> release;
  release.reserve(releaseEvents.size());
  for (const auto &releaseEvent : releaseEvents) {
    auto encodedRelease = materializeActivityEvent(
        builder.getContext(), dataflowIdentity, releaseEvent);
    if (!encodedRelease)
      return encodedRelease.takeError();
    release.push_back(::mapping::SpatialEventPointAttr::get(
        builder.getContext(), *encodedRelease,
        ::mapping::OwnerTypedValueAttr()));
  }
  llvm::sort(release, [](mlir::Attribute left, mlir::Attribute right) {
    return ::loom::mapping::canonicalEventPointKey(
               mlir::cast<::mapping::SpatialEventPointAttr>(left)) <
           ::loom::mapping::canonicalEventPointKey(
               mlir::cast<::mapping::SpatialEventPointAttr>(right));
  });
  auto activation = ::mapping::SpatialRelativeActivationAttr::get(
      builder.getContext(), trigger, builder.getArrayAttr(release));
  builder.setInsertionPointToEnd(&body);
  ::mapping::ResourceUseOp::create(
      builder, location, owner,
      fabricAttr<::mapping::FabricUsePatternRefAttr>(builder.getContext(),
                                                     pattern),
      activation, *parameterAttrs, *sharingAttrs);
  return llvm::Error::success();
}

llvm::Error materializeComputeResourceUses(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Block &body,
    llvm::ArrayRef<::loom::mapping::SpatialComputeUseRequirement> uses,
    const ArtifactIdentity &dataflowIdentity,
    const ::loom::fabric::FabricArtifactView &fabric) {
  for (const auto &use : uses)
    if (llvm::Error error = materializeResourceUse(
            builder, location, body,
            ::mapping::ComputeRealizationRefAttr::get(builder.getContext(),
                                                      use.realization),
            use.trigger, use.release, use.pattern, dataflowIdentity, fabric))
      return error;
  return llvm::Error::success();
}

llvm::Error materializeMemoryResourceUses(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Block &body,
    const SpatialCandidateState &candidate,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric) {
  const auto &problem = candidate.problem();
  const auto &realizations = problem.realizations();
  const auto &memory = problem.memory();
  const auto plans = problem.handshake().memoryOperationPlans();
  const auto patterns = problem.resources().usePatterns();
  std::map<std::string, ::loom::fabric::FabricUsePatternRef> serviceUses;

  for (const auto &realization : realizations.memoryRealizations()) {
    for (PnrIndex localActor = 0; localActor < realization.actorCount;
         ++localActor) {
      const PnrIndex actorOrdinal = realization.actorOffset + localActor;
      if (actorOrdinal >= realizations.memoryActors().size())
        return invalid("memory ResourceUse actor is out of range");
      const auto &actor = realizations.memoryActors()[actorOrdinal];
      auto issue =
          ::loom::mapping::deriveSpatialMemoryIssueEvent(dataflow, actor.actor);
      if (!issue)
        return issue.takeError();
      const ::loom::mapping::SpatialActivityEventRef trigger =
          std::move(*issue);
      const PnrIndex plan = candidate.memoryOperationPlan(actorOrdinal);
      if (plan >= plans.size() || plans[plan].usePattern >= patterns.size())
        return invalid("memory ResourceUse operation plan is out of range");
      if (llvm::Error error = materializeResourceUse(
              builder, location, body,
              ::mapping::MemoryRealizationRefAttr::get(
                  builder.getContext(), realization.reference.entity),
              trigger, {}, patterns[plans[plan].usePattern].reference,
              dataflow.identity(), fabric))
        return error;

      const PnrIndex useBegin = memory.actorUseOffsets()[actorOrdinal];
      const PnrIndex useEnd = memory.actorUseOffsets()[actorOrdinal + 1];
      for (PnrIndex useOrdinal = useBegin; useOrdinal < useEnd; ++useOrdinal) {
        const PnrIndex dispatch = candidate.memoryUseDispatch(useOrdinal);
        if (dispatch >= memory.dispatchOptions().size())
          return invalid("memory ResourceUse dispatch is out of range");
        const auto &option = memory.dispatchOptions()[dispatch];
        if (!option.serviceUsePattern)
          continue;
        const auto &use = memory.rootedUses()[useOrdinal];
        if (!use.logicalBinding)
          return invalid("local service ResourceUse has no MemoryBinding");
        auto actorBytes = ::dataflow::encodeDataflowReference(
            dataflow.identity(), actor.actor);
        if (!actorBytes)
          return actorBytes.takeError();
        std::string key;
        for (unsigned byte = 0; byte < 8; ++byte)
          key.push_back(
              static_cast<char>(*use.logicalBinding >> (8 * (7 - byte))));
        key.append(reinterpret_cast<const char *>(actorBytes->data()),
                   actorBytes->size());
        auto [found, inserted] =
            serviceUses.try_emplace(key, *option.serviceUsePattern);
        if (!inserted) {
          if (found->second != *option.serviceUsePattern)
            return invalid("one memory service activation selects multiple "
                           "UsePatterns");
          continue;
        }
        if (llvm::Error error = materializeResourceUse(
                builder, location, body,
                ::mapping::MemoryBindingRefAttr::get(builder.getContext(),
                                                     *use.logicalBinding),
                trigger, {}, *option.serviceUsePattern, dataflow.identity(),
                fabric))
          return error;
      }
    }
  }
  return llvm::Error::success();
}

struct PhysicalTagUseOrigin final {
  mlir::Attribute owner;
  ::loom::fabric::FabricPhysicalTagAssignmentPointView assignmentPoint;
};

llvm::Expected<PhysicalTagUseOrigin> materializePhysicalTagUseOrigin(
    mlir::MLIRContext *context, const SpatialCandidateState &candidate,
    PnrIndex netOrdinal, const SpatialTagContinuitySegment &segment,
    llvm::ArrayRef<PnrIndex> routeNodeOrdinals,
    const ArtifactIdentity &dataflowIdentity,
    const ::loom::fabric::FabricArtifactView &fabric) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const auto logicalNets = problem.transfers().logicalNets();
  const auto sourceBindings = problem.transfers().logicalNetSourceBindings();
  const auto endpoints = problem.routing().routingEndpoints();
  const auto arcs = problem.routing().routingArcs();
  const auto traversalPoints =
      problem.routing().tagContinuity().traversalPointOrdinals();
  const auto pointOwners = problem.routing().tagContinuity().points();
  if (netOrdinal >= logicalNets.size() || netOrdinal >= sourceBindings.size())
    return invalid("Physical Tag origin names an absent logical net");
  auto logicalNet = dataflowAttr<::mapping::GraphProducerEndpointRefAttr>(
      context, dataflowIdentity, logicalNets[netOrdinal].producer);
  if (!logicalNet)
    return logicalNet.takeError();

  mlir::Attribute owner;
  ::loom::fabric::FabricTransportEndpointRef endpoint;
  auto expectedKind =
      ::loom::fabric::FabricPhysicalTagAssignmentPointKind::Writer;
  if (segment.originKind == SpatialTagContinuityOriginKind::RouteSource) {
    const auto source = candidate.routeTree(netOrdinal).sourceEndpoint();
    if (!source || segment.origin != *source ||
        segment.origin >= endpoints.size())
      return invalid("Physical Tag route-source origin is inconsistent");
    endpoint = endpoints[segment.origin].reference;
    const FrozenSpatialTerminalBinding binding = sourceBindings[netOrdinal];
    if (binding.kind == FrozenSpatialTerminalBindingKind::GraphBoundary) {
      owner = ::mapping::RouteTreeNodeRefAttr::get(context, *logicalNet, 0);
      expectedKind =
          ::loom::fabric::FabricPhysicalTagAssignmentPointKind::Ingress;
    } else {
      const auto demands = problem.ports().portDemands();
      if (binding.index >= demands.size())
        return invalid("Physical Tag route source has an absent port demand");
      const FrozenSpatialPortDemand &demand = demands[binding.index];
      if (demand.logicalNet != netOrdinal)
        return invalid("Physical Tag route source belongs to another net");
      if (demand.kind == FrozenSpatialPortDemandKind::Compute) {
        const auto realizations = problem.realizations().computeRealizations();
        if (demand.realization >= realizations.size())
          return invalid(
              "Physical Tag source has an absent compute realization");
        owner = ::mapping::ComputeRealizationRefAttr::get(
            context, realizations[demand.realization].reference.entity);
      } else {
        const auto realizations = problem.realizations().memoryRealizations();
        if (demand.realization >= realizations.size())
          return invalid(
              "Physical Tag source has an absent memory realization");
        owner = ::mapping::MemoryRealizationRefAttr::get(
            context, realizations[demand.realization].reference.entity);
      }
    }
  } else {
    if (segment.origin >= pointOwners.size())
      return invalid("Physical Tag boundary origin is out of range");
    const RouteTreeState &tree = candidate.routeTree(netOrdinal);
    const auto nodes = tree.nodeStorage();
    std::optional<PnrIndex> originSlot;
    for (auto [slotValue, node] : llvm::enumerate(nodes)) {
      if (!node.isActive() || node.parentArc == getInvalidPnrIndex())
        continue;
      if (node.parentArc >= arcs.size())
        return invalid("Physical Tag boundary origin has an absent arc");
      const EndpointRoutingArc &arc = arcs[node.parentArc];
      if (arc.traversal >= traversalPoints.size())
        return invalid("Physical Tag boundary origin has an absent traversal");
      if (traversalPoints[arc.traversal] != segment.origin)
        continue;
      if (originSlot)
        return invalid(
            "Physical Tag boundary origin occurs twice in one route");
      originSlot = static_cast<PnrIndex>(slotValue);
    }
    if (!originSlot || *originSlot >= routeNodeOrdinals.size() ||
        routeNodeOrdinals[*originSlot] == getInvalidPnrIndex() ||
        nodes[*originSlot].endpoint >= endpoints.size())
      return invalid(
          "Physical Tag boundary origin has no canonical route node");
    endpoint = endpoints[nodes[*originSlot].endpoint].reference;
    owner = ::mapping::RouteTreeNodeRefAttr::get(
        context, *logicalNet, routeNodeOrdinals[*originSlot]);
  }

  auto assignmentPoint = fabric.physicalTagAssignmentPoint(endpoint);
  if (!assignmentPoint || assignmentPoint->kind != expectedKind ||
      assignmentPoint->tagWidthBits != segment.tagWidthBits)
    return invalid("Physical Tag origin has no exact Fabric assignment point");
  return PhysicalTagUseOrigin{owner, *assignmentPoint};
}

llvm::Error materializePhysicalTagResourceUses(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Block &body,
    const SpatialCandidateState &candidate,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric) {
  const auto logicalNets = candidate.problem().transfers().logicalNets();
  for (PnrIndex netOrdinal = 0; netOrdinal < logicalNets.size(); ++netOrdinal) {
    if (candidate.usesRegisterFifo(netOrdinal))
      continue;
    const auto segments = candidate.tagSegments(netOrdinal);
    const auto values = candidate.tagValues(netOrdinal);
    if (segments.size() != values.size())
      return invalid("Physical Tag segments and values have different sizes");
    auto routeNodeOrdinals =
        canonicalRouteNodeOrdinals(candidate.routeTree(netOrdinal));
    if (!routeNodeOrdinals)
      return routeNodeOrdinals.takeError();
    for (auto [segment, value] : llvm::zip_equal(segments, values)) {
      if (!value || !::fabric::isRepresentablePhysicalTagValue(
                        segment.tagWidthBits, *value))
        return invalid("Physical Tag segment has no exact assigned value");
      auto origin = materializePhysicalTagUseOrigin(
          builder.getContext(), candidate, netOrdinal, segment,
          *routeNodeOrdinals, dataflow.identity(), fabric);
      if (!origin)
        return origin.takeError();
      const std::array<::fabric::UsePatternValue, 1> assignment = {
          ::fabric::PhysicalTagPatternValue{
              value->zextOrTrunc(segment.tagWidthBits)}};
      if (llvm::Error error = materializeResourceUse(
              builder, location, body, origin->owner,
              ::loom::mapping::SpatialActivityEventRef(
                  logicalNets[netOrdinal].producer),
              {}, origin->assignmentPoint.pattern, dataflow.identity(), fabric,
              {}, assignment))
        return error;
    }
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<::loom::mapping::FinalizedSpatialMapping>
finalizeSpatialMappingCandidate(
    const SpatialCandidateState &candidate,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::mapping::SpatialMappingConstraintSetView &constraints,
    const ArtifactStore &store,
    const ::loom::fabric::FabricHandshakeContext *handshakeContext) {
  if (llvm::Error error = candidate.verify())
    return std::move(error);
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  if (problem.dataflowIdentity() != dataflow.identity() ||
      problem.techMappingIdentity() != techMapping.identity() ||
      problem.fabricIdentity() != fabric.identity())
    return invalid("candidate and sealed upstream views have different owners");
  if (techMapping.dataflowIdentity() != dataflow.identity() ||
      techMapping.fabricIdentity() != fabric.identity())
    return invalid("TechMapping upstream closure is inconsistent");
  if (candidate.unroutedObligationCount() != 0)
    return invalid("candidate still has unrouted sink obligations");
  auto capacityOveruse = spatialMappingViolationValue(
      candidate, ResolvedPnrViolationKind::CapacityOveruse);
  if (!capacityOveruse)
    return capacityOveruse.takeError();
  if (*capacityOveruse != 0)
    return invalid("candidate still exceeds a resource capacity");
  if (candidate.tagUnassignedCount() != 0)
    return invalid("candidate still has unassigned Physical Tags");
  if (candidate.tagConflictCount() != 0)
    return invalid("candidate still has conflicting Physical Tags");
  auto hardProgress = spatialMappingViolationValue(
      candidate, ResolvedPnrViolationKind::HardProgressViolation);
  if (!hardProgress)
    return hardProgress.takeError();
  if (*hardProgress != 0)
    return invalid("candidate has a hard progress violation");
  mlir::DialectRegistry registry;
  registry.insert<::mapping::MappingDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadDialect<::mapping::MappingDialect>();
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  auto root = ::mapping::SpatialOp::create(
      builder, location, identityAttr(&context, techMapping.identity()),
      identityAttr(&context, dataflow.identity()),
      identityAttr(&context, fabric.identity()));
  auto *body = new mlir::Block();
  root.getBody().push_back(body);

  auto bindings =
      materializeComputeBindings(builder, location, *body, candidate);
  if (!bindings)
    return bindings.takeError();
  auto registerFifoTransfers = materializeRegisterFifoTransfers(
      builder, location, *body, candidate, dataflow.identity());
  if (!registerFifoTransfers)
    return registerFifoTransfers.takeError();
  if (llvm::Error error = materializeMemoryBindings(
          builder, location, *body, candidate, dataflow.identity()))
    return std::move(error);
  if (llvm::Error error = materializeMemoryEngineBindings(
          builder, location, *body, candidate, dataflow.identity()))
    return std::move(error);
  const auto logicalNets = problem.transfers().logicalNets();
  for (PnrIndex net = 0; net < logicalNets.size(); ++net)
    if (!candidate.usesRegisterFifo(net))
      if (llvm::Error error = materializeRouteTree(
              builder, location, *body, candidate, net, dataflow.identity()))
        return std::move(error);
  auto uses = ::loom::mapping::deriveSpatialComputeUseRequirements(
      dataflow, techMapping, fabric, *bindings, *registerFifoTransfers);
  if (!uses)
    return uses.takeError();
  if (llvm::Error error = materializeComputeResourceUses(
          builder, location, *body, *uses, dataflow.identity(), fabric))
    return std::move(error);
  if (llvm::Error error = materializeMemoryResourceUses(
          builder, location, *body, candidate, dataflow, fabric))
    return std::move(error);
  if (llvm::Error error = materializePhysicalTagResourceUses(
          builder, location, *body, candidate, dataflow, fabric))
    return std::move(error);

  return ::loom::mapping::finalizeSpatialMapping(root, dataflow, techMapping,
                                                 fabric, constraints, store,
                                                 handshakeContext);
}

} // namespace loom::pnr
