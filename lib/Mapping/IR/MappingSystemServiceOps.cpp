#include "Mapping/IR/MappingOps.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/SystemMappingIdentity.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace {

std::vector<std::uint8_t> unsignedBytes(DenseI8ArrayAttr record) {
  std::vector<std::uint8_t> result;
  result.reserve(record.size());
  for (std::int8_t byte : record.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

llvm::Expected<::loom::ArtifactIdentity>
identity(mapping::ArtifactIdentityAttr attribute) {
  return ::loom::ArtifactIdentity::fromBytes(
      unsignedBytes(attribute.getRecord()));
}

template <typename Ref, typename Attr>
llvm::Expected<Ref> decodeDataflow(Attr attribute,
                                   const ::loom::ArtifactIdentity &owner) {
  return ::dataflow::decodeDataflowReference<Ref>(
      unsignedBytes(attribute.getRecord()), owner);
}

template <typename Ref, typename Attr>
llvm::Expected<Ref> decodeFabric(Attr attribute) {
  return ::loom::fabric::decodeFabricRef<Ref>(
      unsignedBytes(attribute.getRecord()));
}

Attribute servicePlanElementKey(Operation &operation) {
  MLIRContext *context = operation.getContext();
  if (auto route = dyn_cast<mapping::TransferLegRealizationOp>(operation))
    return mapping::TransferLegElementKeyAttr::get(context, route.getLeg());
  if (auto target = dyn_cast<mapping::MemoryRegionTargetOp>(operation))
    return mapping::MemoryRegionElementKeyAttr::get(
        context, target.getLogicalMemory(), target.getInterval(),
        target.getServiceRegion(), target.getTransformPath());
  if (auto target = dyn_cast<mapping::ConsistencyTargetOp>(operation))
    return mapping::ConsistencyElementKeyAttr::get(
        context, target.getFence(), target.getConsistencyDomain());
  return {};
}

LogicalResult rejectUnknownAttributes(Operation *operation,
                                      ArrayRef<StringRef> allowed) {
  llvm::SmallDenseSet<StringRef, 8> allowedSet(allowed.begin(), allowed.end());
  for (NamedAttribute attribute : operation->getAttrs())
    if (!allowedSet.contains(attribute.getName().getValue()))
      return operation->emitOpError()
             << "unknown persistent field '" << attribute.getName() << "'";
  return success();
}

std::int64_t integerValue(Operation *operation, StringRef name) {
  return operation->getAttrOfType<IntegerAttr>(name).getInt();
}

template <typename Key>
LogicalResult requireSingleBlock(Key operation, ArrayRef<StringRef> attrs) {
  if (failed(rejectUnknownAttributes(operation, attrs)))
    return failure();
  if (operation.getBody().empty() ||
      !llvm::hasSingleElement(operation.getBody()))
    return operation.emitOpError("must contain exactly one declarative block");
  if (operation.getBody().front().getNumArguments() != 0)
    return operation.emitOpError("declarative block must not have arguments");
  return success();
}

} // namespace

LogicalResult mapping::SystemOp::verify() {
  if (failed(rejectUnknownAttributes(*this, {"dataflow", "fabric",
                                             "spatial_mapping_imports",
                                             "root_thread_launches"})))
    return failure();
  if (getBody().empty() || !llvm::hasSingleElement(getBody()))
    return emitOpError("must contain exactly one declarative block");
  if (getBody().front().getNumArguments() != 0)
    return emitOpError("declarative block must not have arguments");
  if (getRootThreadLaunches().empty())
    return emitOpError("requires a non-empty root thread launch set");

  llvm::DenseSet<Attribute> imports;
  for (Attribute attribute : getSpatialMappingImports()) {
    if (!isa<mapping::ArtifactRootReferenceAttr>(attribute))
      return emitOpError(
          "spatial_mapping_imports contains a non-root reference");
    if (!imports.insert(attribute).second)
      return emitOpError("spatial_mapping_imports contains a duplicate");
  }
  llvm::DenseSet<Attribute> roots;
  for (Attribute attribute : getRootThreadLaunches()) {
    if (!isa<mapping::RootThreadLaunchRefAttr>(attribute))
      return emitOpError(
          "root_thread_launches contains a non-root-thread reference");
    if (!roots.insert(attribute).second)
      return emitOpError("root_thread_launches contains a duplicate");
  }

  llvm::DenseSet<Attribute> threadKeys;
  llvm::DenseSet<Attribute> graphKeys;
  llvm::DenseSet<Attribute> serviceKeys;
  llvm::DenseSet<Attribute> resourceUseKeys;
  llvm::SmallVector<mapping::ResourceUseOp> resourceUses;
  for (Operation &child : getBody().front()) {
    if (auto binding = dyn_cast<mapping::ThreadExecutionBindingOp>(child)) {
      if (!roots.contains(binding.getKey()))
        return binding.emitOpError("has a key outside root_thread_launches");
      if (!threadKeys.insert(binding.getKey()).second)
        return binding.emitOpError("duplicates a ThreadExecutionBinding key");
      continue;
    }
    if (auto binding = dyn_cast<mapping::GraphExecutionBindingOp>(child)) {
      if (!graphKeys.insert(binding.getKey()).second)
        return binding.emitOpError("duplicates a GraphExecutionBinding key");
      continue;
    }
    if (auto service = dyn_cast<mapping::ServiceRealizationOp>(child)) {
      if (!serviceKeys.insert(service.getKey()).second)
        return service.emitOpError("duplicates a ServiceRealization key");
      continue;
    }
    if (auto use = dyn_cast<mapping::ResourceUseOp>(child)) {
      Attribute key = ArrayAttr::get(
          getContext(), {use.getOwner(), use.getUseSite(), use.getActivation(),
                         use.getParameters(), use.getSharingAssignments()});
      if (!resourceUseKeys.insert(key).second)
        return use.emitOpError("duplicates a ResourceUse structural key");
      resourceUses.push_back(use);
      continue;
    }
    return child.emitOpError(
        "is not an implemented closed SystemMapping record kind");
  }
  if (threadKeys.size() != roots.size())
    return emitOpError(
        "requires exactly one ThreadExecutionBinding for every root launch");

  auto dataflowOwner = identity(getDataflow());
  if (!dataflowOwner)
    return emitOpError() << llvm::toString(dataflowOwner.takeError());
  for (mapping::ResourceUseOp use : resourceUses) {
    if (auto instruction =
            dyn_cast<mapping::InstructionExecutionResourceOwnerRefAttr>(
                use.getOwner())) {
      if (!roots.contains(instruction.getRoot()))
        return use.emitOpError(
            "InstructionCore owner names a root outside root_thread_launches");
      auto context = decodeFabric<::loom::fabric::InstructionCoreContextRef>(
          instruction.getInstructionContext());
      auto pattern =
          decodeFabric<::loom::fabric::FabricUsePatternRef>(use.getUseSite());
      if (!context || !pattern) {
        if (!context)
          return use.emitOpError() << llvm::toString(context.takeError());
        return use.emitOpError() << llvm::toString(pattern.takeError());
      }
      if (pattern->owner.catalog() !=
          ::loom::fabric::FabricInventoryOwnerRef::of(*context))
        return use.emitOpError(
            "InstructionCore use site belongs to another Fabric owner");
      auto root = decodeDataflow<::dataflow::RootThreadLaunchRef>(
          instruction.getRoot(), *dataflowOwner);
      auto activation =
          dyn_cast<mapping::SystemRelativeActivationAttr>(use.getActivation());
      if (!root || !activation || !activation.getRelease())
        return use.emitOpError(
            "InstructionCore occupancy requires root start and completion");
      auto trigger = decodeDataflow<::dataflow::EventFamilyKey>(
          activation.getTrigger().getEvent(), *dataflowOwner);
      auto release = decodeDataflow<::dataflow::EventFamilyKey>(
          activation.getRelease().getEvent(), *dataflowOwner);
      if (!trigger || !release) {
        if (!trigger)
          return use.emitOpError() << llvm::toString(trigger.takeError());
        return use.emitOpError() << llvm::toString(release.takeError());
      }
      const ::dataflow::EventFamilyKey expectedTrigger(
          ::dataflow::StaticTransferEventRef(
              ::dataflow::ConsumedTransferEventRef{
                  ::dataflow::CanonicalSinkTerminalRef(
                      ::dataflow::RootThreadBoundarySinkRef{
                          ::dataflow::RootThreadBoundaryTransferRef(
                              ::dataflow::RootThreadStartTransferRef{
                                  *root})})}));
      const ::dataflow::EventFamilyKey expectedRelease(
          ::dataflow::StaticTransferEventRef(
              ::dataflow::ProducedTransferEventRef{
                  ::dataflow::CanonicalProducerTerminalRef(
                      ::dataflow::RootThreadBoundarySourceRef{
                          ::dataflow::RootThreadBoundaryTransferRef(
                              ::dataflow::RootThreadCompletionTransferRef{
                                  *root})})}));
      if (*trigger != expectedTrigger || *release != expectedRelease)
        return use.emitOpError(
            "InstructionCore occupancy has the wrong root events");
      continue;
    }

    auto owner = cast<mapping::ServicePlanElementRefAttr>(use.getOwner());
    mapping::ServiceRealizationOp selectedService;
    mapping::ServicePlanOp selectedPlan;
    for (auto service :
         getBody().front().getOps<mapping::ServiceRealizationOp>()) {
      if (service.getKey() != owner.getService())
        continue;
      selectedService = service;
      for (auto plan :
           service.getBody().front().getOps<mapping::ServicePlanOp>())
        if (plan.getPlanOrdinal() == owner.getPlanOrdinal())
          selectedPlan = plan;
    }
    if (!selectedService || !selectedPlan)
      return use.emitOpError("ServicePlanElement owner is absent");
    bool foundElement = false;
    for (Operation &child : selectedPlan.getBody().front())
      if (servicePlanElementKey(child) == owner.getElement()) {
        foundElement = true;
        break;
      }
    if (!foundElement)
      return use.emitOpError("ServicePlanElement owner is absent");
    auto activation =
        cast<mapping::SystemRelativeActivationAttr>(use.getActivation());
    if (activation.getRelease())
      return use.emitOpError(
          "service ResourceUse requires intrinsic completion");
    auto trigger = decodeDataflow<::dataflow::EventFamilyKey>(
        activation.getTrigger().getEvent(), *dataflowOwner);
    if (!trigger)
      return use.emitOpError() << llvm::toString(trigger.takeError());
    if (!std::holds_alternative<::dataflow::ContextualActorTransitionEventRef>(
            *trigger))
      return use.emitOpError(
          "service ResourceUse requires a contextual actor issue event");
  }
  return success();
}

LogicalResult mapping::ServiceRealizationOp::verify() {
  if (failed(requireSingleBlock(*this, {"key"})))
    return failure();
  llvm::DenseSet<std::uint64_t> ordinals;
  llvm::DenseSet<Attribute> selectionKeys;
  llvm::DenseSet<std::uint64_t> selectedOrdinals;
  for (Operation &child : getBody().front()) {
    if (auto plan = dyn_cast<mapping::ServicePlanOp>(child)) {
      const std::int64_t ordinal = integerValue(plan, "plan_ordinal");
      if (ordinal < 0)
        return plan.emitOpError("plan ordinal must be nonnegative");
      if (!ordinals.insert(static_cast<std::uint64_t>(ordinal)).second)
        return plan.emitOpError("duplicates a ServicePlan ordinal");
      continue;
    }
    auto selection = dyn_cast<mapping::ServicePlanSelectionOp>(child);
    if (!selection)
      return child.emitOpError(
          "is not a closed ServiceRealization record kind");
    if (!selectionKeys.insert(selection.getKey()).second)
      return selection.emitOpError("duplicates a ServicePlanSelection key");
    if (auto target =
            selection->getAttrOfType<IntegerAttr>("default_plan_ordinal"))
      selectedOrdinals.insert(static_cast<std::uint64_t>(target.getInt()));
    for (auto clause : selection.getBody()
                           .front()
                           .getOps<mapping::ServicePlanPresburgerClauseOp>())
      selectedOrdinals.insert(static_cast<std::uint64_t>(
          integerValue(clause, "target_plan_ordinal")));
  }
  if (ordinals.empty())
    return emitOpError("requires at least one ServicePlan");
  if (selectionKeys.empty())
    return emitOpError("requires at least one ServicePlanSelection");
  for (std::uint64_t selected : selectedOrdinals)
    if (!ordinals.contains(selected))
      return emitOpError(
          "ServicePlanSelection names an absent ServicePlan ordinal");
  for (std::uint64_t ordinal : ordinals)
    if (!selectedOrdinals.contains(ordinal))
      return emitOpError("contains an unselected ServicePlan ordinal");
  return success();
}

LogicalResult mapping::ServicePlanOp::verify() {
  if (failed(requireSingleBlock(*this, {"plan_ordinal"})))
    return failure();
  if (integerValue(*this, "plan_ordinal") < 0)
    return emitOpError("plan ordinal must be nonnegative");
  llvm::DenseSet<Attribute> legs;
  llvm::DenseSet<Attribute> memoryTargets;
  llvm::DenseSet<Attribute> consistencyTargets;
  for (Operation &child : getBody().front()) {
    if (auto route = dyn_cast<mapping::TransferLegRealizationOp>(child)) {
      if (!legs.insert(route.getLeg()).second)
        return route.emitOpError("duplicates a TransferLegRealization key");
      continue;
    }
    if (auto target = dyn_cast<mapping::MemoryRegionTargetOp>(child)) {
      Attribute key = ArrayAttr::get(
          getContext(), {target.getLogicalMemory(), target.getInterval(),
                         target.getServiceRegion(), target.getTransformPath()});
      if (!memoryTargets.insert(key).second)
        return target.emitOpError("duplicates a MemoryRegionTarget key");
      continue;
    }
    if (auto target = dyn_cast<mapping::ConsistencyTargetOp>(child)) {
      Attribute key = ArrayAttr::get(
          getContext(), {target.getFence(), target.getConsistencyDomain()});
      if (!consistencyTargets.insert(key).second)
        return target.emitOpError("duplicates a ConsistencyTarget key");
      continue;
    }
    return child.emitOpError("is not a closed ServicePlan record kind");
  }
  auto root = (*this)->getParentOfType<mapping::SystemOp>();
  if (!root)
    return emitOpError("must belong to a SystemMapping root");
  auto dataflowOwner = identity(root.getDataflow());
  if (!dataflowOwner)
    return emitOpError() << llvm::toString(dataflowOwner.takeError());
  auto service = (*this)->getParentOfType<mapping::ServiceRealizationOp>();
  if (!service)
    return emitOpError("must belong to a ServiceRealization");
  auto obligation = ::loom::mapping::decodeSystemServiceObligationKey(
      unsignedBytes(service.getKey().getRecord()), *dataflowOwner);
  if (!obligation)
    return emitOpError() << llvm::toString(obligation.takeError());
  if (std::holds_alternative<::loom::mapping::TransferObligationFamilyKey>(
          *obligation)) {
    if (!memoryTargets.empty() || !consistencyTargets.empty())
      return emitOpError("transfer plan must not contain a service target");
    return success();
  }
  if (legs.empty())
    return emitOpError("non-message plan requires a TransferLegRealization");
  const auto &operation =
      std::get<::loom::mapping::OperationServiceObligationFamilyKey>(
          *obligation);
  if (std::holds_alternative<::dataflow::LogicalMemoryRootOrViewRef>(
          operation)) {
    if (memoryTargets.empty() || !consistencyTargets.empty())
      return emitOpError(
          "memory plan requires a MemoryRegionTarget and no ConsistencyTarget");
  } else if (memoryTargets.size() != 0 || consistencyTargets.size() != 1) {
    return emitOpError("fence plan requires exactly one ConsistencyTarget");
  }
  return success();
}

LogicalResult mapping::MemoryRegionTargetOp::verify() {
  if (failed(requireSingleBlock(*this, {"logical_memory", "interval",
                                        "service_region", "transform_path"})))
    return failure();
  llvm::DenseSet<Attribute> transforms;
  for (Attribute transform : getTransformPath()) {
    if (!isa<mapping::SystemServiceTransformRefAttr>(transform))
      return emitOpError(
          "transform_path must contain only SystemServiceTransformRef values");
    if (!transforms.insert(transform).second)
      return emitOpError("transform_path contains a duplicate transform");
  }
  llvm::DenseSet<Attribute> exposures;
  for (Operation &child : getBody().front()) {
    auto exposure = dyn_cast<mapping::SystemMemoryExposureOp>(child);
    if (!exposure)
      return child.emitOpError("is not a System memory-exposure target child");
    Attribute key = ArrayAttr::get(
        getContext(), {exposure.getExposure(), exposure.getTerminal()});
    if (!exposures.insert(key).second)
      return exposure.emitOpError("duplicates a memory exposure target");
  }

  auto root = (*this)->getParentOfType<mapping::SystemOp>();
  auto service = (*this)->getParentOfType<mapping::ServiceRealizationOp>();
  if (!root || !service)
    return emitOpError("must belong to a System ServiceRealization");
  auto dataflowOwner = identity(root.getDataflow());
  if (!dataflowOwner)
    return emitOpError() << llvm::toString(dataflowOwner.takeError());
  auto obligation = ::loom::mapping::decodeSystemServiceObligationKey(
      unsignedBytes(service.getKey().getRecord()), *dataflowOwner);
  if (!obligation)
    return emitOpError() << llvm::toString(obligation.takeError());
  auto logicalMemory = ::dataflow::decodeDataflowReference<
      ::dataflow::LogicalMemoryRootOrViewRef>(
      unsignedBytes(getLogicalMemory().getRecord()), *dataflowOwner);
  if (!logicalMemory)
    return emitOpError() << llvm::toString(logicalMemory.takeError());
  const auto *operation =
      std::get_if<::loom::mapping::OperationServiceObligationFamilyKey>(
          &*obligation);
  const auto *expected =
      operation ? std::get_if<::dataflow::LogicalMemoryRootOrViewRef>(operation)
                : nullptr;
  if (!expected || *expected != *logicalMemory)
    return emitOpError("logical memory differs from its service obligation");
  return success();
}

LogicalResult mapping::SystemMemoryExposureOp::verify() {
  return rejectUnknownAttributes(*this, {"exposure", "terminal"});
}

LogicalResult mapping::ConsistencyTargetOp::verify() {
  if (failed(rejectUnknownAttributes(*this, {"fence", "consistency_domain"})))
    return failure();
  auto root = (*this)->getParentOfType<mapping::SystemOp>();
  auto service = (*this)->getParentOfType<mapping::ServiceRealizationOp>();
  if (!root || !service)
    return emitOpError("must belong to a System ServiceRealization");
  auto dataflowOwner = identity(root.getDataflow());
  if (!dataflowOwner)
    return emitOpError() << llvm::toString(dataflowOwner.takeError());
  auto obligation = ::loom::mapping::decodeSystemServiceObligationKey(
      unsignedBytes(service.getKey().getRecord()), *dataflowOwner);
  if (!obligation)
    return emitOpError() << llvm::toString(obligation.takeError());
  auto fence =
      ::dataflow::decodeDataflowReference<::dataflow::FenceActorFamilyRef>(
          unsignedBytes(getFence().getRecord()), *dataflowOwner);
  if (!fence)
    return emitOpError() << llvm::toString(fence.takeError());
  const auto *operation =
      std::get_if<::loom::mapping::OperationServiceObligationFamilyKey>(
          &*obligation);
  const auto *expected =
      operation ? std::get_if<::dataflow::FenceActorFamilyRef>(operation)
                : nullptr;
  if (!expected || *expected != *fence)
    return emitOpError("fence family differs from its service obligation");
  return success();
}

LogicalResult mapping::ServicePlanSelectionOp::verify() {
  if (failed(requireSingleBlock(
          *this, {"key", "relation_kind", "default_plan_ordinal"})))
    return failure();
  if (getRelationKind() !=
      mapping::SystemBindingRelationKind::PresburgerPartition)
    return emitOpError(
        "StableKeyLookup is unavailable without a Dataflow stable-key owner");
  auto defaultTarget =
      (*this)->getAttrOfType<IntegerAttr>("default_plan_ordinal");
  if (defaultTarget && defaultTarget.getInt() < 0)
    return emitOpError("default plan ordinal must be nonnegative");
  if (getBody().front().empty() && !defaultTarget)
    return emitOpError(
        "Presburger relation requires a clause or default plan ordinal");
  for (Operation &child : getBody().front())
    if (!isa<mapping::ServicePlanPresburgerClauseOp>(child))
      return child.emitOpError("is not a service-plan Presburger clause");

  auto root = (*this)->getParentOfType<mapping::SystemOp>();
  if (!root)
    return emitOpError("must belong to a SystemMapping root");
  auto dataflowOwner = identity(root.getDataflow());
  if (!dataflowOwner)
    return emitOpError() << llvm::toString(dataflowOwner.takeError());
  auto key = ::loom::mapping::decodeServicePlanSelectionKey(
      unsignedBytes(getKey().getRecord()), *dataflowOwner);
  if (!key)
    return emitOpError() << llvm::toString(key.takeError());
  auto service = cast<mapping::ServiceRealizationOp>((*this)->getParentOp());
  auto obligation = ::loom::mapping::decodeSystemServiceObligationKey(
      unsignedBytes(service.getKey().getRecord()), *dataflowOwner);
  if (!obligation)
    return emitOpError() << llvm::toString(obligation.takeError());
  if (std::holds_alternative<::loom::mapping::TransferObligationFamilyKey>(
          *obligation)) {
    const auto *member =
        std::get_if<::loom::mapping::ServiceMemberPlanSelectionAnchor>(
            &key->anchor);
    const auto message =
        ::dataflow::ServiceMemberRef(::dataflow::MessageTransferMemberRef{});
    if (!member || member->member != message)
      return emitOpError(
          "transfer obligation requires its singleton MessageTransfer anchor");
  }
  return success();
}

LogicalResult mapping::ServicePlanPresburgerClauseOp::verify() {
  if (failed(rejectUnknownAttributes(*this, {"cells", "target_plan_ordinal"})))
    return failure();
  if (integerValue(*this, "target_plan_ordinal") < 0)
    return emitOpError("target plan ordinal must be nonnegative");
  if (getCells().empty())
    return emitOpError("requires at least one Presburger cell");
  for (Attribute cell : getCells())
    if (!isa<mapping::SystemPresburgerCellAttr>(cell))
      return emitOpError("cells contains a non-Presburger value");
  return success();
}

LogicalResult mapping::TransferLegRealizationOp::verify() {
  if (failed(requireSingleBlock(*this, {"leg", "root_endpoint"})))
    return failure();
  auto root = (*this)->getParentOfType<mapping::SystemOp>();
  if (!root)
    return emitOpError("must belong to a SystemMapping root");
  auto dataflowOwner = identity(root.getDataflow());
  if (!dataflowOwner)
    return emitOpError() << llvm::toString(dataflowOwner.takeError());
  auto leg = ::loom::mapping::decodeCanonicalServiceLegKey(
      unsignedBytes(getLeg().getRecord()), *dataflowOwner);
  if (!leg)
    return emitOpError() << llvm::toString(leg.takeError());

  llvm::DenseMap<std::uint64_t, mapping::SystemRouteNodeOp> nodes;
  llvm::DenseSet<Attribute> arcs;
  llvm::DenseSet<Attribute> sinkAttachments;
  for (Operation &child : getBody().front()) {
    if (auto node = dyn_cast<mapping::SystemRouteNodeOp>(child)) {
      const std::int64_t ordinal = integerValue(node, "node_ordinal");
      if (ordinal <= 0)
        return node.emitOpError("non-root node ordinal must be positive");
      if (!nodes.try_emplace(static_cast<std::uint64_t>(ordinal), node).second)
        return node.emitOpError("duplicates a System route-node ordinal");
      const std::int64_t parent = integerValue(node, "parent_node_ordinal");
      if (parent < 0)
        return node.emitOpError("parent node ordinal must be nonnegative");
      Attribute arc =
          ArrayAttr::get(getContext(), {node->getAttr("parent_node_ordinal"),
                                        node->getAttr("incoming_traversal")});
      if (!arcs.insert(arc).second)
        return node.emitOpError(
            "duplicates a parent and incoming-traversal arc");
      continue;
    }
    auto sink = dyn_cast<mapping::SystemRouteSinkOp>(child);
    if (!sink)
      return child.emitOpError(
          "is not a closed System transfer-leg route record kind");
    Attribute attachment = ArrayAttr::get(
        getContext(), {sink.getTerminal(), sink->getAttr("node_ordinal")});
    if (!sinkAttachments.insert(attachment).second)
      return sink.emitOpError("duplicates a System route sink attachment");
    auto terminal = ::loom::mapping::decodeSystemTransferTerminalKey(
        unsignedBytes(sink.getTerminal().getRecord()), *dataflowOwner);
    if (!terminal)
      return sink.emitOpError() << llvm::toString(terminal.takeError());
    const auto *sinkKey =
        std::get_if<::loom::mapping::SystemTransferSinkTerminalKey>(&*terminal);
    if (!sinkKey)
      return sink.emitOpError("must name a sink terminal key");
    if (sinkKey->leg != *leg)
      return sink.emitOpError("names a terminal from another service leg");
  }
  if (sinkAttachments.empty())
    return emitOpError("requires at least one System route sink");

  for (auto [ordinal, node] : nodes) {
    (void)ordinal;
    const std::uint64_t parent =
        static_cast<std::uint64_t>(integerValue(node, "parent_node_ordinal"));
    if (parent != 0 && !nodes.contains(parent))
      return node.emitOpError("references an absent parent node");
  }
  for (auto sink : getBody().front().getOps<mapping::SystemRouteSinkOp>()) {
    const std::int64_t ordinal = integerValue(sink, "node_ordinal");
    if (ordinal < 0)
      return sink.emitOpError("node ordinal must be nonnegative");
    if (ordinal != 0 && !nodes.contains(static_cast<std::uint64_t>(ordinal)))
      return sink.emitOpError("references an absent route node");
  }

  llvm::DenseMap<std::uint64_t, std::uint8_t> visitState;
  for (auto [start, unused] : nodes) {
    (void)unused;
    llvm::SmallVector<std::uint64_t, 8> path;
    std::uint64_t current = start;
    while (current != 0 && visitState.lookup(current) == 0) {
      visitState[current] = 1;
      path.push_back(current);
      current = static_cast<std::uint64_t>(
          integerValue(nodes.lookup(current), "parent_node_ordinal"));
    }
    if (current != 0 && visitState.lookup(current) == 1)
      return emitOpError("System route-node parent relation contains a cycle");
    for (std::uint64_t ordinal : path)
      visitState[ordinal] = 2;
  }
  return success();
}

LogicalResult mapping::SystemRouteNodeOp::verify() {
  if (failed(
          rejectUnknownAttributes(*this, {"node_ordinal", "parent_node_ordinal",
                                          "incoming_traversal"})))
    return failure();
  if (integerValue(*this, "node_ordinal") <= 0)
    return emitOpError("non-root node ordinal must be positive");
  if (integerValue(*this, "parent_node_ordinal") < 0)
    return emitOpError("parent node ordinal must be nonnegative");
  return success();
}

LogicalResult mapping::SystemRouteSinkOp::verify() {
  if (failed(rejectUnknownAttributes(*this, {"terminal", "node_ordinal"})))
    return failure();
  if (integerValue(*this, "node_ordinal") < 0)
    return emitOpError("node ordinal must be nonnegative");
  return success();
}
