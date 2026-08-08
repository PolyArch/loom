#include "Mapping/IR/MappingOps.h"

#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

using namespace mlir;

namespace {

std::vector<std::uint8_t> unsignedBytes(DenseI8ArrayAttr record) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(record.size());
  for (std::int8_t byte : record.asArrayRef())
    bytes.push_back(static_cast<std::uint8_t>(byte));
  return bytes;
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

LogicalResult rejectUnknownAttributes(Operation *operation,
                                      ArrayRef<StringRef> allowed) {
  llvm::SmallDenseSet<StringRef, 8> allowedSet(allowed.begin(), allowed.end());
  for (NamedAttribute attribute : operation->getAttrs()) {
    if (!allowedSet.contains(attribute.getName().getValue()))
      return operation->emitOpError()
             << "unknown persistent field '" << attribute.getName() << "'";
  }
  return success();
}

using MemoryEngineRef = ::loom::fabric::FabricMemoryEngineTemplateRef;

LogicalResult verifyMemoryEndpointArray(Operation *operation,
                                        ArrayAttr attributes,
                                        MemoryEngineRef engine,
                                        llvm::StringRef field) {
  for (Attribute attribute : attributes) {
    auto endpoint =
        dyn_cast<mapping::FabricMemoryEngineTemplateEndpointRefAttr>(attribute);
    if (!endpoint)
      return operation->emitOpError()
             << field << " must contain only memory engine endpoint refs";
    auto decoded =
        decodeFabric<::loom::fabric::FabricMemoryEngineTemplateEndpointRef>(
            endpoint);
    if (!decoded)
      return operation->emitOpError() << llvm::toString(decoded.takeError());
    if (decoded->engine != engine)
      return operation->emitOpError()
             << field << " contains an endpoint owned by another engine";
  }
  return success();
}

std::pair<unsigned, std::vector<std::uint8_t>>
graphEndpointKey(Attribute endpoint) {
  if (auto producer = dyn_cast<mapping::GraphProducerEndpointRefAttr>(endpoint))
    return {0, unsignedBytes(producer.getRecord())};
  auto consumer = cast<mapping::GraphConsumerEndpointRefAttr>(endpoint);
  return {1, unsignedBytes(consumer.getRecord())};
}

LogicalResult verifyOwnerTypedValues(Operation *operation, ArrayAttr values,
                                     llvm::StringRef field) {
  for (Attribute value : values)
    if (!isa<mapping::OwnerTypedValueAttr>(value))
      return operation->emitOpError()
             << field << " must contain only owner_typed_value attributes";
  return success();
}

LogicalResult verifyPhysicalRefinements(Operation *operation,
                                        ArrayAttr refinements) {
  llvm::DenseSet<Attribute> domains;
  for (Attribute value : refinements) {
    auto assignment =
        dyn_cast<mapping::PhysicalRefinementAssignmentAttr>(value);
    if (!assignment)
      return operation->emitOpError(
          "refinements must contain only physical_refinement_assignment "
          "attributes");
    if (!domains.insert(assignment.getDomain()).second)
      return operation->emitOpError(
          "refinements contains a duplicate Fabric domain");
  }
  return success();
}

Attribute resourceUseKey(mapping::ResourceUseOp use) {
  return ArrayAttr::get(use.getContext(),
                        {use.getOwner(), use.getUseSite(), use.getActivation(),
                         use.getParameters(), use.getSharingAssignments()});
}

std::int64_t integerValue(Operation *operation, llvm::StringRef field) {
  return operation->getAttrOfType<IntegerAttr>(field).getInt();
}

LogicalResult verifyMemoryDispatchTarget(Operation *operation,
                                         Attribute dispatch,
                                         Attribute bindingTarget) {
  if (auto local = dyn_cast<mapping::LocalMemoryServiceRefAttr>(dispatch)) {
    auto region = dyn_cast<mapping::MemoryLocalRegionAttr>(bindingTarget);
    if (!region)
      return operation->emitOpError(
          "local dispatch requires a LocalRegion MemoryBinding target");
    auto decodedService =
        decodeFabric<::loom::fabric::LocalMemoryServiceRef>(local);
    auto decodedRegion =
        decodeFabric<::loom::fabric::FabricMemoryServiceRegionRef>(
            region.getServiceRegion());
    if (!decodedService)
      return operation->emitOpError()
             << llvm::toString(decodedService.takeError());
    if (!decodedRegion)
      return operation->emitOpError()
             << llvm::toString(decodedRegion.takeError());
    if (decodedRegion->service != decodedService->underlying())
      return operation->emitOpError(
          "local dispatch and LocalRegion name different services");
    return success();
  }
  if (!isa<mapping::ManagerEndpointRefAttr>(dispatch))
    return operation->emitOpError("has an invalid memory dispatch target");
  if (!isa<mapping::MemoryBoundaryProxyAttr>(bindingTarget))
    return operation->emitOpError(
        "manager dispatch requires a BoundaryProxy MemoryBinding target");
  return success();
}

} // namespace

ParseResult mapping::TechOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  std::uint32_t major = 0;
  std::uint32_t minor = 0;
  if (parser.parseKeyword("version") || parser.parseLess() ||
      parser.parseInteger(major) || parser.parseComma() ||
      parser.parseInteger(minor) || parser.parseGreater())
    return failure();
  if (major != 3 || minor != 0)
    return parser.emitError(parser.getCurrentLocation(),
                            "mapping.tech requires schema version 3.0");

  mapping::ArtifactIdentityAttr dataflow;
  mapping::ArtifactIdentityAttr fabric;
  ArrayAttr covers;
  if (parser.parseKeyword("dataflow") || parser.parseLParen() ||
      parser.parseAttribute(dataflow, "dataflow", result.attributes) ||
      parser.parseRParen() || parser.parseKeyword("fabric") ||
      parser.parseLParen() ||
      parser.parseAttribute(fabric, "fabric", result.attributes) ||
      parser.parseRParen() || parser.parseKeyword("covers") ||
      parser.parseLParen() ||
      parser.parseAttribute(covers, "covers", result.attributes) ||
      parser.parseRParen())
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, {}, /*enableNameShadowing=*/false) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void mapping::TechOp::print(OpAsmPrinter &printer) {
  printer << " version<3, 0> dataflow(" << getDataflow() << ") fabric("
          << getFabric() << ") covers(" << getCovers() << ") ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/false);
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                {"dataflow", "fabric", "covers"});
}

LogicalResult mapping::TechOp::verify() {
  if (failed(rejectUnknownAttributes(*this, {"dataflow", "fabric", "covers"})))
    return failure();
  if (getBody().empty() || !llvm::hasSingleElement(getBody()))
    return emitOpError("must contain exactly one declarative block");
  if (getCovers().empty())
    return emitOpError("covers must be non-empty");

  auto dataflowIdentity = identity(getDataflow());
  if (!dataflowIdentity)
    return emitOpError() << llvm::toString(dataflowIdentity.takeError());

  llvm::SmallDenseSet<std::uint64_t, 8> coverIds;
  for (Attribute attribute : getCovers()) {
    auto graph = dyn_cast<mapping::GraphRefAttr>(attribute);
    if (!graph)
      return emitOpError("covers must contain only GraphRef attributes");
    auto decoded =
        decodeDataflow<::dataflow::GraphRef>(graph, *dataflowIdentity);
    if (!decoded)
      return emitOpError() << llvm::toString(decoded.takeError());
    if (!coverIds.insert(decoded->entity.value()).second)
      return emitOpError("covers contains a duplicate GraphRef");
  }

  bool hasRealization = false;
  llvm::SmallDenseSet<std::uint64_t, 8> entityIds;
  for (Operation &child : getBody().front()) {
    std::optional<std::uint64_t> entityId;
    if (auto realization = dyn_cast<mapping::ComputeRealizationOp>(child))
      entityId = realization.getEntityId();
    else if (auto realization = dyn_cast<mapping::MemoryRealizationOp>(child))
      entityId = realization.getEntityId();
    else
      return child.emitOpError("is not a closed TechMapping record kind");
    hasRealization = true;
    if (!entityIds.insert(*entityId).second)
      return child.emitOpError("duplicates a Mapping EntityId");
  }
  if (!hasRealization)
    return emitOpError("must contain at least one realization");
  return success();
}

LogicalResult mapping::ComputeRealizationOp::verify() {
  if (failed(
          rejectUnknownAttributes(*this, {"entity_id", "capability_template"})))
    return failure();
  if (getBody().empty() || !llvm::hasSingleElement(getBody()))
    return emitOpError("must contain exactly one declarative block");
  bool hasActor = false;
  for (Operation &child : getBody().front()) {
    if (isa<mapping::ComputeActorOp>(child)) {
      hasActor = true;
      continue;
    }
    if (!isa<mapping::ComputeBoundaryOp>(child))
      return child.emitOpError(
          "is not a closed Compute Realization child kind");
  }
  if (!hasActor)
    return emitOpError("must contain at least one compute_actor");
  return success();
}

LogicalResult mapping::ComputeActorOp::verify() {
  if (failed(rejectUnknownAttributes(
          *this, {"actor", "fabric_op", "operand_ports", "result_ports"})))
    return failure();
  auto node =
      decodeFabric<::loom::fabric::FabricFuTemplateNodeRef>(getFabricOp());
  if (!node)
    return emitOpError() << llvm::toString(node.takeError());
  if (node->node != ::loom::fabric::FabricFuNodeKind::Op)
    return emitOpError("fabric_op must select an operation node");
  for (std::int64_t port : getOperandPorts())
    if (port < 0)
      return emitOpError("operand port ordinals must be nonnegative");
  for (std::int64_t port : getResultPorts())
    if (port < 0)
      return emitOpError("result port ordinals must be nonnegative");
  return success();
}

ParseResult mapping::ComputeBoundaryOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  mapping::ActorRefAttr actor;
  mapping::FabricFuTemplatePortRefAttr fuPort;
  StringRef directionKeyword;
  std::uint64_t portOrdinal = 0;
  if (parser.parseKeyword("actor") || parser.parseLParen() ||
      parser.parseAttribute(actor, "actor", result.attributes) ||
      parser.parseRParen() || parser.parseKeyword(&directionKeyword) ||
      parser.parseInteger(portOrdinal) || parser.parseKeyword("fu_port") ||
      parser.parseLParen() ||
      parser.parseAttribute(fuPort, "fu_port", result.attributes) ||
      parser.parseRParen())
    return failure();

  std::optional<mapping::PortDirection> direction =
      mapping::symbolizePortDirection(directionKeyword);
  if (!direction)
    return parser.emitError(parser.getCurrentLocation(),
                            "expected input or output port direction");
  Builder &builder = parser.getBuilder();
  result.addAttribute("direction", mapping::PortDirectionAttr::get(
                                       builder.getContext(), *direction));
  result.addAttribute("port_ordinal", builder.getI64IntegerAttr(portOrdinal));
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void mapping::ComputeBoundaryOp::print(OpAsmPrinter &printer) {
  printer << " actor(" << getActor() << ") "
          << mapping::stringifyPortDirection(getDirection()) << ' '
          << getPortOrdinal() << " fu_port(" << getFuPort() << ')';
  printer.printOptionalAttrDict(
      (*this)->getAttrs(), {"actor", "direction", "port_ordinal", "fu_port"});
}

LogicalResult mapping::ComputeBoundaryOp::verify() {
  if (failed(rejectUnknownAttributes(
          *this, {"actor", "direction", "port_ordinal", "fu_port"})))
    return failure();
  auto port =
      decodeFabric<::loom::fabric::FabricFuTemplatePortRef>(getFuPort());
  if (!port)
    return emitOpError() << llvm::toString(port.takeError());
  const auto expected = getDirection() == mapping::PortDirection::Input
                            ? ::loom::fabric::FabricPortDirection::Input
                            : ::loom::fabric::FabricPortDirection::Output;
  if (port->direction != expected)
    return emitOpError("software and FU boundary directions disagree");
  return success();
}

LogicalResult mapping::MemoryRealizationOp::verify() {
  if (failed(rejectUnknownAttributes(*this, {"entity_id", "engine"})))
    return failure();
  if (getBody().empty() || !llvm::hasSingleElement(getBody()))
    return emitOpError("must contain exactly one declarative block");

  auto engine = decodeFabric<MemoryEngineRef>(getEngine());
  if (!engine)
    return emitOpError() << llvm::toString(engine.takeError());

  bool hasActor = false;
  std::set<std::vector<std::uint8_t>> actorKeys;
  std::set<std::pair<unsigned, std::vector<std::uint8_t>>> boundaryKeys;
  std::set<std::pair<std::vector<std::uint8_t>, std::vector<std::uint8_t>>>
      edgeKeys;

  for (Operation &child : getBody().front()) {
    if (auto actor = dyn_cast<mapping::MemoryActorOp>(child)) {
      hasActor = true;
      if (!actorKeys.insert(unsignedBytes(actor.getActor().getRecord())).second)
        return actor.emitOpError("duplicates a memory actor correspondence");
      auto port = decodeFabric<
          ::loom::fabric::FabricMemoryEngineTemplateOperationPortRef>(
          actor.getOperationPort());
      if (!port)
        return actor.emitOpError() << llvm::toString(port.takeError());
      if (port->engine != *engine)
        return actor.emitOpError(
            "operation port is owned by another memory engine template");
      continue;
    }
    if (auto boundary = dyn_cast<mapping::MemoryGraphBoundaryOp>(child)) {
      if (!boundaryKeys.insert(graphEndpointKey(boundary.getTerminal())).second)
        return boundary.emitOpError(
            "duplicates a graph-boundary terminal correspondence");
      auto endpoint =
          decodeFabric<::loom::fabric::FabricMemoryEngineTemplateEndpointRef>(
              boundary.getEndpoint());
      if (!endpoint)
        return boundary.emitOpError() << llvm::toString(endpoint.takeError());
      if (endpoint->engine != *engine)
        return boundary.emitOpError(
            "endpoint is owned by another memory engine template");
      continue;
    }
    if (auto edge = dyn_cast<mapping::MemoryInternalEdgeOp>(child)) {
      auto key = std::make_pair(unsignedBytes(edge.getProducer().getRecord()),
                                unsignedBytes(edge.getConsumer().getRecord()));
      if (!edgeKeys.insert(std::move(key)).second)
        return edge.emitOpError("duplicates a canonical software edge");
      auto connection = decodeFabric<
          ::loom::fabric::FabricMemoryEngineTemplateInternalConnectionRef>(
          edge.getConnection());
      if (!connection)
        return edge.emitOpError() << llvm::toString(connection.takeError());
      if (connection->engine != *engine ||
          connection->source.engine != *engine ||
          connection->sink.engine != *engine)
        return edge.emitOpError(
            "connection is owned by another memory engine template");
      continue;
    }
    return child.emitOpError("is not a closed Memory Realization child kind");
  }
  if (!hasActor)
    return emitOpError("must contain at least one memory_actor");
  return success();
}

LogicalResult mapping::MemoryActorOp::verify() {
  if (failed(rejectUnknownAttributes(*this,
                                     {"actor", "operation_port", "capability",
                                      "operand_ports", "result_ports"})))
    return failure();
  auto port =
      decodeFabric<::loom::fabric::FabricMemoryEngineTemplateOperationPortRef>(
          getOperationPort());
  if (!port)
    return emitOpError() << llvm::toString(port.takeError());
  auto capability = decodeFabric<
      ::loom::fabric::FabricMemoryEngineTemplateCapabilityAlternativeRef>(
      getCapability());
  if (!capability)
    return emitOpError() << llvm::toString(capability.takeError());
  if (capability->port != *port)
    return emitOpError(
        "capability alternative is not owned by the selected operation port");
  if (failed(verifyMemoryEndpointArray(*this, getOperandPorts(), port->engine,
                                       "operand_ports")) ||
      failed(verifyMemoryEndpointArray(*this, getResultPorts(), port->engine,
                                       "result_ports")))
    return failure();
  return success();
}

LogicalResult mapping::MemoryGraphBoundaryOp::verify() {
  return rejectUnknownAttributes(*this, {"terminal", "endpoint"});
}

LogicalResult mapping::MemoryInternalEdgeOp::verify() {
  if (failed(rejectUnknownAttributes(*this,
                                     {"producer", "consumer", "connection"})))
    return failure();
  auto connection = decodeFabric<
      ::loom::fabric::FabricMemoryEngineTemplateInternalConnectionRef>(
      getConnection());
  if (!connection)
    return emitOpError() << llvm::toString(connection.takeError());
  if (connection->source.engine != connection->engine ||
      connection->sink.engine != connection->engine)
    return emitOpError(
        "internal connection endpoints must share the connection owner");
  return success();
}

ParseResult mapping::SpatialOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  std::uint32_t major = 0;
  std::uint32_t minor = 0;
  if (parser.parseKeyword("version") || parser.parseLess() ||
      parser.parseInteger(major) || parser.parseComma() ||
      parser.parseInteger(minor) || parser.parseGreater())
    return failure();
  if (major != 3 || minor != 0)
    return parser.emitError(parser.getCurrentLocation(),
                            "mapping.spatial requires schema version 3.0");

  mapping::ArtifactIdentityAttr techMapping;
  mapping::ArtifactIdentityAttr dataflow;
  mapping::ArtifactIdentityAttr fabric;
  if (parser.parseKeyword("tech_mapping") || parser.parseLParen() ||
      parser.parseAttribute(techMapping, "tech_mapping", result.attributes) ||
      parser.parseRParen() || parser.parseKeyword("dataflow") ||
      parser.parseLParen() ||
      parser.parseAttribute(dataflow, "dataflow", result.attributes) ||
      parser.parseRParen() || parser.parseKeyword("fabric") ||
      parser.parseLParen() ||
      parser.parseAttribute(fabric, "fabric", result.attributes) ||
      parser.parseRParen())
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, {}, /*enableNameShadowing=*/false) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void mapping::SpatialOp::print(OpAsmPrinter &printer) {
  printer << " version<3, 0> tech_mapping(" << getTechMapping() << ") dataflow("
          << getDataflow() << ") fabric(" << getFabric() << ") ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/false);
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                {"tech_mapping", "dataflow", "fabric"});
}

LogicalResult mapping::SpatialOp::verify() {
  if (failed(rejectUnknownAttributes(*this,
                                     {"tech_mapping", "dataflow", "fabric"})))
    return failure();
  if (getBody().empty() || !llvm::hasSingleElement(getBody()))
    return emitOpError("must contain exactly one declarative block");
  if (getBody().front().getNumArguments() != 0)
    return emitOpError("declarative block must not have arguments");

  llvm::SmallDenseSet<std::uint64_t, 8> computeBindings;
  llvm::SmallDenseSet<std::uint64_t, 8> memoryEngineBindings;
  llvm::DenseMap<std::uint64_t, mapping::MemoryBindingOp> memoryBindings;
  llvm::DenseMap<Attribute, llvm::SmallDenseSet<std::uint64_t, 8>> routeNodes;
  llvm::DenseSet<Attribute> resourceUses;
  llvm::SmallVector<mapping::ResourceUseOp, 8> deferredUses;
  for (Operation &child : getBody().front()) {
    if (auto binding = dyn_cast<mapping::ComputeBindingOp>(child)) {
      if (!computeBindings.insert(binding.getRealization().getEntity()).second)
        return binding.emitOpError("duplicates a ComputeBinding key");
      continue;
    }
    if (auto binding = dyn_cast<mapping::MemoryEngineBindingOp>(child)) {
      if (!memoryEngineBindings.insert(binding.getRealization().getEntity())
               .second)
        return binding.emitOpError("duplicates a MemoryEngineBinding key");
      continue;
    }
    if (auto binding = dyn_cast<mapping::MemoryBindingOp>(child)) {
      const std::int64_t entity = integerValue(binding, "entity_id");
      if (entity < 0)
        return binding.emitOpError("entity ID must be nonnegative");
      if (!memoryBindings
               .try_emplace(static_cast<std::uint64_t>(entity), binding)
               .second)
        return binding.emitOpError("duplicates a MemoryBinding EntityId");
      continue;
    }
    if (auto route = dyn_cast<mapping::RouteTreeOp>(child)) {
      auto [position, inserted] = routeNodes.try_emplace(route.getLogicalNet());
      if (!inserted)
        return route.emitOpError("duplicates a RouteTree logical-net key");
      for (auto node : route.getBody().front().getOps<mapping::RouteNodeOp>())
        position->second.insert(
            static_cast<std::uint64_t>(integerValue(node, "node_ordinal")));
      continue;
    }
    if (auto use = dyn_cast<mapping::ResourceUseOp>(child)) {
      if (!resourceUses.insert(resourceUseKey(use)).second)
        return use.emitOpError("duplicates a ResourceUse structural key");
      deferredUses.push_back(use);
      continue;
    }
    return child.emitOpError(
        "is not an implemented closed SpatialMapping record kind");
  }

  for (auto engine :
       getBody().front().getOps<mapping::MemoryEngineBindingOp>()) {
    for (auto operation : engine.getBody()
                              .front()
                              .getOps<mapping::AddressedMemoryOperationOp>()) {
      for (auto use : operation.getBody()
                          .front()
                          .getOps<mapping::AddressedMemoryUseOp>()) {
        auto binding = memoryBindings.find(use.getBinding().getEntity());
        if (binding == memoryBindings.end())
          return use.emitOpError("references an absent MemoryBinding target");
        if (failed(verifyMemoryDispatchTarget(use, use.getDispatchTarget(),
                                              binding->second.getTarget())))
          return failure();
      }
    }
  }

  for (mapping::ResourceUseOp use : deferredUses) {
    if (auto compute =
            dyn_cast<mapping::ComputeRealizationRefAttr>(use.getOwner())) {
      if (computeBindings.contains(compute.getEntity()))
        continue;
      return use.emitOpError("references an absent ComputeBinding owner");
    }
    if (auto engine =
            dyn_cast<mapping::MemoryRealizationRefAttr>(use.getOwner())) {
      if (memoryEngineBindings.contains(engine.getEntity()))
        continue;
      return use.emitOpError("references an absent MemoryEngineBinding owner");
    }
    if (auto binding =
            dyn_cast<mapping::MemoryBindingRefAttr>(use.getOwner())) {
      if (memoryBindings.contains(binding.getEntity()))
        continue;
      return use.emitOpError("references an absent MemoryBinding owner");
    }
    if (auto route = dyn_cast<mapping::RouteTreeNodeRefAttr>(use.getOwner())) {
      auto nodes = routeNodes.find(route.getLogicalNet());
      if (nodes == routeNodes.end())
        return use.emitOpError("references an absent RouteTree owner");
      if (!nodes->second.contains(route.getNodeOrdinal()))
        return use.emitOpError("references an absent RouteTree node owner");
      continue;
    }
    return use.emitOpError(
        "references a Spatial owner family not present in this root");
  }
  return success();
}

LogicalResult mapping::ComputeBindingOp::verify() {
  if (failed(rejectUnknownAttributes(
          *this, {"realization", "occurrence", "context", "refinements"})))
    return failure();
  if (failed(verifyPhysicalRefinements(*this, getRefinements())))
    return failure();
  return success();
}

LogicalResult mapping::MemoryEngineBindingOp::verify() {
  if (failed(rejectUnknownAttributes(*this, {"realization", "occurrence"})))
    return failure();
  if (getBody().empty() || !llvm::hasSingleElement(getBody()))
    return emitOpError("must contain exactly one declarative block");
  if (getBody().front().getNumArguments() != 0)
    return emitOpError("declarative block must not have arguments");
  llvm::DenseSet<Attribute> actors;
  for (Operation &child : getBody().front()) {
    Attribute actor;
    if (auto addressed = dyn_cast<mapping::AddressedMemoryOperationOp>(child))
      actor = addressed.getActor();
    else if (auto fence = dyn_cast<mapping::FenceMemoryOperationOp>(child))
      actor = fence.getActor();
    else
      return child.emitOpError(
          "is not a closed MemoryOperationEntry record kind");
    if (!actors.insert(actor).second)
      return child.emitOpError("duplicates a memory actor entry");
  }
  return success();
}

LogicalResult mapping::AddressedMemoryOperationOp::verify() {
  if (failed(rejectUnknownAttributes(*this, {"actor", "placement"})))
    return failure();
  if (getBody().empty() || !llvm::hasSingleElement(getBody()))
    return emitOpError("must contain exactly one declarative block");
  if (getBody().front().getNumArguments() != 0)
    return emitOpError("declarative block must not have arguments");
  llvm::DenseSet<Attribute> launches;
  for (Operation &child : getBody().front()) {
    auto use = dyn_cast<mapping::AddressedMemoryUseOp>(child);
    if (!use)
      return child.emitOpError("is not an AddressedOperationUse record");
    if (!launches.insert(use.getLaunch()).second)
      return use.emitOpError("duplicates a rooted addressed-memory use");
  }
  if (launches.empty())
    return emitOpError("must contain at least one rooted addressed-memory use");
  return success();
}

LogicalResult mapping::AddressedMemoryUseOp::verify() {
  return rejectUnknownAttributes(*this,
                                 {"launch", "binding", "dispatch_target"});
}

LogicalResult mapping::FenceMemoryOperationOp::verify() {
  if (failed(rejectUnknownAttributes(*this, {"actor", "placement"})))
    return failure();
  if (getBody().empty() || !llvm::hasSingleElement(getBody()))
    return emitOpError("must contain exactly one declarative block");
  if (getBody().front().getNumArguments() != 0)
    return emitOpError("declarative block must not have arguments");
  llvm::DenseSet<Attribute> launches;
  for (Operation &child : getBody().front()) {
    auto use = dyn_cast<mapping::FenceMemoryUseOp>(child);
    if (!use)
      return child.emitOpError("is not a FenceOperationUse record");
    if (!launches.insert(use.getLaunch()).second)
      return use.emitOpError("duplicates a rooted fence use");
  }
  if (launches.empty())
    return emitOpError("must contain at least one rooted fence use");
  return success();
}

LogicalResult mapping::FenceMemoryUseOp::verify() {
  return rejectUnknownAttributes(*this, {"launch", "consistency_target"});
}

LogicalResult mapping::MemoryBindingOp::verify() {
  if (failed(rejectUnknownAttributes(
          *this, {"entity_id", "logical_memory", "interval", "target"})))
    return failure();
  if (integerValue(*this, "entity_id") < 0)
    return emitOpError("entity ID must be nonnegative");
  if (getBody().empty() || !llvm::hasSingleElement(getBody()))
    return emitOpError("must contain exactly one declarative block");
  if (getBody().front().getNumArguments() != 0)
    return emitOpError("declarative block must not have arguments");

  auto parent = (*this)->getParentOfType<mapping::SpatialOp>();
  if (!parent)
    return emitOpError("must belong to a SpatialMapping root");
  auto dataflowOwner = identity(parent.getDataflow());
  if (!dataflowOwner)
    return emitOpError() << llvm::toString(dataflowOwner.takeError());
  auto logicalMemory = decodeDataflow<::dataflow::LogicalMemoryRootOrViewRef>(
      getLogicalMemory(), *dataflowOwner);
  if (!logicalMemory)
    return emitOpError() << llvm::toString(logicalMemory.takeError());

  if (auto local = dyn_cast<mapping::MemoryLocalRegionAttr>(getTarget())) {
    auto region = decodeFabric<::loom::fabric::FabricMemoryServiceRegionRef>(
        local.getServiceRegion());
    if (!region)
      return emitOpError() << llvm::toString(region.takeError());
    if (region->service.kind() !=
        ::loom::fabric::FabricMemoryServiceKind::Local)
      return emitOpError("LocalRegion must name a local memory service");
  }

  llvm::DenseSet<Attribute> exposures;
  for (Operation &child : getBody().front()) {
    auto exposure = dyn_cast<mapping::ExposureEntryOp>(child);
    if (!exposure)
      return child.emitOpError("is not a closed ExposureEntry record kind");
    if (!exposures.insert(exposure.getExposure()).second)
      return exposure.emitOpError("duplicates a memory exposure key");
    if (failed(verifyMemoryDispatchTarget(
            exposure, exposure.getDispatchTarget(), getTarget())))
      return failure();
  }
  return success();
}

LogicalResult mapping::ExposureEntryOp::verify() {
  return rejectUnknownAttributes(*this,
                                 {"exposure", "terminal", "dispatch_target"});
}

LogicalResult mapping::RouteTreeOp::verify() {
  if (failed(rejectUnknownAttributes(*this, {"logical_net", "root_endpoint"})))
    return failure();
  if (getBody().empty() || !llvm::hasSingleElement(getBody()))
    return emitOpError("must contain exactly one declarative block");
  if (getBody().front().getNumArguments() != 0)
    return emitOpError("declarative block must not have arguments");

  llvm::DenseMap<std::uint64_t, mapping::RouteNodeOp> nodes;
  llvm::DenseSet<Attribute> sinks;
  llvm::DenseSet<Attribute> incomingArcs;
  llvm::SmallVector<std::uint64_t, 8> roots;
  for (Operation &child : getBody().front()) {
    if (auto node = dyn_cast<mapping::RouteNodeOp>(child)) {
      const std::int64_t ordinal = integerValue(node, "node_ordinal");
      if (ordinal < 0)
        return node.emitOpError("node ordinal must be nonnegative");
      if (!nodes.try_emplace(static_cast<std::uint64_t>(ordinal), node).second)
        return node.emitOpError("duplicates a route-node ordinal");

      auto parent = node->getAttrOfType<IntegerAttr>("parent_node_ordinal");
      auto traversal = node.getIncomingTraversalAttr();
      if (static_cast<bool>(parent) != static_cast<bool>(traversal))
        return node.emitOpError(
            "parent and incoming traversal must be present together");
      if (!parent) {
        roots.push_back(static_cast<std::uint64_t>(ordinal));
        continue;
      }
      if (parent.getInt() < 0)
        return node.emitOpError("parent node ordinal must be nonnegative");
      Attribute arc = ArrayAttr::get(getContext(), {parent, traversal});
      if (!incomingArcs.insert(arc).second)
        return node.emitOpError(
            "duplicates a parent and incoming-traversal arc");
      continue;
    }
    if (auto sink = dyn_cast<mapping::RouteSinkOp>(child)) {
      if (!sinks.insert(sink.getSink()).second)
        return sink.emitOpError("duplicates a sink obligation");
      continue;
    }
    return child.emitOpError("is not a closed RouteTree record kind");
  }

  if (nodes.empty())
    return emitOpError("must contain at least one route node");
  if (roots.size() != 1)
    return emitOpError("must contain exactly one root route node");
  if (sinks.empty())
    return emitOpError("must contain at least one sink attachment");

  for (auto [ordinal, node] : nodes) {
    (void)ordinal;
    auto parent = node->getAttrOfType<IntegerAttr>("parent_node_ordinal");
    if (parent && !nodes.contains(static_cast<std::uint64_t>(parent.getInt())))
      return node.emitOpError("references an absent parent route node");
  }
  for (auto sink : getBody().front().getOps<mapping::RouteSinkOp>()) {
    const std::int64_t ordinal = integerValue(sink, "node_ordinal");
    if (ordinal < 0)
      return sink.emitOpError("node ordinal must be nonnegative");
    if (!nodes.contains(static_cast<std::uint64_t>(ordinal)))
      return sink.emitOpError("references an absent route node");
  }

  llvm::DenseMap<std::uint64_t, std::uint8_t> visitState;
  for (auto [start, unused] : nodes) {
    (void)unused;
    llvm::SmallVector<std::uint64_t, 8> path;
    std::uint64_t current = start;
    while (visitState.lookup(current) == 0) {
      visitState[current] = 1;
      path.push_back(current);
      auto parent = nodes.lookup(current)->getAttrOfType<IntegerAttr>(
          "parent_node_ordinal");
      if (!parent)
        break;
      current = static_cast<std::uint64_t>(parent.getInt());
    }
    if (visitState.lookup(current) == 1 && current != roots.front())
      return emitOpError("route-node parent relation contains a cycle");
    for (std::uint64_t ordinal : path)
      visitState[ordinal] = 2;
  }
  return success();
}

LogicalResult mapping::RouteNodeOp::verify() {
  if (failed(rejectUnknownAttributes(*this,
                                     {"node_ordinal", "parent_node_ordinal",
                                      "incoming_traversal", "refinements"})))
    return failure();
  if (integerValue(*this, "node_ordinal") < 0)
    return emitOpError("node ordinal must be nonnegative");
  auto parent = (*this)->getAttrOfType<IntegerAttr>("parent_node_ordinal");
  if (parent && parent.getInt() < 0)
    return emitOpError("parent node ordinal must be nonnegative");
  if (static_cast<bool>(parent) !=
      static_cast<bool>(getIncomingTraversalAttr()))
    return emitOpError(
        "parent and incoming traversal must be present together");
  return verifyPhysicalRefinements(*this, getRefinements());
}

LogicalResult mapping::RouteSinkOp::verify() {
  if (failed(rejectUnknownAttributes(*this, {"sink", "node_ordinal"})))
    return failure();
  if (integerValue(*this, "node_ordinal") < 0)
    return emitOpError("node ordinal must be nonnegative");
  return success();
}

LogicalResult mapping::ResourceUseOp::verify() {
  if (failed(rejectUnknownAttributes(*this,
                                     {"owner", "use_site", "activation",
                                      "parameters", "sharing_assignments"})))
    return failure();
  if (failed(verifyOwnerTypedValues(*this, getParameters(), "parameters")) ||
      failed(verifyOwnerTypedValues(*this, getSharingAssignments(), "sharing")))
    return failure();
  Operation *parent = (*this)->getParentOp();
  if (isa_and_nonnull<mapping::SpatialOp>(parent)) {
    if (!isa<mapping::ComputeRealizationRefAttr,
             mapping::MemoryRealizationRefAttr, mapping::MemoryBindingRefAttr,
             mapping::RouteTreeNodeRefAttr>(getOwner()))
      return emitOpError("Spatial ResourceUse requires a Spatial owner");
    if (!isa<mapping::SpatialRelativeActivationAttr>(getActivation()))
      return emitOpError("Spatial ResourceUse requires Spatial activation");
    return success();
  }
  if (isa_and_nonnull<mapping::SystemOp>(parent)) {
    if (!isa<mapping::InstructionExecutionResourceOwnerRefAttr,
             mapping::ServicePlanElementRefAttr>(getOwner()))
      return emitOpError("System ResourceUse requires a System owner");
    if (!isa<mapping::SystemRelativeActivationAttr>(getActivation()))
      return emitOpError("System ResourceUse requires System activation");
    return success();
  }
  return emitOpError("must be a direct child of a Mapping root");
}

ParseResult mapping::SystemOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  std::uint32_t major = 0;
  std::uint32_t minor = 0;
  if (parser.parseKeyword("version") || parser.parseLess() ||
      parser.parseInteger(major) || parser.parseComma() ||
      parser.parseInteger(minor) || parser.parseGreater())
    return failure();
  if (major != 3 || minor != 0)
    return parser.emitError(parser.getCurrentLocation(),
                            "mapping.system requires schema version 3.0");

  mapping::ArtifactIdentityAttr dataflow;
  mapping::ArtifactIdentityAttr fabric;
  ArrayAttr imports;
  ArrayAttr roots;
  if (parser.parseKeyword("dataflow") || parser.parseLParen() ||
      parser.parseAttribute(dataflow, "dataflow", result.attributes) ||
      parser.parseRParen() || parser.parseKeyword("fabric") ||
      parser.parseLParen() ||
      parser.parseAttribute(fabric, "fabric", result.attributes) ||
      parser.parseRParen() || parser.parseKeyword("spatial_mapping_imports") ||
      parser.parseLParen() ||
      parser.parseAttribute(imports, "spatial_mapping_imports",
                            result.attributes) ||
      parser.parseRParen() || parser.parseKeyword("root_thread_launches") ||
      parser.parseLParen() ||
      parser.parseAttribute(roots, "root_thread_launches", result.attributes) ||
      parser.parseRParen())
    return failure();
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, {}, /*enableNameShadowing=*/false) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void mapping::SystemOp::print(OpAsmPrinter &printer) {
  printer << " version<3, 0> dataflow(" << getDataflow() << ") fabric("
          << getFabric() << ") spatial_mapping_imports("
          << getSpatialMappingImports() << ") root_thread_launches("
          << getRootThreadLaunches() << ") ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/false);
  printer.printOptionalAttrDict((*this)->getAttrs(), {"dataflow", "fabric",
                                                      "spatial_mapping_imports",
                                                      "root_thread_launches"});
}

LogicalResult mapping::ThreadExecutionBindingOp::verify() {
  if (failed(rejectUnknownAttributes(
          *this, {"key", "relation_kind", "default_target"})))
    return failure();
  if (getBody().empty() || !llvm::hasSingleElement(getBody()) ||
      getBody().front().getNumArguments() != 0)
    return emitOpError("must contain one argument-free declarative block");
  if (getRelationKind() !=
      mapping::SystemBindingRelationKind::PresburgerPartition)
    return emitOpError(
        "StableKeyLookup is unavailable without a Dataflow stable-key owner");
  if (getBody().front().empty() && !getDefaultTarget())
    return emitOpError(
        "Presburger relation requires a clause or default target");
  for (Operation &child : getBody().front())
    if (!isa<mapping::ThreadPresburgerClauseOp>(child))
      return child.emitOpError("is not a thread Presburger clause");
  return success();
}

LogicalResult mapping::ThreadPresburgerClauseOp::verify() {
  if (failed(rejectUnknownAttributes(*this, {"cells", "target"})))
    return failure();
  if (getCells().empty())
    return emitOpError("requires at least one Presburger cell");
  for (Attribute cell : getCells())
    if (!isa<mapping::SystemPresburgerCellAttr>(cell))
      return emitOpError("cells contains a non-Presburger value");
  return success();
}

LogicalResult mapping::GraphExecutionBindingOp::verify() {
  if (failed(rejectUnknownAttributes(
          *this, {"key", "relation_kind", "default_target"})))
    return failure();
  if (getBody().empty() || !llvm::hasSingleElement(getBody()) ||
      getBody().front().getNumArguments() != 0)
    return emitOpError("must contain one argument-free declarative block");
  if (getRelationKind() !=
      mapping::SystemBindingRelationKind::PresburgerPartition)
    return emitOpError(
        "StableKeyLookup is unavailable without a Dataflow stable-key owner");
  if (getBody().front().empty() && !getDefaultTarget())
    return emitOpError(
        "Presburger relation requires a clause or default target");
  const auto importCount = cast<mapping::SystemOp>((*this)->getParentOp())
                               .getSpatialMappingImports()
                               .size();
  if (getDefaultTarget() && getDefaultTarget()->getOrdinal() >= importCount)
    return emitOpError("default target names an absent SpatialMapping import");
  for (Operation &child : getBody().front()) {
    auto clause = dyn_cast<mapping::GraphPresburgerClauseOp>(child);
    if (!clause)
      return child.emitOpError("is not a graph Presburger clause");
    if (clause.getTarget().getOrdinal() >= importCount)
      return clause.emitOpError("target names an absent SpatialMapping import");
  }
  return success();
}

LogicalResult mapping::GraphPresburgerClauseOp::verify() {
  if (failed(rejectUnknownAttributes(*this, {"cells", "target"})))
    return failure();
  if (getCells().empty())
    return emitOpError("requires at least one Presburger cell");
  for (Attribute cell : getCells())
    if (!isa<mapping::SystemPresburgerCellAttr>(cell))
      return emitOpError("cells contains a non-Presburger value");
  return success();
}

ParseResult mapping::ConstraintsSpatialOp::parse(OpAsmParser &parser,
                                                 OperationState &result) {
  mapping::ArtifactIdentityAttr dataflow;
  mapping::ArtifactIdentityAttr techMapping;
  mapping::ArtifactIdentityAttr fabric;
  if (parser.parseKeyword("dataflow") || parser.parseLParen() ||
      parser.parseAttribute(dataflow, "dataflow", result.attributes) ||
      parser.parseRParen() || parser.parseKeyword("tech_mapping") ||
      parser.parseLParen() ||
      parser.parseAttribute(techMapping, "tech_mapping", result.attributes) ||
      parser.parseRParen() || parser.parseKeyword("fabric") ||
      parser.parseLParen() ||
      parser.parseAttribute(fabric, "fabric", result.attributes) ||
      parser.parseRParen())
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, {}, /*enableNameShadowing=*/false))
    return failure();
  if (body->empty())
    body->emplaceBlock();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void mapping::ConstraintsSpatialOp::print(OpAsmPrinter &printer) {
  printer << " dataflow(" << getDataflow() << ") tech_mapping("
          << getTechMapping() << ") fabric(" << getFabric() << ") ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/false);
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                {"dataflow", "tech_mapping", "fabric"});
}

LogicalResult mapping::ConstraintsSpatialOp::verify() {
  if (failed(rejectUnknownAttributes(*this,
                                     {"dataflow", "tech_mapping", "fabric"})))
    return failure();
  if (getBody().empty() || !llvm::hasSingleElement(getBody()))
    return emitOpError("must contain exactly one declarative block");
  if (getBody().front().getNumArguments() != 0)
    return emitOpError("declarative block must not have arguments");
  for (Operation &child : getBody().front()) {
    if (!isa<mapping::ConstraintDomainRestrictionOp, mapping::ConstraintEqualOp,
             mapping::ConstraintDisjointOp>(child))
      return child.emitOpError(
          "is not a closed Spatial MappingConstraintSet clause kind");
  }
  return success();
}

LogicalResult mapping::ConstraintsSystemOp::verify() {
  if (failed(rejectUnknownAttributes(
          *this, {"dataflow", "fabric", "root_thread_launches",
                  "spatial_mapping_reference_table"})))
    return failure();
  if (getBody().empty() || !llvm::hasSingleElement(getBody()))
    return emitOpError("must contain exactly one declarative block");
  if (getBody().front().getNumArguments() != 0)
    return emitOpError("declarative block must not have arguments");
  if (getRootThreadLaunches().empty())
    return emitOpError("requires a non-empty root thread launch set");
  for (Attribute attribute : getRootThreadLaunches()) {
    if (!isa<mapping::RootThreadLaunchRefAttr>(attribute))
      return emitOpError(
          "root_thread_launches contains a non-root-thread reference");
  }
  for (Attribute attribute : getSpatialMappingReferenceTable()) {
    if (!isa<mapping::ArtifactRootReferenceAttr>(attribute))
      return emitOpError("spatial_mapping_reference_table contains a "
                         "non-ArtifactRootReference value");
  }
  for (Operation &child : getBody().front()) {
    if (!isa<mapping::ConstraintDomainRestrictionOp, mapping::ConstraintEqualOp,
             mapping::ConstraintDisjointOp>(child))
      return child.emitOpError(
          "is not a closed System MappingConstraintSet clause kind");
  }
  return success();
}

namespace {

ParseResult parseConstraintProjection(OpAsmParser &parser,
                                      OperationState &result) {
  StringRef keyword;
  if (parser.parseKeyword("projection") || parser.parseLParen() ||
      parser.parseKeyword(&keyword) || parser.parseRParen())
    return failure();
  std::optional<mapping::SpatialConstraintProjection> spatial =
      mapping::symbolizeSpatialConstraintProjection(keyword);
  if (spatial) {
    result.addAttribute(
        "projection",
        mapping::SpatialConstraintProjectionKeyAttr::get(
            parser.getContext(), static_cast<std::uint32_t>(*spatial)));
    return success();
  }
  std::optional<mapping::SystemConstraintProjection> system =
      mapping::symbolizeSystemConstraintProjection(keyword);
  if (system) {
    result.addAttribute(
        "projection",
        mapping::SystemConstraintProjectionKeyAttr::get(
            parser.getContext(), static_cast<std::uint32_t>(*system)));
    return success();
  }
  return parser.emitError(parser.getCurrentLocation(),
                          "unknown constraint projection");
}

void printConstraintProjection(OpAsmPrinter &printer, Attribute projection) {
  printer << " projection(";
  if (auto spatial =
          dyn_cast<mapping::SpatialConstraintProjectionKeyAttr>(projection))
    printer << mapping::stringifySpatialConstraintProjection(
        static_cast<mapping::SpatialConstraintProjection>(spatial.getValue()));
  else if (auto system =
               dyn_cast<mapping::SystemConstraintProjectionKeyAttr>(projection))
    printer << mapping::stringifySystemConstraintProjection(
        static_cast<mapping::SystemConstraintProjection>(system.getValue()));
  else
    llvm_unreachable("unknown constraint projection attribute");
  printer << ')';
}

bool isSpatialConstraintSubject(mapping::SpatialConstraintProjection projection,
                                Attribute subject) {
  using Projection = mapping::SpatialConstraintProjection;
  switch (projection) {
  case Projection::ComputePlacement:
  case Projection::ComputeParentPe:
  case Projection::ComputeInstructionContext:
  case Projection::ComputeFuContext:
    return isa<mapping::ComputeRealizationRefAttr>(subject);
  case Projection::MemoryPlacement:
    return isa<mapping::MemoryRealizationRefAttr>(subject);
  case Projection::NetAssignedTagValues:
  case Projection::NetSelectedPhysicalTraversals:
  case Projection::NetTraversalResourceStates:
    return isa<mapping::GraphProducerEndpointRefAttr>(subject);
  case Projection::SpatialTransferAttachment:
    return isa<mapping::SpatialTransferTerminalAttr>(subject);
  case Projection::MemoryOperationPort:
    return isa<mapping::ActorRefAttr>(subject);
  case Projection::MemoryBoundServices:
  case Projection::MemoryAddressRegion:
    return isa<mapping::LogicalMemoryRootRefAttr>(subject);
  }
  llvm_unreachable("unknown Spatial constraint projection");
}

bool isSpatialConstraintDomainValue(
    mapping::SpatialConstraintProjection projection, Attribute value) {
  using Projection = mapping::SpatialConstraintProjection;
  switch (projection) {
  case Projection::ComputePlacement:
    return isa<mapping::FabricFuOccurrenceRefAttr>(value);
  case Projection::ComputeParentPe:
    return isa<mapping::FabricPeOccurrenceRefAttr>(value);
  case Projection::ComputeInstructionContext:
    return isa<mapping::InstructionContextRefAttr>(value);
  case Projection::ComputeFuContext:
    return isa<mapping::ConstraintFuContextAttr>(value);
  case Projection::MemoryPlacement:
    return isa<mapping::FabricMemoryOccurrenceRefAttr>(value);
  case Projection::NetAssignedTagValues:
    return isa<mapping::ConstraintUnsignedIntervalAttr>(value);
  case Projection::NetSelectedPhysicalTraversals:
    return isa<mapping::FabricPhysicalTraversalRefAttr>(value);
  case Projection::NetTraversalResourceStates:
    return isa<mapping::FabricResourceStateRefAttr>(value);
  case Projection::SpatialTransferAttachment:
    return isa<mapping::FabricTransportEndpointRefAttr>(value);
  case Projection::MemoryOperationPort:
    return isa<mapping::FabricMemoryOperationPortRefAttr>(value);
  case Projection::MemoryBoundServices:
    return isa<mapping::FabricMemoryServiceRefAttr>(value);
  case Projection::MemoryAddressRegion:
    return isa<mapping::ConstraintAddressRegionAttr>(value);
  }
  llvm_unreachable("unknown Spatial constraint projection");
}

bool isSystemConstraintSubject(mapping::SystemConstraintProjection projection,
                               Attribute subject) {
  using Projection = mapping::SystemConstraintProjection;
  switch (projection) {
  case Projection::ThreadTargetAccCore:
    return isa<mapping::RootThreadLaunchRefAttr>(subject);
  case Projection::GraphSelectedSpatialMapping:
  case Projection::GraphTargetSpatialCore:
    return isa<mapping::RootedGraphLaunchRefAttr>(subject);
  case Projection::ServiceTargetRegion:
    return isa<mapping::SystemServiceObligationKeyAttr>(subject);
  case Projection::TransferTerminalAttachment:
    return isa<mapping::SystemTransferTerminalKeyAttr>(subject);
  case Projection::TransferSelectedTraversals:
  case Projection::TransferResourceStates:
  case Projection::TransferAssignedTagValues:
    return isa<mapping::CanonicalServiceLegKeyAttr>(subject);
  }
  llvm_unreachable("unknown System constraint projection");
}

bool isSystemConstraintDomainValue(
    mapping::SystemConstraintProjection projection, Attribute value) {
  using Projection = mapping::SystemConstraintProjection;
  switch (projection) {
  case Projection::ThreadTargetAccCore:
    return isa<mapping::FabricAccCoreOccurrenceRefAttr>(value);
  case Projection::GraphSelectedSpatialMapping:
    return isa<mapping::ConstraintSpatialMappingReferenceAttr>(value);
  case Projection::GraphTargetSpatialCore:
    return isa<mapping::FabricSpatialCoreOccurrenceRefAttr>(value);
  case Projection::ServiceTargetRegion:
    return isa<mapping::FabricMemoryServiceRegionRefAttr>(value);
  case Projection::TransferTerminalAttachment:
    return isa<mapping::FabricTransportEndpointRefAttr>(value);
  case Projection::TransferSelectedTraversals:
    return isa<mapping::FabricPhysicalTraversalRefAttr>(value);
  case Projection::TransferResourceStates:
    return isa<mapping::FabricResourceStateRefAttr>(value);
  case Projection::TransferAssignedTagValues:
    return isa<mapping::ConstraintUnsignedIntervalAttr>(value);
  }
  llvm_unreachable("unknown System constraint projection");
}

LogicalResult verifyConstraintProjectionParent(Operation *operation,
                                               Attribute projection) {
  Operation *parent = operation->getParentOp();
  if (isa<mapping::SpatialConstraintProjectionKeyAttr>(projection)) {
    if (!isa_and_nonnull<mapping::ConstraintsSpatialOp>(parent))
      return operation->emitOpError(
          "Spatial projection requires a Spatial MappingConstraintSet root");
    return success();
  }
  if (!isa<mapping::SystemConstraintProjectionKeyAttr>(projection))
    return operation->emitOpError("has an unknown projection attribute");
  if (!isa_and_nonnull<mapping::ConstraintsSystemOp>(parent))
    return operation->emitOpError(
        "System projection requires a System MappingConstraintSet root");
  return success();
}

bool isConstraintSubject(Attribute projection, Attribute subject) {
  if (auto spatial =
          dyn_cast<mapping::SpatialConstraintProjectionKeyAttr>(projection))
    return isSpatialConstraintSubject(
        static_cast<mapping::SpatialConstraintProjection>(spatial.getValue()),
        subject);
  return isSystemConstraintSubject(
      static_cast<mapping::SystemConstraintProjection>(
          cast<mapping::SystemConstraintProjectionKeyAttr>(projection)
              .getValue()),
      subject);
}

bool isConstraintDomainValue(Attribute projection, Attribute value) {
  if (auto spatial =
          dyn_cast<mapping::SpatialConstraintProjectionKeyAttr>(projection))
    return isSpatialConstraintDomainValue(
        static_cast<mapping::SpatialConstraintProjection>(spatial.getValue()),
        value);
  return isSystemConstraintDomainValue(
      static_cast<mapping::SystemConstraintProjection>(
          cast<mapping::SystemConstraintProjectionKeyAttr>(projection)
              .getValue()),
      value);
}

StringRef constraintProjectionName(Attribute projection) {
  if (auto spatial =
          dyn_cast<mapping::SpatialConstraintProjectionKeyAttr>(projection))
    return mapping::stringifySpatialConstraintProjection(
        static_cast<mapping::SpatialConstraintProjection>(spatial.getValue()));
  return mapping::stringifySystemConstraintProjection(
      static_cast<mapping::SystemConstraintProjection>(
          cast<mapping::SystemConstraintProjectionKeyAttr>(projection)
              .getValue()));
}

LogicalResult verifyConstraintSubjects(Operation *operation,
                                       Attribute projection,
                                       ArrayAttr subjects) {
  if (failed(verifyConstraintProjectionParent(operation, projection)))
    return failure();
  if (subjects.size() < 2)
    return operation->emitOpError("requires at least two subjects");
  for (Attribute subject : subjects)
    if (!isConstraintSubject(projection, subject))
      return operation->emitOpError(
          "contains a subject of the wrong typed projection domain");
  return success();
}

} // namespace

ParseResult
mapping::ConstraintDomainRestrictionOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  Attribute subject;
  ArrayAttr domain;
  if (failed(parseConstraintProjection(parser, result)) ||
      parser.parseKeyword("subject") || parser.parseLParen() ||
      parser.parseAttribute(subject, "subject", result.attributes) ||
      parser.parseRParen() || parser.parseKeyword("admissible_domain") ||
      parser.parseLParen() ||
      parser.parseAttribute(domain, "admissible_domain", result.attributes) ||
      parser.parseRParen() || parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void mapping::ConstraintDomainRestrictionOp::print(OpAsmPrinter &printer) {
  printConstraintProjection(printer, getProjection());
  printer << " subject(" << getSubject() << ") admissible_domain("
          << getAdmissibleDomain() << ')';
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                {"projection", "subject", "admissible_domain"});
}

ParseResult mapping::ConstraintEqualOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  ArrayAttr subjects;
  if (failed(parseConstraintProjection(parser, result)) ||
      parser.parseKeyword("subjects") || parser.parseLParen() ||
      parser.parseAttribute(subjects, "subjects", result.attributes) ||
      parser.parseRParen() || parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void mapping::ConstraintEqualOp::print(OpAsmPrinter &printer) {
  printConstraintProjection(printer, getProjection());
  printer << " subjects(" << getSubjects() << ')';
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                {"projection", "subjects"});
}

ParseResult mapping::ConstraintDisjointOp::parse(OpAsmParser &parser,
                                                 OperationState &result) {
  ArrayAttr subjects;
  if (failed(parseConstraintProjection(parser, result)) ||
      parser.parseKeyword("subjects") || parser.parseLParen() ||
      parser.parseAttribute(subjects, "subjects", result.attributes) ||
      parser.parseRParen() || parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void mapping::ConstraintDisjointOp::print(OpAsmPrinter &printer) {
  printConstraintProjection(printer, getProjection());
  printer << " subjects(" << getSubjects() << ')';
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                {"projection", "subjects"});
}

LogicalResult mapping::ConstraintDomainRestrictionOp::verify() {
  if (failed(rejectUnknownAttributes(
          *this, {"projection", "subject", "admissible_domain"})))
    return failure();
  if (failed(verifyConstraintProjectionParent(*this, getProjection())))
    return failure();
  if (!isConstraintSubject(getProjection(), getSubject()))
    return emitOpError("subject has the wrong typed projection domain");
  for (Attribute value : getAdmissibleDomain())
    if (!isConstraintDomainValue(getProjection(), value))
      return emitOpError(
                 "admissible_domain contains a value of the wrong carrier "
                 "type for projection ")
             << constraintProjectionName(getProjection()) << ": " << value;
  return success();
}

LogicalResult mapping::ConstraintEqualOp::verify() {
  if (failed(rejectUnknownAttributes(*this, {"projection", "subjects"})))
    return failure();
  return verifyConstraintSubjects(*this, getProjection(), getSubjects());
}

LogicalResult mapping::ConstraintDisjointOp::verify() {
  if (failed(rejectUnknownAttributes(*this, {"projection", "subjects"})))
    return failure();
  return verifyConstraintSubjects(*this, getProjection(), getSubjects());
}

#define GET_OP_CLASSES
#include "Mapping/IR/MappingOps.cpp.inc"
