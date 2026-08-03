#include "Mapping/IR/MappingOps.h"

#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/DenseSet.h"

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

} // namespace

ParseResult mapping::TechOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  std::uint32_t major = 0;
  std::uint32_t minor = 0;
  if (parser.parseKeyword("version") || parser.parseLess() ||
      parser.parseInteger(major) || parser.parseComma() ||
      parser.parseInteger(minor) || parser.parseGreater())
    return failure();
  if (major != 2 || minor != 0)
    return parser.emitError(parser.getCurrentLocation(),
                            "mapping.tech requires schema version 2.0");

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
  printer << " version<2, 0> dataflow(" << getDataflow() << ") fabric("
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
  if (major != 2 || minor != 0)
    return parser.emitError(parser.getCurrentLocation(),
                            "mapping.spatial requires schema version 2.0");

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
  printer << " version<2, 0> tech_mapping(" << getTechMapping() << ") dataflow("
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
  llvm::DenseSet<Attribute> resourceUses;
  llvm::SmallVector<mapping::ResourceUseOp, 8> deferredUses;
  for (Operation &child : getBody().front()) {
    if (auto binding = dyn_cast<mapping::ComputeBindingOp>(child)) {
      if (!computeBindings.insert(binding.getRealization().getEntity()).second)
        return binding.emitOpError("duplicates a ComputeBinding key");
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

  for (mapping::ResourceUseOp use : deferredUses) {
    auto compute = dyn_cast<mapping::ComputeRealizationRefAttr>(use.getOwner());
    if (!compute)
      return use.emitOpError(
          "references a Spatial owner family not present in this root");
    if (!computeBindings.contains(compute.getEntity()))
      return use.emitOpError("references an absent ComputeBinding owner");
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

LogicalResult mapping::ResourceUseOp::verify() {
  if (failed(rejectUnknownAttributes(*this,
                                     {"owner", "use_site", "activation",
                                      "parameters", "sharing_assignments"})))
    return failure();
  if (failed(verifyOwnerTypedValues(*this, getParameters(), "parameters")) ||
      failed(verifyOwnerTypedValues(*this, getSharingAssignments(), "sharing")))
    return failure();
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

namespace {

ParseResult parseSpatialProjection(OpAsmParser &parser,
                                   OperationState &result) {
  StringRef keyword;
  if (parser.parseKeyword("projection") || parser.parseLParen() ||
      parser.parseKeyword(&keyword) || parser.parseRParen())
    return failure();
  std::optional<mapping::SpatialConstraintProjection> projection =
      mapping::symbolizeSpatialConstraintProjection(keyword);
  if (!projection)
    return parser.emitError(parser.getCurrentLocation(),
                            "unknown Spatial constraint projection");
  result.addAttribute("projection",
                      mapping::SpatialConstraintProjectionAttr::get(
                          parser.getContext(), *projection));
  return success();
}

void printSpatialProjection(OpAsmPrinter &printer,
                            mapping::SpatialConstraintProjection projection) {
  printer << " projection("
          << mapping::stringifySpatialConstraintProjection(projection) << ')';
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

LogicalResult
verifyConstraintSubjects(Operation *operation,
                         mapping::SpatialConstraintProjection projection,
                         ArrayAttr subjects) {
  if (subjects.size() < 2)
    return operation->emitOpError("requires at least two subjects");
  for (Attribute subject : subjects)
    if (!isSpatialConstraintSubject(projection, subject))
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
  if (failed(parseSpatialProjection(parser, result)) ||
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
  printSpatialProjection(printer, getProjection());
  printer << " subject(" << getSubject() << ") admissible_domain("
          << getAdmissibleDomain() << ')';
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                {"projection", "subject", "admissible_domain"});
}

ParseResult mapping::ConstraintEqualOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  ArrayAttr subjects;
  if (failed(parseSpatialProjection(parser, result)) ||
      parser.parseKeyword("subjects") || parser.parseLParen() ||
      parser.parseAttribute(subjects, "subjects", result.attributes) ||
      parser.parseRParen() || parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void mapping::ConstraintEqualOp::print(OpAsmPrinter &printer) {
  printSpatialProjection(printer, getProjection());
  printer << " subjects(" << getSubjects() << ')';
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                {"projection", "subjects"});
}

ParseResult mapping::ConstraintDisjointOp::parse(OpAsmParser &parser,
                                                 OperationState &result) {
  ArrayAttr subjects;
  if (failed(parseSpatialProjection(parser, result)) ||
      parser.parseKeyword("subjects") || parser.parseLParen() ||
      parser.parseAttribute(subjects, "subjects", result.attributes) ||
      parser.parseRParen() || parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void mapping::ConstraintDisjointOp::print(OpAsmPrinter &printer) {
  printSpatialProjection(printer, getProjection());
  printer << " subjects(" << getSubjects() << ')';
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                {"projection", "subjects"});
}

LogicalResult mapping::ConstraintDomainRestrictionOp::verify() {
  if (failed(rejectUnknownAttributes(
          *this, {"projection", "subject", "admissible_domain"})))
    return failure();
  if (!isSpatialConstraintSubject(getProjection(), getSubject()))
    return emitOpError("subject has the wrong typed projection domain");
  for (Attribute value : getAdmissibleDomain())
    if (!isSpatialConstraintDomainValue(getProjection(), value))
      return emitOpError(
                 "admissible_domain contains a value of the wrong carrier "
                 "type for projection ")
             << mapping::stringifySpatialConstraintProjection(getProjection())
             << ": " << value;
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
