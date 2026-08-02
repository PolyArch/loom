#include "Mapping/IR/MappingOps.h"

#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/DenseSet.h"

#include <cstdint>
#include <optional>
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
    auto realization = dyn_cast<mapping::ComputeRealizationOp>(child);
    if (!realization)
      return child.emitOpError("is not a closed TechMapping record kind");
    hasRealization = true;
    const std::uint64_t id = realization.getEntityId();
    if (!entityIds.insert(id).second)
      return realization.emitOpError("duplicates a Mapping EntityId");
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
  for (std::int32_t port : getOperandPorts())
    if (port < 0)
      return emitOpError("operand port ordinals must be nonnegative");
  for (std::int32_t port : getResultPorts())
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

#define GET_OP_CLASSES
#include "Mapping/IR/MappingOps.cpp.inc"
