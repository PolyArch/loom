#include "Hardware/RTL/Transport.h"

#include "Fabric/Identity/FabricRefImport.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::hardware::rtl {
namespace {

llvm::Error transportError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "rtl_transport_invalid: " + message);
}

llvm::Error validateSignal(llvm::StringRef name, std::uint32_t width,
                           const std::optional<mlir::Value> &signal) {
  if (width == 0)
    return signal ? transportError(name + " must be absent at width zero")
                  : llvm::Error::success();
  if (!signal || !*signal)
    return transportError(name + " is absent");
  const auto integer = mlir::dyn_cast<mlir::IntegerType>(signal->getType());
  if (!integer || !integer.isSignless() || integer.getWidth() != width)
    return transportError(name + " has the wrong signless integer width");
  return llvm::Error::success();
}

llvm::Error validateCirctCapacity(llvm::StringRef endpoint,
                                  ::fabric::DataPathType type) {
  if (type.payloadWidthBits > mlir::IntegerType::kMaxWidth)
    return transportError(endpoint + " payload width exceeds CIRCT capacity");
  if (type.kind == ::fabric::DataPathKind::BitsTag &&
      type.tagWidthBits > mlir::IntegerType::kMaxWidth)
    return transportError(endpoint + " tag width exceeds CIRCT capacity");
  return llvm::Error::success();
}

llvm::Expected<std::optional<mlir::Value>>
adaptSignal(mlir::OpBuilder &builder, mlir::Location location,
            std::uint32_t sourceWidth, std::uint32_t destinationWidth,
            std::optional<mlir::Value> source) {
  if (destinationWidth == 0)
    return std::optional<mlir::Value>{};
  if (sourceWidth == 0)
    return std::optional<mlir::Value>{circt::hw::ConstantOp::create(
        builder, location, llvm::APInt(destinationWidth, 0))};
  if (sourceWidth == destinationWidth)
    return source;
  if (sourceWidth > destinationWidth)
    return std::optional<mlir::Value>{circt::comb::ExtractOp::create(
        builder, location, *source, 0, destinationWidth)};

  mlir::Value highZeros = circt::hw::ConstantOp::create(
      builder, location, llvm::APInt(destinationWidth - sourceWidth, 0));
  llvm::SmallVector<mlir::Value, 2> pieces{highZeros, *source};
  return std::optional<mlir::Value>{
      circt::comb::ConcatOp::create(builder, location, pieces)};
}

struct ValidatedBoundary final {
  loom::fabric::FabricModuleBoundaryEndpointRef reference;
  ::fabric::DataPathType dataPath;
};

circt::hw::PortInfo makePort(mlir::OpBuilder &builder, llvm::StringRef name,
                             mlir::Type type,
                             circt::hw::ModulePort::Direction direction) {
  return circt::hw::PortInfo{{builder.getStringAttr(name), type, direction}};
}

std::string
boundaryPortName(const loom::fabric::FabricModuleBoundaryEndpointRef &endpoint,
                 llvm::StringRef signal) {
  const llvm::StringRef direction =
      endpoint.direction == loom::fabric::FabricPortDirection::Input ? "input"
                                                                     : "output";
  return (llvm::Twine(direction) + "_" + llvm::Twine(endpoint.ordinal) + "_" +
          signal)
      .str();
}

} // namespace

llvm::Expected<std::vector<ModuleBoundaryTransportPortProjection>>
deriveModuleBoundaryTransportPorts(
    mlir::OpBuilder &builder,
    const loom::fabric::FabricArtifactView &artifact) {
  const auto module = artifact.moduleRootTemplate();
  if (!module)
    return transportError(
        "Module boundary projection requires exactly one Module root");

  std::vector<ValidatedBoundary> boundaries;
  for (loom::fabric::FabricPortDirection direction :
       {loom::fabric::FabricPortDirection::Input,
        loom::fabric::FabricPortDirection::Output}) {
    const std::uint64_t count =
        artifact.moduleBoundaryEndpointCount(*module, direction);
    for (std::uint64_t ordinal = 0; ordinal < count; ++ordinal) {
      const loom::fabric::FabricModuleBoundaryEndpointRef reference{
          *module, direction, ordinal};
      const auto plane = artifact.moduleBoundaryEndpointPlane(reference);
      if (!plane)
        return transportError("Module boundary endpoint is invalid");
      if (*plane ==
          loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Memory)
        continue;
      if (*plane !=
          loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Transport)
        return transportError("Module boundary endpoint has an unknown plane");
      const auto dataPath = artifact.moduleBoundaryEndpointDataPath(reference);
      if (!dataPath || !dataPath->isWellFormed())
        return transportError(
            "Module token boundary has no canonical data path");
      if (dataPath->payloadWidthBits > mlir::IntegerType::kMaxWidth ||
          dataPath->tagWidthBits > mlir::IntegerType::kMaxWidth)
        return transportError(
            "Module boundary integer bitwidth exceeds CIRCT capacity");
      boundaries.push_back({reference, *dataPath});
    }
  }

  std::vector<ModuleBoundaryTransportPortProjection> projections;
  projections.reserve(boundaries.size());
  for (const ValidatedBoundary &boundary : boundaries) {
    const bool isInput = boundary.reference.direction ==
                         loom::fabric::FabricPortDirection::Input;
    const auto forwardDirection =
        isInput ? circt::hw::ModulePort::Direction::Input
                : circt::hw::ModulePort::Direction::Output;
    const auto readyDirection = isInput
                                    ? circt::hw::ModulePort::Direction::Output
                                    : circt::hw::ModulePort::Direction::Input;
    const mlir::Type bit = builder.getI1Type();

    std::optional<circt::hw::PortInfo> data;
    if (boundary.dataPath.payloadWidthBits != 0)
      data =
          makePort(builder, boundaryPortName(boundary.reference, "data"),
                   builder.getIntegerType(boundary.dataPath.payloadWidthBits),
                   forwardDirection);

    std::optional<circt::hw::PortInfo> tag;
    if (boundary.dataPath.kind == ::fabric::DataPathKind::BitsTag)
      tag = makePort(builder, boundaryPortName(boundary.reference, "tag"),
                     builder.getIntegerType(boundary.dataPath.tagWidthBits),
                     forwardDirection);

    projections.push_back(ModuleBoundaryTransportPortProjection{
        boundary.reference, std::move(data), std::move(tag),
        makePort(builder, boundaryPortName(boundary.reference, "valid"), bit,
                 forwardDirection),
        makePort(builder, boundaryPortName(boundary.reference, "ready"), bit,
                 readyDirection)});
  }
  return projections;
}

llvm::Expected<ForwardTransportSignals>
adaptForwardTransportSignals(mlir::OpBuilder &builder, mlir::Location location,
                             ::fabric::DataPathType sourceType,
                             ::fabric::DataPathType destinationType,
                             ForwardTransportSignals sourceSignals) {
  if (!sourceType.isWellFormed())
    return transportError("source type is malformed");
  if (!destinationType.isWellFormed())
    return transportError("destination type is malformed");
  if (sourceType.kind != destinationType.kind)
    return transportError("cannot adapt different Fabric transport kinds");
  if (llvm::Error error = validateCirctCapacity("source", sourceType))
    return std::move(error);
  if (llvm::Error error = validateCirctCapacity("destination", destinationType))
    return std::move(error);

  if (llvm::Error error =
          validateSignal("source valid signal", 1,
                         std::optional<mlir::Value>{sourceSignals.valid}))
    return std::move(error);
  if (llvm::Error error =
          validateSignal("source payload signal", sourceType.payloadWidthBits,
                         sourceSignals.payload))
    return std::move(error);
  const std::uint32_t sourceTagWidth =
      sourceType.kind == ::fabric::DataPathKind::BitsTag
          ? sourceType.tagWidthBits
          : 0;
  if (llvm::Error error = validateSignal("source tag signal", sourceTagWidth,
                                         sourceSignals.tag))
    return std::move(error);

  auto payload =
      adaptSignal(builder, location, sourceType.payloadWidthBits,
                  destinationType.payloadWidthBits, sourceSignals.payload);
  if (!payload)
    return payload.takeError();
  const std::uint32_t destinationTagWidth =
      destinationType.kind == ::fabric::DataPathKind::BitsTag
          ? destinationType.tagWidthBits
          : 0;
  auto tag = adaptSignal(builder, location, sourceTagWidth, destinationTagWidth,
                         sourceSignals.tag);
  if (!tag)
    return tag.takeError();
  return ForwardTransportSignals{sourceSignals.valid, std::move(*payload),
                                 std::move(*tag)};
}

llvm::Expected<ForwardTransportSignals>
adaptFabricPointConnectionForwardSignals(
    mlir::OpBuilder &builder, mlir::Location location,
    const loom::fabric::FabricArtifactView &artifact,
    const loom::fabric::FabricPointConnectionPayload &connection,
    ForwardTransportSignals sourceSignals) {
  if (!artifact.hasPointConnection(connection.source, connection.destination))
    return transportError("point connection is absent from the exact Fabric");
  const auto sourceType = artifact.transportEndpointDataPath(connection.source);
  const auto destinationType =
      artifact.transportEndpointDataPath(connection.destination);
  if (!sourceType || !destinationType)
    return transportError(
        "point connection endpoint has no canonical transport type");
  return adaptForwardTransportSignals(builder, location, *sourceType,
                                      *destinationType,
                                      std::move(sourceSignals));
}

} // namespace loom::hardware::rtl
