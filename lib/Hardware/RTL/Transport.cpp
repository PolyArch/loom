#include "Hardware/RTL/Transport.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <optional>
#include <utility>

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

} // namespace

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

} // namespace loom::hardware::rtl
