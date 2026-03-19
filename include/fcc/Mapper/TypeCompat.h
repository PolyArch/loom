#ifndef FCC_MAPPER_TYPECOMPAT_H
#define FCC_MAPPER_TYPECOMPAT_H

#include "fcc/Dialect/Fabric/FabricTypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/MathExtras.h"

#include <optional>

namespace fcc {

namespace detail {

inline std::optional<unsigned> getScalarWidth(mlir::Type t) {
  if (auto bits = mlir::dyn_cast<fcc::fabric::BitsType>(t))
    return bits.getWidth();
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(t))
    return intTy.getWidth();
  if (mlir::isa<mlir::Float32Type>(t))
    return 32u;
  if (mlir::isa<mlir::Float64Type>(t))
    return 64u;
  if (mlir::isa<mlir::Float16Type, mlir::BFloat16Type>(t))
    return 16u;
  if (t.isIndex())
    return fcc::fabric::getConfiguredIndexBitWidth();
  if (mlir::isa<mlir::NoneType>(t))
    return 0u;
  return std::nullopt;
}

inline std::optional<unsigned> getTagWidth(mlir::Type t) {
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(t))
    return intTy.getWidth();
  return std::nullopt;
}

inline std::optional<unsigned> getMemRefElementWidth(mlir::Type t) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(t);
  if (!memref)
    return std::nullopt;
  return getScalarWidth(memref.getElementType());
}

inline std::optional<unsigned> getMemRefElementByteWidthLog2(mlir::Type t) {
  auto bits = getMemRefElementWidth(t);
  if (!bits || *bits == 0 || (*bits % 8) != 0)
    return std::nullopt;
  unsigned bytes = *bits / 8;
  if (!llvm::isPowerOf2_32(bytes))
    return std::nullopt;
  return llvm::Log2_32(bytes);
}

struct PortTypeInfo {
  bool isTagged = false;
  unsigned valueWidth = 0;
  unsigned tagWidth = 0;
};

inline std::optional<PortTypeInfo> getPortTypeInfo(mlir::Type t) {
  if (auto tagged = mlir::dyn_cast<fcc::fabric::TaggedType>(t)) {
    auto valueWidth = getScalarWidth(tagged.getValueType());
    auto tagWidth = getTagWidth(tagged.getTagType());
    if (!valueWidth || !tagWidth)
      return std::nullopt;
    return PortTypeInfo{true, *valueWidth, *tagWidth};
  }

  auto valueWidth = getScalarWidth(t);
  if (!valueWidth)
    return std::nullopt;
  return PortTypeInfo{false, *valueWidth, 0};
}

} // namespace detail

inline bool isHardwarePortCompatible(mlir::Type src, mlir::Type dst) {
  if (mlir::isa<mlir::MemRefType>(src) || mlir::isa<mlir::MemRefType>(dst)) {
    auto srcWidth = detail::getMemRefElementWidth(src);
    auto dstWidth = detail::getMemRefElementWidth(dst);
    return srcWidth && dstWidth;
  }
  auto srcInfo = detail::getPortTypeInfo(src);
  auto dstInfo = detail::getPortTypeInfo(dst);
  if (!srcInfo || !dstInfo)
    return false;
  if (srcInfo->isTagged != dstInfo->isTagged)
    return false;
  if (!srcInfo->isTagged)
    return true;
  return true;
}

inline bool canMapSoftwareTypeToHardware(mlir::Type swType, mlir::Type hwType) {
  if (swType == hwType)
    return true;

  if (mlir::isa<mlir::MemRefType>(swType) || mlir::isa<mlir::MemRefType>(hwType)) {
    auto swWidth = detail::getMemRefElementWidth(swType);
    auto hwWidth = detail::getMemRefElementWidth(hwType);
    if (!swWidth || !hwWidth)
      return false;
    return *swWidth <= *hwWidth;
  }

  auto swInfo = detail::getPortTypeInfo(swType);
  auto hwInfo = detail::getPortTypeInfo(hwType);
  if (!swInfo || !hwInfo)
    return false;
  if (swInfo->isTagged != hwInfo->isTagged)
    return false;
  if (swInfo->valueWidth > hwInfo->valueWidth)
    return false;
  if (swInfo->isTagged && swInfo->tagWidth > hwInfo->tagWidth)
    return false;
  return true;
}

/// Bridge boundaries for software memory/extmemory may appear on already-tagged
/// hardware ports. In that case the software type maps against the tagged
/// payload's value type, while the runtime tag value is supplied by the bridge
/// path rather than by the software op itself.
inline bool canMapSoftwareTypeToBridgeHardware(mlir::Type swType,
                                               mlir::Type hwType) {
  if (canMapSoftwareTypeToHardware(swType, hwType))
    return true;

  auto taggedHwType = mlir::dyn_cast<fcc::fabric::TaggedType>(hwType);
  if (!taggedHwType)
    return false;
  return canMapSoftwareTypeToHardware(swType, taggedHwType.getValueType());
}

} // namespace fcc

#endif // FCC_MAPPER_TYPECOMPAT_H
