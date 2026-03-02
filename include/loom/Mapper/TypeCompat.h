//===-- TypeCompat.h - MLIR type width-compatibility helpers -----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Shared type-compatibility utilities used across the mapper pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_MAPPER_TYPECOMPAT_H
#define LOOM_MAPPER_TYPECOMPAT_H

#include "loom/Dialect/Dataflow/DataflowTypes.h"
#include "loom/Hardware/Common/FabricConstants.h"

#include "mlir/IR/BuiltinTypes.h"

#include <optional>

namespace loom {

/// Check if two MLIR types are width-compatible.
/// Allows matching native types (i32, f32) with bits<N> types of the same
/// width. IndexType maps to ADDR_BIT_WIDTH (57) for width comparison.
/// For tagged types, tag types must match AND value widths must match.
inline bool isTypeWidthCompatible(mlir::Type a, mlir::Type b) {
  if (a == b)
    return true;

  auto getWidth = [](mlir::Type t) -> std::optional<unsigned> {
    if (auto bits = mlir::dyn_cast<loom::dataflow::BitsType>(t))
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
      return static_cast<unsigned>(loom::ADDR_BIT_WIDTH);
    if (mlir::isa<mlir::NoneType>(t))
      return 0u;
    return std::nullopt;
  };

  auto tagA = mlir::dyn_cast<loom::dataflow::TaggedType>(a);
  auto tagB = mlir::dyn_cast<loom::dataflow::TaggedType>(b);
  if (tagA && tagB) {
    if (tagA.getTagType() != tagB.getTagType())
      return false;
    auto wA = getWidth(tagA.getValueType());
    auto wB = getWidth(tagB.getValueType());
    return wA && wB && *wA == *wB;
  }
  if (tagA || tagB)
    return false;

  auto wA = getWidth(a);
  auto wB = getWidth(b);
  return wA && wB && *wA == *wB;
}

} // namespace loom

#endif // LOOM_MAPPER_TYPECOMPAT_H
