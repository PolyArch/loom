#ifndef FCC_DIALECT_FABRIC_FABRICTYPES_H
#define FCC_DIALECT_FABRIC_FABRICTYPES_H

#include "fcc/Dialect/Fabric/FabricDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "fcc/Dialect/Fabric/FabricTypes.h.inc"

namespace fcc {
namespace fabric {

enum class IndexWidthPreset : unsigned {
  INDEX_WIDTH_32 = 32u,
  INDEX_WIDTH_48 = 48u,
  INDEX_WIDTH_57 = 57u,
  INDEX_WIDTH_64 = 64u,
};

constexpr unsigned MIN_INDEX_BIT_WIDTH = 32u;
constexpr unsigned MAX_INDEX_BIT_WIDTH = 64u;
constexpr IndexWidthPreset DEFAULT_INDEX_WIDTH =
    IndexWidthPreset::INDEX_WIDTH_32;

constexpr unsigned getDefaultIndexBitWidth() {
  return static_cast<unsigned>(DEFAULT_INDEX_WIDTH);
}

bool isSupportedIndexBitWidth(unsigned width);
unsigned getConfiguredIndexBitWidth();
mlir::IntegerType getIndexIntegerType(mlir::MLIRContext *ctx);

// Minimum network granularity
constexpr unsigned MIN_BITS_WIDTH = 8;

} // namespace fabric
} // namespace fcc

#endif
