#ifndef FCC_DIALECT_FABRIC_FABRICTYPES_H
#define FCC_DIALECT_FABRIC_FABRICTYPES_H

#include "fcc/Dialect/Fabric/FabricDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "fcc/Dialect/Fabric/FabricTypes.h.inc"

namespace fcc {
namespace fabric {

// Address bit width (same as loom: 57 bits for index type)
constexpr unsigned ADDR_BIT_WIDTH = 57;

// Minimum network granularity
constexpr unsigned MIN_BITS_WIDTH = 8;

} // namespace fabric
} // namespace fcc

#endif
