//===-- FabricTypeUtils.h - Shared type helpers for Fabric verifiers -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Utility functions shared across Fabric dialect verifier files (SwitchOps,
// FifoOps, TopOps, PEOps, TagOps). These helpers handle type compatibility
// checks for routing nodes, bits-type validation, and sparse format
// verification.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_DIALECT_FABRIC_FABRICTYPEUTILS_H
#define LOOM_DIALECT_FABRIC_FABRICTYPEUTILS_H

#include "loom/Dialect/Dataflow/DataflowTypes.h"
#include "loom/Hardware/Common/FabricConstants.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include <optional>

namespace loom {
namespace fabric {

/// Get the bit width of a type for routing compatibility checks.
/// Returns std::nullopt for types without a well-defined bit width (e.g. none).
inline std::optional<unsigned> getNativeBitWidth(mlir::Type t) {
  if (auto bitsType = mlir::dyn_cast<dataflow::BitsType>(t))
    return bitsType.getWidth();
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(t))
    return intTy.getWidth();
  if (mlir::isa<mlir::Float16Type, mlir::BFloat16Type>(t))
    return 16u;
  if (mlir::isa<mlir::Float32Type>(t))
    return 32u;
  if (mlir::isa<mlir::Float64Type>(t))
    return 64u;
  if (t.isIndex())
    return static_cast<unsigned>(loom::ADDR_BIT_WIDTH);
  return std::nullopt;
}

/// Check whether two types are compatible for routing through pass-through
/// nodes (switch, temporal_sw, fifo). Rules:
///   - Exact match: always compatible.
///   - Both native with known bit width: compatible if widths match.
///   - Both tagged: compatible if value bit widths AND tag bit widths match.
///   - One native, one tagged (category mismatch): never compatible.
inline bool isRoutingTypeCompatible(mlir::Type a, mlir::Type b) {
  if (a == b)
    return true;

  bool isTaggedA = mlir::isa<dataflow::TaggedType>(a);
  bool isTaggedB = mlir::isa<dataflow::TaggedType>(b);

  if (isTaggedA != isTaggedB)
    return false;

  if (isTaggedA) {
    auto tagA = mlir::cast<dataflow::TaggedType>(a);
    auto tagB = mlir::cast<dataflow::TaggedType>(b);
    auto valWidthA = getNativeBitWidth(tagA.getValueType());
    auto valWidthB = getNativeBitWidth(tagB.getValueType());
    if (!valWidthA || !valWidthB)
      return false;
    return *valWidthA == *valWidthB &&
           tagA.getTagType().getWidth() == tagB.getTagType().getWidth();
  }

  auto widthA = getNativeBitWidth(a);
  auto widthB = getNativeBitWidth(b);
  if (!widthA || !widthB)
    return false;
  return *widthA == *widthB;
}

/// Verify that all port types on a routing node are bit-width compatible.
/// Accepts TypeRange for both named forms (function_type attribute) and inline
/// forms (SSA operands/results).
inline mlir::LogicalResult
verifyRoutingCompatibleTypes(mlir::Operation *op, mlir::TypeRange inputTypes,
                             mlir::TypeRange outputTypes) {
  llvm::SmallVector<mlir::Type> allTypes;
  for (mlir::Type t : inputTypes)
    allTypes.push_back(t);
  for (mlir::Type t : outputTypes)
    allTypes.push_back(t);
  if (allTypes.empty())
    return mlir::success();
  mlir::Type first = allTypes.front();
  for (mlir::Type t : allTypes) {
    if (!isRoutingTypeCompatible(first, t))
      return op->emitOpError(
                 "all ports must have bit-width-compatible types; got ")
             << first << " and " << t;
  }
  return mlir::success();
}

/// Overload accepting OperandRange and ResultRange for inline forms.
inline mlir::LogicalResult
verifyRoutingCompatibleTypes(mlir::Operation *op, mlir::OperandRange inputs,
                             mlir::ResultRange outputs) {
  llvm::SmallVector<mlir::Type> inputTypes, outputTypes;
  for (auto v : inputs)
    inputTypes.push_back(v.getType());
  for (auto v : outputs)
    outputTypes.push_back(v.getType());
  return verifyRoutingCompatibleTypes(op, mlir::TypeRange(inputTypes),
                                      mlir::TypeRange(outputTypes));
}

/// Verify sparse format rules for human-readable config entries.
/// Checks: mixed format, ascending slot order, implicit hole consistency.
inline mlir::LogicalResult
verifySparseFormat(mlir::Operation *op, mlir::ArrayAttr entries,
                   llvm::StringRef prefix, llvm::StringRef mixedCode,
                   llvm::StringRef orderCode, llvm::StringRef holeCode) {
  bool hasHex = false, hasHuman = false, hasExplicitInvalid = false;
  llvm::SmallVector<int64_t> slotIndices;

  for (auto entry : entries) {
    auto str = mlir::dyn_cast<mlir::StringAttr>(entry);
    if (!str)
      continue;
    llvm::StringRef s = str.getValue();
    if (s.starts_with("0x") || s.starts_with("0X")) {
      hasHex = true;
    } else {
      hasHuman = true;
      if (s.contains("invalid"))
        hasExplicitInvalid = true;
      size_t lb = s.find('['), rb = s.find(']');
      if (lb != llvm::StringRef::npos && rb != llvm::StringRef::npos &&
          rb > lb) {
        unsigned idx;
        if (!s.substr(lb + 1, rb - lb - 1).getAsInteger(10, idx))
          slotIndices.push_back(idx);
      }
    }
  }

  if (hasHex && hasHuman)
    return op->emitOpError(mixedCode)
           << " " << prefix
           << " entries mix human-readable and hex formats";

  for (unsigned i = 1; i < slotIndices.size(); ++i) {
    if (slotIndices[i] <= slotIndices[i - 1])
      return op->emitOpError(orderCode)
             << " " << prefix
             << " slot indices must be strictly ascending; got "
             << slotIndices[i - 1] << " followed by " << slotIndices[i];
  }

  if (hasExplicitInvalid) {
    for (unsigned i = 1; i < slotIndices.size(); ++i) {
      if (slotIndices[i] != slotIndices[i - 1] + 1)
        return op->emitOpError(holeCode)
               << " " << prefix << " has implicit hole between slot "
               << slotIndices[i - 1] << " and " << slotIndices[i]
               << "; all holes must be explicit when explicit invalid"
                  " entries exist";
    }
  }
  return mlir::success();
}

} // namespace fabric
} // namespace loom

#endif // LOOM_DIALECT_FABRIC_FABRICTYPEUTILS_H
