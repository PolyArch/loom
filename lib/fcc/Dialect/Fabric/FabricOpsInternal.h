//===- FabricOpsInternal.h - Shared helpers for FabricOps impl --*- C++ -*-===//
//
// Internal header for FabricOps implementation files.  Declares helper
// functions and templates that are used across multiple translation units.
//
//===----------------------------------------------------------------------===//

#ifndef FCC_DIALECT_FABRIC_FABRICOPSINTERNAL_H
#define FCC_DIALECT_FABRIC_FABRICOPSINTERNAL_H

#include "fcc/Dialect/Fabric/FabricOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

namespace fcc {
namespace fabric {
namespace detail {

/// Maximum number of ports allowed on a switch operation.
constexpr unsigned kMaxSwitchPorts = 32;

/// Return the scalar bit-width of a fabric-level type, or std::nullopt
/// for unsupported types.
std::optional<unsigned> getFabricScalarWidth(mlir::Type type);

/// Return the payload width of a type used on spatial switch ports.
/// For TaggedType, extracts the value type first.
std::optional<unsigned> getSpatialSwitchPayloadWidth(mlir::Type type);

/// Verify that an ArrayAttr encoding a binary row table has the expected
/// dimensions and only contains '0'/'1' characters.
mlir::LogicalResult verifyBinaryRowTable(mlir::ArrayAttr tableAttr,
                                         unsigned expectedRows,
                                         unsigned expectedCols,
                                         mlir::Operation *op,
                                         llvm::StringRef name);

/// Return the operation that directly contains the region holding \p op.
mlir::Operation *getDirectRegionParent(mlir::Operation *op);

/// Return true if the direct region parent of \p op is one of the
/// listed operation types.
template <typename... OpTys>
bool hasDirectRegionParentOfType(mlir::Operation *op) {
  if (mlir::Operation *parent = getDirectRegionParent(op))
    return mlir::isa<OpTys...>(parent);
  return false;
}

/// Parse an optional parenthesized operand list.  Sets \p hasOperands
/// to true when the leading '(' is present.
mlir::ParseResult parseOptionalOperandListInParens(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &ops,
    bool &hasOperands);

/// Verify that the operation is placed at the module level (inside
/// mlir::ModuleOp or fabric::ModuleOp) and that inline_instantiation is
/// only used inside fabric::ModuleOp.
template <typename OpTy>
mlir::LogicalResult verifyModuleLevelComponentPlacement(OpTy op) {
  mlir::Operation *parent = op->getParentOp();
  if (!mlir::isa_and_nonnull<mlir::ModuleOp, fcc::fabric::ModuleOp>(parent)) {
    return op.emitOpError(
        "must appear directly inside the top-level module or fabric.module");
  }

  if (op->hasAttr("inline_instantiation") &&
      !mlir::isa<fcc::fabric::ModuleOp>(parent)) {
    return op.emitOpError(
        "inline instantiation must appear directly inside fabric.module");
  }

  return mlir::success();
}

/// Normalize legacy "addr_offset_table" attribute to "addrOffsetTable".
mlir::ParseResult normalizeMemoryConfigAttrs(mlir::OpAsmParser &parser,
                                             mlir::OperationState &result);

/// Print named attributes excluding the given set, aliasing
/// "addrOffsetTable" back to "addr_offset_table".
void printNamedAttrsWithAliases(mlir::OpAsmPrinter &p, mlir::Operation *op,
                                mlir::ArrayRef<mlir::StringRef> excludes);

} // namespace detail
} // namespace fabric
} // namespace fcc

#endif // FCC_DIALECT_FABRIC_FABRICOPSINTERNAL_H
