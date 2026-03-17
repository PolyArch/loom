// Type conversion utilities for the LLVM-to-CF pass.
// Handles pointer-to-memref conversion, element type inference,
// and the PointerInfo representation.

#ifndef FCC_CONVERSION_LLVMTOCFTYPES_H
#define FCC_CONVERSION_LLVMTOCFTYPES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"

namespace fcc {

// Represents a converted LLVM pointer as a memref base + element offset.
// This decouples the memref structure from pointer semantics, enabling
// correct handling of GEP arithmetic, bitcasts, and type punning.
struct PointerInfo {
  mlir::Value base;       // memref<?xElementType, strided<[1], offset: ?>>
  mlir::Value index;      // index-typed element offset into base
  mlir::Type elementType; // element type of the memref

  bool isValid() const { return base != nullptr; }
};

// Build memref<?xElementT, strided<[1], offset: ?>> for a given element type.
mlir::MemRefType buildStridedMemRefType(mlir::MLIRContext *ctx,
                                        mlir::Type elementType);

// Normalize an LLVM scalar type to standard MLIR type.
// Handles: LLVM integer -> mlir integer, LLVM float -> mlir float,
// LLVM pointer -> index (for pointer-as-integer patterns).
mlir::Type normalizeScalarType(mlir::MLIRContext *ctx, mlir::Type llvmType);

// Flatten an LLVM aggregate element type to a memref-compatible scalar element.
// For LLVM array types, returns the leaf scalar element type and multiplies
// `elementCount` by the total number of contained scalar elements.
// Returns null on unsupported aggregates.
mlir::Type flattenAllocaElementType(mlir::MLIRContext *ctx,
                                    mlir::Type llvmType,
                                    uint64_t &elementCount);

// Get byte width of a type (for offset scaling in GEP conversion).
unsigned getTypeBitWidth(mlir::Type type);

// Infer the element type for each pointer-typed function argument
// by analyzing downstream uses (GEPs, loads, stores, calls).
// Returns a map from argument index to inferred element type.
// Falls back to i8 if no evidence found.
llvm::DenseMap<unsigned, mlir::Type>
inferPointerElementTypes(mlir::LLVM::LLVMFuncOp funcOp);

// Map LLVM ICmpPredicate to arith CmpIPredicate.
mlir::arith::CmpIPredicate
convertICmpPredicate(mlir::LLVM::ICmpPredicate pred);

// Map LLVM FCmpPredicate to arith CmpFPredicate.
mlir::arith::CmpFPredicate
convertFCmpPredicate(mlir::LLVM::FCmpPredicate pred);

} // namespace fcc

#endif
