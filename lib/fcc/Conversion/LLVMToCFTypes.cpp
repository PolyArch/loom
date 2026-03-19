// Type conversion utilities for LLVMToCF pass.

#include "LLVMToCFTypes.h"
#include "fcc/Dialect/Fabric/FabricTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace fcc {

MemRefType buildStridedMemRefType(MLIRContext *ctx, Type elementType) {
  auto layout = StridedLayoutAttr::get(ctx,
      /*offset=*/ShapedType::kDynamic,
      /*strides=*/{1});
  return MemRefType::get({ShapedType::kDynamic}, elementType, layout);
}

Type normalizeScalarType(MLIRContext *ctx, Type llvmType) {
  // LLVM integer types map directly
  if (auto intTy = dyn_cast<IntegerType>(llvmType))
    return intTy;
  // LLVM float types map directly
  if (isa<Float16Type, Float32Type, Float64Type, Float128Type,
          BFloat16Type>(llvmType))
    return llvmType;
  // LLVM pointer -> configured index-width integer for pointer-as-integer
  if (isa<LLVM::LLVMPointerType>(llvmType))
    return fcc::fabric::getIndexIntegerType(ctx);
  // Fallback
  return llvmType;
}

Type flattenAllocaElementType(MLIRContext *ctx, Type llvmType,
                              uint64_t &elementCount) {
  if (auto arrayTy = dyn_cast<LLVM::LLVMArrayType>(llvmType)) {
    elementCount *= arrayTy.getNumElements();
    return flattenAllocaElementType(ctx, arrayTy.getElementType(),
                                    elementCount);
  }

  Type scalarTy = normalizeScalarType(ctx, llvmType);
  if (isa<IntegerType, Float16Type, Float32Type, Float64Type, Float128Type,
          BFloat16Type>(scalarTy))
    return scalarTy;
  return nullptr;
}

unsigned getTypeBitWidth(Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return intTy.getWidth();
  if (isa<Float16Type, BFloat16Type>(type))
    return 16;
  if (isa<Float32Type>(type))
    return 32;
  if (isa<Float64Type>(type))
    return 64;
  if (isa<Float128Type>(type))
    return 128;
  if (isa<IndexType>(type))
    return fcc::fabric::getConfiguredIndexBitWidth();
  return 0;
}

// Trace a value through the function to find GEP uses that reveal
// the element type for a pointer argument.
static Type inferFromUses(Value ptrVal, unsigned depth = 0) {
  if (depth > 8)
    return nullptr;

  Type bestType = nullptr;

  for (auto &use : ptrVal.getUses()) {
    Operation *user = use.getOwner();

    // GEP reveals element type directly
    if (auto gep = dyn_cast<LLVM::GEPOp>(user)) {
      if (use.getOperandNumber() == 0) { // base operand
        Type elemTy = gep.getElemType();
        if (!isa<IntegerType>(elemTy) ||
            cast<IntegerType>(elemTy).getWidth() != 8) {
          // Prefer non-i8 types
          return elemTy;
        }
        if (!bestType)
          bestType = elemTy;
        // Also look at GEP result uses
        Type fromGepUses = inferFromUses(gep.getResult(), depth + 1);
        if (fromGepUses)
          return fromGepUses;
      }
    }

    // Load reveals access type
    if (auto load = dyn_cast<LLVM::LoadOp>(user)) {
      Type loadTy = normalizeScalarType(user->getContext(),
                                        load.getResult().getType());
      if (!isa<IntegerType>(loadTy) ||
          cast<IntegerType>(loadTy).getWidth() != 8)
        return loadTy;
      if (!bestType)
        bestType = loadTy;
    }

    // Store reveals value type
    if (auto store = dyn_cast<LLVM::StoreOp>(user)) {
      if (use.getOperandNumber() == 1) { // addr operand
        Type valTy = normalizeScalarType(user->getContext(),
                                         store.getValue().getType());
        if (!isa<IntegerType>(valTy) ||
            cast<IntegerType>(valTy).getWidth() != 8)
          return valTy;
        if (!bestType)
          bestType = valTy;
      }
    }

    // Branch: trace through block arguments
    if (auto br = dyn_cast<LLVM::BrOp>(user)) {
      unsigned idx = use.getOperandNumber();
      Block *dest = br.getDest();
      if (idx < dest->getNumArguments()) {
        Type fromDest = inferFromUses(dest->getArgument(idx), depth + 1);
        if (fromDest)
          return fromDest;
      }
    }
  }

  return bestType;
}

llvm::DenseMap<unsigned, Type>
inferPointerElementTypes(LLVM::LLVMFuncOp funcOp) {
  llvm::DenseMap<unsigned, Type> result;

  if (funcOp.isExternal())
    return result;

  Block &entry = funcOp.getBody().front();
  for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
    auto arg = entry.getArgument(i);
    if (!isa<LLVM::LLVMPointerType>(arg.getType()))
      continue;

    // First try to infer from GEP elem_type (most reliable)
    Type inferred = nullptr;
    funcOp.walk([&](LLVM::GEPOp gepOp) {
      if (gepOp.getBase() == arg) {
        Type elemTy = gepOp.getElemType();
        if (elemTy && (!inferred || (isa<IntegerType>(inferred) &&
                                     cast<IntegerType>(inferred).getWidth() == 8))) {
          inferred = elemTy;
        }
      }
    });

    if (!inferred) {
      // Fallback: trace uses
      inferred = inferFromUses(arg);
    }

    if (!inferred) {
      // Final fallback: i8
      inferred = IntegerType::get(funcOp.getContext(), 8);
    }

    result[i] = inferred;
  }

  return result;
}

arith::CmpIPredicate convertICmpPredicate(LLVM::ICmpPredicate pred) {
  switch (pred) {
  case LLVM::ICmpPredicate::eq:
    return arith::CmpIPredicate::eq;
  case LLVM::ICmpPredicate::ne:
    return arith::CmpIPredicate::ne;
  case LLVM::ICmpPredicate::slt:
    return arith::CmpIPredicate::slt;
  case LLVM::ICmpPredicate::sle:
    return arith::CmpIPredicate::sle;
  case LLVM::ICmpPredicate::sgt:
    return arith::CmpIPredicate::sgt;
  case LLVM::ICmpPredicate::sge:
    return arith::CmpIPredicate::sge;
  case LLVM::ICmpPredicate::ult:
    return arith::CmpIPredicate::ult;
  case LLVM::ICmpPredicate::ule:
    return arith::CmpIPredicate::ule;
  case LLVM::ICmpPredicate::ugt:
    return arith::CmpIPredicate::ugt;
  case LLVM::ICmpPredicate::uge:
    return arith::CmpIPredicate::uge;
  }
  llvm_unreachable("unhandled ICmpPredicate");
}

arith::CmpFPredicate convertFCmpPredicate(LLVM::FCmpPredicate pred) {
  switch (pred) {
  case LLVM::FCmpPredicate::_false:
    return arith::CmpFPredicate::AlwaysFalse;
  case LLVM::FCmpPredicate::oeq:
    return arith::CmpFPredicate::OEQ;
  case LLVM::FCmpPredicate::ogt:
    return arith::CmpFPredicate::OGT;
  case LLVM::FCmpPredicate::oge:
    return arith::CmpFPredicate::OGE;
  case LLVM::FCmpPredicate::olt:
    return arith::CmpFPredicate::OLT;
  case LLVM::FCmpPredicate::ole:
    return arith::CmpFPredicate::OLE;
  case LLVM::FCmpPredicate::one:
    return arith::CmpFPredicate::ONE;
  case LLVM::FCmpPredicate::ord:
    return arith::CmpFPredicate::ORD;
  case LLVM::FCmpPredicate::ueq:
    return arith::CmpFPredicate::UEQ;
  case LLVM::FCmpPredicate::ugt:
    return arith::CmpFPredicate::UGT;
  case LLVM::FCmpPredicate::uge:
    return arith::CmpFPredicate::UGE;
  case LLVM::FCmpPredicate::ult:
    return arith::CmpFPredicate::ULT;
  case LLVM::FCmpPredicate::ule:
    return arith::CmpFPredicate::ULE;
  case LLVM::FCmpPredicate::une:
    return arith::CmpFPredicate::UNE;
  case LLVM::FCmpPredicate::uno:
    return arith::CmpFPredicate::UNO;
  case LLVM::FCmpPredicate::_true:
    return arith::CmpFPredicate::AlwaysTrue;
  }
  llvm_unreachable("unhandled FCmpPredicate");
}

} // namespace fcc
