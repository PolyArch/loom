// Type conversion utilities for LLVMToCF pass.

#include "LLVMToCFTypes.h"
#include "loom/Dialect/Fabric/FabricTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace loom {

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
    return loom::fabric::getIndexIntegerType(ctx);
  // LLVM struct type -> i8 (byte-addressable representation)
  if (isa<LLVM::LLVMStructType>(llvmType))
    return IntegerType::get(ctx, 8);
  // LLVM array type -> element type
  if (auto arrayTy = dyn_cast<LLVM::LLVMArrayType>(llvmType))
    return normalizeScalarType(ctx, arrayTy.getElementType());
  // Vector type -> element type (vectors are decomposed into scalars)
  if (auto vecTy = dyn_cast<VectorType>(llvmType))
    return normalizeScalarType(ctx, vecTy.getElementType());
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

  // Flatten struct types: treat as byte array
  if (auto structTy = dyn_cast<LLVM::LLVMStructType>(llvmType)) {
    unsigned totalBits = getTypeBitWidth(structTy);
    unsigned totalBytes = (totalBits + 7) / 8;
    if (totalBytes == 0) totalBytes = 1;
    elementCount *= totalBytes;
    return IntegerType::get(ctx, 8);
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
    return loom::fabric::getConfiguredIndexBitWidth();
  if (isa<LLVM::LLVMPointerType>(type))
    return 64; // Assume 64-bit pointers
  if (auto arrTy = dyn_cast<LLVM::LLVMArrayType>(type))
    return arrTy.getNumElements() * getTypeBitWidth(arrTy.getElementType());
  if (auto structTy = dyn_cast<LLVM::LLVMStructType>(type)) {
    // Sum up field sizes (simplified, no padding calculation)
    unsigned totalBits = 0;
    for (Type fieldTy : structTy.getBody()) {
      unsigned fieldBits = getTypeBitWidth(fieldTy);
      // Align to field boundary (natural alignment)
      unsigned alignBits = fieldBits > 0 ? fieldBits : 8;
      if (alignBits > 64) alignBits = 64; // Cap alignment at 64 bits
      if (totalBits % alignBits != 0)
        totalBits += alignBits - (totalBits % alignBits);
      totalBits += fieldBits;
    }
    return totalBits;
  }
  return 0;
}

unsigned getStructFieldByteOffset(Type structType, unsigned fieldIndex) {
  auto structTy = dyn_cast<LLVM::LLVMStructType>(structType);
  if (!structTy)
    return 0;
  auto body = structTy.getBody();
  if (fieldIndex >= body.size())
    return 0;

  unsigned byteOffset = 0;
  for (unsigned i = 0; i < fieldIndex; ++i) {
    unsigned fieldBits = getTypeBitWidth(body[i]);
    unsigned fieldBytes = (fieldBits + 7) / 8;
    // Natural alignment (capped at 8 bytes)
    unsigned alignBytes = fieldBytes;
    if (alignBytes > 8) alignBytes = 8;
    if (alignBytes == 0) alignBytes = 1;
    if (byteOffset % alignBytes != 0)
      byteOffset += alignBytes - (byteOffset % alignBytes);
    byteOffset += fieldBytes;
  }
  // Align to the target field's alignment
  unsigned targetBits = getTypeBitWidth(body[fieldIndex]);
  unsigned targetBytes = (targetBits + 7) / 8;
  unsigned targetAlign = targetBytes;
  if (targetAlign > 8) targetAlign = 8;
  if (targetAlign == 0) targetAlign = 1;
  if (byteOffset % targetAlign != 0)
    byteOffset += targetAlign - (byteOffset % targetAlign);
  return byteOffset;
}

Type getStructFieldType(Type structType, unsigned fieldIndex) {
  auto structTy = dyn_cast<LLVM::LLVMStructType>(structType);
  if (!structTy)
    return nullptr;
  auto body = structTy.getBody();
  if (fieldIndex >= body.size())
    return nullptr;
  return body[fieldIndex];
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
        // For struct-typed GEPs, skip past to look at downstream uses
        if (isa<LLVM::LLVMStructType>(elemTy)) {
          Type fromGepUses = inferFromUses(gep.getResult(), depth + 1);
          if (fromGepUses)
            return fromGepUses;
          if (!bestType)
            bestType = IntegerType::get(ptrVal.getContext(), 8);
          continue;
        }
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
      Type rawLoadTy = load.getResult().getType();
      // If loading a pointer, trace its uses to find the scalar type
      if (isa<LLVM::LLVMPointerType>(rawLoadTy)) {
        Type fromLoadUses = inferFromUses(load.getResult(), depth + 1);
        if (fromLoadUses) {
          if (!bestType)
            bestType = fromLoadUses;
        }
        continue;
      }
      // If loading a vector type, infer from the vector element type
      if (auto vecTy = dyn_cast<VectorType>(rawLoadTy)) {
        Type elemTy = normalizeScalarType(user->getContext(),
                                          vecTy.getElementType());
        if (!isa<IntegerType>(elemTy) ||
            cast<IntegerType>(elemTy).getWidth() != 8)
          return elemTy;
        if (!bestType)
          bestType = elemTy;
        continue;
      }
      Type loadTy = normalizeScalarType(user->getContext(), rawLoadTy);
      if (!isa<IntegerType>(loadTy) ||
          cast<IntegerType>(loadTy).getWidth() != 8)
        return loadTy;
      if (!bestType)
        bestType = loadTy;
    }

    // Store reveals value type
    if (auto store = dyn_cast<LLVM::StoreOp>(user)) {
      if (use.getOperandNumber() == 1) { // addr operand
        Type rawValTy = store.getValue().getType();
        // If storing a vector type, infer from element type
        if (auto vecTy = dyn_cast<VectorType>(rawValTy)) {
          Type elemTy = normalizeScalarType(user->getContext(),
                                            vecTy.getElementType());
          if (!isa<IntegerType>(elemTy) ||
              cast<IntegerType>(elemTy).getWidth() != 8)
            return elemTy;
          if (!bestType)
            bestType = elemTy;
          continue;
        }
        Type valTy = normalizeScalarType(user->getContext(), rawValTy);
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

    // First try to infer from GEP elem_type (most reliable).
    // For struct-typed GEPs, look through the struct to find the scalar type
    // from downstream uses or from the struct's field types.
    Type inferred = nullptr;
    funcOp.walk([&](LLVM::GEPOp gepOp) {
      if (gepOp.getBase() == arg) {
        Type elemTy = gepOp.getElemType();
        if (elemTy && isa<LLVM::LLVMStructType>(elemTy)) {
          // For homogeneous structs (all fields same type), use the scalar type.
          // This enables vector load/store decomposition for struct-as-array.
          auto structTy = cast<LLVM::LLVMStructType>(elemTy);
          auto body = structTy.getBody();
          if (!body.empty()) {
            Type firstTy = normalizeScalarType(funcOp.getContext(), body[0]);
            bool allSame = true;
            for (unsigned fi = 1; fi < body.size(); ++fi) {
              if (normalizeScalarType(funcOp.getContext(), body[fi]) != firstTy) {
                allSame = false;
                break;
              }
            }
            if (allSame && (!inferred || (isa<IntegerType>(inferred) &&
                            cast<IntegerType>(inferred).getWidth() == 8))) {
              inferred = firstTy;
              return;
            }
          }
          // Heterogeneous struct or empty: use i8 byte addressing
          if (!inferred)
            inferred = IntegerType::get(funcOp.getContext(), 8);
          return;
        }
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

    // Normalize to ensure memref-compatible type
    inferred = normalizeScalarType(funcOp.getContext(), inferred);
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

} // namespace loom
