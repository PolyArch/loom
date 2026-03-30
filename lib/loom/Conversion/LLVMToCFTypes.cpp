// Type conversion utilities for LLVMToCF pass.

#include "LLVMToCFTypes.h"
#include "loom/Dialect/Fabric/FabricTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace loom {

namespace {

static bool isByteType(Type type) {
  auto intTy = dyn_cast<IntegerType>(type);
  return intTy && intTy.getWidth() == 8;
}

static Type getHomogeneousStructElementType(LLVM::LLVMStructType structTy,
                                            MLIRContext *ctx) {
  auto body = structTy.getBody();
  if (body.empty())
    return nullptr;

  Type firstTy = normalizeScalarType(ctx, body[0]);
  if (isByteType(firstTy))
    return nullptr;

  for (unsigned i = 1; i < body.size(); ++i) {
    if (normalizeScalarType(ctx, body[i]) != firstTy)
      return nullptr;
  }
  return firstTy;
}

} // namespace

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
  bool sawByteEvidence = false;

  auto recordType = [&](Type ty) {
    if (!ty)
      return;
    ty = normalizeScalarType(ptrVal.getContext(), ty);
    if (isByteType(ty)) {
      sawByteEvidence = true;
      return;
    }
    if (!bestType)
      bestType = ty;
  };

  for (auto &use : ptrVal.getUses()) {
    Operation *user = use.getOwner();

    if (auto bitcast = dyn_cast<LLVM::BitcastOp>(user)) {
      if (use.getOperandNumber() == 0 && isa<LLVM::LLVMPointerType>(
                                             bitcast.getType())) {
        Type fromBitcastUses = inferFromUses(bitcast.getResult(), depth + 1);
        if (fromBitcastUses) {
          if (!isByteType(fromBitcastUses))
            return fromBitcastUses;
          sawByteEvidence = true;
        }
      }
    }

    if (auto cast = dyn_cast<LLVM::AddrSpaceCastOp>(user)) {
      if (use.getOperandNumber() == 0 && isa<LLVM::LLVMPointerType>(
                                             cast.getType())) {
        Type fromCastUses = inferFromUses(cast.getResult(), depth + 1);
        if (fromCastUses) {
          if (!isByteType(fromCastUses))
            return fromCastUses;
          sawByteEvidence = true;
        }
      }
    }

    // GEP reveals element type directly
    if (auto gep = dyn_cast<LLVM::GEPOp>(user)) {
      if (use.getOperandNumber() == 0) { // base operand
        Type elemTy = gep.getElemType();
        if (isa<LLVM::LLVMStructType>(elemTy)) {
          Type structTy = getHomogeneousStructElementType(
              cast<LLVM::LLVMStructType>(elemTy), ptrVal.getContext());
          if (structTy)
            recordType(structTy);

          Type fromGepUses = inferFromUses(gep.getResult(), depth + 1);
          if (fromGepUses) {
            if (!isByteType(fromGepUses))
              return fromGepUses;
            sawByteEvidence = true;
          }
          continue;
        }
        recordType(elemTy);
        // Also look at GEP result uses
        Type fromGepUses = inferFromUses(gep.getResult(), depth + 1);
        if (fromGepUses) {
          if (!isByteType(fromGepUses))
            return fromGepUses;
          sawByteEvidence = true;
        }
      }
    }

    // Load reveals access type
    if (auto load = dyn_cast<LLVM::LoadOp>(user)) {
      Type rawLoadTy = load.getResult().getType();
      // If loading a pointer, trace its uses to find the scalar type
      if (isa<LLVM::LLVMPointerType>(rawLoadTy)) {
        Type fromLoadUses = inferFromUses(load.getResult(), depth + 1);
        if (fromLoadUses) {
          if (!isByteType(fromLoadUses))
            return fromLoadUses;
          sawByteEvidence = true;
        }
        continue;
      }
      // If loading a vector type, infer from the vector element type
      if (auto vecTy = dyn_cast<VectorType>(rawLoadTy)) {
        recordType(vecTy.getElementType());
        continue;
      }
      recordType(rawLoadTy);
    }

    // Store reveals value type
    if (auto store = dyn_cast<LLVM::StoreOp>(user)) {
      if (use.getOperandNumber() == 1) { // addr operand
        Type rawValTy = store.getValue().getType();
        // If storing a vector type, infer from element type
        if (auto vecTy = dyn_cast<VectorType>(rawValTy)) {
          recordType(vecTy.getElementType());
          continue;
        }
        recordType(rawValTy);
      }
    }

    // Branch: trace through block arguments
    if (auto br = dyn_cast<LLVM::BrOp>(user)) {
      unsigned idx = use.getOperandNumber();
      Block *dest = br.getDest();
      if (idx < dest->getNumArguments()) {
        Type fromDest = inferFromUses(dest->getArgument(idx), depth + 1);
        if (fromDest) {
          if (!isByteType(fromDest))
            return fromDest;
          sawByteEvidence = true;
        }
      }
    }

    if (auto condBr = dyn_cast<LLVM::CondBrOp>(user)) {
      unsigned idx = use.getOperandNumber();
      if (idx > 0) {
        unsigned trueCount = condBr.getTrueDestOperands().size();
        if (idx - 1 < trueCount) {
          Block *dest = condBr.getTrueDest();
          unsigned argIdx = idx - 1;
          if (argIdx < dest->getNumArguments()) {
            Type fromDest = inferFromUses(dest->getArgument(argIdx),
                                          depth + 1);
            if (fromDest) {
              if (!isByteType(fromDest))
                return fromDest;
              sawByteEvidence = true;
            }
          }
        } else {
          Block *dest = condBr.getFalseDest();
          unsigned argIdx = idx - 1 - trueCount;
          if (argIdx < dest->getNumArguments()) {
            Type fromDest = inferFromUses(dest->getArgument(argIdx), depth + 1);
            if (fromDest) {
              if (!isByteType(fromDest))
                return fromDest;
              sawByteEvidence = true;
            }
          }
        }
      }
    }
  }

  if (bestType)
    return bestType;
  if (sawByteEvidence)
    return IntegerType::get(ptrVal.getContext(), 8);
  return nullptr;
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

    // Prefer evidence from the full use graph; only fall back to byte
    // addressing when no stronger scalar type can be proven.
    Type inferred = inferFromUses(arg);
    if (!inferred)
      inferred = IntegerType::get(funcOp.getContext(), 8);

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
