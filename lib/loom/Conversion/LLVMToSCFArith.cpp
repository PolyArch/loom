//===-- LLVMToSCFArith.cpp - Arithmetic op conversion -------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Implements arithmetic, cast, and select op handlers for FunctionConverter.
//
//===----------------------------------------------------------------------===//

#include "LLVMToSCFInternal.h"

using namespace mlir;

namespace loom::llvm_to_scf {

FailureOr<bool> FunctionConverter::handleArithOps(Operation &op, Location loc) {
  auto convertBinaryOp = [&](Value llvmLhs, Value llvmRhs, Value llvmRes,
                             StringRef name,
                             function_ref<Value(Value, Value)> mkScalar)
      -> LogicalResult {
    if (llvm::isa<VectorType>(llvmRes.getType())) {
      auto *lhsL = LookupVector(vectorMap, llvmLhs);
      auto *rhsL = LookupVector(vectorMap, llvmRhs);
      if (!lhsL || !rhsL) {
        op.emitError("missing vector ") << name << " operand";
        return failure();
      }
      SmallVector<Value, 8> out;
      for (size_t i = 0, e = lhsL->size(); i < e; ++i)
        out.push_back(mkScalar((*lhsL)[i], (*rhsL)[i]));
      vectorMap[llvmRes] = std::move(out);
      return success();
    }
    auto lhs = LookupValue(valueMap, llvmLhs);
    auto rhs = LookupValue(valueMap, llvmRhs);
    if (!lhs || !rhs) {
      op.emitError("missing ") << name << " operand";
      return failure();
    }
    valueMap[llvmRes] = mkScalar(*lhs, *rhs);
    return success();
  };

  if (auto addOp = llvm::dyn_cast<LLVM::AddOp>(op)) {
    if (failed(convertBinaryOp(addOp.getLhs(), addOp.getRhs(),
            addOp.getResult(), "add",
            [&](Value a, Value b) -> Value {
              return arith::AddIOp::create(builder, loc, a, b);
            })))
      return failure();
    return true;
  }

  if (auto subOp = llvm::dyn_cast<LLVM::SubOp>(op)) {
    if (failed(convertBinaryOp(subOp.getLhs(), subOp.getRhs(),
            subOp.getResult(), "sub",
            [&](Value a, Value b) -> Value {
              return arith::SubIOp::create(builder, loc, a, b);
            })))
      return failure();
    return true;
  }

  if (auto mulOp = llvm::dyn_cast<LLVM::MulOp>(op)) {
    if (failed(convertBinaryOp(mulOp.getLhs(), mulOp.getRhs(),
            mulOp.getResult(), "mul",
            [&](Value a, Value b) -> Value {
              return arith::MulIOp::create(builder, loc, a, b);
            })))
      return failure();
    return true;
  }

  if (auto shlOp = llvm::dyn_cast<LLVM::ShlOp>(op)) {
    if (failed(convertBinaryOp(shlOp.getLhs(), shlOp.getRhs(),
            shlOp.getResult(), "shl",
            [&](Value a, Value b) -> Value {
              return arith::ShLIOp::create(builder, loc, a, b);
            })))
      return failure();
    return true;
  }

  if (auto lshrOp = llvm::dyn_cast<LLVM::LShrOp>(op)) {
    if (failed(convertBinaryOp(lshrOp.getLhs(), lshrOp.getRhs(),
            lshrOp.getResult(), "lshr",
            [&](Value a, Value b) -> Value {
              return arith::ShRUIOp::create(builder, loc, a, b);
            })))
      return failure();
    return true;
  }

  if (auto ashrOp = llvm::dyn_cast<LLVM::AShrOp>(op)) {
    if (failed(convertBinaryOp(ashrOp.getLhs(), ashrOp.getRhs(),
            ashrOp.getResult(), "ashr",
            [&](Value a, Value b) -> Value {
              return arith::ShRSIOp::create(builder, loc, a, b);
            })))
      return failure();
    return true;
  }

  if (auto andOp = llvm::dyn_cast<LLVM::AndOp>(op)) {
    if (failed(convertBinaryOp(andOp.getLhs(), andOp.getRhs(),
            andOp.getResult(), "and",
            [&](Value a, Value b) -> Value {
              return arith::AndIOp::create(builder, loc, a, b);
            })))
      return failure();
    return true;
  }

  if (auto orOp = llvm::dyn_cast<LLVM::OrOp>(op)) {
    if (failed(convertBinaryOp(orOp.getLhs(), orOp.getRhs(),
            orOp.getResult(), "or",
            [&](Value a, Value b) -> Value {
              return arith::OrIOp::create(builder, loc, a, b);
            })))
      return failure();
    return true;
  }

  if (auto xorOp = llvm::dyn_cast<LLVM::XOrOp>(op)) {
    if (failed(convertBinaryOp(xorOp.getLhs(), xorOp.getRhs(),
            xorOp.getResult(), "xor",
            [&](Value a, Value b) -> Value {
              return arith::XOrIOp::create(builder, loc, a, b);
            })))
      return failure();
    return true;
  }

  if (auto faddOp = llvm::dyn_cast<LLVM::FAddOp>(op)) {
    if (failed(convertBinaryOp(faddOp.getLhs(), faddOp.getRhs(),
            faddOp.getResult(), "fadd",
            [&](Value a, Value b) -> Value {
              return arith::AddFOp::create(builder, loc, a, b);
            })))
      return failure();
    return true;
  }

  if (auto fnegOp = llvm::dyn_cast<LLVM::FNegOp>(op)) {
    if (llvm::isa<VectorType>(fnegOp.getResult().getType())) {
      auto *srcL = LookupVector(vectorMap, fnegOp.getOperand());
      if (!srcL)
        return fnegOp.emitError("missing vector fneg operand"), failure();
      SmallVector<Value, 8> out;
      for (Value v : *srcL)
        out.push_back(arith::NegFOp::create(builder, loc, v));
      vectorMap[fnegOp.getResult()] = std::move(out);
      return true;
    }
    auto operand = LookupValue(valueMap, fnegOp.getOperand());
    if (!operand)
      return fnegOp.emitError("missing fneg operand"), failure();
    valueMap[fnegOp.getResult()] = arith::NegFOp::create(builder, loc,
                                                          *operand);
    return true;
  }

  if (auto fsubOp = llvm::dyn_cast<LLVM::FSubOp>(op)) {
    if (failed(convertBinaryOp(fsubOp.getLhs(), fsubOp.getRhs(),
            fsubOp.getResult(), "fsub",
            [&](Value a, Value b) -> Value {
              return arith::SubFOp::create(builder, loc, a, b);
            })))
      return failure();
    return true;
  }

  if (auto fmulOp = llvm::dyn_cast<LLVM::FMulOp>(op)) {
    if (failed(convertBinaryOp(fmulOp.getLhs(), fmulOp.getRhs(),
            fmulOp.getResult(), "fmul",
            [&](Value a, Value b) -> Value {
              return arith::MulFOp::create(builder, loc, a, b);
            })))
      return failure();
    return true;
  }

  if (auto fdivOp = llvm::dyn_cast<LLVM::FDivOp>(op)) {
    if (failed(convertBinaryOp(fdivOp.getLhs(), fdivOp.getRhs(),
            fdivOp.getResult(), "fdiv",
            [&](Value a, Value b) -> Value {
              return arith::DivFOp::create(builder, loc, a, b);
            })))
      return failure();
    return true;
  }

  if (auto sdivOp = llvm::dyn_cast<LLVM::SDivOp>(op)) {
    if (failed(convertBinaryOp(sdivOp.getLhs(), sdivOp.getRhs(),
            sdivOp.getResult(), "sdiv",
            [&](Value a, Value b) -> Value {
              return arith::DivSIOp::create(builder, loc, a, b);
            })))
      return failure();
    return true;
  }

  if (auto udivOp = llvm::dyn_cast<LLVM::UDivOp>(op)) {
    if (failed(convertBinaryOp(udivOp.getLhs(), udivOp.getRhs(),
            udivOp.getResult(), "udiv",
            [&](Value a, Value b) -> Value {
              return arith::DivUIOp::create(builder, loc, a, b);
            })))
      return failure();
    return true;
  }

  if (auto uremOp = llvm::dyn_cast<LLVM::URemOp>(op)) {
    if (failed(convertBinaryOp(uremOp.getLhs(), uremOp.getRhs(),
            uremOp.getResult(), "urem",
            [&](Value a, Value b) -> Value {
              return arith::RemUIOp::create(builder, loc, a, b);
            })))
      return failure();
    return true;
  }

  if (auto icmpOp = llvm::dyn_cast<LLVM::ICmpOp>(op)) {
    auto pred = ConvertICmpPredicate(icmpOp.getPredicate());
    if (failed(convertBinaryOp(icmpOp.getLhs(), icmpOp.getRhs(),
            icmpOp.getResult(), "icmp",
            [&](Value a, Value b) -> Value {
              return arith::CmpIOp::create(builder, loc, pred, a, b);
            })))
      return failure();
    return true;
  }

  if (auto fcmpOp = llvm::dyn_cast<LLVM::FCmpOp>(op)) {
    auto pred = ConvertFCmpPredicate(fcmpOp.getPredicate());
    if (failed(convertBinaryOp(fcmpOp.getLhs(), fcmpOp.getRhs(),
            fcmpOp.getResult(), "fcmp",
            [&](Value a, Value b) -> Value {
              return arith::CmpFOp::create(builder, loc, pred, a, b);
            })))
      return failure();
    return true;
  }

  return false;
}

FailureOr<bool> FunctionConverter::handleSelectOp(Operation &op, Location loc) {
  auto selectOp = llvm::dyn_cast<LLVM::SelectOp>(op);
  if (!selectOp)
    return false;

  if (llvm::dyn_cast<VectorType>(selectOp.getResult().getType())) {
    auto *condL = LookupVector(vectorMap, selectOp.getCondition());
    auto *trueL = LookupVector(vectorMap, selectOp.getTrueValue());
    auto *falseL = LookupVector(vectorMap, selectOp.getFalseValue());
    if (!condL || !trueL || !falseL)
      return selectOp.emitError("missing vector select operand"), failure();
    SmallVector<Value, 8> out;
    for (size_t i = 0, e = condL->size(); i < e; ++i)
      out.push_back(arith::SelectOp::create(
          builder, loc, (*condL)[i], (*trueL)[i], (*falseL)[i]));
    vectorMap[selectOp.getResult()] = std::move(out);
    return true;
  }

  auto cond = LookupValue(valueMap, selectOp.getCondition());
  if (!cond)
    return selectOp.emitError("missing select condition"), failure();

  auto lhsPtr = LookupPointer(pointerMap, selectOp.getTrueValue());
  auto rhsPtr = LookupPointer(pointerMap, selectOp.getFalseValue());
  if (lhsPtr || rhsPtr) {
    if (!lhsPtr || !rhsPtr)
      return selectOp.emitError("missing select pointer operand"), failure();
    if (lhsPtr->elementType != rhsPtr->elementType)
      return selectOp.emitError("select pointer type mismatch"), failure();
    auto lhsBaseType =
        llvm::dyn_cast<MemRefType>(lhsPtr->base.getType());
    auto rhsBaseType =
        llvm::dyn_cast<MemRefType>(rhsPtr->base.getType());
    if (!lhsBaseType || !rhsBaseType)
      return selectOp.emitError("select pointer base type invalid"), failure();
    if (lhsBaseType.getMemorySpace() != rhsBaseType.getMemorySpace())
      return selectOp.emitError("select pointer memory space mismatch"),
             failure();
    auto commonBaseType = MakeStridedMemRefType(
        lhsPtr->elementType, lhsBaseType.getMemorySpace());
    Value lhsBase = lhsPtr->base;
    if (lhsBaseType != commonBaseType) {
      if (!memref::CastOp::areCastCompatible(lhsBaseType, commonBaseType))
        return selectOp.emitError("select pointer base mismatch"), failure();
      lhsBase =
          memref::CastOp::create(builder, loc, commonBaseType, lhsBase);
    }
    Value rhsBase = rhsPtr->base;
    if (rhsBaseType != commonBaseType) {
      if (!memref::CastOp::areCastCompatible(rhsBaseType, commonBaseType))
        return selectOp.emitError("select pointer base mismatch"), failure();
      rhsBase =
          memref::CastOp::create(builder, loc, commonBaseType, rhsBase);
    }
    Value baseSel =
        arith::SelectOp::create(builder, loc, *cond, lhsBase, rhsBase);
    Value indexSel = arith::SelectOp::create(builder, loc, *cond,
                                             lhsPtr->index, rhsPtr->index);
    pointerMap[selectOp.getResult()] =
        PointerInfo{baseSel, indexSel, lhsPtr->elementType};
    return true;
  }

  auto lhs = LookupValue(valueMap, selectOp.getTrueValue());
  auto rhs = LookupValue(valueMap, selectOp.getFalseValue());
  if (!lhs || !rhs)
    return selectOp.emitError("missing select operand"), failure();
  auto sel = arith::SelectOp::create(builder, loc, *cond, *lhs, *rhs);
  valueMap[selectOp.getResult()] = sel.getResult();
  return true;
}

FailureOr<bool> FunctionConverter::handleCastOps(Operation &op, Location loc) {
  if (auto zextOp = llvm::dyn_cast<LLVM::ZExtOp>(op)) {
    auto src = LookupValue(valueMap, zextOp.getArg());
    if (!src)
      return zextOp.emitError("missing zext operand"), failure();
    auto dstType = NormalizeScalarType(zextOp.getType(), module.getContext());
    auto ext = arith::ExtUIOp::create(builder, loc, dstType, *src);
    valueMap[zextOp.getResult()] = ext.getResult();
    return true;
  }

  if (auto sextOp = llvm::dyn_cast<LLVM::SExtOp>(op)) {
    auto src = LookupValue(valueMap, sextOp.getArg());
    if (!src)
      return sextOp.emitError("missing sext operand"), failure();
    auto dstType = NormalizeScalarType(sextOp.getType(), module.getContext());
    auto ext = arith::ExtSIOp::create(builder, loc, dstType, *src);
    valueMap[sextOp.getResult()] = ext.getResult();
    return true;
  }

  if (auto truncOp = llvm::dyn_cast<LLVM::TruncOp>(op)) {
    auto src = LookupValue(valueMap, truncOp.getArg());
    if (!src)
      return truncOp.emitError("missing trunc operand"), failure();
    auto dstType = NormalizeScalarType(truncOp.getType(), module.getContext());
    auto trunc = arith::TruncIOp::create(builder, loc, dstType, *src);
    valueMap[truncOp.getResult()] = trunc.getResult();
    return true;
  }

  if (auto fpextOp = llvm::dyn_cast<LLVM::FPExtOp>(op)) {
    auto src = LookupValue(valueMap, fpextOp.getArg());
    if (!src)
      return fpextOp.emitError("missing fpext operand"), failure();
    auto dstType = NormalizeScalarType(fpextOp.getType(), module.getContext());
    auto ext = arith::ExtFOp::create(builder, loc, dstType, *src);
    valueMap[fpextOp.getResult()] = ext.getResult();
    return true;
  }

  if (auto fptruncOp = llvm::dyn_cast<LLVM::FPTruncOp>(op)) {
    auto src = LookupValue(valueMap, fptruncOp.getArg());
    if (!src)
      return fptruncOp.emitError("missing fptrunc operand"), failure();
    auto dstType = NormalizeScalarType(fptruncOp.getType(), module.getContext());
    auto trunc = arith::TruncFOp::create(builder, loc, dstType, *src);
    valueMap[fptruncOp.getResult()] = trunc.getResult();
    return true;
  }

  if (auto uitofpOp = llvm::dyn_cast<LLVM::UIToFPOp>(op)) {
    auto src = LookupValue(valueMap, uitofpOp.getArg());
    if (!src)
      return uitofpOp.emitError("missing uitofp operand"), failure();
    auto dstType = NormalizeScalarType(uitofpOp.getType(), module.getContext());
    auto cast = arith::UIToFPOp::create(builder, loc, dstType, *src);
    valueMap[uitofpOp.getResult()] = cast.getResult();
    return true;
  }

  if (auto sitofpOp = llvm::dyn_cast<LLVM::SIToFPOp>(op)) {
    auto src = LookupValue(valueMap, sitofpOp.getArg());
    if (!src)
      return sitofpOp.emitError("missing sitofp operand"), failure();
    auto dstType = NormalizeScalarType(sitofpOp.getType(), module.getContext());
    auto cast = arith::SIToFPOp::create(builder, loc, dstType, *src);
    valueMap[sitofpOp.getResult()] = cast.getResult();
    return true;
  }

  if (auto fptosiOp = llvm::dyn_cast<LLVM::FPToSIOp>(op)) {
    auto src = LookupValue(valueMap, fptosiOp.getArg());
    if (!src)
      return fptosiOp.emitError("missing fptosi operand"), failure();
    auto dstType = NormalizeScalarType(fptosiOp.getType(), module.getContext());
    auto cast = arith::FPToSIOp::create(builder, loc, dstType, *src);
    valueMap[fptosiOp.getResult()] = cast.getResult();
    return true;
  }

  if (auto fptouiOp = llvm::dyn_cast<LLVM::FPToUIOp>(op)) {
    auto src = LookupValue(valueMap, fptouiOp.getArg());
    if (!src)
      return fptouiOp.emitError("missing fptoui operand"), failure();
    auto dstType = NormalizeScalarType(fptouiOp.getType(), module.getContext());
    auto cast = arith::FPToUIOp::create(builder, loc, dstType, *src);
    valueMap[fptouiOp.getResult()] = cast.getResult();
    return true;
  }

  return false;
}

} // namespace loom::llvm_to_scf
