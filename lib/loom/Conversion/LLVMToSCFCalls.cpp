//===-- LLVMToSCFCalls.cpp - Call and intrinsic conversion ---*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Implements call op and intrinsic call handlers for FunctionConverter.
//
//===----------------------------------------------------------------------===//

#include "LLVMToSCFInternal.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace loom::llvm_to_scf {

FailureOr<bool> FunctionConverter::handleCallOp(Operation &op, Location loc) {
  // InlineAsmOp: skip void/no-result, error otherwise
  if (auto asmOp = llvm::dyn_cast<LLVM::InlineAsmOp>(op)) {
    if (asmOp.getNumResults() == 0)
      return true;
    if (asmOp.getNumResults() == 1) {
      Type resultType = asmOp.getResult(0).getType();
      if (llvm::isa<LLVM::LLVMVoidType>(resultType))
        return true;
    }
    return asmOp.emitError("unsupported inline asm result"), failure();
  }

  auto callOp = llvm::dyn_cast<LLVM::CallOp>(op);
  if (!callOp)
    return false;

  auto calleeAttr = callOp.getCalleeAttr();
  if (!calleeAttr)
    return callOp.emitError("indirect calls are not supported"), failure();
  StringRef callee = calleeAttr.getValue();
  bool returnsPointer =
      callOp.getNumResults() == 1 &&
      IsPointerType(callOp.getResults().front().getType());
  if (returnsPointer) {
    StdMinMaxKind minMaxKind = StdMinMaxKind::Minimum;
    if (IsStdMinMaxName(callee, minMaxKind)) {
      if (callOp.getNumOperands() != 2)
        return callOp.emitError("std::min/max expects 2 operands"), failure();
      auto lhsPtr = LookupPointer(pointerMap, callOp.getOperand(0));
      auto rhsPtr = LookupPointer(pointerMap, callOp.getOperand(1));
      if (!lhsPtr || !rhsPtr)
        return callOp.emitError("missing std::min/max pointer operand"),
               failure();
      Value lhsVal =
          memref::LoadOp::create(builder, loc, lhsPtr->base, lhsPtr->index);
      Value rhsVal =
          memref::LoadOp::create(builder, loc, rhsPtr->base, rhsPtr->index);
      if (lhsVal.getType() != rhsVal.getType())
        return callOp.emitError("std::min/max operand type mismatch"),
               failure();
      StdMinMaxScalarKind scalarKind = ParseStdMinMaxScalarKind(callee);
      Value cmp;
      if (llvm::isa<FloatType>(lhsVal.getType())) {
        if (scalarKind != StdMinMaxScalarKind::FloatKind &&
            scalarKind != StdMinMaxScalarKind::UnknownKind)
          return callOp.emitError(
                     "std::min/max float operand kind mismatch"),
                 failure();
        Value cmpLhs =
            (minMaxKind == StdMinMaxKind::Minimum) ? rhsVal : lhsVal;
        Value cmpRhs =
            (minMaxKind == StdMinMaxKind::Minimum) ? lhsVal : rhsVal;
        cmp = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OLT,
                                     cmpLhs, cmpRhs);
      } else if (llvm::isa<IntegerType>(lhsVal.getType())) {
        bool isUnsigned = scalarKind == StdMinMaxScalarKind::UnsignedIntKind;
        bool isSigned = scalarKind == StdMinMaxScalarKind::SignedIntKind;
        if (!isUnsigned && !isSigned)
          return callOp.emitError(
                     "std::min/max integer operand kind unknown"),
                 failure();
        arith::CmpIPredicate pred =
            isUnsigned ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt;
        Value cmpLhs =
            (minMaxKind == StdMinMaxKind::Minimum) ? rhsVal : lhsVal;
        Value cmpRhs =
            (minMaxKind == StdMinMaxKind::Minimum) ? lhsVal : rhsVal;
        cmp = arith::CmpIOp::create(builder, loc, pred, cmpLhs, cmpRhs);
      } else {
        return callOp.emitError("std::min/max unsupported operand type"),
               failure();
      }
      auto selected =
          BuildPointerSelect(builder, loc, cmp, *lhsPtr, *rhsPtr, true);
      if (!selected)
        return callOp.emitError("std::min/max pointer select failed"),
               failure();
      pointerMap[callOp.getResults().front()] = *selected;
      if (auto def = selected->base.getDefiningOp())
        CopyLoomAnnotations(&op, def);
      if (auto def = selected->index.getDefiningOp())
        CopyLoomAnnotations(&op, def);
      return true;
    }
  }

  SmallVector<Value, 8> mappedOperands;
  mappedOperands.reserve(callOp.getNumOperands());
  bool hasPointerOperand = false;
  for (Value operand : callOp.getOperands()) {
    if (LookupPointer(pointerMap, operand)) {
      hasPointerOperand = true;
      break;
    }
    auto mapped = LookupValue(valueMap, operand);
    if (!mapped)
      return callOp.emitError("missing call operand"), failure();
    mappedOperands.push_back(*mapped);
  }

  if (!hasPointerOperand) {
    if (auto mathResult =
            ConvertMathCall(builder, loc, callee, mappedOperands)) {
      if (callOp.getNumResults() != 1)
        return callOp.emitError("unexpected math call results"), failure();
      valueMap[callOp->getResult(0)] = *mathResult;
      CopyLoomAnnotations(&op, mathResult->getDefiningOp());
      return true;
    }
    if (auto minMaxResult = ConvertStdMinMaxScalarCall(
            builder, loc, callee, mappedOperands)) {
      if (callOp.getNumResults() != 1)
        return callOp.emitError("unexpected min/max call results"), failure();
      valueMap[callOp->getResult(0)] = *minMaxResult;
      CopyLoomAnnotations(&op, minMaxResult->getDefiningOp());
      return true;
    }
  }

  bool isVarArg = varargFunctions.contains(callee);
  bool isRawPtrCall = IsRawPointerCallee(callee);
  if (returnsPointer)
    return callOp.emitError("unsupported pointer return call"), failure();
  SmallVector<Value, 8> operands;
  operands.reserve(callOp.getNumOperands());
  for (Value operand : callOp.getOperands()) {
    if (auto ptrInfo = LookupPointer(pointerMap, operand)) {
      if (isVarArg || isRawPtrCall)
        operands.push_back(MaterializeLLVMPointer(builder, loc, *ptrInfo));
      else
        operands.push_back(MaterializeMemrefPointer(builder, loc, *ptrInfo));
      continue;
    }
    auto mapped = LookupValue(valueMap, operand);
    if (!mapped)
      return callOp.emitError("missing call operand"), failure();
    operands.push_back(*mapped);
  }

  if (isVarArg || isRawPtrCall) {
    auto newCall = LLVM::CallOp::create(builder, loc,
                                         callOp.getResultTypes(), calleeAttr,
                                         operands);
    if (auto varType = callOp.getVarCalleeType())
      newCall.setVarCalleeType(varType);
    CopyLoomAnnotations(&op, newCall.getOperation());
    for (auto [oldRes, newRes] :
         llvm::zip(callOp.getResults(), newCall.getResults()))
      valueMap[oldRes] = newRes;
    return true;
  }

  SmallVector<Type, 4> resultTypes;
  for (Type type : callOp.getResultTypes())
    resultTypes.push_back(NormalizeScalarType(type, module.getContext()));
  auto call =
      func::CallOp::create(builder, loc, callee, resultTypes, operands);
  CopyLoomAnnotations(&op, call.getOperation());
  for (auto [oldRes, newRes] :
       llvm::zip(callOp.getResults(), call.getResults()))
    valueMap[oldRes] = newRes;
  return true;
}

FailureOr<bool> FunctionConverter::handleIntrinsicCall(Operation &op,
                                                        Location loc) {
  auto callOp = llvm::dyn_cast<LLVM::CallIntrinsicOp>(op);
  if (!callOp)
    return false;

  StringRef callee = callOp.getIntrin();
  if (callee.starts_with("llvm.var.annotation") ||
      callee.starts_with("llvm.ptr.annotation") ||
      callee.starts_with("llvm.annotation")) {
    return true;
  }
  if (callee.starts_with("llvm.lifetime.start") ||
      callee.starts_with("llvm.lifetime.end")) {
    return true;
  }
  if (callee.starts_with("llvm.stacksave")) {
    if (callOp.getNumResults() != 1)
      return callOp.emitError("unexpected stacksave results"), failure();
    auto memrefType = MakeMemRefType(builder.getI8Type());
    Value one = BuildIndexConstant(builder, loc, 1);
    auto alloc =
        memref::AllocaOp::create(builder, loc, memrefType, ValueRange{one});
    PointerInfo info{alloc, BuildIndexConstant(builder, loc, 0),
                     builder.getI8Type()};
    pointerMap[callOp->getResult(0)] = info;
    return true;
  }
  if (callee.starts_with("llvm.stackrestore")) {
    return true;
  }
  if (callee.starts_with("llvm.memcpy")) {
    if (callOp.getNumOperands() < 4)
      return callOp.emitError("invalid memcpy operands"), failure();
    auto dstInfo = LookupPointer(pointerMap, callOp.getOperand(0));
    auto srcInfo = LookupPointer(pointerMap, callOp.getOperand(1));
    auto sizeVal = LookupValue(valueMap, callOp.getOperand(2));
    if (!dstInfo || !srcInfo || !sizeVal)
      return callOp.emitError("invalid memcpy operands"), failure();
    int64_t elemSize = GetByteSize(dstInfo->elementType);
    if (elemSize == 0)
      return callOp.emitError("unsupported memcpy element type"), failure();
    Value sizeIndex = ToIndexValue(builder, loc, *sizeVal);
    if (!sizeIndex)
      return callOp.emitError("invalid memcpy size"), failure();
    Value elemSizeVal = BuildIndexConstant(builder, loc, elemSize);
    Value length =
        arith::DivUIOp::create(builder, loc, sizeIndex, elemSizeVal);
    Value zero = BuildIndexConstant(builder, loc, 0);
    Value one = BuildIndexConstant(builder, loc, 1);
    auto loop = scf::ForOp::create(builder, loc, zero, length, one);
    builder.setInsertionPointToStart(loop.getBody());
    Value iv = loop.getInductionVar();
    Value srcIndex = iv;
    if (!IsZeroIndex(srcInfo->index))
      srcIndex = arith::AddIOp::create(builder, loc, srcInfo->index, iv);
    Value dstIndex = iv;
    if (!IsZeroIndex(dstInfo->index))
      dstIndex = arith::AddIOp::create(builder, loc, dstInfo->index, iv);
    Value val = memref::LoadOp::create(builder, loc, srcInfo->base, srcIndex);
    memref::StoreOp::create(builder, loc, val, dstInfo->base, dstIndex);
    builder.setInsertionPointAfter(loop);
    return true;
  }
  if (callee.starts_with("llvm.bswap")) {
    if (callOp.getNumOperands() != 1)
      return callOp.emitError("invalid bswap operands"), failure();
    auto operand = LookupValue(valueMap, callOp.getOperand(0));
    if (!operand)
      return callOp.emitError("missing bswap operand"), failure();
    auto intTy = llvm::dyn_cast<IntegerType>(operand->getType());
    if (!intTy)
      return callOp.emitError("bswap expects integer operand"), failure();
    unsigned bitWidth = intTy.getWidth();
    if (bitWidth % 8 != 0)
      return callOp.emitError("bswap expects byte-multiple width"), failure();
    unsigned byteCount = bitWidth / 8;
    Value result;
    for (unsigned i = 0; i < byteCount; ++i) {
      unsigned inShift = i * 8;
      unsigned outShift = (byteCount - 1 - i) * 8;
      auto inShiftVal = arith::ConstantOp::create(
          builder, loc, builder.getIntegerAttr(intTy, inShift));
      auto outShiftVal = arith::ConstantOp::create(
          builder, loc, builder.getIntegerAttr(intTy, outShift));
      auto byteMask = arith::ConstantOp::create(
          builder, loc, builder.getIntegerAttr(intTy, 0xFF));
      Value shiftedIn =
          arith::ShRUIOp::create(builder, loc, *operand, inShiftVal);
      Value masked = arith::AndIOp::create(builder, loc, shiftedIn, byteMask);
      Value shiftedOut =
          arith::ShLIOp::create(builder, loc, masked, outShiftVal);
      if (result)
        result = arith::OrIOp::create(builder, loc, result, shiftedOut);
      else
        result = shiftedOut;
    }
    valueMap[callOp->getResult(0)] = result;
    return true;
  }
  if (callee.starts_with("llvm.umin")) {
    if (callOp.getNumOperands() != 2)
      return callOp.emitError("invalid umin operands"), failure();
    auto lhs = LookupValue(valueMap, callOp.getOperand(0));
    auto rhs = LookupValue(valueMap, callOp.getOperand(1));
    if (!lhs || !rhs)
      return callOp.emitError("missing umin operand"), failure();
    auto cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult,
                                      *lhs, *rhs);
    auto sel = arith::SelectOp::create(builder, loc, cmp, *lhs, *rhs);
    valueMap[callOp->getResult(0)] = sel.getResult();
    return true;
  }
  if (callee.starts_with("llvm.usub.sat")) {
    if (callOp.getNumOperands() != 2)
      return callOp.emitError("invalid usub.sat operands"), failure();
    auto lhs = LookupValue(valueMap, callOp.getOperand(0));
    auto rhs = LookupValue(valueMap, callOp.getOperand(1));
    if (!lhs || !rhs)
      return callOp.emitError("missing usub.sat operand"), failure();
    auto intTy = llvm::dyn_cast<IntegerType>(lhs->getType());
    if (!intTy)
      return callOp.emitError("usub.sat expects integer operands"), failure();
    auto zero = arith::ConstantOp::create(
        builder, loc, builder.getIntegerAttr(intTy, 0));
    auto cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult,
                                      *lhs, *rhs);
    auto sub = arith::SubIOp::create(builder, loc, *lhs, *rhs);
    auto sel = arith::SelectOp::create(builder, loc, cmp, zero, sub);
    valueMap[callOp->getResult(0)] = sel.getResult();
    return true;
  }
  if (callee.starts_with("llvm.memset")) {
    if (callOp.getNumOperands() < 4)
      return callOp.emitError("invalid memset operands"), failure();
    auto dstInfo = LookupPointer(pointerMap, callOp.getOperand(0));
    auto val = LookupValue(valueMap, callOp.getOperand(1));
    auto sizeVal = LookupValue(valueMap, callOp.getOperand(2));
    if (!dstInfo || !val || !sizeVal)
      return callOp.emitError("invalid memset operands"), failure();
    int64_t elemSize = GetByteSize(dstInfo->elementType);
    if (elemSize == 0)
      return callOp.emitError("unsupported memset element type"), failure();
    Value sizeIndex = ToIndexValue(builder, loc, *sizeVal);
    if (!sizeIndex)
      return callOp.emitError("invalid memset size"), failure();
    Value elemSizeVal = BuildIndexConstant(builder, loc, elemSize);
    Value length =
        arith::DivUIOp::create(builder, loc, sizeIndex, elemSizeVal);
    Type elemType = dstInfo->elementType;
    auto fillVal = BuildMemsetFillValue(builder, loc, *val, elemType);
    if (!fillVal)
      return callOp.emitError("unsupported memset value"), failure();
    Value zero = BuildIndexConstant(builder, loc, 0);
    Value step = BuildIndexConstant(builder, loc, 1);
    auto loop = scf::ForOp::create(builder, loc, zero, length, step);
    builder.setInsertionPointToStart(loop.getBody());
    Value iv = loop.getInductionVar();
    Value dstIndex = iv;
    if (!IsZeroIndex(dstInfo->index))
      dstIndex = arith::AddIOp::create(builder, loc, dstInfo->index, iv);
    memref::StoreOp::create(builder, loc, *fillVal, dstInfo->base, dstIndex);
    builder.setInsertionPointAfter(loop);
    return true;
  }
  if (callee.starts_with("llvm.fmuladd")) {
    if (callOp.getNumOperands() != 3)
      return callOp.emitError("invalid fmuladd operands"), failure();
    auto *lhsL = LookupVector(vectorMap, callOp.getOperand(0));
    if (lhsL) {
      auto *rhsL = LookupVector(vectorMap, callOp.getOperand(1));
      auto *addL = LookupVector(vectorMap, callOp.getOperand(2));
      if (!rhsL || !addL)
        return callOp.emitError("missing vector fmuladd operand"), failure();
      SmallVector<Value, 8> out;
      for (size_t i = 0, e = lhsL->size(); i < e; ++i)
        out.push_back(math::FmaOp::create(builder, loc, (*lhsL)[i],
                                           (*rhsL)[i], (*addL)[i]));
      vectorMap[callOp->getResult(0)] = std::move(out);
      return true;
    }
    auto lhs = LookupValue(valueMap, callOp.getOperand(0));
    auto rhs = LookupValue(valueMap, callOp.getOperand(1));
    auto addend = LookupValue(valueMap, callOp.getOperand(2));
    if (!lhs || !rhs || !addend)
      return callOp.emitError("missing fmuladd operand"), failure();
    if (!llvm::isa<FloatType>(lhs->getType()) ||
        !llvm::isa<FloatType>(rhs->getType()) ||
        !llvm::isa<FloatType>(addend->getType()))
      return callOp.emitError("fmuladd expects float operands"), failure();
    auto fma = math::FmaOp::create(builder, loc, *lhs, *rhs, *addend);
    valueMap[callOp->getResult(0)] = fma.getResult();
    return true;
  }
  if (callee.starts_with("llvm.fshl")) {
    if (callOp.getNumOperands() != 3)
      return callOp.emitError("invalid fshl operands"), failure();
    auto lhs = LookupValue(valueMap, callOp.getOperand(0));
    auto rhs = LookupValue(valueMap, callOp.getOperand(1));
    auto shift = LookupValue(valueMap, callOp.getOperand(2));
    if (!lhs || !rhs || !shift)
      return callOp.emitError("missing fshl operand"), failure();
    auto intTy = llvm::dyn_cast<IntegerType>(lhs->getType());
    if (!intTy)
      return callOp.emitError("fshl expects integer operands"), failure();
    auto width = arith::ConstantOp::create(
        builder, loc, builder.getIntegerAttr(intTy, intTy.getWidth()));
    auto zero = arith::ConstantOp::create(
        builder, loc, builder.getIntegerAttr(intTy, 0));
    Value shiftMod = arith::RemUIOp::create(builder, loc, *shift, width);
    auto isZero = arith::CmpIOp::create(builder, loc,
                                         arith::CmpIPredicate::eq,
                                         shiftMod, zero);
    OpBuilder::InsertionGuard guard(builder);
    auto ifOp = scf::IfOp::create(builder, loc, TypeRange{intTy}, isZero,
                                   /*withElseRegion=*/true);
    builder.setInsertionPointToStart(ifOp.thenBlock());
    scf::YieldOp::create(builder, loc, *lhs);
    builder.setInsertionPointToStart(ifOp.elseBlock());
    Value left = arith::ShLIOp::create(builder, loc, *lhs, shiftMod);
    Value rightShift = arith::SubIOp::create(builder, loc, width, shiftMod);
    Value right = arith::ShRUIOp::create(builder, loc, *rhs, rightShift);
    Value combined = arith::OrIOp::create(builder, loc, left, right);
    scf::YieldOp::create(builder, loc, combined);
    valueMap[callOp->getResult(0)] = ifOp.getResult(0);
    return true;
  }
  if (callee.starts_with("llvm.fabs")) {
    if (callOp.getNumOperands() != 1)
      return callOp.emitError("invalid fabs operands"), failure();
    auto *srcL = LookupVector(vectorMap, callOp.getOperand(0));
    if (srcL) {
      SmallVector<Value, 8> out;
      for (Value v : *srcL)
        out.push_back(math::AbsFOp::create(builder, loc, v));
      vectorMap[callOp->getResult(0)] = std::move(out);
      return true;
    }
    auto operand = LookupValue(valueMap, callOp.getOperand(0));
    if (!operand)
      return callOp.emitError("missing fabs operand"), failure();
    auto abs = math::AbsFOp::create(builder, loc, *operand);
    valueMap[callOp->getResult(0)] = abs.getResult();
    return true;
  }
  if (callee.starts_with("llvm.vector.reduce.xor")) {
    if (callOp.getNumOperands() != 1)
      return callOp.emitError("invalid reduce.xor operands"), failure();
    auto *srcL = LookupVector(vectorMap, callOp.getOperand(0));
    if (!srcL || srcL->empty())
      return callOp.emitError("missing reduce.xor operand"), failure();
    Value result = (*srcL)[0];
    for (size_t i = 1, e = srcL->size(); i < e; ++i)
      result = arith::XOrIOp::create(builder, loc, result, (*srcL)[i]);
    valueMap[callOp->getResult(0)] = result;
    return true;
  }
  if (callee.starts_with("llvm.vector.reduce.add")) {
    if (callOp.getNumOperands() != 1)
      return callOp.emitError("invalid reduce.add operands"), failure();
    auto *srcL = LookupVector(vectorMap, callOp.getOperand(0));
    if (!srcL || srcL->empty())
      return callOp.emitError("missing reduce.add operand"), failure();
    Value result = (*srcL)[0];
    for (size_t i = 1, e = srcL->size(); i < e; ++i)
      result = arith::AddIOp::create(builder, loc, result, (*srcL)[i]);
    valueMap[callOp->getResult(0)] = result;
    return true;
  }
  if (callee.starts_with("llvm.bitreverse")) {
    if (callOp.getNumOperands() != 1)
      return callOp.emitError("invalid bitreverse operands"), failure();
    auto operand = LookupValue(valueMap, callOp.getOperand(0));
    if (!operand)
      return callOp.emitError("missing bitreverse operand"), failure();
    auto bitrev = LLVM::BitReverseOp::create(builder, loc,
                                              operand->getType(), *operand);
    valueMap[callOp->getResult(0)] = bitrev.getResult();
    return true;
  }
  if (callee.starts_with("llvm.smin")) {
    if (callOp.getNumOperands() != 2)
      return callOp.emitError("invalid smin operands"), failure();
    auto lhs = LookupValue(valueMap, callOp.getOperand(0));
    auto rhs = LookupValue(valueMap, callOp.getOperand(1));
    if (!lhs || !rhs)
      return callOp.emitError("missing smin operand"), failure();
    auto result = arith::MinSIOp::create(builder, loc, *lhs, *rhs);
    valueMap[callOp->getResult(0)] = result.getResult();
    return true;
  }
  if (callee.starts_with("llvm.smax")) {
    if (callOp.getNumOperands() != 2)
      return callOp.emitError("invalid smax operands"), failure();
    auto lhs = LookupValue(valueMap, callOp.getOperand(0));
    auto rhs = LookupValue(valueMap, callOp.getOperand(1));
    if (!lhs || !rhs)
      return callOp.emitError("missing smax operand"), failure();
    auto result = arith::MaxSIOp::create(builder, loc, *lhs, *rhs);
    valueMap[callOp->getResult(0)] = result.getResult();
    return true;
  }
  if (callee.starts_with("llvm.umax")) {
    if (callOp.getNumOperands() != 2)
      return callOp.emitError("invalid umax operands"), failure();
    auto lhs = LookupValue(valueMap, callOp.getOperand(0));
    auto rhs = LookupValue(valueMap, callOp.getOperand(1));
    if (!lhs || !rhs)
      return callOp.emitError("missing umax operand"), failure();
    auto result = arith::MaxUIOp::create(builder, loc, *lhs, *rhs);
    valueMap[callOp->getResult(0)] = result.getResult();
    return true;
  }
  if (callee.starts_with("llvm.sadd.sat")) {
    if (callOp.getNumOperands() != 2)
      return callOp.emitError("invalid sadd.sat operands"), failure();
    auto lhs = LookupValue(valueMap, callOp.getOperand(0));
    auto rhs = LookupValue(valueMap, callOp.getOperand(1));
    if (!lhs || !rhs)
      return callOp.emitError("missing sadd.sat operand"), failure();
    auto intTy = llvm::dyn_cast<IntegerType>(lhs->getType());
    if (!intTy)
      return callOp.emitError("sadd.sat expects integer operands"), failure();
    unsigned bitWidth = intTy.getWidth();
    auto sum = arith::AddIOp::create(builder, loc, *lhs, *rhs);
    APInt maxVal = APInt::getSignedMaxValue(bitWidth);
    APInt minVal = APInt::getSignedMinValue(bitWidth);
    auto intMax = arith::ConstantOp::create(
        builder, loc, builder.getIntegerAttr(intTy, maxVal));
    auto intMin = arith::ConstantOp::create(
        builder, loc, builder.getIntegerAttr(intTy, minVal));
    auto zero = arith::ConstantOp::create(
        builder, loc, builder.getIntegerAttr(intTy, 0));
    auto lhsNeg = arith::CmpIOp::create(builder, loc,
                                          arith::CmpIPredicate::slt, *lhs,
                                          zero);
    auto rhsNeg = arith::CmpIOp::create(builder, loc,
                                          arith::CmpIPredicate::slt, *rhs,
                                          zero);
    auto sumNeg =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt, sum,
                               zero);
    auto posOv = arith::AndIOp::create(
        builder, loc,
        arith::AndIOp::create(
            builder, loc,
            arith::XOrIOp::create(
                builder, loc, lhsNeg,
                arith::ConstantOp::create(
                    builder, loc,
                    builder.getIntegerAttr(builder.getI1Type(), 1))),
            arith::XOrIOp::create(
                builder, loc, rhsNeg,
                arith::ConstantOp::create(
                    builder, loc,
                    builder.getIntegerAttr(builder.getI1Type(), 1)))),
        sumNeg);
    auto negOv = arith::AndIOp::create(
        builder, loc, arith::AndIOp::create(builder, loc, lhsNeg, rhsNeg),
        arith::XOrIOp::create(
            builder, loc, sumNeg,
            arith::ConstantOp::create(
                builder, loc,
                builder.getIntegerAttr(builder.getI1Type(), 1))));
    auto overflow = arith::OrIOp::create(builder, loc, posOv, negOv);
    auto satVal =
        arith::SelectOp::create(builder, loc, posOv, intMax, intMin);
    auto result =
        arith::SelectOp::create(builder, loc, overflow, satVal, sum);
    valueMap[callOp->getResult(0)] = result.getResult();
    return true;
  }
  if (callee.starts_with("llvm.ssub.sat")) {
    if (callOp.getNumOperands() != 2)
      return callOp.emitError("invalid ssub.sat operands"), failure();
    auto lhs = LookupValue(valueMap, callOp.getOperand(0));
    auto rhs = LookupValue(valueMap, callOp.getOperand(1));
    if (!lhs || !rhs)
      return callOp.emitError("missing ssub.sat operand"), failure();
    auto intTy = llvm::dyn_cast<IntegerType>(lhs->getType());
    if (!intTy)
      return callOp.emitError("ssub.sat expects integer operands"), failure();
    unsigned bitWidth = intTy.getWidth();
    auto diff = arith::SubIOp::create(builder, loc, *lhs, *rhs);
    APInt maxVal = APInt::getSignedMaxValue(bitWidth);
    APInt minVal = APInt::getSignedMinValue(bitWidth);
    auto intMax = arith::ConstantOp::create(
        builder, loc, builder.getIntegerAttr(intTy, maxVal));
    auto intMin = arith::ConstantOp::create(
        builder, loc, builder.getIntegerAttr(intTy, minVal));
    auto zero = arith::ConstantOp::create(
        builder, loc, builder.getIntegerAttr(intTy, 0));
    auto lhsNeg = arith::CmpIOp::create(builder, loc,
                                          arith::CmpIPredicate::slt, *lhs,
                                          zero);
    auto rhsNeg = arith::CmpIOp::create(builder, loc,
                                          arith::CmpIPredicate::slt, *rhs,
                                          zero);
    auto diffNeg = arith::CmpIOp::create(builder, loc,
                                           arith::CmpIPredicate::slt, diff,
                                           zero);
    auto posOv = arith::AndIOp::create(
        builder, loc,
        arith::AndIOp::create(
            builder, loc,
            arith::XOrIOp::create(
                builder, loc, lhsNeg,
                arith::ConstantOp::create(
                    builder, loc,
                    builder.getIntegerAttr(builder.getI1Type(), 1))),
            rhsNeg),
        diffNeg);
    auto negOv = arith::AndIOp::create(
        builder, loc,
        arith::AndIOp::create(
            builder, loc, lhsNeg,
            arith::XOrIOp::create(
                builder, loc, rhsNeg,
                arith::ConstantOp::create(
                    builder, loc,
                    builder.getIntegerAttr(builder.getI1Type(), 1)))),
        arith::XOrIOp::create(
            builder, loc, diffNeg,
            arith::ConstantOp::create(
                builder, loc,
                builder.getIntegerAttr(builder.getI1Type(), 1))));
    auto overflow = arith::OrIOp::create(builder, loc, posOv, negOv);
    auto satVal =
        arith::SelectOp::create(builder, loc, posOv, intMax, intMin);
    auto result =
        arith::SelectOp::create(builder, loc, overflow, satVal, diff);
    valueMap[callOp->getResult(0)] = result.getResult();
    return true;
  }
  if (callee.starts_with("llvm.ctlz")) {
    if (callOp.getNumOperands() < 1)
      return callOp.emitError("invalid ctlz operands"), failure();
    auto operand = LookupValue(valueMap, callOp.getOperand(0));
    if (!operand)
      return callOp.emitError("missing ctlz operand"), failure();
    auto result = math::CountLeadingZerosOp::create(builder, loc, *operand);
    valueMap[callOp->getResult(0)] = result.getResult();
    return true;
  }
  if (callee.starts_with("llvm.cttz")) {
    if (callOp.getNumOperands() < 1)
      return callOp.emitError("invalid cttz operands"), failure();
    auto operand = LookupValue(valueMap, callOp.getOperand(0));
    if (!operand)
      return callOp.emitError("missing cttz operand"), failure();
    auto result = math::CountTrailingZerosOp::create(builder, loc, *operand);
    valueMap[callOp->getResult(0)] = result.getResult();
    return true;
  }
  if (callee.starts_with("llvm.ctpop")) {
    if (callOp.getNumOperands() < 1)
      return callOp.emitError("invalid ctpop operands"), failure();
    auto operand = LookupValue(valueMap, callOp.getOperand(0));
    if (!operand)
      return callOp.emitError("missing ctpop operand"), failure();
    auto result = math::CtPopOp::create(builder, loc, *operand);
    valueMap[callOp->getResult(0)] = result.getResult();
    return true;
  }
  if (callee.starts_with("llvm.copysign")) {
    if (callOp.getNumOperands() != 2)
      return callOp.emitError("invalid copysign operands"), failure();
    auto lhs = LookupValue(valueMap, callOp.getOperand(0));
    auto rhs = LookupValue(valueMap, callOp.getOperand(1));
    if (!lhs || !rhs)
      return callOp.emitError("missing copysign operand"), failure();
    auto result = math::CopySignOp::create(builder, loc, *lhs, *rhs);
    valueMap[callOp->getResult(0)] = result.getResult();
    return true;
  }
  return callOp.emitError("unsupported intrinsic"), failure();
}

} // namespace loom::llvm_to_scf
