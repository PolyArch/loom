//===-- LLVMToSCFFunction.cpp - LLVM function conversion ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This file implements conversion of LLVM functions into SCF, memref, arith,
// and func dialects, using shared helper utilities.
//
//===----------------------------------------------------------------------===//

#include "loom/Conversion/LLVMToSCF.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace loom::llvm_to_scf {

LogicalResult convertFunction(ModuleOp module, LLVM::LLVMFuncOp func,
                              OpBuilder &builder,
                              llvm::StringMap<ConvertedGlobal> &globals,
                              const llvm::StringSet<> &varargFunctions) {
  StdMinMaxKind minMaxKind = StdMinMaxKind::Minimum;
  Type funcReturnType = func.getFunctionType().getReturnType();
  if (!llvm::isa<LLVM::LLVMVoidType>(funcReturnType) &&
      IsPointerType(funcReturnType) &&
      IsStdMinMaxName(func.getName(), minMaxKind)) {
    func.erase();
    return success();
  }
  if (func.isExternal() && !llvm::isa<LLVM::LLVMVoidType>(funcReturnType) &&
      IsPointerType(funcReturnType)) {
    func.erase();
    return success();
  }

  std::string originalName = func.getName().str();
  std::string renamedName = originalName + ".llvm";
  func.setSymName(renamedName);

  SmallVector<Type, 8> argTypes;
  SmallVector<std::optional<Type>, 8> argElemTypes;
  auto paramTypes = func.getFunctionType().getParams();
  for (size_t index = 0; index < paramTypes.size(); ++index) {
    Type paramType = paramTypes[index];
    if (IsPointerType(paramType)) {
      std::optional<Type> elemType =
          func.isExternal()
              ? std::nullopt
              : InferPointerElementType(func.getArgument(index));
      std::optional<Type> callSiteType =
          InferPointerElementTypeFromCallSites(func, index);
      if (!elemType || IsI8Type(*elemType)) {
        if (callSiteType)
          elemType = callSiteType;
      }
      if (!elemType)
        elemType = IntegerType::get(module.getContext(), 8);
      argTypes.push_back(MakeStridedMemRefType(*elemType));
      argElemTypes.push_back(*elemType);
      continue;
    }
    argTypes.push_back(
        NormalizeScalarType(paramType, module.getContext()));
    argElemTypes.push_back(std::nullopt);
  }

  SmallVector<Type, 4> resultTypes;
  for (Type type : func.getFunctionType().getReturnTypes()) {
    if (llvm::isa<LLVM::LLVMVoidType>(type))
      continue;
    if (IsPointerType(type)) {
      func.emitError("pointer return types are not supported");
      return failure();
    }
    resultTypes.push_back(NormalizeScalarType(type, module.getContext()));
  }

  auto funcType = builder.getFunctionType(argTypes, resultTypes);
  builder.setInsertionPoint(func);
  func::FuncOp newFunc =
      func::FuncOp::create(builder, func.getLoc(), originalName, funcType);
  newFunc.setSymVisibilityAttr(func.getSymVisibilityAttr());
  CopyLoomAnnotations(func.getOperation(), newFunc.getOperation());
  for (unsigned i = 0, e = func.getNumArguments(); i < e; ++i) {
    if (func.getArgAttr(i, "llvm.noalias"))
      newFunc.setArgAttr(i, "loom.noalias", builder.getUnitAttr());
  }
  if (func.isExternal()) {
    newFunc.setSymVisibilityAttr(builder.getStringAttr("private"));
    func.erase();
    return success();
  }

  DenseMap<Value, Value> valueMap;
  VectorMapT vectorMap;
  DenseMap<Value, PointerInfo> pointerMap;
  DenseMap<Value, PointerInfo> pointerSlotValues;
  llvm::DenseSet<Value> pointerSlots;
  DenseMap<Value, ArrayAttr> pointerSlotAnnotations;
  DenseMap<unsigned, SmallVector<StringAttr, 4>> argAnnotations;
  DenseMap<Block *, Block *> blockMap;

  for (Block &oldBlock : func.getBody()) {
    Block *newBlock = new Block();
    newFunc.getBody().push_back(newBlock);
    blockMap[&oldBlock] = newBlock;
    OpBuilder blockBuilder(module.getContext());
    blockBuilder.setInsertionPointToStart(newBlock);
    Value zeroIndex =
        BuildIndexConstant(blockBuilder, func.getLoc(), 0);
    bool isEntryBlock = (&oldBlock == &func.getBody().front());
    for (BlockArgument arg : oldBlock.getArguments()) {
      Type oldType = arg.getType();
      if (IsPointerType(oldType)) {
        std::optional<Type> elemType;
        if (isEntryBlock && arg.getArgNumber() < argElemTypes.size())
          elemType = argElemTypes[arg.getArgNumber()];
        if (!elemType)
          elemType = InferPointerElementType(arg);
        if (!elemType)
          elemType = IntegerType::get(module.getContext(), 8);
        auto memrefType = MakeStridedMemRefType(*elemType);
        auto newArg = newBlock->addArgument(memrefType, arg.getLoc());
        PointerInfo info{newArg, zeroIndex, *elemType};
        pointerMap[arg] = info;
        continue;
      }
      auto newArg = newBlock->addArgument(
          NormalizeScalarType(oldType, module.getContext()), arg.getLoc());
      valueMap[arg] = newArg;
    }
  }

  SmallVector<Block *, 16> blockOrder;
  llvm::DenseSet<Block *> visitedBlocks;
  auto dfs = [&](auto &&self, Block *block) -> void {
    if (!block || visitedBlocks.contains(block))
      return;
    visitedBlocks.insert(block);
    if (auto *term = block->getTerminator()) {
      for (Block *succ : term->getSuccessors())
        self(self, succ);
    }
    blockOrder.push_back(block);
  };
  dfs(dfs, &func.getBody().front());
  std::reverse(blockOrder.begin(), blockOrder.end());
  for (Block &oldBlock : func.getBody()) {
    if (!visitedBlocks.contains(&oldBlock))
      blockOrder.push_back(&oldBlock);
  }

  for (Block *oldBlock : blockOrder) {
    Block *newBlock = blockMap[oldBlock];
    builder.setInsertionPointToStart(newBlock);

    for (Operation &op : oldBlock->getOperations()) {
      builder.setInsertionPointToEnd(newBlock);
      Location loc = op.getLoc();

      if (auto constOp = llvm::dyn_cast<LLVM::ConstantOp>(op)) {
        if (auto denseAttr =
                llvm::dyn_cast<DenseElementsAttr>(constOp.getValue())) {
          auto lanes = ScalarizeDenseConstant(builder, loc, denseAttr);
          if (lanes.empty())
            return constOp.emitError("unsupported vector constant"), failure();
          vectorMap[constOp.getResult()] = std::move(lanes);
          continue;
        }
        auto value = ConvertLLVMConstant(builder, constOp);
        if (!value)
          return constOp.emitError("unsupported constant"), failure();
        valueMap[constOp.getResult()] = *value;
        continue;
      }

      if (auto zeroOp = llvm::dyn_cast<LLVM::ZeroOp>(op)) {
        Type type = zeroOp.getType();
        if (IsPointerType(type)) {
          auto zeroInt = arith::ConstantOp::create(builder,
              loc, builder.getI64Type(),
              builder.getIntegerAttr(builder.getI64Type(), 0));
          auto ptr = LLVM::IntToPtrOp::create(builder, loc, type,
                                                      zeroInt.getResult());
          valueMap[zeroOp.getResult()] = ptr.getResult();
          continue;
        }
        if (auto intTy = llvm::dyn_cast<IntegerType>(type)) {
          auto attr = builder.getIntegerAttr(intTy, 0);
          auto zero = arith::ConstantOp::create(builder, loc, attr);
          valueMap[zeroOp.getResult()] = zero.getResult();
          continue;
        }
        if (auto floatTy = llvm::dyn_cast<FloatType>(type)) {
          auto attr = builder.getFloatAttr(floatTy, 0.0);
          auto zero = arith::ConstantOp::create(builder, loc, attr);
          valueMap[zeroOp.getResult()] = zero.getResult();
          continue;
        }
        return zeroOp.emitError("unsupported zero type"), failure();
      }

      if (auto poisonOp = llvm::dyn_cast<LLVM::PoisonOp>(op)) {
        if (auto vecTy =
                llvm::dyn_cast<VectorType>(poisonOp.getType())) {
          Type elemTy = vecTy.getElementType();
          int64_t n = vecTy.getNumElements();
          SmallVector<Value, 8> lanes;
          for (int64_t i = 0; i < n; ++i)
            lanes.push_back(ub::PoisonOp::create(builder, loc, elemTy));
          vectorMap[poisonOp.getResult()] = std::move(lanes);
          continue;
        }
        auto poison = ub::PoisonOp::create(builder, loc, poisonOp.getType());
        valueMap[poisonOp.getResult()] = poison.getResult();
        continue;
      }

      if (auto undefOp = llvm::dyn_cast<LLVM::UndefOp>(op)) {
        if (auto vecTy =
                llvm::dyn_cast<VectorType>(undefOp.getType())) {
          Type elemTy = vecTy.getElementType();
          int64_t n = vecTy.getNumElements();
          SmallVector<Value, 8> lanes;
          for (int64_t i = 0; i < n; ++i)
            lanes.push_back(ub::PoisonOp::create(builder, loc, elemTy));
          vectorMap[undefOp.getResult()] = std::move(lanes);
          continue;
        }
        auto poison = ub::PoisonOp::create(builder, loc, undefOp.getType());
        valueMap[undefOp.getResult()] = poison.getResult();
        continue;
      }

      if (auto insertOp = llvm::dyn_cast<LLVM::InsertElementOp>(op)) {
        auto *srcLanes = LookupVector(vectorMap, insertOp.getVector());
        if (!srcLanes)
          return insertOp.emitError("missing insertelement vector"), failure();
        auto scalarVal = LookupValue(valueMap, insertOp.getValue());
        if (!scalarVal)
          return insertOp.emitError("missing insertelement value"), failure();
        auto posVal = LookupValue(valueMap, insertOp.getPosition());
        if (!posVal)
          return insertOp.emitError("missing insertelement index"), failure();
        auto idx = GetConstantIntValue(*posVal);
        if (!idx)
          return insertOp.emitError("non-constant insertelement index"),
                 failure();
        SmallVector<Value, 8> lanes(*srcLanes);
        lanes[*idx] = *scalarVal;
        vectorMap[insertOp.getResult()] = std::move(lanes);
        continue;
      }

      if (auto extractOp = llvm::dyn_cast<LLVM::ExtractElementOp>(op)) {
        auto *srcLanes = LookupVector(vectorMap, extractOp.getVector());
        if (!srcLanes)
          return extractOp.emitError("missing extractelement vector"),
                 failure();
        auto posVal = LookupValue(valueMap, extractOp.getPosition());
        if (!posVal)
          return extractOp.emitError("missing extractelement index"), failure();
        auto idx = GetConstantIntValue(*posVal);
        if (!idx)
          return extractOp.emitError("non-constant extractelement index"),
                 failure();
        valueMap[extractOp.getResult()] = (*srcLanes)[*idx];
        continue;
      }

      if (auto shuffleOp = llvm::dyn_cast<LLVM::ShuffleVectorOp>(op)) {
        auto *v1Lanes = LookupVector(vectorMap, shuffleOp.getV1());
        auto *v2Lanes = LookupVector(vectorMap, shuffleOp.getV2());
        if (!v1Lanes || !v2Lanes)
          return shuffleOp.emitError("missing shufflevector operand"),
                 failure();
        SmallVector<Value, 16> pool;
        pool.append(v1Lanes->begin(), v1Lanes->end());
        pool.append(v2Lanes->begin(), v2Lanes->end());
        SmallVector<Value, 8> outLanes;
        for (int32_t maskIdx : shuffleOp.getMask()) {
          if (maskIdx < 0 || static_cast<size_t>(maskIdx) >= pool.size())
            return shuffleOp.emitError("shufflevector mask out of range"),
                   failure();
          outLanes.push_back(pool[maskIdx]);
        }
        vectorMap[shuffleOp.getResult()] = std::move(outLanes);
        continue;
      }

      if (auto addrOp = llvm::dyn_cast<LLVM::AddressOfOp>(op)) {
        auto it = globals.find(addrOp.getGlobalName());
        if (it == globals.end())
          return addrOp.emitError("unknown global"), failure();
        auto memrefType = it->second.type;
        auto getGlobal = memref::GetGlobalOp::create(builder,
            loc, memrefType, addrOp.getGlobalName());
        CopyLoomAnnotations(it->second.oldGlobal.getOperation(),
                            getGlobal.getOperation());
        PointerInfo info{getGlobal, BuildIndexConstant(builder, loc, 0),
                         memrefType.getElementType()};
        pointerMap[addrOp.getResult()] = info;
        continue;
      }

      if (auto allocaOp = llvm::dyn_cast<LLVM::AllocaOp>(op)) {
        if (IsPointerType(allocaOp.getElemType())) {
          pointerSlots.insert(allocaOp.getResult());
          if (auto ann =
                  allocaOp->getAttrOfType<ArrayAttr>("loom.annotations")) {
            pointerSlotAnnotations[allocaOp.getResult()] = ann;
          }
          continue;
        }
        SmallVector<int64_t, 4> dims;
        Type scalar = GetScalarType(allocaOp.getElemType(), dims);
        scalar = NormalizeScalarType(scalar, module.getContext());
        if (!llvm::isa<IntegerType>(scalar) &&
            !llvm::isa<FloatType>(scalar)) {
          return allocaOp.emitError("unsupported alloca element type"),
                 failure();
        }
        int64_t elementCount = 1;
        for (int64_t dim : dims)
          elementCount *= dim;
        Value countValue = BuildIndexConstant(builder, loc, elementCount);
        std::optional<int64_t> staticCount = elementCount;
        if (Value size = allocaOp.getArraySize()) {
          auto mappedSize = LookupValue(valueMap, size);
          if (!mappedSize)
            return allocaOp.emitError("missing alloca size"), failure();
          if (auto constSize = GetConstantIntValue(*mappedSize)) {
            if (*constSize <= 0) {
              staticCount = std::nullopt;
            } else if (staticCount) {
              staticCount = *staticCount * (*constSize);
            }
          } else {
            staticCount = std::nullopt;
          }
          Value sizeIndex = ToIndexValue(builder, loc, *mappedSize);
          if (!sizeIndex)
            return allocaOp.emitError("invalid alloca size"), failure();
          countValue = arith::MulIOp::create(builder, loc, sizeIndex, countValue);
        }
        MemRefType memrefType;
        memref::AllocaOp alloc;
        if (staticCount && *staticCount > 0) {
          memrefType =
              MemRefType::get({*staticCount}, scalar,
                              MemRefLayoutAttrInterface(), Attribute());
          alloc = memref::AllocaOp::create(builder, loc, memrefType);
        } else {
          memrefType = MakeMemRefType(scalar);
          alloc = memref::AllocaOp::create(builder, loc, memrefType,
                                                    ValueRange{countValue});
        }
        PointerInfo info{alloc, BuildIndexConstant(builder, loc, 0), scalar};
        pointerMap[allocaOp.getResult()] = info;
        continue;
      }

      if (auto gepOp = llvm::dyn_cast<LLVM::GEPOp>(op)) {
        auto baseInfo = LookupPointer(pointerMap, gepOp.getBase());
        if (!baseInfo)
          return gepOp.emitError("missing GEP base"), failure();

        SmallVector<int64_t, 4> dims;
        Type scalar = GetScalarType(gepOp.getElemType(), dims);
        scalar = NormalizeScalarType(scalar, module.getContext());
        if (!llvm::isa<IntegerType>(scalar) &&
            !llvm::isa<FloatType>(scalar))
          return gepOp.emitError("unsupported GEP element type"), failure();

        LLVM::GEPIndicesAdaptor<ValueRange> indicesAdaptor(
            gepOp.getRawConstantIndicesAttr(), gepOp.getDynamicIndices());
        SmallVector<Value, 4> indices;
        indices.reserve(indicesAdaptor.size());
        for (auto item : indicesAdaptor) {
          if (auto attr = llvm::dyn_cast<IntegerAttr>(item)) {
            indices.push_back(BuildIndexConstant(builder, loc, attr.getInt()));
            continue;
          }
          if (auto val = llvm::dyn_cast<Value>(item)) {
            auto mapped = LookupValue(valueMap, val);
            if (!mapped)
              return gepOp.emitError("missing GEP index"), failure();
            Value idx = ToIndexValue(builder, loc, *mapped);
            if (!idx)
              return gepOp.emitError("invalid GEP index type"), failure();
            indices.push_back(idx);
            continue;
          }
          return gepOp.emitError("unsupported GEP index"), failure();
        }

        if (indices.size() > dims.size() + 1)
          return gepOp.emitError("unsupported GEP rank"), failure();

        int64_t totalStride = 1;
        for (int64_t dim : dims)
          totalStride *= dim;

        Value offset = nullptr;
        auto addTerm = [&](Value term) {
          if (!offset)
            offset = term;
          else
            offset = arith::AddIOp::create(builder, loc, offset, term);
        };

        if (!indices.empty()) {
          Value stride0 = BuildIndexConstant(builder, loc, totalStride);
          Value term = indices[0];
          if (totalStride != 1)
            term = arith::MulIOp::create(builder, loc, term, stride0);
          addTerm(term);
        }

        for (size_t i = 1; i < indices.size(); ++i) {
          int64_t stride = 1;
          if (i - 1 < dims.size()) {
            for (size_t j = i; j < dims.size(); ++j)
              stride *= dims[j];
          }
          Value term = indices[i];
          if (stride != 1) {
            Value strideVal = BuildIndexConstant(builder, loc, stride);
            term = arith::MulIOp::create(builder, loc, term, strideVal);
          }
          addTerm(term);
        }

        if (!offset)
          offset = BuildIndexConstant(builder, loc, 0);

        Value baseIndex =
            ScaleIndexBetweenElementTypes(builder, loc, baseInfo->index,
                                          baseInfo->elementType, scalar);
        Value newIndex = baseIndex;
        if (!IsZeroIndex(offset))
          newIndex = arith::AddIOp::create(builder, loc, baseIndex, offset);

        PointerInfo info{baseInfo->base, newIndex, scalar};
        pointerMap[gepOp.getResult()] = info;
        continue;
      }

      if (auto bitcastOp = llvm::dyn_cast<LLVM::BitcastOp>(op)) {
        if (IsPointerType(bitcastOp.getType())) {
          auto srcInfo = LookupPointer(pointerMap, bitcastOp.getArg());
          if (!srcInfo)
            return bitcastOp.emitError("missing bitcast source"), failure();
          Type targetElem = srcInfo->elementType;
          if (auto inferred = InferPointerElementType(bitcastOp.getResult()))
            targetElem = *inferred;
          PointerInfo info = *srcInfo;
          if (targetElem != srcInfo->elementType) {
            info.index = ScaleIndexBetweenElementTypes(
                builder, loc, srcInfo->index, srcInfo->elementType,
                targetElem);
            info.elementType = targetElem;
          }
          pointerMap[bitcastOp.getResult()] = info;
          continue;
        }
        // vector<NxiK> -> iM bitcast: pack N i-K lanes into one iM integer
        auto srcVecTy =
            llvm::dyn_cast<VectorType>(bitcastOp.getArg().getType());
        auto dstIntTy =
            llvm::dyn_cast<IntegerType>(bitcastOp.getType());
        if (srcVecTy && dstIntTy) {
          auto *lanes = LookupVector(vectorMap, bitcastOp.getArg());
          if (!lanes)
            return bitcastOp.emitError("missing vector bitcast source"),
                   failure();
          unsigned laneWidth =
              srcVecTy.getElementType().getIntOrFloatBitWidth();
          Value result = arith::ConstantOp::create(builder, loc,
              builder.getIntegerAttr(dstIntTy, 0));
          for (size_t i = 0, e = lanes->size(); i < e; ++i) {
            Value lane = (*lanes)[i];
            if (lane.getType() != dstIntTy)
              lane = arith::ExtUIOp::create(builder, loc, dstIntTy, lane);
            if (i > 0) {
              Value shift = arith::ConstantOp::create(builder, loc,
                  builder.getIntegerAttr(dstIntTy, i * laneWidth));
              lane = arith::ShLIOp::create(builder, loc, lane, shift);
            }
            result = arith::OrIOp::create(builder, loc, result, lane);
          }
          valueMap[bitcastOp.getResult()] = result;
          continue;
        }
      }

      if (auto freezeOp = llvm::dyn_cast<LLVM::FreezeOp>(op)) {
        // freeze is a no-op for hardware: pass through the value
        auto src = LookupValue(valueMap, freezeOp.getVal());
        if (!src)
          return freezeOp.emitError("missing freeze operand"), failure();
        valueMap[freezeOp.getResult()] = *src;
        continue;
      }

      if (auto loadOp = llvm::dyn_cast<LLVM::LoadOp>(op)) {
        if (pointerSlots.contains(loadOp.getAddr())) {
          auto it = pointerSlotValues.find(loadOp.getAddr());
          if (it == pointerSlotValues.end())
            return loadOp.emitError("missing pointer slot value"), failure();
          pointerMap[loadOp.getResult()] = it->second;
          continue;
        }
        auto ptrInfo = LookupPointer(pointerMap, loadOp.getAddr());
        if (!ptrInfo)
          return loadOp.emitError("missing load pointer for address ")
                     << loadOp.getAddr(),
                 failure();
        Type accessType =
            NormalizeScalarType(loadOp.getResult().getType(),
                                module.getContext());
        Value index = ptrInfo->index;
        if (ptrInfo->elementType != accessType) {
          index = ScaleIndexBetweenElementTypes(builder, loc, ptrInfo->index,
                                                ptrInfo->elementType,
                                                accessType);
        }
        Value val =
            memref::LoadOp::create(builder, loc, ptrInfo->base, index);
        valueMap[loadOp.getResult()] = val;
        continue;
      }

      if (auto storeOp = llvm::dyn_cast<LLVM::StoreOp>(op)) {
        if (pointerSlots.contains(storeOp.getAddr())) {
          auto storedPtr = LookupPointer(pointerMap, storeOp.getValue());
          if (!storedPtr)
            return storeOp.emitError("missing pointer store value"), failure();
          auto annIt = pointerSlotAnnotations.find(storeOp.getAddr());
          if (annIt != pointerSlotAnnotations.end()) {
            if (auto arg = llvm::dyn_cast<BlockArgument>(storeOp.getValue())) {
              if (arg.getOwner() == &func.getBody().front()) {
                MergeLoomAnnotationList(argAnnotations[arg.getArgNumber()],
                                        annIt->second);
              }
            }
          }
          pointerSlotValues[storeOp.getAddr()] = *storedPtr;
          continue;
        }
        auto ptrInfo = LookupPointer(pointerMap, storeOp.getAddr());
        if (!ptrInfo)
          return storeOp.emitError("missing store pointer"), failure();
        auto storedVal = LookupValue(valueMap, storeOp.getValue());
        if (!storedVal)
          return storeOp.emitError("missing store value"), failure();
        Type accessType = storedVal->getType();
        Value index = ptrInfo->index;
        if (ptrInfo->elementType != accessType) {
          index = ScaleIndexBetweenElementTypes(builder, loc, ptrInfo->index,
                                                ptrInfo->elementType,
                                                accessType);
        }
        memref::StoreOp::create(builder, loc, *storedVal, ptrInfo->base,
                                        index);
        continue;
      }

      if (auto callOp = llvm::dyn_cast<LLVM::CallOp>(op)) {
        auto calleeAttr = callOp.getCalleeAttr();
        if (!calleeAttr)
          return callOp.emitError("indirect calls are not supported"),
                 failure();
        StringRef callee = calleeAttr.getValue();
        bool returnsPointer =
            callOp.getNumResults() == 1 &&
            IsPointerType(callOp.getResults().front().getType());
        if (returnsPointer) {
          StdMinMaxKind minMaxKind = StdMinMaxKind::Minimum;
          if (IsStdMinMaxName(callee, minMaxKind)) {
            if (callOp.getNumOperands() != 2)
              return callOp.emitError("std::min/max expects 2 operands"),
                     failure();
            auto lhsPtr = LookupPointer(pointerMap, callOp.getOperand(0));
            auto rhsPtr = LookupPointer(pointerMap, callOp.getOperand(1));
            if (!lhsPtr || !rhsPtr)
              return callOp.emitError("missing std::min/max pointer operand"),
                     failure();
            Value lhsVal = memref::LoadOp::create(builder,
                loc, lhsPtr->base, lhsPtr->index);
            Value rhsVal = memref::LoadOp::create(builder,
                loc, rhsPtr->base, rhsPtr->index);
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
              cmp = arith::CmpFOp::create(builder,
                  loc, arith::CmpFPredicate::OLT, cmpLhs, cmpRhs);
            } else if (llvm::isa<IntegerType>(lhsVal.getType())) {
              bool isUnsigned =
                  scalarKind == StdMinMaxScalarKind::UnsignedIntKind;
              bool isSigned = scalarKind == StdMinMaxScalarKind::SignedIntKind;
              if (!isUnsigned && !isSigned)
                return callOp.emitError(
                           "std::min/max integer operand kind unknown"),
                       failure();
              arith::CmpIPredicate pred =
                  isUnsigned ? arith::CmpIPredicate::ult
                             : arith::CmpIPredicate::slt;
              Value cmpLhs =
                  (minMaxKind == StdMinMaxKind::Minimum) ? rhsVal : lhsVal;
              Value cmpRhs =
                  (minMaxKind == StdMinMaxKind::Minimum) ? lhsVal : rhsVal;
              cmp = arith::CmpIOp::create(builder, loc, pred, cmpLhs, cmpRhs);
            } else {
              return callOp.emitError(
                         "std::min/max unsupported operand type"),
                     failure();
            }
            auto selected = BuildPointerSelect(builder, loc, cmp, *lhsPtr,
                                               *rhsPtr, true);
            if (!selected)
              return callOp.emitError(
                         "std::min/max pointer select failed"),
                     failure();
            pointerMap[callOp.getResults().front()] = *selected;
            if (auto def = selected->base.getDefiningOp())
              CopyLoomAnnotations(&op, def);
            if (auto def = selected->index.getDefiningOp())
              CopyLoomAnnotations(&op, def);
            continue;
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
              return callOp.emitError("unexpected math call results"),
                     failure();
            valueMap[callOp->getResult(0)] = *mathResult;
            CopyLoomAnnotations(&op, mathResult->getDefiningOp());
            continue;
          }
          if (auto minMaxResult = ConvertStdMinMaxScalarCall(
                  builder, loc, callee, mappedOperands)) {
            if (callOp.getNumResults() != 1)
              return callOp.emitError("unexpected min/max call results"),
                     failure();
            valueMap[callOp->getResult(0)] = *minMaxResult;
            CopyLoomAnnotations(&op, minMaxResult->getDefiningOp());
            continue;
          }
        }

        bool isVarArg = varargFunctions.contains(callee);
        bool isRawPtrCall = IsRawPointerCallee(callee);
        if (returnsPointer)
          return callOp.emitError("unsupported pointer return call"),
                 failure();
        SmallVector<Value, 8> operands;
        operands.reserve(callOp.getNumOperands());
        for (Value operand : callOp.getOperands()) {
          if (auto ptrInfo = LookupPointer(pointerMap, operand)) {
            if (isVarArg || isRawPtrCall) {
              operands.push_back(MaterializeLLVMPointer(builder, loc, *ptrInfo));
            } else {
              operands.push_back(MaterializeMemrefPointer(builder, loc, *ptrInfo));
            }
            continue;
          }
          auto mapped = LookupValue(valueMap, operand);
          if (!mapped)
            return callOp.emitError("missing call operand"), failure();
          operands.push_back(*mapped);
        }

        if (isVarArg || isRawPtrCall) {
          auto newCall = LLVM::CallOp::create(builder, loc, callOp.getResultTypes(),
                                                      calleeAttr, operands);
          if (auto varType = callOp.getVarCalleeType())
            newCall.setVarCalleeType(varType);
          CopyLoomAnnotations(&op, newCall.getOperation());
          for (auto [oldRes, newRes] :
               llvm::zip(callOp.getResults(), newCall.getResults()))
            valueMap[oldRes] = newRes;
          continue;
        }

        SmallVector<Type, 4> resultTypes;
        for (Type type : callOp.getResultTypes())
          resultTypes.push_back(NormalizeScalarType(type, module.getContext()));
        auto call = func::CallOp::create(builder, loc, callee,
                                                 resultTypes, operands);
        CopyLoomAnnotations(&op, call.getOperation());
        for (auto [oldRes, newRes] :
             llvm::zip(callOp.getResults(), call.getResults()))
          valueMap[oldRes] = newRes;
        continue;
      }

      if (auto asmOp = llvm::dyn_cast<LLVM::InlineAsmOp>(op)) {
        if (asmOp.getNumResults() == 0)
          continue;
        if (asmOp.getNumResults() == 1) {
          Type resultType = asmOp.getResult(0).getType();
          if (llvm::isa<LLVM::LLVMVoidType>(resultType))
            continue;
        }
        return asmOp.emitError("unsupported inline asm result"), failure();
      }

      if (auto callOp = llvm::dyn_cast<LLVM::CallIntrinsicOp>(op)) {
        StringRef callee = callOp.getIntrin();
        if (callee.starts_with("llvm.var.annotation") ||
            callee.starts_with("llvm.ptr.annotation") ||
            callee.starts_with("llvm.annotation")) {
          continue;
        }
        if (callee.starts_with("llvm.lifetime.start") ||
            callee.starts_with("llvm.lifetime.end")) {
          continue;
        }
        if (callee.starts_with("llvm.stacksave")) {
          if (callOp.getNumResults() != 1)
            return callOp.emitError("unexpected stacksave results"), failure();
          auto memrefType = MakeMemRefType(builder.getI8Type());
          Value one = BuildIndexConstant(builder, loc, 1);
          auto alloc =
              memref::AllocaOp::create(builder, loc, memrefType,
                                               ValueRange{one});
          PointerInfo info{alloc, BuildIndexConstant(builder, loc, 0),
                           builder.getI8Type()};
          pointerMap[callOp->getResult(0)] = info;
          continue;
        }
        if (callee.starts_with("llvm.stackrestore")) {
          continue;
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
            return callOp.emitError("unsupported memcpy element type"),
                   failure();
          Value sizeIndex = ToIndexValue(builder, loc, *sizeVal);
          if (!sizeIndex)
            return callOp.emitError("invalid memcpy size"), failure();
          Value elemSizeVal = BuildIndexConstant(builder, loc, elemSize);
          Value length = arith::DivUIOp::create(builder, loc, sizeIndex,
                                                        elemSizeVal);
          Value zero = BuildIndexConstant(builder, loc, 0);
          Value one = BuildIndexConstant(builder, loc, 1);
          auto loop = scf::ForOp::create(builder, loc, zero, length, one);
          builder.setInsertionPointToStart(loop.getBody());
          Value iv = loop.getInductionVar();
          Value srcIndex = iv;
          if (!IsZeroIndex(srcInfo->index))
            srcIndex =
                arith::AddIOp::create(builder, loc, srcInfo->index, iv);
          Value dstIndex = iv;
          if (!IsZeroIndex(dstInfo->index))
            dstIndex =
                arith::AddIOp::create(builder, loc, dstInfo->index, iv);
          Value val =
              memref::LoadOp::create(builder, loc, srcInfo->base, srcIndex);
          memref::StoreOp::create(builder, loc, val, dstInfo->base, dstIndex);
          builder.setInsertionPointAfter(loop);
          continue;
        }
        if (callee.starts_with("llvm.bswap")) {
          if (callOp.getNumOperands() != 1)
            return callOp.emitError("invalid bswap operands"), failure();
          auto operand = LookupValue(valueMap, callOp.getOperand(0));
          if (!operand)
            return callOp.emitError("missing bswap operand"), failure();
          auto intTy = llvm::dyn_cast<IntegerType>(operand->getType());
          if (!intTy)
            return callOp.emitError("bswap expects integer operand"),
                   failure();
          unsigned bitWidth = intTy.getWidth();
          if (bitWidth % 8 != 0)
            return callOp.emitError("bswap expects byte-multiple width"),
                   failure();
          unsigned byteCount = bitWidth / 8;
          Value result;
          for (unsigned i = 0; i < byteCount; ++i) {
            unsigned inShift = i * 8;
            unsigned outShift = (byteCount - 1 - i) * 8;
            auto inShiftVal = arith::ConstantOp::create(builder,
                loc, builder.getIntegerAttr(intTy, inShift));
            auto outShiftVal = arith::ConstantOp::create(builder,
                loc, builder.getIntegerAttr(intTy, outShift));
            auto byteMask = arith::ConstantOp::create(builder,
                loc, builder.getIntegerAttr(intTy, 0xFF));
            Value shiftedIn = arith::ShRUIOp::create(builder,
                loc, *operand, inShiftVal);
            Value masked =
                arith::AndIOp::create(builder, loc, shiftedIn, byteMask);
            Value shiftedOut =
                arith::ShLIOp::create(builder, loc, masked, outShiftVal);
            if (result)
              result =
                  arith::OrIOp::create(builder, loc, result, shiftedOut);
            else
              result = shiftedOut;
          }
          valueMap[callOp->getResult(0)] = result;
          continue;
        }
        if (callee.starts_with("llvm.umin")) {
          if (callOp.getNumOperands() != 2)
            return callOp.emitError("invalid umin operands"), failure();
          auto lhs = LookupValue(valueMap, callOp.getOperand(0));
          auto rhs = LookupValue(valueMap, callOp.getOperand(1));
          if (!lhs || !rhs)
            return callOp.emitError("missing umin operand"), failure();
          auto cmp = arith::CmpIOp::create(builder,
              loc, arith::CmpIPredicate::ult, *lhs, *rhs);
          auto sel =
              arith::SelectOp::create(builder, loc, cmp, *lhs, *rhs);
          valueMap[callOp->getResult(0)] = sel.getResult();
          continue;
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
            return callOp.emitError("usub.sat expects integer operands"),
                   failure();
          auto zero = arith::ConstantOp::create(builder,
              loc, builder.getIntegerAttr(intTy, 0));
          auto cmp = arith::CmpIOp::create(builder,
              loc, arith::CmpIPredicate::ult, *lhs, *rhs);
          auto sub = arith::SubIOp::create(builder, loc, *lhs, *rhs);
          auto sel =
              arith::SelectOp::create(builder, loc, cmp, zero, sub);
          valueMap[callOp->getResult(0)] = sel.getResult();
          continue;
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
            return callOp.emitError("unsupported memset element type"),
                   failure();
          Value sizeIndex = ToIndexValue(builder, loc, *sizeVal);
          if (!sizeIndex)
            return callOp.emitError("invalid memset size"), failure();
          Value elemSizeVal = BuildIndexConstant(builder, loc, elemSize);
          Value length = arith::DivUIOp::create(builder, loc, sizeIndex,
                                                        elemSizeVal);
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
            dstIndex =
                arith::AddIOp::create(builder, loc, dstInfo->index, iv);
          memref::StoreOp::create(builder, loc, *fillVal, dstInfo->base,
                                          dstIndex);
          builder.setInsertionPointAfter(loop);
          continue;
        }
        if (callee.starts_with("llvm.fmuladd")) {
          if (callOp.getNumOperands() != 3)
            return callOp.emitError("invalid fmuladd operands"), failure();
          // Vector fmuladd: per-lane math.fma
          auto *lhsL = LookupVector(vectorMap, callOp.getOperand(0));
          if (lhsL) {
            auto *rhsL = LookupVector(vectorMap, callOp.getOperand(1));
            auto *addL = LookupVector(vectorMap, callOp.getOperand(2));
            if (!rhsL || !addL)
              return callOp.emitError("missing vector fmuladd operand"),
                     failure();
            SmallVector<Value, 8> out;
            for (size_t i = 0, e = lhsL->size(); i < e; ++i)
              out.push_back(math::FmaOp::create(
                  builder, loc, (*lhsL)[i], (*rhsL)[i], (*addL)[i]));
            vectorMap[callOp->getResult(0)] = std::move(out);
            continue;
          }
          auto lhs = LookupValue(valueMap, callOp.getOperand(0));
          auto rhs = LookupValue(valueMap, callOp.getOperand(1));
          auto addend = LookupValue(valueMap, callOp.getOperand(2));
          if (!lhs || !rhs || !addend)
            return callOp.emitError("missing fmuladd operand"), failure();
          if (!llvm::isa<FloatType>(lhs->getType()) ||
              !llvm::isa<FloatType>(rhs->getType()) ||
              !llvm::isa<FloatType>(addend->getType()))
            return callOp.emitError("fmuladd expects float operands"),
                   failure();
          auto fma =
              math::FmaOp::create(builder, loc, *lhs, *rhs, *addend);
          valueMap[callOp->getResult(0)] = fma.getResult();
          continue;
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
            return callOp.emitError("fshl expects integer operands"),
                   failure();
          auto width = arith::ConstantOp::create(builder,
              loc, builder.getIntegerAttr(intTy, intTy.getWidth()));
          auto zero = arith::ConstantOp::create(builder,
              loc, builder.getIntegerAttr(intTy, 0));
          Value shiftMod =
              arith::RemUIOp::create(builder, loc, *shift, width);
          auto isZero = arith::CmpIOp::create(builder,
              loc, arith::CmpIPredicate::eq, shiftMod, zero);
          OpBuilder::InsertionGuard guard(builder);
          auto ifOp =
              scf::IfOp::create(builder, loc, TypeRange{intTy}, isZero,
                                        /*withElseRegion=*/true);
          builder.setInsertionPointToStart(ifOp.thenBlock());
          scf::YieldOp::create(builder, loc, *lhs);
          builder.setInsertionPointToStart(ifOp.elseBlock());
          Value left = arith::ShLIOp::create(builder, loc, *lhs, shiftMod);
          Value rightShift =
              arith::SubIOp::create(builder, loc, width, shiftMod);
          Value right =
              arith::ShRUIOp::create(builder, loc, *rhs, rightShift);
          Value combined =
              arith::OrIOp::create(builder, loc, left, right);
          scf::YieldOp::create(builder, loc, combined);
          valueMap[callOp->getResult(0)] = ifOp.getResult(0);
          continue;
        }
        if (callee.starts_with("llvm.fabs")) {
          if (callOp.getNumOperands() != 1)
            return callOp.emitError("invalid fabs operands"), failure();
          // Vector fabs: per-lane math.absf
          auto *srcL = LookupVector(vectorMap, callOp.getOperand(0));
          if (srcL) {
            SmallVector<Value, 8> out;
            for (Value v : *srcL)
              out.push_back(math::AbsFOp::create(builder, loc, v));
            vectorMap[callOp->getResult(0)] = std::move(out);
            continue;
          }
          auto operand = LookupValue(valueMap, callOp.getOperand(0));
          if (!operand)
            return callOp.emitError("missing fabs operand"), failure();
          auto abs = math::AbsFOp::create(builder, loc, *operand);
          valueMap[callOp->getResult(0)] = abs.getResult();
          continue;
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
          continue;
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
          continue;
        }
        if (callee.starts_with("llvm.bitreverse")) {
          if (callOp.getNumOperands() != 1)
            return callOp.emitError("invalid bitreverse operands"), failure();
          auto operand = LookupValue(valueMap, callOp.getOperand(0));
          if (!operand)
            return callOp.emitError("missing bitreverse operand"), failure();
          auto bitrev = LLVM::BitReverseOp::create(builder, loc,
                                                    operand->getType(),
                                                    *operand);
          valueMap[callOp->getResult(0)] = bitrev.getResult();
          continue;
        }
        return callOp.emitError("unsupported intrinsic"), failure();
      }

      // --- Binary ops with vector scalarization support ---
      // Handles both vector (per-lane) and scalar paths for binary ops.
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
        continue;
      }

      if (auto subOp = llvm::dyn_cast<LLVM::SubOp>(op)) {
        if (failed(convertBinaryOp(subOp.getLhs(), subOp.getRhs(),
                subOp.getResult(), "sub",
                [&](Value a, Value b) -> Value {
                  return arith::SubIOp::create(builder, loc, a, b);
                })))
          return failure();
        continue;
      }

      if (auto mulOp = llvm::dyn_cast<LLVM::MulOp>(op)) {
        if (failed(convertBinaryOp(mulOp.getLhs(), mulOp.getRhs(),
                mulOp.getResult(), "mul",
                [&](Value a, Value b) -> Value {
                  return arith::MulIOp::create(builder, loc, a, b);
                })))
          return failure();
        continue;
      }

      if (auto shlOp = llvm::dyn_cast<LLVM::ShlOp>(op)) {
        if (failed(convertBinaryOp(shlOp.getLhs(), shlOp.getRhs(),
                shlOp.getResult(), "shl",
                [&](Value a, Value b) -> Value {
                  return arith::ShLIOp::create(builder, loc, a, b);
                })))
          return failure();
        continue;
      }

      if (auto lshrOp = llvm::dyn_cast<LLVM::LShrOp>(op)) {
        if (failed(convertBinaryOp(lshrOp.getLhs(), lshrOp.getRhs(),
                lshrOp.getResult(), "lshr",
                [&](Value a, Value b) -> Value {
                  return arith::ShRUIOp::create(builder, loc, a, b);
                })))
          return failure();
        continue;
      }

      if (auto ashrOp = llvm::dyn_cast<LLVM::AShrOp>(op)) {
        if (failed(convertBinaryOp(ashrOp.getLhs(), ashrOp.getRhs(),
                ashrOp.getResult(), "ashr",
                [&](Value a, Value b) -> Value {
                  return arith::ShRSIOp::create(builder, loc, a, b);
                })))
          return failure();
        continue;
      }

      if (auto andOp = llvm::dyn_cast<LLVM::AndOp>(op)) {
        if (failed(convertBinaryOp(andOp.getLhs(), andOp.getRhs(),
                andOp.getResult(), "and",
                [&](Value a, Value b) -> Value {
                  return arith::AndIOp::create(builder, loc, a, b);
                })))
          return failure();
        continue;
      }

      if (auto orOp = llvm::dyn_cast<LLVM::OrOp>(op)) {
        if (failed(convertBinaryOp(orOp.getLhs(), orOp.getRhs(),
                orOp.getResult(), "or",
                [&](Value a, Value b) -> Value {
                  return arith::OrIOp::create(builder, loc, a, b);
                })))
          return failure();
        continue;
      }

      if (auto xorOp = llvm::dyn_cast<LLVM::XOrOp>(op)) {
        if (failed(convertBinaryOp(xorOp.getLhs(), xorOp.getRhs(),
                xorOp.getResult(), "xor",
                [&](Value a, Value b) -> Value {
                  return arith::XOrIOp::create(builder, loc, a, b);
                })))
          return failure();
        continue;
      }

      if (auto faddOp = llvm::dyn_cast<LLVM::FAddOp>(op)) {
        if (failed(convertBinaryOp(faddOp.getLhs(), faddOp.getRhs(),
                faddOp.getResult(), "fadd",
                [&](Value a, Value b) -> Value {
                  return arith::AddFOp::create(builder, loc, a, b);
                })))
          return failure();
        continue;
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
          continue;
        }
        auto operand = LookupValue(valueMap, fnegOp.getOperand());
        if (!operand)
          return fnegOp.emitError("missing fneg operand"), failure();
        valueMap[fnegOp.getResult()] = arith::NegFOp::create(builder, loc,
                                                              *operand);
        continue;
      }

      if (auto fsubOp = llvm::dyn_cast<LLVM::FSubOp>(op)) {
        if (failed(convertBinaryOp(fsubOp.getLhs(), fsubOp.getRhs(),
                fsubOp.getResult(), "fsub",
                [&](Value a, Value b) -> Value {
                  return arith::SubFOp::create(builder, loc, a, b);
                })))
          return failure();
        continue;
      }

      if (auto fmulOp = llvm::dyn_cast<LLVM::FMulOp>(op)) {
        if (failed(convertBinaryOp(fmulOp.getLhs(), fmulOp.getRhs(),
                fmulOp.getResult(), "fmul",
                [&](Value a, Value b) -> Value {
                  return arith::MulFOp::create(builder, loc, a, b);
                })))
          return failure();
        continue;
      }

      if (auto fdivOp = llvm::dyn_cast<LLVM::FDivOp>(op)) {
        if (failed(convertBinaryOp(fdivOp.getLhs(), fdivOp.getRhs(),
                fdivOp.getResult(), "fdiv",
                [&](Value a, Value b) -> Value {
                  return arith::DivFOp::create(builder, loc, a, b);
                })))
          return failure();
        continue;
      }

      if (auto sdivOp = llvm::dyn_cast<LLVM::SDivOp>(op)) {
        if (failed(convertBinaryOp(sdivOp.getLhs(), sdivOp.getRhs(),
                sdivOp.getResult(), "sdiv",
                [&](Value a, Value b) -> Value {
                  return arith::DivSIOp::create(builder, loc, a, b);
                })))
          return failure();
        continue;
      }

      if (auto udivOp = llvm::dyn_cast<LLVM::UDivOp>(op)) {
        if (failed(convertBinaryOp(udivOp.getLhs(), udivOp.getRhs(),
                udivOp.getResult(), "udiv",
                [&](Value a, Value b) -> Value {
                  return arith::DivUIOp::create(builder, loc, a, b);
                })))
          return failure();
        continue;
      }

      if (auto uremOp = llvm::dyn_cast<LLVM::URemOp>(op)) {
        if (failed(convertBinaryOp(uremOp.getLhs(), uremOp.getRhs(),
                uremOp.getResult(), "urem",
                [&](Value a, Value b) -> Value {
                  return arith::RemUIOp::create(builder, loc, a, b);
                })))
          return failure();
        continue;
      }

      if (auto icmpOp = llvm::dyn_cast<LLVM::ICmpOp>(op)) {
        auto pred = ConvertICmpPredicate(icmpOp.getPredicate());
        if (failed(convertBinaryOp(icmpOp.getLhs(), icmpOp.getRhs(),
                icmpOp.getResult(), "icmp",
                [&](Value a, Value b) -> Value {
                  return arith::CmpIOp::create(builder, loc, pred, a, b);
                })))
          return failure();
        continue;
      }

      if (auto fcmpOp = llvm::dyn_cast<LLVM::FCmpOp>(op)) {
        auto pred = ConvertFCmpPredicate(fcmpOp.getPredicate());
        if (failed(convertBinaryOp(fcmpOp.getLhs(), fcmpOp.getRhs(),
                fcmpOp.getResult(), "fcmp",
                [&](Value a, Value b) -> Value {
                  return arith::CmpFOp::create(builder, loc, pred, a, b);
                })))
          return failure();
        continue;
      }

      if (auto selectOp = llvm::dyn_cast<LLVM::SelectOp>(op)) {
        // Vector select: per-lane arith.select
        if (auto vecTy = llvm::dyn_cast<VectorType>(
                selectOp.getResult().getType())) {
          auto *condL = LookupVector(vectorMap, selectOp.getCondition());
          auto *trueL = LookupVector(vectorMap, selectOp.getTrueValue());
          auto *falseL = LookupVector(vectorMap, selectOp.getFalseValue());
          if (!condL || !trueL || !falseL)
            return selectOp.emitError("missing vector select operand"),
                   failure();
          SmallVector<Value, 8> out;
          for (size_t i = 0, e = condL->size(); i < e; ++i)
            out.push_back(arith::SelectOp::create(
                builder, loc, (*condL)[i], (*trueL)[i], (*falseL)[i]));
          vectorMap[selectOp.getResult()] = std::move(out);
          continue;
        }

        auto cond = LookupValue(valueMap, selectOp.getCondition());
        if (!cond)
          return selectOp.emitError("missing select condition"), failure();

        auto lhsPtr = LookupPointer(pointerMap, selectOp.getTrueValue());
        auto rhsPtr = LookupPointer(pointerMap, selectOp.getFalseValue());
        if (lhsPtr || rhsPtr) {
          if (!lhsPtr || !rhsPtr)
            return selectOp.emitError("missing select pointer operand"),
                   failure();
          if (lhsPtr->elementType != rhsPtr->elementType)
            return selectOp.emitError("select pointer type mismatch"),
                   failure();
          auto lhsBaseType =
              llvm::dyn_cast<MemRefType>(lhsPtr->base.getType());
          auto rhsBaseType =
              llvm::dyn_cast<MemRefType>(rhsPtr->base.getType());
          if (!lhsBaseType || !rhsBaseType)
            return selectOp.emitError("select pointer base type invalid"),
                   failure();
          if (lhsBaseType.getMemorySpace() != rhsBaseType.getMemorySpace())
            return selectOp.emitError("select pointer memory space mismatch"),
                   failure();
          auto commonBaseType = MakeStridedMemRefType(
              lhsPtr->elementType, lhsBaseType.getMemorySpace());
          Value lhsBase = lhsPtr->base;
          if (lhsBaseType != commonBaseType) {
            if (!memref::CastOp::areCastCompatible(
                    lhsBaseType, commonBaseType))
              return selectOp.emitError("select pointer base mismatch"),
                     failure();
            lhsBase = memref::CastOp::create(builder, loc, commonBaseType,
                                                     lhsBase);
          }
          Value rhsBase = rhsPtr->base;
          if (rhsBaseType != commonBaseType) {
            if (!memref::CastOp::areCastCompatible(
                    rhsBaseType, commonBaseType))
              return selectOp.emitError("select pointer base mismatch"),
                     failure();
            rhsBase = memref::CastOp::create(builder, loc, commonBaseType,
                                                     rhsBase);
          }
          Value baseSel =
              arith::SelectOp::create(builder, loc, *cond, lhsBase, rhsBase);
          Value indexSel =
              arith::SelectOp::create(builder, loc, *cond, lhsPtr->index,
                                              rhsPtr->index);
          pointerMap[selectOp.getResult()] =
              PointerInfo{baseSel, indexSel, lhsPtr->elementType};
          continue;
        }

        auto lhs = LookupValue(valueMap, selectOp.getTrueValue());
        auto rhs = LookupValue(valueMap, selectOp.getFalseValue());
        if (!lhs || !rhs)
          return selectOp.emitError("missing select operand"), failure();
        auto sel = arith::SelectOp::create(builder, loc, *cond, *lhs, *rhs);
        valueMap[selectOp.getResult()] = sel.getResult();
        continue;
      }

      if (auto zextOp = llvm::dyn_cast<LLVM::ZExtOp>(op)) {
        auto src = LookupValue(valueMap, zextOp.getArg());
        if (!src)
          return zextOp.emitError("missing zext operand"), failure();
        auto dstType = NormalizeScalarType(zextOp.getType(), module.getContext());
        auto ext = arith::ExtUIOp::create(builder, loc, dstType, *src);
        valueMap[zextOp.getResult()] = ext.getResult();
        continue;
      }

      if (auto sextOp = llvm::dyn_cast<LLVM::SExtOp>(op)) {
        auto src = LookupValue(valueMap, sextOp.getArg());
        if (!src)
          return sextOp.emitError("missing sext operand"), failure();
        auto dstType = NormalizeScalarType(sextOp.getType(), module.getContext());
        auto ext = arith::ExtSIOp::create(builder, loc, dstType, *src);
        valueMap[sextOp.getResult()] = ext.getResult();
        continue;
      }

      if (auto truncOp = llvm::dyn_cast<LLVM::TruncOp>(op)) {
        auto src = LookupValue(valueMap, truncOp.getArg());
        if (!src)
          return truncOp.emitError("missing trunc operand"), failure();
        auto dstType = NormalizeScalarType(truncOp.getType(), module.getContext());
        auto trunc = arith::TruncIOp::create(builder, loc, dstType, *src);
        valueMap[truncOp.getResult()] = trunc.getResult();
        continue;
      }

      if (auto fpextOp = llvm::dyn_cast<LLVM::FPExtOp>(op)) {
        auto src = LookupValue(valueMap, fpextOp.getArg());
        if (!src)
          return fpextOp.emitError("missing fpext operand"), failure();
        auto dstType = NormalizeScalarType(fpextOp.getType(), module.getContext());
        auto ext = arith::ExtFOp::create(builder, loc, dstType, *src);
        valueMap[fpextOp.getResult()] = ext.getResult();
        continue;
      }

      if (auto fptruncOp = llvm::dyn_cast<LLVM::FPTruncOp>(op)) {
        auto src = LookupValue(valueMap, fptruncOp.getArg());
        if (!src)
          return fptruncOp.emitError("missing fptrunc operand"), failure();
        auto dstType = NormalizeScalarType(fptruncOp.getType(), module.getContext());
        auto trunc = arith::TruncFOp::create(builder, loc, dstType, *src);
        valueMap[fptruncOp.getResult()] = trunc.getResult();
        continue;
      }

      if (auto uitofpOp = llvm::dyn_cast<LLVM::UIToFPOp>(op)) {
        auto src = LookupValue(valueMap, uitofpOp.getArg());
        if (!src)
          return uitofpOp.emitError("missing uitofp operand"), failure();
        auto dstType = NormalizeScalarType(uitofpOp.getType(), module.getContext());
        auto cast = arith::UIToFPOp::create(builder, loc, dstType, *src);
        valueMap[uitofpOp.getResult()] = cast.getResult();
        continue;
      }

      if (auto sitofpOp = llvm::dyn_cast<LLVM::SIToFPOp>(op)) {
        auto src = LookupValue(valueMap, sitofpOp.getArg());
        if (!src)
          return sitofpOp.emitError("missing sitofp operand"), failure();
        auto dstType = NormalizeScalarType(sitofpOp.getType(), module.getContext());
        auto cast = arith::SIToFPOp::create(builder, loc, dstType, *src);
        valueMap[sitofpOp.getResult()] = cast.getResult();
        continue;
      }

      if (auto fptosiOp = llvm::dyn_cast<LLVM::FPToSIOp>(op)) {
        auto src = LookupValue(valueMap, fptosiOp.getArg());
        if (!src)
          return fptosiOp.emitError("missing fptosi operand"), failure();
        auto dstType = NormalizeScalarType(fptosiOp.getType(), module.getContext());
        auto cast = arith::FPToSIOp::create(builder, loc, dstType, *src);
        valueMap[fptosiOp.getResult()] = cast.getResult();
        continue;
      }

      if (auto fptouiOp = llvm::dyn_cast<LLVM::FPToUIOp>(op)) {
        auto src = LookupValue(valueMap, fptouiOp.getArg());
        if (!src)
          return fptouiOp.emitError("missing fptoui operand"), failure();
        auto dstType = NormalizeScalarType(fptouiOp.getType(), module.getContext());
        auto cast = arith::FPToUIOp::create(builder, loc, dstType, *src);
        valueMap[fptouiOp.getResult()] = cast.getResult();
        continue;
      }

      if (auto switchOp = llvm::dyn_cast<LLVM::SwitchOp>(op)) {
        auto flag = LookupValue(valueMap, switchOp.getValue());
        if (!flag)
          return switchOp.emitError("missing switch flag"), failure();

        SmallVector<Value, 4> defaultOperands;
        for (Value operand : switchOp.getDefaultOperands()) {
          if (auto ptrInfo = LookupPointer(pointerMap, operand)) {
            defaultOperands.push_back(
                MaterializeMemrefPointer(builder, loc, *ptrInfo));
            continue;
          }
          auto mapped = LookupValue(valueMap, operand);
          if (!mapped)
            return switchOp.emitError("missing default operand"), failure();
          defaultOperands.push_back(*mapped);
        }

        SmallVector<Block *, 4> caseDests;
        caseDests.reserve(switchOp.getCaseDestinations().size());
        for (Block *dest : switchOp.getCaseDestinations())
          caseDests.push_back(blockMap[dest]);

        SmallVector<SmallVector<Value, 4>, 4> caseOperandsStorage;
        SmallVector<ValueRange, 4> caseOperands;
        for (auto caseOps : switchOp.getCaseOperands()) {
          caseOperandsStorage.emplace_back();
          auto &mappedOps = caseOperandsStorage.back();
          for (Value operand : caseOps) {
            if (auto ptrInfo = LookupPointer(pointerMap, operand)) {
              mappedOps.push_back(
                  MaterializeMemrefPointer(builder, loc, *ptrInfo));
              continue;
            }
            auto mapped = LookupValue(valueMap, operand);
            if (!mapped)
              return switchOp.emitError("missing case operand"), failure();
            mappedOps.push_back(*mapped);
          }
          caseOperands.emplace_back(mappedOps);
        }

        auto caseValues = switchOp.getCaseValuesAttr();
        cf::SwitchOp::create(builder, loc, *flag,
                                     blockMap[switchOp.getDefaultDestination()],
                                     defaultOperands, caseValues, caseDests,
                                     caseOperands);
        break;
      }

      if (auto brOp = llvm::dyn_cast<LLVM::BrOp>(op)) {
        Block *dest = blockMap[brOp.getDest()];
        SmallVector<Value, 4> operands;
        for (Value operand : brOp.getOperands()) {
          if (auto ptrInfo = LookupPointer(pointerMap, operand)) {
            operands.push_back(MaterializeMemrefPointer(builder, loc, *ptrInfo));
            continue;
          }
          auto mapped = LookupValue(valueMap, operand);
          if (!mapped)
            return brOp.emitError("missing branch operand"), failure();
          operands.push_back(*mapped);
        }
        cf::BranchOp::create(builder, loc, dest, operands);
        break;
      }

      if (auto condBrOp = llvm::dyn_cast<LLVM::CondBrOp>(op)) {
        auto cond = LookupValue(valueMap, condBrOp.getCondition());
        if (!cond)
          return condBrOp.emitError("missing condition"), failure();

        SmallVector<Value, 4> trueOperands;
        for (Value operand : condBrOp.getTrueDestOperands()) {
          if (auto ptrInfo = LookupPointer(pointerMap, operand)) {
            trueOperands.push_back(
                MaterializeMemrefPointer(builder, loc, *ptrInfo));
            continue;
          }
          auto mapped = LookupValue(valueMap, operand);
          if (!mapped)
            return condBrOp.emitError("missing true operand"), failure();
          trueOperands.push_back(*mapped);
        }

        SmallVector<Value, 4> falseOperands;
        for (Value operand : condBrOp.getFalseDestOperands()) {
          if (auto ptrInfo = LookupPointer(pointerMap, operand)) {
            falseOperands.push_back(
                MaterializeMemrefPointer(builder, loc, *ptrInfo));
            continue;
          }
          auto mapped = LookupValue(valueMap, operand);
          if (!mapped)
            return condBrOp.emitError("missing false operand"), failure();
          falseOperands.push_back(*mapped);
        }

        cf::CondBranchOp::create(builder, loc, *cond,
                                         blockMap[condBrOp.getTrueDest()],
                                         trueOperands,
                                         blockMap[condBrOp.getFalseDest()],
                                         falseOperands);
        break;
      }

      if (auto retOp = llvm::dyn_cast<LLVM::ReturnOp>(op)) {
        SmallVector<Value, 2> results;
        for (Value operand : retOp.getOperands()) {
          auto mapped = LookupValue(valueMap, operand);
          if (!mapped)
            return retOp.emitError("missing return operand"), failure();
          results.push_back(*mapped);
        }
        func::ReturnOp::create(builder, loc, results);
        break;
      }

      if (llvm::isa<LLVM::InlineAsmOp>(op))
        continue;
      if (llvm::isa<LLVM::DbgDeclareOp, LLVM::DbgValueOp,
                    LLVM::DbgLabelOp>(op))
        continue;

      op.emitError("unsupported LLVM operation in scf conversion: ")
          << op.getName();
      return failure();
    }
  }

  for (auto &entry : argAnnotations) {
    unsigned argIndex = entry.first;
    if (entry.second.empty())
      continue;
    SmallVector<StringAttr, 4> merged;
    if (auto existing =
            newFunc.getArgAttrOfType<ArrayAttr>(argIndex, "loom.annotations")) {
      MergeLoomAnnotationList(merged, existing);
    }
    auto incoming = BuildLoomAnnotationArray(module.getContext(), entry.second);
    MergeLoomAnnotationList(merged, incoming);
    newFunc.setArgAttr(argIndex, "loom.annotations",
                       BuildLoomAnnotationArray(module.getContext(), merged));
  }

  func.erase();
  return success();
}

} // namespace loom::llvm_to_scf
