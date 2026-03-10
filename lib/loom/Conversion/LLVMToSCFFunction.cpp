//===-- LLVMToSCFFunction.cpp - LLVM function conversion ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This file implements the FunctionConverter class methods for the core
// dispatch loop and the constant, vector, memory, load/store, and terminator
// op handlers.
//
//===----------------------------------------------------------------------===//

#include "LLVMToSCFInternal.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/RegionGraphTraits.h"

using namespace mlir;

namespace loom::llvm_to_scf {

// Free function entry point -- delegates to FunctionConverter.
LogicalResult convertFunction(ModuleOp module, LLVM::LLVMFuncOp func,
                              OpBuilder &builder,
                              llvm::StringMap<ConvertedGlobal> &globals,
                              const llvm::StringSet<> &varargFunctions) {
  FunctionConverter converter(module, func, builder, globals, varargFunctions);
  return converter.convert();
}

// --- FunctionConverter::convert ---

LogicalResult FunctionConverter::convert() {
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

  // Pre-scan: find pointer-valued loads from non-alloca addresses and
  // promote them to extra function arguments (argument promotion).
  struct PromotedPtrLoad {
    LLVM::LoadOp loadOp;
    Type elemType;
  };
  SmallVector<PromotedPtrLoad, 4> promotedLoads;
  if (!func.isExternal()) {
    for (Block &block : func.getBody()) {
      for (Operation &scanOp : block.getOperations()) {
        auto loadOp = llvm::dyn_cast<LLVM::LoadOp>(scanOp);
        if (!loadOp)
          continue;
        if (!IsPointerType(loadOp.getResult().getType()))
          continue;
        Value addr = loadOp.getAddr();
        if (auto allocaOp = addr.getDefiningOp<LLVM::AllocaOp>()) {
          if (IsPointerType(allocaOp.getElemType()))
            continue;
        }
        auto elemType = InferPointerElementType(loadOp.getResult());
        if (!elemType)
          elemType = IntegerType::get(module.getContext(), 8);
        promotedLoads.push_back({loadOp, *elemType});
      }
    }
    for (auto &promoted : promotedLoads)
      argTypes.push_back(MakeStridedMemRefType(promoted.elemType));
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
  newFunc =
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

  // --- Block and argument initialization ---
  // Two-pass approach: first add all block arguments (which may cause
  // reallocation of the block's internal storage), then retrieve stable
  // BlockArgument handles by index to populate the maps.
  struct ArgMapping {
    unsigned newArgIdx;
    BlockArgument oldArg;
    bool isPointer;
    Type elemType; // only valid if isPointer
  };
  for (Block &oldBlock : func.getBody()) {
    Block *newBlock = new Block();
    newFunc.getBody().push_back(newBlock);
    blockMap[&oldBlock] = newBlock;
    bool isEntryBlock = (&oldBlock == &func.getBody().front());

    // Pass 1: add all block arguments.
    SmallVector<ArgMapping, 8> mappings;
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
        unsigned idx = newBlock->getNumArguments();
        newBlock->addArgument(memrefType, arg.getLoc());
        mappings.push_back({idx, arg, true, *elemType});
        continue;
      }
      unsigned idx = newBlock->getNumArguments();
      newBlock->addArgument(
          NormalizeScalarType(oldType, module.getContext()), arg.getLoc());
      mappings.push_back({idx, arg, false, {}});
    }
    // Add promoted pointer load arguments on the entry block.
    if (isEntryBlock) {
      for (size_t i = 0; i < promotedLoads.size(); ++i) {
        newBlock->addArgument(MakeStridedMemRefType(promotedLoads[i].elemType),
                              func.getLoc());
      }
    }

    // Pass 2: retrieve stable BlockArgument handles and populate maps.
    OpBuilder blockBuilder(module.getContext());
    blockBuilder.setInsertionPointToStart(newBlock);
    Value zeroIndex = BuildIndexConstant(blockBuilder, func.getLoc(), 0);
    for (auto &m : mappings) {
      Value newArg = newBlock->getArgument(m.newArgIdx);
      if (m.isPointer) {
        pointerMap[m.oldArg] = {newArg, zeroIndex, m.elemType};
      } else {
        valueMap[m.oldArg] = newArg;
      }
    }
    if (isEntryBlock) {
      unsigned baseArgCount = oldBlock.getNumArguments();
      for (size_t i = 0; i < promotedLoads.size(); ++i) {
        unsigned newArgIdx = baseArgCount + i;
        Value newArg = newBlock->getArgument(newArgIdx);
        PointerInfo info{newArg, zeroIndex, promotedLoads[i].elemType};
        pointerMap[promotedLoads[i].loadOp.getResult()] = info;
        promotedPtrLoads.insert(promotedLoads[i].loadOp.getResult());
        newFunc.setArgAttr(newArgIdx, "loom.noalias", builder.getUnitAttr());
      }
    }
  }

  // --- Block ordering (iterative reverse post-order) ---
  SmallVector<Block *, 16> blockOrder;
  llvm::DenseSet<Block *> visitedBlocks;
  llvm::ReversePostOrderTraversal<Block *> rpot(&func.getBody().front());
  for (Block *block : rpot) {
    blockOrder.push_back(block);
    visitedBlocks.insert(block);
  }
  for (Block &oldBlock : func.getBody()) {
    if (!visitedBlocks.contains(&oldBlock))
      blockOrder.push_back(&oldBlock);
  }

  // --- Dispatch loop ---
  auto tryHandle =
      [](FailureOr<bool> result) -> std::optional<LogicalResult> {
    if (failed(result))
      return failure();
    if (*result)
      return success();
    return std::nullopt;
  };

  for (Block *oldBlock : blockOrder) {
    Block *newBlock = blockMap[oldBlock];
    builder.setInsertionPointToStart(newBlock);

    for (Operation &op : oldBlock->getOperations()) {
      builder.setInsertionPointToEnd(newBlock);
      Location loc = op.getLoc();

      if (auto r = tryHandle(handleConstantOps(op, loc))) {
        if (failed(*r)) return failure();
        continue;
      }
      if (auto r = tryHandle(handleVectorOps(op, loc))) {
        if (failed(*r)) return failure();
        continue;
      }
      if (auto r = tryHandle(handleMemoryOps(op, loc))) {
        if (failed(*r)) return failure();
        continue;
      }
      if (auto r = tryHandle(handleLoadStore(op, loc))) {
        if (failed(*r)) return failure();
        continue;
      }
      if (auto r = tryHandle(handleCallOp(op, loc))) {
        if (failed(*r)) return failure();
        continue;
      }
      if (auto r = tryHandle(handleIntrinsicCall(op, loc))) {
        if (failed(*r)) return failure();
        continue;
      }
      if (auto r = tryHandle(handleArithOps(op, loc))) {
        if (failed(*r)) return failure();
        continue;
      }
      if (auto r = tryHandle(handleSelectOp(op, loc))) {
        if (failed(*r)) return failure();
        continue;
      }
      if (auto r = tryHandle(handleCastOps(op, loc))) {
        if (failed(*r)) return failure();
        continue;
      }
      if (auto r = tryHandle(handleTerminatorOps(op, loc))) {
        if (failed(*r)) return failure();
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

  // --- Epilogue: merge arg annotations ---
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

// --- Constant ops ---

FailureOr<bool> FunctionConverter::handleConstantOps(Operation &op,
                                                      Location loc) {
  if (auto constOp = llvm::dyn_cast<LLVM::ConstantOp>(op)) {
    if (auto denseAttr =
            llvm::dyn_cast<DenseElementsAttr>(constOp.getValue())) {
      auto lanes = ScalarizeDenseConstant(builder, loc, denseAttr);
      if (lanes.empty())
        return constOp.emitError("unsupported vector constant"), failure();
      vectorMap[constOp.getResult()] = std::move(lanes);
      return true;
    }
    auto value = ConvertLLVMConstant(builder, constOp);
    if (!value)
      return constOp.emitError("unsupported constant"), failure();
    valueMap[constOp.getResult()] = *value;
    return true;
  }

  if (auto zeroOp = llvm::dyn_cast<LLVM::ZeroOp>(op)) {
    Type type = zeroOp.getType();
    if (IsPointerType(type)) {
      auto zeroInt = arith::ConstantOp::create(
          builder, loc, builder.getI64Type(),
          builder.getIntegerAttr(builder.getI64Type(), 0));
      auto ptr = LLVM::IntToPtrOp::create(builder, loc, type,
                                           zeroInt.getResult());
      valueMap[zeroOp.getResult()] = ptr.getResult();
      return true;
    }
    if (auto intTy = llvm::dyn_cast<IntegerType>(type)) {
      auto attr = builder.getIntegerAttr(intTy, 0);
      auto zero = arith::ConstantOp::create(builder, loc, attr);
      valueMap[zeroOp.getResult()] = zero.getResult();
      return true;
    }
    if (auto floatTy = llvm::dyn_cast<FloatType>(type)) {
      auto attr = builder.getFloatAttr(floatTy, 0.0);
      auto zero = arith::ConstantOp::create(builder, loc, attr);
      valueMap[zeroOp.getResult()] = zero.getResult();
      return true;
    }
    return zeroOp.emitError("unsupported zero type"), failure();
  }

  if (auto poisonOp = llvm::dyn_cast<LLVM::PoisonOp>(op)) {
    if (auto vecTy = llvm::dyn_cast<VectorType>(poisonOp.getType())) {
      Type elemTy = vecTy.getElementType();
      int64_t n = vecTy.getNumElements();
      SmallVector<Value, 8> lanes;
      for (int64_t i = 0; i < n; ++i)
        lanes.push_back(ub::PoisonOp::create(builder, loc, elemTy));
      vectorMap[poisonOp.getResult()] = std::move(lanes);
      return true;
    }
    auto poison = ub::PoisonOp::create(builder, loc, poisonOp.getType());
    valueMap[poisonOp.getResult()] = poison.getResult();
    return true;
  }

  if (auto undefOp = llvm::dyn_cast<LLVM::UndefOp>(op)) {
    if (auto vecTy = llvm::dyn_cast<VectorType>(undefOp.getType())) {
      Type elemTy = vecTy.getElementType();
      int64_t n = vecTy.getNumElements();
      SmallVector<Value, 8> lanes;
      for (int64_t i = 0; i < n; ++i)
        lanes.push_back(ub::PoisonOp::create(builder, loc, elemTy));
      vectorMap[undefOp.getResult()] = std::move(lanes);
      return true;
    }
    auto poison = ub::PoisonOp::create(builder, loc, undefOp.getType());
    valueMap[undefOp.getResult()] = poison.getResult();
    return true;
  }

  return false;
}

// --- Vector ops ---

FailureOr<bool> FunctionConverter::handleVectorOps(Operation &op,
                                                    Location loc) {
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
      return insertOp.emitError("non-constant insertelement index"), failure();
    SmallVector<Value, 8> lanes(*srcLanes);
    lanes[*idx] = *scalarVal;
    vectorMap[insertOp.getResult()] = std::move(lanes);
    return true;
  }

  if (auto extractOp = llvm::dyn_cast<LLVM::ExtractElementOp>(op)) {
    auto *srcLanes = LookupVector(vectorMap, extractOp.getVector());
    if (!srcLanes)
      return extractOp.emitError("missing extractelement vector"), failure();
    auto posVal = LookupValue(valueMap, extractOp.getPosition());
    if (!posVal)
      return extractOp.emitError("missing extractelement index"), failure();
    auto idx = GetConstantIntValue(*posVal);
    if (!idx)
      return extractOp.emitError("non-constant extractelement index"),
             failure();
    valueMap[extractOp.getResult()] = (*srcLanes)[*idx];
    return true;
  }

  if (auto shuffleOp = llvm::dyn_cast<LLVM::ShuffleVectorOp>(op)) {
    auto *v1Lanes = LookupVector(vectorMap, shuffleOp.getV1());
    auto *v2Lanes = LookupVector(vectorMap, shuffleOp.getV2());
    if (!v1Lanes || !v2Lanes)
      return shuffleOp.emitError("missing shufflevector operand"), failure();
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
    return true;
  }

  return false;
}

// --- Memory ops ---

FailureOr<bool> FunctionConverter::handleMemoryOps(Operation &op,
                                                    Location loc) {
  if (auto addrOp = llvm::dyn_cast<LLVM::AddressOfOp>(op)) {
    auto it = globals.find(addrOp.getGlobalName());
    if (it == globals.end())
      return addrOp.emitError("unknown global"), failure();
    auto memrefType = it->second.type;
    auto getGlobal = memref::GetGlobalOp::create(
        builder, loc, memrefType, addrOp.getGlobalName());
    CopyLoomAnnotations(it->second.oldGlobal.getOperation(),
                        getGlobal.getOperation());
    PointerInfo info{getGlobal, BuildIndexConstant(builder, loc, 0),
                     memrefType.getElementType()};
    pointerMap[addrOp.getResult()] = info;
    return true;
  }

  if (auto allocaOp = llvm::dyn_cast<LLVM::AllocaOp>(op)) {
    if (IsPointerType(allocaOp.getElemType())) {
      pointerSlots.insert(allocaOp.getResult());
      if (auto ann =
              allocaOp->getAttrOfType<ArrayAttr>("loom.annotations")) {
        pointerSlotAnnotations[allocaOp.getResult()] = ann;
      }
      return true;
    }
    SmallVector<int64_t, 4> dims;
    Type scalar = GetScalarType(allocaOp.getElemType(), dims);
    scalar = NormalizeScalarType(scalar, module.getContext());
    if (!llvm::isa<IntegerType>(scalar) && !llvm::isa<FloatType>(scalar)) {
      return allocaOp.emitError("unsupported alloca element type"), failure();
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
    return true;
  }

  if (auto gepOp = llvm::dyn_cast<LLVM::GEPOp>(op)) {
    auto baseInfo = LookupPointer(pointerMap, gepOp.getBase());
    if (!baseInfo)
      return gepOp.emitError("missing GEP base"), failure();
    if (!isValidPointerInfo(*baseInfo))
      return gepOp.emitError("invalid GEP base pointer"), failure();

    // Struct-containing GEP: compute byte offsets by walking type hierarchy.
    if (ContainsStructType(gepOp.getElemType())) {
      LLVM::GEPIndicesAdaptor<ValueRange> ixAdaptor(
          gepOp.getRawConstantIndicesAttr(), gepOp.getDynamicIndices());
      Value byteOff = nullptr;
      Type currType = gepOp.getElemType();

      auto addByteOffset = [&](Value term) {
        if (!byteOff)
          byteOff = term;
        else
          byteOff = arith::AddIOp::create(builder, loc, byteOff, term);
      };

      for (size_t i = 0; i < ixAdaptor.size(); ++i) {
        Value ixVal;
        int64_t constIx = 0;
        bool isConst = false;
        auto item = ixAdaptor[i];
        if (auto attr = llvm::dyn_cast<IntegerAttr>(item)) {
          constIx = attr.getInt();
          isConst = true;
          ixVal = BuildIndexConstant(builder, loc, constIx);
        } else if (auto val = llvm::dyn_cast<Value>(item)) {
          auto mapped = LookupValue(valueMap, val);
          if (!mapped)
            return gepOp.emitError("missing GEP index"), failure();
          ixVal = ToIndexValue(builder, loc, *mapped);
          if (!ixVal)
            return gepOp.emitError("invalid GEP index type"), failure();
        } else {
          return gepOp.emitError("unsupported GEP index"), failure();
        }

        if (i == 0) {
          // First index: skip N * sizeof(elemType) bytes.
          int64_t elemSize = GetLLVMTypeByteSize(currType);
          if (elemSize > 0 && !(isConst && constIx == 0)) {
            Value sizeVal = BuildIndexConstant(builder, loc, elemSize);
            addByteOffset(arith::MulIOp::create(builder, loc, ixVal, sizeVal));
          }
        } else if (auto structTy =
                       llvm::dyn_cast<LLVM::LLVMStructType>(currType)) {
          if (!isConst)
            return gepOp.emitError("non-constant struct field index"),
                   failure();
          int64_t fieldOff = GetStructFieldByteOffset(structTy, constIx);
          if (fieldOff > 0)
            addByteOffset(BuildIndexConstant(builder, loc, fieldOff));
          currType = structTy.getBody()[constIx];
        } else if (auto arrayTy =
                       llvm::dyn_cast<LLVM::LLVMArrayType>(currType)) {
          int64_t eSize = GetLLVMTypeByteSize(arrayTy.getElementType());
          if (eSize > 0 && !(isConst && constIx == 0)) {
            Value sizeVal = BuildIndexConstant(builder, loc, eSize);
            addByteOffset(arith::MulIOp::create(builder, loc, ixVal, sizeVal));
          }
          currType = arrayTy.getElementType();
        } else {
          // Primitive type at this level -- treat as i8 offset.
          if (!(isConst && constIx == 0))
            addByteOffset(ixVal);
        }
      }

      if (!byteOff)
        byteOff = BuildIndexConstant(builder, loc, 0);

      // Add base's existing byte offset.
      if (!IsZeroIndex(baseInfo->index)) {
        Value baseBytes = baseInfo->index;
        int64_t baseElemSize = GetByteSize(baseInfo->elementType);
        if (baseElemSize > 1) {
          Value s = BuildIndexConstant(builder, loc, baseElemSize);
          baseBytes = arith::MulIOp::create(builder, loc, baseInfo->index, s);
        }
        byteOff = arith::AddIOp::create(builder, loc, baseBytes, byteOff);
      }

      // Determine the final element type after walking all indices.
      Type finalElem = IntegerType::get(module.getContext(), 8);
      if (llvm::isa<IntegerType>(currType) || llvm::isa<FloatType>(currType))
        finalElem = NormalizeScalarType(currType, module.getContext());

      // If base is i8 memref and final type is non-i8, use memref.view for
      // typed access so downstream loads/stores work at the right type.
      auto baseMemref =
          llvm::dyn_cast<MemRefType>(baseInfo->base.getType());
      if (baseMemref && IsI8Type(baseMemref.getElementType()) &&
          !IsI8Type(finalElem) && baseMemref.getLayout().isIdentity()) {
        int64_t eSize = GetByteSize(finalElem);
        Value totalBytes =
            memref::DimOp::create(builder, loc, baseInfo->base, 0);
        Value remaining =
            arith::SubIOp::create(builder, loc, totalBytes, byteOff);
        Value elemCount = remaining;
        if (eSize > 1) {
          Value s = BuildIndexConstant(builder, loc, eSize);
          elemCount = arith::DivUIOp::create(builder, loc, remaining, s);
        }
        auto viewType =
            MakeMemRefType(finalElem, baseMemref.getMemorySpace());
        auto viewed = memref::ViewOp::create(builder, loc, viewType,
                                             baseInfo->base, byteOff,
                                             ValueRange{elemCount});
        PointerInfo info{viewed, BuildIndexConstant(builder, loc, 0),
                         finalElem};
        pointerMap[gepOp.getResult()] = info;
        return true;
      }

      // Base is i8 and final is i8, or non-identity layout: use byte offset.
      if (baseMemref && IsI8Type(baseInfo->elementType)) {
        PointerInfo info{baseInfo->base, byteOff, finalElem};
        pointerMap[gepOp.getResult()] = info;
        return true;
      }

      // Base has non-i8 element: scale byte offset to base element type.
      Value scaled = ScaleIndexBetweenElementTypes(
          builder, loc, byteOff, IntegerType::get(module.getContext(), 8),
          baseInfo->elementType);
      PointerInfo info{baseInfo->base, scaled, baseInfo->elementType};
      pointerMap[gepOp.getResult()] = info;
      return true;
    }

    SmallVector<int64_t, 4> dims;
    Type scalar = GetScalarType(gepOp.getElemType(), dims);
    scalar = NormalizeScalarType(scalar, module.getContext());
    if (!llvm::isa<IntegerType>(scalar) && !llvm::isa<FloatType>(scalar))
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

    if (IsI8Type(scalar) && !IsI8Type(baseInfo->elementType)) {
      int64_t baseElemSize = GetByteSize(baseInfo->elementType);
      if (baseElemSize > 1) {
        Value byteOffset = offset;
        Value elemSizeVal = BuildIndexConstant(builder, loc, baseElemSize);
        Value scaledOffset =
            arith::DivUIOp::create(builder, loc, byteOffset, elemSizeVal);
        Value newIndex = baseInfo->index;
        if (!IsZeroIndex(scaledOffset)) {
          if (IsZeroIndex(newIndex))
            newIndex = scaledOffset;
          else
            newIndex = arith::AddIOp::create(builder, loc, newIndex,
                                             scaledOffset);
        }
        PointerInfo info{baseInfo->base, newIndex, baseInfo->elementType};
        pointerMap[gepOp.getResult()] = info;
        return true;
      }
    }

    Value baseIndex = ScaleIndexBetweenElementTypes(
        builder, loc, baseInfo->index, baseInfo->elementType, scalar);
    Value newIndex = baseIndex;
    if (!IsZeroIndex(offset))
      newIndex = arith::AddIOp::create(builder, loc, baseIndex, offset);

    PointerInfo info{baseInfo->base, newIndex, scalar};
    pointerMap[gepOp.getResult()] = info;
    return true;
  }

  if (auto bitcastOp = llvm::dyn_cast<LLVM::BitcastOp>(op)) {
    if (IsPointerType(bitcastOp.getType())) {
      auto srcInfo = LookupPointer(pointerMap, bitcastOp.getArg());
      if (!srcInfo)
        return bitcastOp.emitError("missing bitcast source"), failure();
      if (!isValidPointerInfo(*srcInfo))
        return bitcastOp.emitError("invalid bitcast source pointer"), failure();
      Type targetElem = srcInfo->elementType;
      if (auto inferred = InferPointerElementType(bitcastOp.getResult()))
        targetElem = *inferred;
      PointerInfo info = *srcInfo;
      if (targetElem != srcInfo->elementType) {
        info.index = ScaleIndexBetweenElementTypes(
            builder, loc, srcInfo->index, srcInfo->elementType, targetElem);
        info.elementType = targetElem;
      }
      pointerMap[bitcastOp.getResult()] = info;
      return true;
    }
    auto srcVecTy =
        llvm::dyn_cast<VectorType>(bitcastOp.getArg().getType());
    auto dstIntTy = llvm::dyn_cast<IntegerType>(bitcastOp.getType());
    if (srcVecTy && dstIntTy) {
      auto *lanes = LookupVector(vectorMap, bitcastOp.getArg());
      if (!lanes)
        return bitcastOp.emitError("missing vector bitcast source"), failure();
      unsigned laneWidth = srcVecTy.getElementType().getIntOrFloatBitWidth();
      Value result = arith::ConstantOp::create(
          builder, loc, builder.getIntegerAttr(dstIntTy, 0));
      for (size_t i = 0, e = lanes->size(); i < e; ++i) {
        Value lane = (*lanes)[i];
        if (lane.getType() != dstIntTy)
          lane = arith::ExtUIOp::create(builder, loc, dstIntTy, lane);
        if (i > 0) {
          Value shift = arith::ConstantOp::create(
              builder, loc,
              builder.getIntegerAttr(dstIntTy, i * laneWidth));
          lane = arith::ShLIOp::create(builder, loc, lane, shift);
        }
        result = arith::OrIOp::create(builder, loc, result, lane);
      }
      valueMap[bitcastOp.getResult()] = result;
      return true;
    }
    return false;
  }

  if (auto freezeOp = llvm::dyn_cast<LLVM::FreezeOp>(op)) {
    auto src = LookupValue(valueMap, freezeOp.getVal());
    if (!src)
      return freezeOp.emitError("missing freeze operand"), failure();
    valueMap[freezeOp.getResult()] = *src;
    return true;
  }

  return false;
}

// --- Load / Store ---

FailureOr<bool> FunctionConverter::handleLoadStore(Operation &op,
                                                    Location loc) {
  if (auto loadOp = llvm::dyn_cast<LLVM::LoadOp>(op)) {
    // Promoted pointer load -- already mapped via argument promotion.
    if (promotedPtrLoads.contains(loadOp.getResult()))
      return true;

    if (pointerSlots.contains(loadOp.getAddr())) {
      auto it = pointerSlotValues.find(loadOp.getAddr());
      if (it == pointerSlotValues.end())
        return loadOp.emitError("missing pointer slot value"), failure();
      pointerMap[loadOp.getResult()] = it->second;
      return true;
    }
    auto ptrInfo = LookupPointer(pointerMap, loadOp.getAddr());
    if (!ptrInfo)
      return loadOp.emitError("missing load pointer for address ")
                 << loadOp.getAddr(),
             failure();
    if (!isValidPointerInfo(*ptrInfo))
      return loadOp.emitError("invalid pointer base for load"), failure();
    Type accessType =
        NormalizeScalarType(loadOp.getResult().getType(), module.getContext());
    Value index = ptrInfo->index;
    if (ptrInfo->elementType != accessType) {
      if (ptrInfo->elementType.isIntOrFloat() && accessType.isIntOrFloat() &&
          ptrInfo->elementType.getIntOrFloatBitWidth() ==
              accessType.getIntOrFloatBitWidth()) {
        // Same bitwidth: load as memref element type, then bitcast.
      } else {
        index = ScaleIndexBetweenElementTypes(builder, loc, ptrInfo->index,
                                              ptrInfo->elementType, accessType);
      }
    }
    Value val = memref::LoadOp::create(builder, loc, ptrInfo->base, index);
    if (ptrInfo->elementType != accessType &&
        ptrInfo->elementType.isIntOrFloat() && accessType.isIntOrFloat() &&
        ptrInfo->elementType.getIntOrFloatBitWidth() ==
            accessType.getIntOrFloatBitWidth()) {
      val = arith::BitcastOp::create(builder, loc, accessType, val);
    }
    valueMap[loadOp.getResult()] = val;
    return true;
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
      return true;
    }
    auto ptrInfo = LookupPointer(pointerMap, storeOp.getAddr());
    if (!ptrInfo)
      return storeOp.emitError("missing store pointer"), failure();
    if (!isValidPointerInfo(*ptrInfo))
      return storeOp.emitError("invalid pointer base for store"), failure();

    // Pointer-valued store to non-slot address: skip gracefully.
    // The memref model cannot store a memref into a memref.
    if (LookupPointer(pointerMap, storeOp.getValue()))
      return true;

    auto storedVal = LookupValue(valueMap, storeOp.getValue());
    if (!storedVal)
      return storeOp.emitError("missing store value"), failure();
    Value storeVal = *storedVal;
    Type accessType = storeVal.getType();
    Value index = ptrInfo->index;
    if (ptrInfo->elementType != accessType) {
      if (ptrInfo->elementType.isIntOrFloat() && accessType.isIntOrFloat() &&
          ptrInfo->elementType.getIntOrFloatBitWidth() ==
              accessType.getIntOrFloatBitWidth()) {
        // Same bitwidth, different kind (e.g., i64 vs f64): bitcast the value.
        storeVal = arith::BitcastOp::create(builder, loc, ptrInfo->elementType,
                                            storeVal);
      } else {
        index = ScaleIndexBetweenElementTypes(builder, loc, ptrInfo->index,
                                              ptrInfo->elementType, accessType);
      }
    }
    memref::StoreOp::create(builder, loc, storeVal, ptrInfo->base, index);
    return true;
  }

  return false;
}

// --- Terminator ops ---

FailureOr<bool> FunctionConverter::handleTerminatorOps(Operation &op,
                                                        Location loc) {
  if (auto switchOp = llvm::dyn_cast<LLVM::SwitchOp>(op)) {
    auto flag = LookupValue(valueMap, switchOp.getValue());
    if (!flag)
      return switchOp.emitError("missing switch flag"), failure();

    Block *defaultDest = blockMap[switchOp.getDefaultDestination()];
    if (!defaultDest)
      return switchOp.emitError("unmapped default destination"), failure();
    SmallVector<Value, 4> defaultOperands;
    unsigned defaultArgIdx = 0;
    for (Value operand : switchOp.getDefaultOperands()) {
      if (defaultArgIdx >= defaultDest->getNumArguments())
        return switchOp.emitError("too many default operands"), failure();
      if (auto ptrInfo = LookupPointer(pointerMap, operand)) {
        if (!isValidPointerInfo(*ptrInfo))
          return switchOp.emitError("invalid pointer for default operand"),
                 failure();
        Type destArgTy = defaultDest->getArgument(defaultArgIdx).getType();
        Value brPtr =
            MaterializeBranchPointer(builder, loc, *ptrInfo, destArgTy);
        if (!brPtr)
          return switchOp.emitError("invalid pointer for default operand"),
                 failure();
        defaultOperands.push_back(brPtr);
        ++defaultArgIdx;
        continue;
      }
      auto mapped = LookupValue(valueMap, operand);
      if (!mapped)
        return switchOp.emitError("missing default operand"), failure();
      defaultOperands.push_back(*mapped);
      ++defaultArgIdx;
    }

    SmallVector<Block *, 4> caseDests;
    caseDests.reserve(switchOp.getCaseDestinations().size());
    for (Block *dest : switchOp.getCaseDestinations())
      caseDests.push_back(blockMap[dest]);

    SmallVector<SmallVector<Value, 4>, 4> caseOperandsStorage;
    SmallVector<ValueRange, 4> caseOperands;
    for (size_t caseIdx = 0; caseIdx < switchOp.getCaseOperands().size();
         ++caseIdx) {
      auto caseOps = switchOp.getCaseOperands()[caseIdx];
      Block *caseDest = caseDests[caseIdx];
      caseOperandsStorage.emplace_back();
      auto &mappedOps = caseOperandsStorage.back();
      unsigned caseArgIdx = 0;
      for (Value operand : caseOps) {
        if (auto ptrInfo = LookupPointer(pointerMap, operand)) {
          if (!isValidPointerInfo(*ptrInfo))
            return switchOp.emitError("invalid pointer for case operand"),
                   failure();
          Type destArgTy = caseDest->getArgument(caseArgIdx).getType();
          Value brPtr =
              MaterializeBranchPointer(builder, loc, *ptrInfo, destArgTy);
          if (!brPtr)
            return switchOp.emitError("invalid pointer for case operand"),
                   failure();
          mappedOps.push_back(brPtr);
          ++caseArgIdx;
          continue;
        }
        auto mapped = LookupValue(valueMap, operand);
        if (!mapped)
          return switchOp.emitError("missing case operand"), failure();
        mappedOps.push_back(*mapped);
        ++caseArgIdx;
      }
      caseOperands.emplace_back(mappedOps);
    }

    auto caseValues = switchOp.getCaseValuesAttr();
    cf::SwitchOp::create(builder, loc, *flag, defaultDest, defaultOperands,
                          caseValues, caseDests, caseOperands);
    return true;
  }

  if (auto brOp = llvm::dyn_cast<LLVM::BrOp>(op)) {
    Block *dest = blockMap[brOp.getDest()];
    if (!dest)
      return brOp.emitError("unmapped branch destination"), failure();
    SmallVector<Value, 4> operands;
    unsigned argIdx = 0;
    for (Value operand : brOp.getOperands()) {
      if (argIdx >= dest->getNumArguments())
        return brOp.emitError("too many branch operands"), failure();
      if (auto ptrInfo = LookupPointer(pointerMap, operand)) {
        if (!isValidPointerInfo(*ptrInfo))
          return brOp.emitError("invalid pointer base for branch operand"),
                 failure();
        Type destArgTy = dest->getArgument(argIdx).getType();
        Value brPtr =
            MaterializeBranchPointer(builder, loc, *ptrInfo, destArgTy);
        if (!brPtr)
          return brOp.emitError("invalid pointer for branch operand"),
                 failure();
        operands.push_back(brPtr);
        ++argIdx;
        continue;
      }
      auto mapped = LookupValue(valueMap, operand);
      if (!mapped)
        return brOp.emitError("missing branch operand"), failure();
      operands.push_back(*mapped);
      ++argIdx;
    }
    cf::BranchOp::create(builder, loc, dest, operands);
    return true;
  }

  if (auto condBrOp = llvm::dyn_cast<LLVM::CondBrOp>(op)) {
    auto cond = LookupValue(valueMap, condBrOp.getCondition());
    if (!cond)
      return condBrOp.emitError("missing condition"), failure();

    Block *trueDest = blockMap[condBrOp.getTrueDest()];
    if (!trueDest)
      return condBrOp.emitError("unmapped true destination"), failure();
    SmallVector<Value, 4> trueOperands;
    unsigned trueArgIdx = 0;
    for (Value operand : condBrOp.getTrueDestOperands()) {
      if (trueArgIdx >= trueDest->getNumArguments())
        return condBrOp.emitError("too many true branch operands"), failure();
      if (auto ptrInfo = LookupPointer(pointerMap, operand)) {
        if (!isValidPointerInfo(*ptrInfo))
          return condBrOp.emitError("invalid pointer for true operand"),
                 failure();
        Type destArgTy = trueDest->getArgument(trueArgIdx).getType();
        Value brPtr =
            MaterializeBranchPointer(builder, loc, *ptrInfo, destArgTy);
        if (!brPtr)
          return condBrOp.emitError("invalid pointer for true operand"),
                 failure();
        trueOperands.push_back(brPtr);
        ++trueArgIdx;
        continue;
      }
      auto mapped = LookupValue(valueMap, operand);
      if (!mapped)
        return condBrOp.emitError("missing true operand"), failure();
      trueOperands.push_back(*mapped);
      ++trueArgIdx;
    }

    Block *falseDest = blockMap[condBrOp.getFalseDest()];
    if (!falseDest)
      return condBrOp.emitError("unmapped false destination"), failure();
    SmallVector<Value, 4> falseOperands;
    unsigned falseArgIdx = 0;
    for (Value operand : condBrOp.getFalseDestOperands()) {
      if (falseArgIdx >= falseDest->getNumArguments())
        return condBrOp.emitError("too many false branch operands"), failure();
      if (auto ptrInfo = LookupPointer(pointerMap, operand)) {
        if (!isValidPointerInfo(*ptrInfo))
          return condBrOp.emitError("invalid pointer for false operand"),
                 failure();
        Type destArgTy = falseDest->getArgument(falseArgIdx).getType();
        Value brPtr =
            MaterializeBranchPointer(builder, loc, *ptrInfo, destArgTy);
        if (!brPtr)
          return condBrOp.emitError("invalid pointer for false operand"),
                 failure();
        falseOperands.push_back(brPtr);
        ++falseArgIdx;
        continue;
      }
      auto mapped = LookupValue(valueMap, operand);
      if (!mapped)
        return condBrOp.emitError("missing false operand"), failure();
      falseOperands.push_back(*mapped);
      ++falseArgIdx;
    }

    cf::CondBranchOp::create(builder, loc, *cond, trueDest, trueOperands,
                              falseDest, falseOperands);
    return true;
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
    return true;
  }

  return false;
}

} // namespace loom::llvm_to_scf
