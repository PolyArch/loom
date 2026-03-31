// LLVM dialect to CF-stage conversion pass.
// Converts LLVM dialect to func/cf/arith/memref/math.
// General-purpose: handles arbitrary C programs, not just vecadd.
// Functions with unconvertible ops (varargs, inline asm) are skipped.

#include "LLVMToCFTypes.h"
#include "loom/Conversion/Passes.h"
#include "loom/Dialect/Fabric/FabricTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace loom;

namespace {

static bool isByteLikeType(Type type) {
  auto intTy = dyn_cast<IntegerType>(type);
  return intTy && intTy.getWidth() == 8;
}

static Operation *resolveCalleeOp(Operation *site, FlatSymbolRefAttr calleeAttr) {
  if (!site || !calleeAttr)
    return nullptr;

  if (Operation *callee =
          SymbolTable::lookupNearestSymbolFrom(site, calleeAttr))
    return callee;

  auto module = site->getParentOfType<ModuleOp>();
  if (!module)
    return nullptr;

  StringRef calleeName = calleeAttr.getAttr().getValue();
  for (auto llvmFunc : module.getOps<LLVM::LLVMFuncOp>()) {
    if (llvmFunc.getSymName() == calleeName)
      return llvmFunc;
  }
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.getSymName() == calleeName)
      return func;
  }
  return nullptr;
}

static Type inferPointerElemTypeFromUses(Value ptrVal, Operation *scope = nullptr,
                                         unsigned depth = 0) {
  if (depth > 8)
    return nullptr;

  Type bestType = nullptr;
  for (auto &use : ptrVal.getUses()) {
    Operation *user = use.getOwner();

    if (auto bitcast = dyn_cast<LLVM::BitcastOp>(user)) {
      if (use.getOperandNumber() == 0 &&
          isa<LLVM::LLVMPointerType>(bitcast.getType())) {
        Type fromBitcastUses = inferPointerElemTypeFromUses(bitcast.getResult(),
                                                            scope, depth + 1);
        if (fromBitcastUses)
          return fromBitcastUses;
      }
    }

    if (auto cast = dyn_cast<LLVM::AddrSpaceCastOp>(user)) {
      if (use.getOperandNumber() == 0 &&
          isa<LLVM::LLVMPointerType>(cast.getType())) {
        Type fromCastUses = inferPointerElemTypeFromUses(cast.getResult(), scope,
                                                         depth + 1);
        if (fromCastUses)
          return fromCastUses;
      }
    }

    if (auto gep = dyn_cast<LLVM::GEPOp>(user)) {
      if (use.getOperandNumber() == 0) {
        Type elemTy = gep.getElemType();
        // Skip struct-typed GEP elem types and look through to uses
        if (isa<LLVM::LLVMStructType>(elemTy)) {
          Type fromGepUses = inferPointerElemTypeFromUses(gep.getResult(), scope,
                                                          depth + 1);
          if (fromGepUses)
            return fromGepUses;
          continue;
        }
        elemTy = normalizeScalarType(user->getContext(), elemTy);
        if (!isByteLikeType(elemTy))
          return elemTy;
        if (!bestType)
          bestType = elemTy;
        Type fromGepUses = inferPointerElemTypeFromUses(gep.getResult(), scope,
                                                        depth + 1);
        if (fromGepUses)
          return fromGepUses;
      }
    }

    if (auto load = dyn_cast<LLVM::LoadOp>(user)) {
      Type loadTy =
          normalizeScalarType(user->getContext(), load.getResult().getType());
      if (!isa<IntegerType>(loadTy) || cast<IntegerType>(loadTy).getWidth() != 8)
        return loadTy;
      if (!bestType)
        bestType = loadTy;
    }

    if (auto store = dyn_cast<LLVM::StoreOp>(user)) {
      if (use.getOperandNumber() == 1) {
        Type valTy =
            normalizeScalarType(user->getContext(), store.getValue().getType());
        if (!isa<IntegerType>(valTy) || cast<IntegerType>(valTy).getWidth() != 8)
          return valTy;
        if (!bestType)
          bestType = valTy;
      }
    }

    if (auto br = dyn_cast<LLVM::BrOp>(user)) {
      unsigned idx = use.getOperandNumber();
      Block *dest = br.getDest();
      if (idx < dest->getNumArguments()) {
        Type fromDest = inferPointerElemTypeFromUses(dest->getArgument(idx),
                                                     scope, depth + 1);
        if (fromDest)
          return fromDest;
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
            Type fromDest = inferPointerElemTypeFromUses(dest->getArgument(argIdx),
                                                         scope, depth + 1);
            if (fromDest)
              return fromDest;
          }
        } else {
          Block *dest = condBr.getFalseDest();
          unsigned argIdx = idx - 1 - trueCount;
          if (argIdx < dest->getNumArguments()) {
            Type fromDest = inferPointerElemTypeFromUses(dest->getArgument(argIdx),
                                                         scope, depth + 1);
            if (fromDest)
              return fromDest;
          }
        }
      }
    }

    if (auto call = dyn_cast<LLVM::CallOp>(user)) {
      if (!scope)
        continue;
      auto calleeAttr = call.getCalleeAttr();
      if (!calleeAttr)
        continue;

      unsigned argIdx = use.getOperandNumber();
      Operation *calleeOp = resolveCalleeOp(call, calleeAttr);
      if (!calleeOp)
        continue;

      if (auto calleeLLVM = dyn_cast<LLVM::LLVMFuncOp>(calleeOp)) {
        if (calleeLLVM.isExternal())
          continue;
        Block &calleeEntry = calleeLLVM.getBody().front();
        if (argIdx >= calleeEntry.getNumArguments())
          continue;
        Type fromCallee = inferPointerElemTypeFromUses(
            calleeEntry.getArgument(argIdx), scope, depth + 1);
        if (fromCallee)
          return fromCallee;
        continue;
      }

      if (auto calleeFunc = dyn_cast<func::FuncOp>(calleeOp)) {
        auto fnTy = calleeFunc.getFunctionType();
        if (argIdx >= fnTy.getNumInputs())
          continue;
        auto memRefTy = dyn_cast<MemRefType>(fnTy.getInput(argIdx));
        if (!memRefTy)
          continue;
        Type elemTy = memRefTy.getElementType();
        if (!isByteLikeType(elemTy))
          return elemTy;
        if (!bestType)
          bestType = elemTy;
      }
    }
  }

  return bestType;
}

static Type inferPointerElemTypeFromIncoming(BlockArgument blockArg,
                                             Operation *scope,
                                             unsigned depth = 0) {
  if (depth > 8)
    return nullptr;

  Block *block = blockArg.getOwner();
  unsigned argIdx = blockArg.getArgNumber();
  Type bestType = nullptr;
  auto recordIncoming = [&](Value incoming) {
    Type incomingTy =
        inferPointerElemTypeFromUses(incoming, scope, depth + 1);
    if (!incomingTy)
      return false;
    if (!isByteLikeType(incomingTy)) {
      bestType = incomingTy;
      return true;
    }
    if (!bestType)
      bestType = incomingTy;
    return false;
  };

  for (Block *pred : block->getPredecessors()) {
    Operation *term = pred->getTerminator();
    if (auto br = dyn_cast<LLVM::BrOp>(term)) {
      if (br.getDest() == block && argIdx < br.getDestOperands().size() &&
          recordIncoming(br.getDestOperands()[argIdx]))
        return bestType;
      continue;
    }

    auto condBr = dyn_cast<LLVM::CondBrOp>(term);
    if (!condBr)
      continue;

    if (condBr.getTrueDest() == block &&
        argIdx < condBr.getTrueDestOperands().size() &&
        recordIncoming(condBr.getTrueDestOperands()[argIdx]))
      return bestType;

    if (condBr.getFalseDest() == block &&
        argIdx < condBr.getFalseDestOperands().size() &&
        recordIncoming(condBr.getFalseDestOperands()[argIdx]))
      return bestType;
  }

  return bestType;
}

static LogicalResult materializeReadOnlyArrayInit(Attribute initValue,
                                                  Type elementType,
                                                  int64_t numElements,
                                                  ElementsAttr &denseInit) {
  auto tensorType = RankedTensorType::get({numElements}, elementType);

  if (auto elements = dyn_cast<ElementsAttr>(initValue)) {
    auto shapedType = dyn_cast<ShapedType>(elements.getType());
    if (shapedType && shapedType.getRank() == 1 &&
        shapedType.getNumElements() == numElements &&
        shapedType.getElementType() == elementType) {
      denseInit = elements;
      return success();
    }

    if (auto splat = dyn_cast<SplatElementsAttr>(elements)) {
      Attribute splatValue = splat.getSplatValue<Attribute>();
      if (!isa<IntegerAttr, FloatAttr>(splatValue))
        return failure();
      denseInit = DenseElementsAttr::get(
          tensorType, SmallVector<Attribute>(numElements, splatValue));
      return success();
    }

    SmallVector<Attribute> values;
    values.reserve(numElements);
    for (Attribute value : elements.getValues<Attribute>()) {
      if (!isa<IntegerAttr, FloatAttr>(value))
        return failure();
      values.push_back(value);
    }
    if (static_cast<int64_t>(values.size()) != numElements)
      return failure();
    denseInit = DenseElementsAttr::get(tensorType, values);
    return success();
  }

  if (auto array = dyn_cast<ArrayAttr>(initValue)) {
    SmallVector<Attribute> values;
    values.reserve(array.size());
    for (Attribute value : array) {
      if (!isa<IntegerAttr, FloatAttr>(value))
        return failure();
      values.push_back(value);
    }
    if (static_cast<int64_t>(values.size()) != numElements)
      return failure();
    denseInit = DenseElementsAttr::get(tensorType, values);
    return success();
  }

  return failure();
}

struct ReadOnlyGlobalInfo {
  MemRefType memrefType;
  ElementsAttr initialValue;
  Type pointeeType;
};

static LogicalResult materializeReadOnlyGlobal(LLVM::GlobalOp global,
                                               ReadOnlyGlobalInfo &info) {
  if (!global.getConstant())
    return failure();
  if (global.getAddrSpace() != 0)
    return failure();

  auto arrayType = dyn_cast<LLVM::LLVMArrayType>(global.getType());
  if (!arrayType)
    return failure();

  Type elementType = arrayType.getElementType();
  if (!isa<IntegerType, FloatType>(elementType))
    return failure();

  Attribute initValue = global.getValueOrNull();
  if (!initValue) {
    if (Block *block = global.getInitializerBlock()) {
      auto ret = dyn_cast_or_null<LLVM::ReturnOp>(block->getTerminator());
      if (!ret || ret.getNumOperands() != 1)
        return failure();
      Value returned = ret.getOperand(0);
      if (auto cst = returned.getDefiningOp<LLVM::ConstantOp>()) {
        initValue = cst.getValue();
      } else if (isa<LLVM::UndefOp, LLVM::PoisonOp>(returned.getDefiningOp())) {
        auto tensorType =
            RankedTensorType::get({arrayType.getNumElements()}, elementType);
        SmallVector<Attribute> zeros;
        zeros.reserve(arrayType.getNumElements());
        for (uint64_t i = 0, e = arrayType.getNumElements(); i < e; ++i) {
          if (auto intTy = dyn_cast<IntegerType>(elementType)) {
            zeros.push_back(IntegerAttr::get(intTy, 0));
          } else if (auto floatTy = dyn_cast<FloatType>(elementType)) {
            zeros.push_back(FloatAttr::get(
                floatTy, APFloat::getZero(floatTy.getFloatSemantics())));
          } else {
            return failure();
          }
        }
        info.memrefType = MemRefType::get(
            {static_cast<int64_t>(arrayType.getNumElements())}, elementType);
        info.initialValue = DenseElementsAttr::get(tensorType, zeros);
        info.pointeeType = elementType;
        return success();
      } else {
        return failure();
      }
    } else {
      return failure();
    }
  }

  ElementsAttr denseInit;
  if (failed(materializeReadOnlyArrayInit(initValue, elementType,
                                          arrayType.getNumElements(),
                                          denseInit)))
    return failure();

  info.memrefType = MemRefType::get(
      {static_cast<int64_t>(arrayType.getNumElements())}, elementType);
  info.initialValue = denseInit;
  info.pointeeType = elementType;
  return success();
}

static Type getHomogeneousStructElementType(LLVM::LLVMStructType structTy,
                                            MLIRContext *ctx) {
  auto body = structTy.getBody();
  if (body.empty())
    return nullptr;

  Type firstTy = normalizeScalarType(ctx, body.front());
  if (isByteLikeType(firstTy))
    return nullptr;
  for (Type fieldTy : body.drop_front()) {
    if (normalizeScalarType(ctx, fieldTy) != firstTy)
      return nullptr;
  }
  return firstTy;
}

static LogicalResult materializeReadOnlyStructGlobal(LLVM::GlobalOp global,
                                                     ReadOnlyGlobalInfo &info) {
  if (!global.getConstant() || global.getAddrSpace() != 0)
    return failure();

  auto structType = dyn_cast<LLVM::LLVMStructType>(global.getType());
  if (!structType)
    return failure();

  Type elementType =
      getHomogeneousStructElementType(structType, global.getContext());
  if (!elementType)
    return failure();

  SmallVector<Attribute> attrs;
  auto collectFromValue = [&](auto &&self, Value value) -> LogicalResult {
    if (auto ret = value.getDefiningOp<LLVM::ReturnOp>()) {
      if (ret.getNumOperands() != 1)
        return failure();
      return self(self, ret.getOperand(0));
    }
    if (isa<LLVM::UndefOp, LLVM::PoisonOp>(value.getDefiningOp())) {
      attrs.clear();
      attrs.reserve(structType.getBody().size());
      for (Type fieldTy : structType.getBody()) {
        Type normalized = normalizeScalarType(value.getContext(), fieldTy);
        if (auto intTy = dyn_cast<IntegerType>(normalized))
          attrs.push_back(IntegerAttr::get(intTy, 0));
        else if (auto floatTy = dyn_cast<FloatType>(normalized))
          attrs.push_back(
              FloatAttr::get(floatTy,
                             APFloat::getZero(floatTy.getFloatSemantics())));
        else
          return failure();
      }
      return success();
    }
    if (auto insert = value.getDefiningOp<LLVM::InsertValueOp>()) {
      if (insert.getPosition().size() != 1)
        return failure();
      if (failed(self(self, insert.getContainer())))
        return failure();
      unsigned fieldIdx = insert.getPosition()[0];
      if (fieldIdx >= structType.getBody().size() || fieldIdx >= attrs.size())
        return failure();
      Value insertedValue = insert.getValue();
      Attribute fieldAttr;
      if (auto constant = insertedValue.getDefiningOp<LLVM::ConstantOp>())
        fieldAttr = constant.getValue();
      else if (auto zero = insertedValue.getDefiningOp<LLVM::ZeroOp>()) {
        Type normalized =
            normalizeScalarType(value.getContext(), zero.getType());
        if (auto intTy = dyn_cast<IntegerType>(normalized))
          fieldAttr = IntegerAttr::get(intTy, 0);
        else if (auto floatTy = dyn_cast<FloatType>(normalized))
          fieldAttr = FloatAttr::get(
              floatTy, APFloat::getZero(floatTy.getFloatSemantics()));
      }
      if (!fieldAttr || !isa<IntegerAttr, FloatAttr>(fieldAttr))
        return failure();
      attrs[fieldIdx] = fieldAttr;
      return success();
    }
    return failure();
  };

  Attribute initAttr = global.getValueOrNull();
  if (initAttr) {
    auto denseAttr = dyn_cast<DenseElementsAttr>(initAttr);
    if (!denseAttr)
      return failure();
    int64_t numFields = static_cast<int64_t>(structType.getBody().size());
    auto tensorType = RankedTensorType::get({numFields}, elementType);
    info.memrefType = MemRefType::get({numFields}, elementType);
    info.initialValue = denseAttr.reshape(tensorType);
    info.pointeeType = elementType;
    return success();
  }

  Block *initBlock = global.getInitializerBlock();
  if (!initBlock)
    return failure();
  auto ret = dyn_cast_or_null<LLVM::ReturnOp>(initBlock->getTerminator());
  if (!ret || ret.getNumOperands() != 1)
    return failure();
  if (failed(collectFromValue(collectFromValue, ret.getOperand(0))))
    return failure();

  int64_t numFields = static_cast<int64_t>(attrs.size());
  auto tensorType = RankedTensorType::get({numFields}, elementType);
  info.memrefType = MemRefType::get({numFields}, elementType);
  info.initialValue = DenseElementsAttr::get(tensorType, attrs);
  info.pointeeType = elementType;
  return success();
}

static void flattenAggregateLeafTypes(Type type, MLIRContext *ctx,
                                      SmallVectorImpl<Type> &leafTypes) {
  if (auto structTy = dyn_cast<LLVM::LLVMStructType>(type)) {
    for (Type fieldTy : structTy.getBody())
      flattenAggregateLeafTypes(fieldTy, ctx, leafTypes);
    return;
  }
  if (auto arrayTy = dyn_cast<LLVM::LLVMArrayType>(type)) {
    for (uint64_t i = 0; i < arrayTy.getNumElements(); ++i)
      flattenAggregateLeafTypes(arrayTy.getElementType(), ctx, leafTypes);
    return;
  }
  leafTypes.push_back(normalizeScalarType(ctx, type));
}

static unsigned countAggregateLeaves(Type type, MLIRContext *ctx) {
  SmallVector<Type, 8> leaves;
  flattenAggregateLeafTypes(type, ctx, leaves);
  return leaves.size();
}

static bool computeAggregateOffset(Type aggregateType, ArrayRef<int64_t> pos,
                                   MLIRContext *ctx, unsigned &offset,
                                   Type &subType) {
  offset = 0;
  subType = aggregateType;
  for (int64_t idx : pos) {
    if (auto structTy = dyn_cast<LLVM::LLVMStructType>(subType)) {
      if (idx < 0 || static_cast<size_t>(idx) >= structTy.getBody().size())
        return false;
      auto body = structTy.getBody();
      for (int64_t i = 0; i < idx; ++i)
        offset += countAggregateLeaves(body[i], ctx);
      subType = body[idx];
      continue;
    }
    if (auto arrayTy = dyn_cast<LLVM::LLVMArrayType>(subType)) {
      if (idx < 0 || static_cast<uint64_t>(idx) >= arrayTy.getNumElements())
        return false;
      unsigned elemLeaves = countAggregateLeaves(arrayTy.getElementType(), ctx);
      offset += static_cast<unsigned>(idx) * elemLeaves;
      subType = arrayTy.getElementType();
      continue;
    }
    return false;
  }
  return true;
}

// Per-function converter state.
struct FunctionConverter {
  LLVM::LLVMFuncOp llvmFunc;
  func::FuncOp newFunc;
  OpBuilder &builder;
  MLIRContext *ctx;
  const llvm::StringMap<ReadOnlyGlobalInfo> *readonlyGlobals;

  // Value mapping from old SSA values to new ones
  IRMapping valueMap;

  // Pointer tracking: maps LLVM pointer values to PointerInfo
  llvm::DenseMap<Value, PointerInfo> pointerMap;

  // Block mapping: old blocks -> new blocks
  llvm::DenseMap<Block *, Block *> blockMap;

  // Argument element types inferred from GEP usage
  llvm::DenseMap<unsigned, Type> argElemTypes;

  // Argument memref types (for pointer args)
  llvm::DenseMap<unsigned, MemRefType> argMemRefTypes;

  // Vector element tracking: maps LLVM vector values to their scalar components.
  // Used to decompose vector operations (from SRoA struct decomposition)
  // into individual scalar operations.
  llvm::DenseMap<Value, SmallVector<Value, 4>> vectorMap;

  // Aggregate element tracking for insertvalue/extractvalue.
  llvm::DenseMap<Value, SmallVector<Value, 4>> aggregateMap;

  Operation *failedOp = nullptr;

  FunctionConverter(LLVM::LLVMFuncOp func, OpBuilder &b,
                    const llvm::StringMap<ReadOnlyGlobalInfo> &globals)
      : llvmFunc(func), builder(b), ctx(func.getContext()),
        readonlyGlobals(&globals) {}

  LogicalResult convert();

private:
  LogicalResult createFuncOp();
  LogicalResult createBlocks();
  LogicalResult convertOps();
  LogicalResult convertOp(Operation *op);

  // Pointer operations
  LogicalResult convertGEP(LLVM::GEPOp op);
  LogicalResult convertLoad(LLVM::LoadOp op);
  LogicalResult convertStore(LLVM::StoreOp op);
  LogicalResult convertAlloca(LLVM::AllocaOp op);

  // Vector operations (from SRoA struct decomposition)
  LogicalResult convertExtractElement(LLVM::ExtractElementOp op);
  LogicalResult convertInsertElement(LLVM::InsertElementOp op);
  LogicalResult convertExtractValue(LLVM::ExtractValueOp op);
  LogicalResult convertInsertValue(LLVM::InsertValueOp op);

  // Arithmetic
  LogicalResult convertBinaryIntOp(Operation *op);
  LogicalResult convertBinaryFloatOp(Operation *op);
  LogicalResult convertCast(Operation *op);
  LogicalResult convertICmp(LLVM::ICmpOp op);
  LogicalResult convertFCmp(LLVM::FCmpOp op);
  LogicalResult convertSelect(LLVM::SelectOp op);
  LogicalResult convertConstant(LLVM::ConstantOp op);
  LogicalResult convertIntrinsic(LLVM::CallIntrinsicOp op);

  // Control flow
  LogicalResult convertBr(LLVM::BrOp op);
  LogicalResult convertCondBr(LLVM::CondBrOp op);
  LogicalResult convertReturn(LLVM::ReturnOp op);

  // Helpers
  Value lookup(Value v);
  PointerInfo lookupPtr(Value v);
  void mapValue(Value oldVal, Value newVal);
  void mapPointer(Value oldPtr, PointerInfo info);
  void mapVector(Value oldVec, SmallVector<Value, 4> elements);
  void mapAggregate(Value oldAgg, SmallVector<Value, 4> elements);
  SmallVector<Value, 4> lookupVector(Value v);
  SmallVector<Value, 4> lookupAggregate(Value v);
  SmallVector<Value> materializeBranchArgs(Location loc, OperandRange args,
                                           Block *dest);
  Value createIndexCast(Location loc, Value intVal);
  Value scaleIndex(Location loc, Value idx, Type fromElem, Type toElem);
  Value buildByteSwap(Location loc, Value value);
  Value buildFunnelShift(Location loc, Value lhs, Value rhs, Value amount,
                         bool isLeft);
  Value buildBitReverse(Location loc, Value value);
  Value buildCtPop(Location loc, Value value);
  Value buildCountZeros(Location loc, Value value, bool leading);
};

// Check if an LLVM function can be converted. Returns false for varargs,
// functions with inline asm, or other unsupported patterns.
static bool canConvertFunction(LLVM::LLVMFuncOp funcOp) {
  if (funcOp.isExternal())
    return false;
  // Check for vararg
  if (funcOp.getFunctionType().isVarArg())
    return false;
  // Check for inline asm
  bool hasInlineAsm = false;
  funcOp.walk([&](LLVM::InlineAsmOp) {
    hasInlineAsm = true;
    return WalkResult::interrupt();
  });
  return !hasInlineAsm;
}

//===----------------------------------------------------------------------===//
// FunctionConverter implementation
//===----------------------------------------------------------------------===//

LogicalResult FunctionConverter::convert() {
  argElemTypes = inferPointerElementTypes(llvmFunc);
  if (failed(createFuncOp()))
    return failure();
  if (failed(createBlocks()))
    return failure();
  if (failed(convertOps()))
    return failure();
  return success();
}

LogicalResult FunctionConverter::createFuncOp() {
  Location loc = llvmFunc.getLoc();
  auto llvmFuncType = llvmFunc.getFunctionType();
  auto llvmArgTypes = llvmFuncType.getParams();
  auto oldArgAttrs = llvmFunc.getArgAttrsAttr();

  SmallVector<Type> newArgTypes;
  SmallVector<DictionaryAttr> newArgAttrs;

  for (unsigned i = 0; i < llvmArgTypes.size(); ++i) {
    Type argTy = llvmArgTypes[i];

    // Extract LLVM argument attributes
    bool isNoAlias = false;
    if (oldArgAttrs) {
      auto dictAttr = cast<DictionaryAttr>(oldArgAttrs[i]);
      isNoAlias = dictAttr.get("llvm.noalias") != nullptr;
    }

    if (isa<LLVM::LLVMPointerType>(argTy)) {
      Type elemTy = argElemTypes.count(i) ? argElemTypes[i]
                                          : IntegerType::get(ctx, 8);
      // Normalize the element type to ensure it's memref-compatible
      elemTy = normalizeScalarType(ctx, elemTy);
      auto memrefTy = buildStridedMemRefType(ctx, elemTy);
      newArgTypes.push_back(memrefTy);
      argMemRefTypes[i] = memrefTy;

      SmallVector<NamedAttribute> attrs;
      if (isNoAlias)
        attrs.push_back({StringAttr::get(ctx, "loom.noalias"),
                         UnitAttr::get(ctx)});
      newArgAttrs.push_back(DictionaryAttr::get(ctx, attrs));
    } else {
      newArgTypes.push_back(normalizeScalarType(ctx, argTy));
      newArgAttrs.push_back(DictionaryAttr::get(ctx, {}));
    }
  }

  // Result types
  SmallVector<Type> newResultTypes;
  Type retTy = llvmFuncType.getReturnType();
  if (!isa<LLVM::LLVMVoidType>(retTy))
    newResultTypes.push_back(normalizeScalarType(ctx, retTy));

  auto funcType = FunctionType::get(ctx, newArgTypes, newResultTypes);
  builder.setInsertionPoint(llvmFunc);
  newFunc = func::FuncOp::create(builder, loc, llvmFunc.getSymName(), funcType);
  newFunc.setArgAttrsAttr(builder.getArrayAttr(
      SmallVector<Attribute>(newArgAttrs.begin(), newArgAttrs.end())));
  return success();
}

LogicalResult FunctionConverter::createBlocks() {
  Region &srcRegion = llvmFunc.getBody();
  Region &dstRegion = newFunc.getBody();

  // Create all destination blocks and map them
  for (Block &srcBlock : srcRegion) {
    Block *dstBlock = new Block();
    dstRegion.push_back(dstBlock);
    blockMap[&srcBlock] = dstBlock;
  }

  // Add block arguments with converted types and set up mappings
  for (Block &srcBlock : srcRegion) {
    Block *dstBlock = blockMap[&srcBlock];
    bool isEntry = (&srcBlock == &srcRegion.front());

    for (unsigned i = 0; i < srcBlock.getNumArguments(); ++i) {
      auto srcArg = srcBlock.getArgument(i);
      Type rawTy = srcArg.getType();

      if (isEntry && isa<LLVM::LLVMPointerType>(rawTy)) {
        Type newTy;
        if (argMemRefTypes.count(i))
          newTy = argMemRefTypes[i];
        else
          newTy = buildStridedMemRefType(ctx, IntegerType::get(ctx, 8));
        auto dstArg = dstBlock->addArgument(newTy, srcArg.getLoc());
        OpBuilder blockBuilder = OpBuilder::atBlockBegin(dstBlock);
        auto zeroIdx =
            arith::ConstantIndexOp::create(blockBuilder, srcArg.getLoc(), 0);
        Type elemTy;
        if (argElemTypes.count(i))
          elemTy = argElemTypes[i];
        else {
          elemTy = inferPointerElemTypeFromUses(srcArg);
          if (!elemTy) elemTy = IntegerType::get(ctx, 8);
          elemTy = normalizeScalarType(ctx, elemTy);
        }
        mapPointer(srcArg, {dstArg, zeroIdx, elemTy});
      } else if (isa<LLVM::LLVMPointerType>(rawTy)) {
        Type elemTy = inferPointerElemTypeFromUses(srcArg, llvmFunc);
        if (!elemTy || isByteLikeType(elemTy)) {
          Type incomingTy = inferPointerElemTypeFromIncoming(srcArg, llvmFunc);
          if (incomingTy)
            elemTy = incomingTy;
        }
        if (!elemTy) elemTy = IntegerType::get(ctx, 8);
        elemTy = normalizeScalarType(ctx, elemTy);
        Type newTy = buildStridedMemRefType(ctx, elemTy);
        auto dstArg = dstBlock->addArgument(newTy, srcArg.getLoc());
        OpBuilder blockBuilder = OpBuilder::atBlockBegin(dstBlock);
        auto zeroIdx =
            arith::ConstantIndexOp::create(blockBuilder, srcArg.getLoc(), 0);
        mapPointer(srcArg, {dstArg, zeroIdx, elemTy});
      } else if (auto vecTy = dyn_cast<VectorType>(rawTy)) {
        // Vector block arg (from phi): expand into N scalar block args
        Type scalarTy = normalizeScalarType(ctx, vecTy.getElementType());
        unsigned numElems = vecTy.getNumElements();
        SmallVector<Value, 4> scalarArgs;
        for (unsigned j = 0; j < numElems; ++j) {
          auto dstArg = dstBlock->addArgument(scalarTy, srcArg.getLoc());
          scalarArgs.push_back(dstArg);
        }
        mapVector(srcArg, std::move(scalarArgs));
      } else {
        Type newTy = normalizeScalarType(ctx, rawTy);
        auto dstArg = dstBlock->addArgument(newTy, srcArg.getLoc());
        mapValue(srcArg, dstArg);
      }
    }
  }

  return success();
}

LogicalResult FunctionConverter::convertOps() {
  Block &entryBlock = llvmFunc.getBody().front();
  for (Block *srcBlock : llvm::ReversePostOrderTraversal<Block *>(&entryBlock)) {
    Block *dstBlock = blockMap[srcBlock];
    builder.setInsertionPointToEnd(dstBlock);

    for (Operation &op : *srcBlock) {
      if (failed(convertOp(&op))) {
        failedOp = &op;
        return failure();
      }
    }
  }
  return success();
}

LogicalResult FunctionConverter::convertOp(Operation *op) {
  Location loc = op->getLoc();

  // Debug intrinsics: skip
  if (isa<LLVM::DbgValueOp, LLVM::DbgDeclareOp, LLVM::DbgLabelOp>(op))
    return success();

  // LLVM intrinsic calls (debug, lifetime, etc.): skip void ones,
  // fail on non-void ones we don't handle
  if (auto callIntr = dyn_cast<LLVM::CallIntrinsicOp>(op)) {
    StringRef name = callIntr.getIntrin();
    // Debug/lifetime/annotation intrinsics: skip
    if (name.starts_with("llvm.dbg.") || name.starts_with("llvm.lifetime.") ||
        name.starts_with("llvm.annotation") ||
        name.starts_with("llvm.var.annotation") ||
        name.starts_with("llvm.assume") ||
        name.starts_with("llvm.experimental"))
      return success();
    if (succeeded(convertIntrinsic(callIntr)))
      return success();
    // Void intrinsics we don't understand: skip with warning
    if (callIntr.getNumResults() == 0)
      return success();
    return op->emitError("unsupported LLVM intrinsic: ") << name;
  }

  // AddressOf (global references): track as pointer
  if (auto addrOf = dyn_cast<LLVM::AddressOfOp>(op)) {
    StringRef globalName =
        addrOf.getGlobalNameAttr().getLeafReference().getValue();
    auto it = readonlyGlobals->find(globalName);
    if (addrOf.getType().getAddressSpace() != 0)
      return op->emitError("unsupported global address space");

    MemRefType memrefType;
    Type pointeeType;
    if (it != readonlyGlobals->end()) {
      memrefType = it->second.memrefType;
      pointeeType = it->second.pointeeType;
    } else if (auto memrefGlobal =
                   SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
                       addrOf, addrOf.getGlobalNameAttr())) {
      memrefType = memrefGlobal.getType();
      pointeeType = memrefType.getElementType();
    } else if (auto llvmGlobal =
                   SymbolTable::lookupNearestSymbolFrom<LLVM::GlobalOp>(
                       addrOf, addrOf.getGlobalNameAttr())) {
      ReadOnlyGlobalInfo info;
      if (failed(materializeReadOnlyGlobal(llvmGlobal, info)))
        return failure();

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(llvmGlobal);
      auto materialized = memref::GlobalOp::create(
          builder, llvmGlobal.getLoc(), llvmGlobal.getSymName(),
          builder.getStringAttr("private"), info.memrefType,
          info.initialValue, /*constant=*/true, IntegerAttr());
      memrefType = materialized.getType();
      pointeeType = info.pointeeType;
      llvmGlobal.erase();
    } else {
      return failure();
    }

    auto getGlobal =
        memref::GetGlobalOp::create(builder, loc, memrefType, globalName);
    Value zeroIdx = arith::ConstantIndexOp::create(builder, loc, 0);
    mapPointer(addrOf.getResult(), {getGlobal, zeroIdx, pointeeType});
    return success();
  }

  // Constants
  if (auto c = dyn_cast<LLVM::ConstantOp>(op))
    return convertConstant(c);
  if (isa<LLVM::ZeroOp>(op)) {
    Type rawTy = op->getResult(0).getType();
    // Handle vector zero: create zero elements for each lane
    if (auto vecTy = dyn_cast<VectorType>(rawTy)) {
      Type elemTy = normalizeScalarType(ctx, vecTy.getElementType());
      unsigned numElems = vecTy.getNumElements();
      SmallVector<Value, 4> elems;
      for (unsigned i = 0; i < numElems; ++i) {
        Value zero;
        if (auto intTy = dyn_cast<IntegerType>(elemTy))
          zero = arith::ConstantIntOp::create(builder, loc, intTy, 0);
        else if (auto fTy = dyn_cast<FloatType>(elemTy))
          zero = arith::ConstantFloatOp::create(builder,
              loc, fTy, APFloat::getZero(fTy.getFloatSemantics()));
        else
          return op->emitError("unsupported zero vector element type");
        elems.push_back(zero);
      }
      mapVector(op->getResult(0), std::move(elems));
      return success();
    }
    Type ty = normalizeScalarType(ctx, rawTy);
    Value zero;
    if (auto intTy = dyn_cast<IntegerType>(ty))
      zero = arith::ConstantIntOp::create(builder, loc, intTy, 0);
    else if (auto fTy = dyn_cast<FloatType>(ty))
      zero = arith::ConstantFloatOp::create(builder,
          loc, fTy, APFloat::getZero(fTy.getFloatSemantics()));
    else
      return op->emitError("unsupported zero type");
    mapValue(op->getResult(0), zero);
    return success();
  }
  if (isa<LLVM::UndefOp, LLVM::PoisonOp>(op)) {
    // Undef/poison: create arith.constant 0 as placeholder
    Type rawTy = op->getResult(0).getType();
    if (isa<LLVM::LLVMPointerType>(rawTy)) {
      // Pointer undef: skip (will be handled if used)
      return success();
    }
    // Handle vector undef/poison by creating zero-initialized vector elements
    if (auto vecTy = dyn_cast<VectorType>(rawTy)) {
      Type elemTy = normalizeScalarType(ctx, vecTy.getElementType());
      unsigned numElems = vecTy.getNumElements();
      SmallVector<Value, 4> elems;
      for (unsigned i = 0; i < numElems; ++i) {
        Value zero;
        if (auto intTy = dyn_cast<IntegerType>(elemTy))
          zero = arith::ConstantIntOp::create(builder, loc, intTy, 0);
        else if (auto fTy = dyn_cast<FloatType>(elemTy))
          zero = arith::ConstantFloatOp::create(builder,
              loc, fTy, APFloat::getZero(fTy.getFloatSemantics()));
        else
          return success();
        elems.push_back(zero);
      }
      mapVector(op->getResult(0), std::move(elems));
      return success();
    }
    Type ty = normalizeScalarType(ctx, rawTy);
    Value zero;
    if (auto intTy = dyn_cast<IntegerType>(ty))
      zero = arith::ConstantIntOp::create(builder, loc, intTy, 0);
    else if (auto fTy = dyn_cast<FloatType>(ty))
      zero = arith::ConstantFloatOp::create(builder,
          loc, fTy, APFloat::getZero(fTy.getFloatSemantics()));
    else
      return success(); // skip unsupported undef types
    mapValue(op->getResult(0), zero);
    return success();
  }

  // Vector element operations (from SRoA struct decomposition)
  if (auto extractElem = dyn_cast<LLVM::ExtractElementOp>(op))
    return convertExtractElement(extractElem);
  if (auto insertElem = dyn_cast<LLVM::InsertElementOp>(op))
    return convertInsertElement(insertElem);
  if (auto extractVal = dyn_cast<LLVM::ExtractValueOp>(op))
    return convertExtractValue(extractVal);
  if (auto insertVal = dyn_cast<LLVM::InsertValueOp>(op))
    return convertInsertValue(insertVal);

  // Memory operations
  if (auto gep = dyn_cast<LLVM::GEPOp>(op))
    return convertGEP(gep);
  if (auto load = dyn_cast<LLVM::LoadOp>(op))
    return convertLoad(load);
  if (auto store = dyn_cast<LLVM::StoreOp>(op))
    return convertStore(store);
  if (auto alloca = dyn_cast<LLVM::AllocaOp>(op))
    return convertAlloca(alloca);

  // Integer binary ops
  if (isa<LLVM::AddOp, LLVM::SubOp, LLVM::MulOp, LLVM::SDivOp,
          LLVM::UDivOp, LLVM::SRemOp, LLVM::URemOp, LLVM::AndOp,
          LLVM::OrOp, LLVM::XOrOp, LLVM::ShlOp, LLVM::LShrOp,
          LLVM::AShrOp>(op))
    return convertBinaryIntOp(op);

  // Float binary ops
  if (isa<LLVM::FAddOp, LLVM::FSubOp, LLVM::FMulOp, LLVM::FDivOp,
          LLVM::FRemOp, LLVM::FNegOp>(op))
    return convertBinaryFloatOp(op);

  // Comparisons
  if (auto icmp = dyn_cast<LLVM::ICmpOp>(op))
    return convertICmp(icmp);
  if (auto fcmp = dyn_cast<LLVM::FCmpOp>(op))
    return convertFCmp(fcmp);

  // Casts
  if (isa<LLVM::ZExtOp, LLVM::SExtOp, LLVM::TruncOp, LLVM::FPExtOp,
          LLVM::FPTruncOp, LLVM::UIToFPOp, LLVM::SIToFPOp,
          LLVM::FPToSIOp, LLVM::FPToUIOp>(op))
    return convertCast(op);

  // Select
  if (auto sel = dyn_cast<LLVM::SelectOp>(op))
    return convertSelect(sel);

  // Control flow
  if (auto br = dyn_cast<LLVM::BrOp>(op))
    return convertBr(br);
  if (auto condBr = dyn_cast<LLVM::CondBrOp>(op))
    return convertCondBr(condBr);
  if (auto ret = dyn_cast<LLVM::ReturnOp>(op))
    return convertReturn(ret);

  // Calls: convert non-vararg function calls
  if (auto call = dyn_cast<LLVM::CallOp>(op)) {
    auto callee = call.getCalleeAttr();

    // Map results of skipped calls to zero/dummy values so downstream
    // uses don't fail.
    auto mapSkippedCallResults = [&]() -> LogicalResult {
      for (Value oldRes : call.getResults()) {
        Type oldTy = oldRes.getType();
        if (isa<LLVM::LLVMVoidType>(oldTy))
          continue;

        if (isa<LLVM::LLVMPointerType>(oldTy)) {
          Type elemTy = IntegerType::get(ctx, 8);
          auto plainMemRefTy = MemRefType::get({ShapedType::kDynamic}, elemTy);
          Value oneIdx = arith::ConstantIndexOp::create(builder, loc, 1);
          Value alloc = memref::AllocaOp::create(builder, loc, plainMemRefTy,
                                                 ValueRange{oneIdx});
          auto memrefTy = buildStridedMemRefType(ctx, elemTy);
          Value base = alloc;
          if (plainMemRefTy != memrefTy)
            base = memref::CastOp::create(builder, loc, memrefTy, alloc);
          Value zeroIdx = arith::ConstantIndexOp::create(builder, loc, 0);
          mapPointer(oldRes, {base, zeroIdx, elemTy});

          Type scalarTy = normalizeScalarType(ctx, oldTy);
          if (auto intTy = dyn_cast<IntegerType>(scalarTy))
            mapValue(oldRes,
                     arith::ConstantIntOp::create(builder, loc, intTy, 0));
          else if (isa<IndexType>(scalarTy))
            mapValue(oldRes, arith::ConstantIndexOp::create(builder, loc, 0));
          else
            return call.emitError("unsupported skipped pointer call result type");
          continue;
        }

        Type resTy = normalizeScalarType(ctx, oldTy);
        Value zero;
        if (auto intTy = dyn_cast<IntegerType>(resTy))
          zero = arith::ConstantIntOp::create(builder, loc, intTy, 0);
        else if (auto floatTy = dyn_cast<FloatType>(resTy))
          zero = arith::ConstantFloatOp::create(
              builder, loc, floatTy,
              APFloat::getZero(floatTy.getFloatSemantics()));
        else if (isa<IndexType>(resTy))
          zero = arith::ConstantIndexOp::create(builder, loc, 0);
        else
          return call.emitError("unsupported skipped call result type");
        mapValue(oldRes, zero);
      }
      return success();
    };

    // Skip runtime calls that don't map to CGRA dataflow.
    if (callee) {
      StringRef name = callee.getAttr().getValue();
      if (name == "free" || name == "memset" || name == "memcpy" ||
          name == "memmove")
        return mapSkippedCallResults();

      if (name == "printf" || name == "fprintf" || name == "sprintf" ||
          name == "snprintf" || name == "vprintf" || name == "vfprintf" ||
          name == "puts" || name == "fputs" || name == "putchar" ||
          name == "perror" || name == "fwrite" || name == "fread")
        return mapSkippedCallResults();

      auto *calleeOp = resolveCalleeOp(call, callee);
      auto calleeFunc = dyn_cast_or_null<LLVM::LLVMFuncOp>(calleeOp);
      if (calleeFunc && calleeFunc.isExternal() &&
          calleeFunc.getFunctionType().isVarArg())
        return mapSkippedCallResults();
    }

    SmallVector<Value> args;
    for (Value v : call.getOperands()) {
      if (isa<LLVM::LLVMPointerType>(v.getType())) {
        auto pi = lookupPtr(v);
        if (!pi.isValid())
          return call.emitError("cannot convert pointer call argument");
        args.push_back(pi.base);
      } else {
        args.push_back(lookup(v));
      }
    }
    if (!callee)
      return call.emitError("indirect calls not supported");

    SmallVector<Type> resultTypes;
    SmallVector<unsigned> ptrResultIndices;
    for (auto [ri, t] : llvm::enumerate(call.getResultTypes())) {
      if (isa<LLVM::LLVMVoidType>(t))
        continue;
      if (isa<LLVM::LLVMPointerType>(t))
        ptrResultIndices.push_back(ri);
      resultTypes.push_back(normalizeScalarType(ctx, t));
    }
    auto newCall = func::CallOp::create(builder,
        op->getLoc(), callee.getAttr(), resultTypes, args);
    for (unsigned i = 0; i < newCall->getNumResults(); ++i) {
      Value oldRes = call->getResult(i);
      Value newRes = newCall->getResult(i);

      // If the original result was a pointer, track it in the pointer map
      if (isa<LLVM::LLVMPointerType>(oldRes.getType())) {
        // Infer element type from downstream uses
        Type ptrElemTy = inferPointerElemTypeFromUses(oldRes, llvmFunc);
        if (!ptrElemTy)
          ptrElemTy = IntegerType::get(ctx, 8);
        ptrElemTy = normalizeScalarType(ctx, ptrElemTy);

        // Convert integer result to index for base offset
        Value baseIdx = createIndexCast(call.getLoc(), newRes);
        unsigned elemBytes = getTypeBitWidth(ptrElemTy) / 8;
        if (elemBytes > 1) {
          Value divisor = arith::ConstantIndexOp::create(builder,
              call.getLoc(), elemBytes);
          baseIdx = arith::DivUIOp::create(builder, call.getLoc(),
              baseIdx, divisor);
        }

        // Create a memref for the returned pointer using reinterpret_cast.
        // For malloc-like calls, the returned pointer starts a new array.
        auto newMemRefTy = buildStridedMemRefType(ctx, ptrElemTy);
        auto plainMemRefTy = MemRefType::get({ShapedType::kDynamic}, ptrElemTy);
        Value scratchSize =
            arith::ConstantIndexOp::create(builder, call.getLoc(), 1);
        Value scratch = memref::AllocaOp::create(builder, call.getLoc(),
                                                 plainMemRefTy,
                                                 ValueRange{scratchSize});
        Value memrefBase = scratch;
        if (plainMemRefTy != newMemRefTy) {
          memrefBase = memref::CastOp::create(builder, call.getLoc(),
                                              newMemRefTy, scratch);
        }
        mapPointer(oldRes, {memrefBase, baseIdx, ptrElemTy});
        // Also map as a scalar value for non-pointer uses
        mapValue(oldRes, newRes);
      } else {
        mapValue(oldRes, newRes);
      }
    }
    return success();
  }

  // Skip LLVM metadata/annotation ops silently
  if (isa<LLVM::GlobalDtorsOp, LLVM::GlobalCtorsOp,
          LLVM::ComdatOp, LLVM::ModuleFlagsOp>(op))
    return success();

  // Freeze: pass through
  if (auto freeze = dyn_cast<LLVM::FreezeOp>(op)) {
    if (isa<LLVM::LLVMPointerType>(freeze.getType())) {
      auto pi = lookupPtr(freeze.getVal());
      if (pi.isValid())
        mapPointer(freeze.getResult(), pi);
    } else {
      mapValue(freeze.getResult(), lookup(freeze.getVal()));
    }
    return success();
  }

  return op->emitError("unsupported LLVM op: ") << op->getName();
}

//===----------------------------------------------------------------------===//
// Memory operations
//===----------------------------------------------------------------------===//

LogicalResult FunctionConverter::convertGEP(LLVM::GEPOp op) {
  Location loc = op.getLoc();
  auto baseInfo = lookupPtr(op.getBase());
  if (!baseInfo.isValid())
    return op.emitError("GEP base pointer not found");

  Type elemTy = op.getElemType();
  auto indices = op.getIndices();
  auto dynamicIndices = op.getDynamicIndices();

  // Simple case: single dynamic index, non-struct element type
  // gep base[idx] where base is a typed pointer to a scalar/array
  if (dynamicIndices.size() == 1 && indices.size() == 1 &&
      !isa<LLVM::LLVMStructType>(elemTy)) {
    Value idx = lookup(dynamicIndices[0]);
    Value idxAsIndex = createIndexCast(loc, idx);

    // Scale index if element types differ
    Value finalIdx;
    if (baseInfo.elementType == elemTy) {
      finalIdx = arith::AddIOp::create(builder, loc, baseInfo.index, idxAsIndex);
    } else {
      // Scale: new_offset = base.index * sizeof(baseElem) / sizeof(elemTy)
      //                     + idx
      Value scaledBase = scaleIndex(loc, baseInfo.index,
                                     baseInfo.elementType, elemTy);
      finalIdx = arith::AddIOp::create(builder, loc, scaledBase, idxAsIndex);
    }

    mapPointer(op.getResult(), {baseInfo.base, finalIdx, elemTy});
    return success();
  }

  // Multi-index GEP or struct-typed GEP: compute byte offset.
  // LLVM GEP semantics:
  //   gep elemTy, ptr, idx0, idx1, ...
  //   - idx0 indexes into an array of elemTy (scales by sizeof(elemTy))
  //   - idx1 indexes into elemTy:
  //       if elemTy is struct -> constant field index (byte offset of field)
  //       if elemTy is array  -> dynamic index (scales by array element size)
  //   - etc. for deeper nesting

  // Start with base's existing offset in bytes
  Value byteOffset;
  unsigned baseElemBytes = getTypeBitWidth(baseInfo.elementType) / 8;
  if (baseElemBytes == 0) baseElemBytes = 1;
  if (baseElemBytes != 1) {
    Value scale = arith::ConstantIndexOp::create(builder, loc, baseElemBytes);
    byteOffset = arith::MulIOp::create(builder, loc, baseInfo.index, scale);
  } else {
    byteOffset = baseInfo.index;
  }

  // Walk through GEP indices, tracking the current type at each level
  Type currentType = elemTy;
  unsigned dynIdx = 0;
  bool isFirstIndex = true;

  for (auto idxEntry : indices) {
    if (isFirstIndex) {
      // First index: scales by sizeof(elemTy) (array of the element type)
      isFirstIndex = false;

      Value idxVal;
      if (auto constIdx = dyn_cast<IntegerAttr>(idxEntry)) {
        idxVal = arith::ConstantIndexOp::create(builder, loc,
                                                 constIdx.getInt());
      } else {
        if (dynIdx >= dynamicIndices.size())
          return op.emitError("GEP: more indices than dynamic operands");
        idxVal = createIndexCast(loc, lookup(dynamicIndices[dynIdx++]));
      }

      unsigned strideBytes = getTypeBitWidth(currentType) / 8;
      if (strideBytes == 0) strideBytes = 1;

      if (strideBytes != 1) {
        Value stride = arith::ConstantIndexOp::create(builder, loc, strideBytes);
        Value contribution = arith::MulIOp::create(builder, loc, idxVal, stride);
        byteOffset = arith::AddIOp::create(builder, loc, byteOffset, contribution);
      } else {
        byteOffset = arith::AddIOp::create(builder, loc, byteOffset, idxVal);
      }
      // currentType stays as elemTy (we're now pointing into that type)
      continue;
    }

    // Subsequent indices: depend on the current type
    if (auto structTy = dyn_cast<LLVM::LLVMStructType>(currentType)) {
      // Struct field access: index must be a constant
      auto constIdx = dyn_cast<IntegerAttr>(idxEntry);
      if (!constIdx)
        return op.emitError("GEP: struct field index must be constant");

      unsigned fieldIdx = constIdx.getInt();
      unsigned fieldOffset = getStructFieldByteOffset(currentType, fieldIdx);
      if (fieldOffset != 0) {
        Value offsetVal = arith::ConstantIndexOp::create(builder, loc, fieldOffset);
        byteOffset = arith::AddIOp::create(builder, loc, byteOffset, offsetVal);
      }

      // Advance to field type
      Type fieldTy = getStructFieldType(currentType, fieldIdx);
      if (!fieldTy)
        return op.emitError("GEP: invalid struct field index");
      currentType = fieldTy;
    } else if (auto arrayTy = dyn_cast<LLVM::LLVMArrayType>(currentType)) {
      // Array element access
      Value idxVal;
      if (auto constIdx = dyn_cast<IntegerAttr>(idxEntry)) {
        idxVal = arith::ConstantIndexOp::create(builder, loc,
                                                 constIdx.getInt());
      } else {
        if (dynIdx >= dynamicIndices.size())
          return op.emitError("GEP: more indices than dynamic operands");
        idxVal = createIndexCast(loc, lookup(dynamicIndices[dynIdx++]));
      }

      Type innerTy = arrayTy.getElementType();
      unsigned strideBytes = getTypeBitWidth(innerTy) / 8;
      if (strideBytes == 0) strideBytes = 1;

      if (strideBytes != 1) {
        Value stride = arith::ConstantIndexOp::create(builder, loc, strideBytes);
        Value contribution = arith::MulIOp::create(builder, loc, idxVal, stride);
        byteOffset = arith::AddIOp::create(builder, loc, byteOffset, contribution);
      } else {
        byteOffset = arith::AddIOp::create(builder, loc, byteOffset, idxVal);
      }
      currentType = innerTy;
    } else {
      // Scalar type: treat as array index
      Value idxVal;
      if (auto constIdx = dyn_cast<IntegerAttr>(idxEntry)) {
        idxVal = arith::ConstantIndexOp::create(builder, loc,
                                                 constIdx.getInt());
      } else {
        if (dynIdx >= dynamicIndices.size())
          return op.emitError("GEP: more indices than dynamic operands");
        idxVal = createIndexCast(loc, lookup(dynamicIndices[dynIdx++]));
      }

      unsigned strideBytes = getTypeBitWidth(currentType) / 8;
      if (strideBytes == 0) strideBytes = 1;

      if (strideBytes != 1) {
        Value stride = arith::ConstantIndexOp::create(builder, loc, strideBytes);
        Value contribution = arith::MulIOp::create(builder, loc, idxVal, stride);
        byteOffset = arith::AddIOp::create(builder, loc, byteOffset, contribution);
      } else {
        byteOffset = arith::AddIOp::create(builder, loc, byteOffset, idxVal);
      }
    }
  }

  // Determine the final element type for the resulting pointer.
  // For struct types used as arrays of homogeneous scalars (e.g., cmplx_t
  // with two floats), use the scalar element type. This enables seamless
  // vector load/store decomposition.
  Type finalElemTy = currentType;
  if (auto structTy = dyn_cast<LLVM::LLVMStructType>(finalElemTy)) {
    // Check if all fields have the same type
    auto body = structTy.getBody();
    if (!body.empty()) {
      Type firstFieldTy = normalizeScalarType(ctx, body[0]);
      bool allSame = true;
      for (unsigned fi = 1; fi < body.size(); ++fi) {
        if (normalizeScalarType(ctx, body[fi]) != firstFieldTy) {
          allSame = false;
          break;
        }
      }
      if (allSame)
        finalElemTy = firstFieldTy;
      else
        finalElemTy = IntegerType::get(ctx, 8);
    } else {
      finalElemTy = IntegerType::get(ctx, 8);
    }
  } else if (isa<LLVM::LLVMArrayType>(finalElemTy)) {
    finalElemTy = IntegerType::get(ctx, 8);
  } else {
    finalElemTy = normalizeScalarType(ctx, finalElemTy);
  }

  // Convert byte offset to element offset
  unsigned finalElemBytes = getTypeBitWidth(finalElemTy) / 8;
  if (finalElemBytes == 0) finalElemBytes = 1;
  Value finalIdx;
  if (finalElemBytes != 1) {
    Value divisor = arith::ConstantIndexOp::create(builder, loc, finalElemBytes);
    finalIdx = arith::DivUIOp::create(builder, loc, byteOffset, divisor);
  } else {
    finalIdx = byteOffset;
  }

  // If the base memref element type differs from finalElemTy, we may need to
  // reinterpret. Keep the base memref as-is and just record the new element type.
  mapPointer(op.getResult(), {baseInfo.base, finalIdx, finalElemTy});
  return success();
}

LogicalResult FunctionConverter::convertLoad(LLVM::LoadOp op) {
  Location loc = op.getLoc();
  auto addrInfo = lookupPtr(op.getAddr());
  if (!addrInfo.isValid())
    return op.emitError("load address pointer not found");

  // Handle vector-typed loads by decomposing into scalar loads.
  // This handles patterns from SRoA where struct loads become vector loads
  // (e.g., load <2 x float> from a cmplx_t pointer).
  if (auto vecTy = dyn_cast<VectorType>(op.getResult().getType())) {
    Type scalarTy = normalizeScalarType(ctx, vecTy.getElementType());
    unsigned numElems = vecTy.getNumElements();
    Value baseIdx = addrInfo.index;

    // Scale base index from current element type to scalar element type
    if (addrInfo.elementType != scalarTy) {
      unsigned srcBits = getTypeBitWidth(addrInfo.elementType);
      unsigned dstBits = getTypeBitWidth(scalarTy);
      if (srcBits != dstBits && srcBits > 0 && dstBits > 0)
        baseIdx = scaleIndex(loc, baseIdx, addrInfo.elementType, scalarTy);
    }

    SmallVector<Value, 4> elems;
    for (unsigned i = 0; i < numElems; ++i) {
      Value elemIdx = baseIdx;
      if (i != 0) {
        Value offset = arith::ConstantIndexOp::create(builder, loc, i);
        elemIdx = arith::AddIOp::create(builder, loc, baseIdx, offset);
      }

      // Create a memref with the correct scalar element type
      auto scalarMemRefTy = buildStridedMemRefType(ctx, scalarTy);
      Value scalarBase = addrInfo.base;
      if (scalarBase.getType() != scalarMemRefTy)
        scalarBase = memref::CastOp::create(builder, loc, scalarMemRefTy,
                                            addrInfo.base);

      auto loadOp = memref::LoadOp::create(builder, loc, scalarBase,
                                            ValueRange{elemIdx});
      Value result = loadOp.getResult();
      if (result.getType() != scalarTy)
        result = arith::BitcastOp::create(builder, loc, scalarTy, result);
      elems.push_back(result);
    }

    mapVector(op.getResult(), std::move(elems));
    return success();
  }

  // If the load result is a pointer type, handle indirect pointer access.
  // This occurs in patterns like CSR graph access: g->row_ptr[i] where
  // row_ptr is a pointer field loaded from a struct.
  // We model the loaded pointer as a new memref base by loading the index
  // value (pointer-as-integer) and creating a PointerInfo that can be
  // used for subsequent GEP/load/store operations.
  if (isa<LLVM::LLVMPointerType>(op.getResult().getType())) {
    // Infer the element type from downstream uses of this loaded pointer
    Type ptrElemTy = inferPointerElemTypeFromUses(op.getResult(), llvmFunc);
    if (!ptrElemTy)
      ptrElemTy = IntegerType::get(ctx, 8);
    ptrElemTy = normalizeScalarType(ctx, ptrElemTy);

    // Load the pointer value as an integer (index-width)
    Type idxIntTy = loom::fabric::getIndexIntegerType(ctx);
    Value idx = addrInfo.index;

    // Scale index to match the load access width (pointer = index-width integer)
    if (addrInfo.elementType != idxIntTy) {
      unsigned srcBits = getTypeBitWidth(addrInfo.elementType);
      unsigned dstBits = getTypeBitWidth(idxIntTy);
      if (srcBits != dstBits && srcBits > 0 && dstBits > 0)
        idx = scaleIndex(loc, idx, addrInfo.elementType, idxIntTy);
    }

    auto loadOp = memref::LoadOp::create(builder, loc, addrInfo.base,
                                          ValueRange{idx});
    Value loadedVal = loadOp.getResult();

    // Convert loaded value to index-width integer
    if (loadedVal.getType() != idxIntTy) {
      unsigned loadBits = getTypeBitWidth(loadedVal.getType());
      unsigned idxBits = getTypeBitWidth(idxIntTy);
      if (isa<FloatType>(loadedVal.getType())) {
        // Float memref element: bitcast to same-width integer first
        auto intOfSameWidth = IntegerType::get(ctx, loadBits);
        loadedVal = arith::BitcastOp::create(builder, loc, intOfSameWidth,
                                              loadedVal);
        if (loadBits < idxBits)
          loadedVal = arith::ExtUIOp::create(builder, loc, idxIntTy, loadedVal);
        else if (loadBits > idxBits)
          loadedVal = arith::TruncIOp::create(builder, loc, idxIntTy, loadedVal);
      } else if (loadBits < idxBits)
        loadedVal = arith::ExtUIOp::create(builder, loc, idxIntTy, loadedVal);
      else if (loadBits > idxBits)
        loadedVal = arith::TruncIOp::create(builder, loc, idxIntTy, loadedVal);
    }

    // Convert the loaded integer to an index value for use as a base offset.
    // Scale from byte address to element offset.
    Value baseIdx = createIndexCast(loc, loadedVal);
    unsigned elemBytes = getTypeBitWidth(ptrElemTy) / 8;
    if (elemBytes > 1) {
      Value divisor = arith::ConstantIndexOp::create(builder, loc, elemBytes);
      baseIdx = arith::DivUIOp::create(builder, loc, baseIdx, divisor);
    }

    // Create a new memref for the target array via reinterpret_cast.
    // In the flat memory model, the loaded pointer addresses the same
    // memory space, so we reinterpret the root memref as a new type.
    auto newMemRefTy = buildStridedMemRefType(ctx, ptrElemTy);

    // Find the root memref base (walk up the base chain)
    Value rootBase = addrInfo.base;
    auto rootCast = memref::CastOp::create(builder, loc, newMemRefTy, rootBase);

    mapPointer(op.getResult(), {rootCast, baseIdx, ptrElemTy});
    // Also map as a regular value (loaded integer) for non-pointer uses
    mapValue(op.getResult(), loadedVal);
    return success();
  }

  Type accessType = normalizeScalarType(ctx, op.getResult().getType());
  Value idx = addrInfo.index;

  // Scale index if element types differ
  if (addrInfo.elementType != accessType) {
    unsigned srcBits = getTypeBitWidth(addrInfo.elementType);
    unsigned dstBits = getTypeBitWidth(accessType);
    if (srcBits != dstBits && srcBits > 0 && dstBits > 0) {
      idx = scaleIndex(loc, idx, addrInfo.elementType, accessType);
    }
  }

  auto loadOp = memref::LoadOp::create(builder, loc, addrInfo.base,
                                                ValueRange{idx});
  Value result = loadOp.getResult();

  // Cast if types differ
  if (result.getType() != accessType) {
    unsigned resBits = getTypeBitWidth(result.getType());
    unsigned accBits = getTypeBitWidth(accessType);
    if (resBits == accBits) {
      result = arith::BitcastOp::create(builder, loc, accessType, result);
    } else if (isa<IntegerType>(result.getType()) && isa<IntegerType>(accessType)) {
      if (resBits < accBits)
        result = arith::ExtUIOp::create(builder, loc, accessType, result);
      else
        result = arith::TruncIOp::create(builder, loc, accessType, result);
    }
    // For other type mismatches (different width float/int), skip the cast
    // and use the loaded value as-is (will be caught by MLIR verifier)
  }

  mapValue(op.getResult(), result);
  return success();
}

LogicalResult FunctionConverter::convertStore(LLVM::StoreOp op) {
  Location loc = op.getLoc();
  auto addrInfo = lookupPtr(op.getAddr());
  if (!addrInfo.isValid())
    return op.emitError("store address pointer not found");

  // Skip pointer stores
  if (isa<LLVM::LLVMPointerType>(op.getValue().getType()))
    return success();

  // Handle vector-typed stores by decomposing into scalar stores.
  if (isa<VectorType>(op.getValue().getType())) {
    auto elems = lookupVector(op.getValue());
    if (elems.empty())
      return op.emitError("vector store value not tracked");

    Type scalarTy = elems[0].getType();
    Value baseIdx = addrInfo.index;

    // Scale base index from current element type to scalar element type
    if (addrInfo.elementType != scalarTy) {
      unsigned srcBits = getTypeBitWidth(addrInfo.elementType);
      unsigned dstBits = getTypeBitWidth(scalarTy);
      if (srcBits != dstBits && srcBits > 0 && dstBits > 0)
        baseIdx = scaleIndex(loc, baseIdx, addrInfo.elementType, scalarTy);
    }

    auto scalarMemRefTy = buildStridedMemRefType(ctx, scalarTy);
    Value scalarBase = addrInfo.base;
    if (scalarBase.getType() != scalarMemRefTy)
      scalarBase = memref::CastOp::create(builder, loc, scalarMemRefTy,
                                          addrInfo.base);

    for (unsigned i = 0; i < elems.size(); ++i) {
      Value elemIdx = baseIdx;
      if (i != 0) {
        Value offset = arith::ConstantIndexOp::create(builder, loc, i);
        elemIdx = arith::AddIOp::create(builder, loc, baseIdx, offset);
      }
      Value val = elems[i];
      if (val.getType() != scalarTy)
        val = arith::BitcastOp::create(builder, loc, scalarTy, val);
      memref::StoreOp::create(builder, loc, val, scalarBase, ValueRange{elemIdx});
    }
    return success();
  }

  Value val = lookup(op.getValue());
  Value idx = addrInfo.index;

  Type accessType = val.getType();
  if (addrInfo.elementType != accessType) {
    unsigned srcBits = getTypeBitWidth(addrInfo.elementType);
    unsigned dstBits = getTypeBitWidth(accessType);
    if (srcBits != dstBits && srcBits > 0 && dstBits > 0) {
      idx = scaleIndex(loc, idx, addrInfo.elementType, accessType);
    }
    if (addrInfo.elementType != accessType &&
        getTypeBitWidth(addrInfo.elementType) == getTypeBitWidth(accessType)) {
      val = arith::BitcastOp::create(builder, loc, addrInfo.elementType, val);
    }
  }

  memref::StoreOp::create(builder, loc, val, addrInfo.base, ValueRange{idx});
  return success();
}

LogicalResult FunctionConverter::convertAlloca(LLVM::AllocaOp op) {
  Location loc = op.getLoc();
  uint64_t elementCount = 1;
  Type elemTy = op.getElemType();
  if (!elemTy)
    elemTy = IntegerType::get(ctx, 8);
  elemTy = flattenAllocaElementType(ctx, elemTy, elementCount);
  if (!elemTy)
    return op.emitError("unsupported alloca element type");

  Value size = lookup(op.getArraySize());
  Value sizeIdx = createIndexCast(loc, size);
  if (elementCount != 1) {
    Value countVal =
        arith::ConstantIndexOp::create(builder, loc, elementCount);
    sizeIdx = arith::MulIOp::create(builder, loc, sizeIdx, countVal);
  }

  auto plainMemRefTy = MemRefType::get({ShapedType::kDynamic}, elemTy);
  auto alloc = memref::AllocaOp::create(builder, loc, plainMemRefTy,
                                        ValueRange{sizeIdx});
  Value base = alloc;
  auto memrefTy = buildStridedMemRefType(ctx, elemTy);
  if (plainMemRefTy != memrefTy)
    base = memref::CastOp::create(builder, loc, memrefTy, alloc);
  Value zeroIdx = arith::ConstantIndexOp::create(builder, loc, 0);
  mapPointer(op.getResult(), {base, zeroIdx, elemTy});
  return success();
}

//===----------------------------------------------------------------------===//
// Vector operations (from SRoA struct decomposition)
//===----------------------------------------------------------------------===//

LogicalResult FunctionConverter::convertExtractElement(LLVM::ExtractElementOp op) {
  auto elems = lookupVector(op.getVector());
  if (elems.empty())
    return op.emitError("extractelement: vector operand not tracked");

  // Get the constant index
  Value position = op.getPosition();
  // Try to resolve the index from the value map
  auto *defOp = position.getDefiningOp();
  int64_t constIdx = -1;
  if (defOp) {
    if (auto constOp = dyn_cast<LLVM::ConstantOp>(defOp)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
        constIdx = intAttr.getInt();
    }
    if (constIdx < 0) {
      // Try from mapped arith constant
      Value mapped = lookup(position);
      if (mapped) {
        if (auto *mappedDef = mapped.getDefiningOp()) {
          if (auto arithConst = dyn_cast<arith::ConstantOp>(mappedDef)) {
            if (auto intAttr = dyn_cast<IntegerAttr>(arithConst.getValue()))
              constIdx = intAttr.getInt();
          }
        }
      }
    }
  }

  if (constIdx < 0 || constIdx >= (int64_t)elems.size())
    return op.emitError("extractelement: non-constant or out-of-range index");

  mapValue(op.getResult(), elems[constIdx]);
  return success();
}

LogicalResult FunctionConverter::convertInsertElement(LLVM::InsertElementOp op) {
  // Get existing vector elements (or create zero-initialized if from poison)
  auto elems = lookupVector(op.getVector());
  if (elems.empty()) {
    // The vector might be undef/poison - create zero-initialized elements
    auto vecTy = dyn_cast<VectorType>(op.getVector().getType());
    if (!vecTy)
      return op.emitError("insertelement: non-fixed vector type");
    Type scalarTy = normalizeScalarType(ctx, vecTy.getElementType());
    unsigned numElems = vecTy.getNumElements();
    for (unsigned i = 0; i < numElems; ++i) {
      Value zero;
      if (auto intTy = dyn_cast<IntegerType>(scalarTy))
        zero = arith::ConstantIntOp::create(builder, op.getLoc(), intTy, 0);
      else if (auto fTy = dyn_cast<FloatType>(scalarTy))
        zero = arith::ConstantFloatOp::create(builder, op.getLoc(), fTy,
            APFloat::getZero(fTy.getFloatSemantics()));
      else
        return op.emitError("insertelement: unsupported element type");
      elems.push_back(zero);
    }
  }

  // Get the constant index
  Value position = op.getPosition();
  auto *defOp = position.getDefiningOp();
  int64_t constIdx = -1;
  if (defOp) {
    if (auto constOp = dyn_cast<LLVM::ConstantOp>(defOp)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
        constIdx = intAttr.getInt();
    }
    if (constIdx < 0) {
      Value mapped = lookup(position);
      if (mapped) {
        if (auto *mappedDef = mapped.getDefiningOp()) {
          if (auto arithConst = dyn_cast<arith::ConstantOp>(mappedDef)) {
            if (auto intAttr = dyn_cast<IntegerAttr>(arithConst.getValue()))
              constIdx = intAttr.getInt();
          }
        }
      }
    }
  }

  if (constIdx < 0 || constIdx >= (int64_t)elems.size())
    return op.emitError("insertelement: non-constant or out-of-range index");

  // Replace the element at the given index
  Value newVal = lookup(op.getValue());
  if (!newVal)
    return op.emitError("insertelement: value operand not mapped");
  SmallVector<Value, 4> newElems(elems.begin(), elems.end());
  newElems[constIdx] = newVal;
  mapVector(op.getResult(), std::move(newElems));
  return success();
}

LogicalResult FunctionConverter::convertExtractValue(LLVM::ExtractValueOp op) {
  auto elems = lookupAggregate(op.getContainer());
  if (elems.empty())
    return op.emitError("extractvalue: aggregate operand not tracked");

  unsigned offset = 0;
  Type subType;
  if (!computeAggregateOffset(op.getContainer().getType(), op.getPosition(),
                              ctx, offset, subType))
    return op.emitError("extractvalue: unsupported aggregate position");

  unsigned subLeaves = countAggregateLeaves(subType, ctx);
  if (offset + subLeaves > elems.size())
    return op.emitError("extractvalue: aggregate index out of range");

  if (subLeaves == 1) {
    Value v = elems[offset];
    Type dstTy = normalizeScalarType(ctx, op.getType());
    if (v.getType() != dstTy) {
      if (isa<IndexType>(dstTy))
        v = createIndexCast(op.getLoc(), v);
      else if (isa<IndexType>(v.getType()))
        v = arith::IndexCastOp::create(builder, op.getLoc(), dstTy, v);
      else if (getTypeBitWidth(v.getType()) == getTypeBitWidth(dstTy))
        v = arith::BitcastOp::create(builder, op.getLoc(), dstTy, v);
    }
    mapValue(op.getResult(), v);
    return success();
  }

  SmallVector<Value, 4> sliced;
  sliced.reserve(subLeaves);
  for (unsigned i = 0; i < subLeaves; ++i)
    sliced.push_back(elems[offset + i]);
  mapAggregate(op.getResult(), std::move(sliced));
  return success();
}

LogicalResult FunctionConverter::convertInsertValue(LLVM::InsertValueOp op) {
  SmallVector<Type, 8> leafTypes;
  flattenAggregateLeafTypes(op.getType(), ctx, leafTypes);
  if (leafTypes.empty())
    return op.emitError("insertvalue: unsupported aggregate result type");

  SmallVector<Value, 8> elems;
  auto existing = lookupAggregate(op.getContainer());
  if (!existing.empty() && existing.size() == leafTypes.size()) {
    elems.assign(existing.begin(), existing.end());
  } else if (isa<LLVM::UndefOp, LLVM::PoisonOp>(
                 op.getContainer().getDefiningOp())) {
    elems.reserve(leafTypes.size());
    for (Type leafTy : leafTypes) {
      Value zero;
      if (auto intTy = dyn_cast<IntegerType>(leafTy))
        zero = arith::ConstantIntOp::create(builder, op.getLoc(), intTy, 0);
      else if (auto floatTy = dyn_cast<FloatType>(leafTy))
        zero = arith::ConstantFloatOp::create(
            builder, op.getLoc(), floatTy,
            APFloat::getZero(floatTy.getFloatSemantics()));
      else
        return op.emitError("insertvalue: unsupported leaf type");
      elems.push_back(zero);
    }
  } else {
    return op.emitError("insertvalue: aggregate container not tracked");
  }

  unsigned offset = 0;
  Type subType;
  if (!computeAggregateOffset(op.getType(), op.getPosition(), ctx, offset,
                              subType))
    return op.emitError("insertvalue: unsupported aggregate position");

  unsigned subLeaves = countAggregateLeaves(subType, ctx);
  if (offset + subLeaves > elems.size())
    return op.emitError("insertvalue: aggregate index out of range");

  if (subLeaves == 1) {
    Value inserted = lookup(op.getValue());
    if (!inserted)
      return op.emitError("insertvalue: value operand not mapped");
    Type expectedTy = leafTypes[offset];
    if (inserted.getType() != expectedTy) {
      if (isa<IndexType>(expectedTy))
        inserted = createIndexCast(op.getLoc(), inserted);
      else if (isa<IndexType>(inserted.getType()))
        inserted = arith::IndexCastOp::create(builder, op.getLoc(), expectedTy,
                                              inserted);
      else if (getTypeBitWidth(inserted.getType()) == getTypeBitWidth(expectedTy))
        inserted = arith::BitcastOp::create(builder, op.getLoc(), expectedTy,
                                            inserted);
    }
    elems[offset] = inserted;
  } else {
    auto insertedAgg = lookupAggregate(op.getValue());
    if (insertedAgg.size() != subLeaves)
      return op.emitError("insertvalue: aggregate value operand not tracked");
    for (unsigned i = 0; i < subLeaves; ++i)
      elems[offset + i] = insertedAgg[i];
  }

  SmallVector<Value, 4> mapped(elems.begin(), elems.end());
  mapAggregate(op.getResult(), mapped);
  if (mapped.size() == 1)
    mapValue(op.getResult(), mapped[0]);
  return success();
}

//===----------------------------------------------------------------------===//
// Arithmetic operations
//===----------------------------------------------------------------------===//

LogicalResult FunctionConverter::convertBinaryIntOp(Operation *op) {
  Location loc = op->getLoc();
  Value lhs = lookup(op->getOperand(0));
  Value rhs = lookup(op->getOperand(1));
  if (!lhs || !rhs)
    return op->emitError("integer binary op has unmapped operand");
  Value result;

  if (isa<LLVM::AddOp>(op))
    result = arith::AddIOp::create(builder, loc, lhs, rhs);
  else if (isa<LLVM::SubOp>(op))
    result = arith::SubIOp::create(builder, loc, lhs, rhs);
  else if (isa<LLVM::MulOp>(op))
    result = arith::MulIOp::create(builder, loc, lhs, rhs);
  else if (isa<LLVM::SDivOp>(op))
    result = arith::DivSIOp::create(builder, loc, lhs, rhs);
  else if (isa<LLVM::UDivOp>(op))
    result = arith::DivUIOp::create(builder, loc, lhs, rhs);
  else if (isa<LLVM::SRemOp>(op))
    result = arith::RemSIOp::create(builder, loc, lhs, rhs);
  else if (isa<LLVM::URemOp>(op))
    result = arith::RemUIOp::create(builder, loc, lhs, rhs);
  else if (isa<LLVM::AndOp>(op))
    result = arith::AndIOp::create(builder, loc, lhs, rhs);
  else if (isa<LLVM::OrOp>(op))
    result = arith::OrIOp::create(builder, loc, lhs, rhs);
  else if (isa<LLVM::XOrOp>(op))
    result = arith::XOrIOp::create(builder, loc, lhs, rhs);
  else if (isa<LLVM::ShlOp>(op))
    result = arith::ShLIOp::create(builder, loc, lhs, rhs);
  else if (isa<LLVM::LShrOp>(op))
    result = arith::ShRUIOp::create(builder, loc, lhs, rhs);
  else if (isa<LLVM::AShrOp>(op))
    result = arith::ShRSIOp::create(builder, loc, lhs, rhs);
  else
    return op->emitError("unhandled integer binary op");

  mapValue(op->getResult(0), result);
  return success();
}

LogicalResult FunctionConverter::convertBinaryFloatOp(Operation *op) {
  Location loc = op->getLoc();

  if (isa<LLVM::FNegOp>(op)) {
    Value operand = lookup(op->getOperand(0));
    if (!operand)
      return op->emitError("floating-point unary op has unmapped operand");
    auto result = arith::NegFOp::create(builder, loc, operand);
    mapValue(op->getResult(0), result);
    return success();
  }

  Value lhs = lookup(op->getOperand(0));
  Value rhs = lookup(op->getOperand(1));
  if (!lhs || !rhs)
    return op->emitError("floating-point binary op has unmapped operand");
  Value result;

  if (isa<LLVM::FAddOp>(op))
    result = arith::AddFOp::create(builder, loc, lhs, rhs);
  else if (isa<LLVM::FSubOp>(op))
    result = arith::SubFOp::create(builder, loc, lhs, rhs);
  else if (isa<LLVM::FMulOp>(op))
    result = arith::MulFOp::create(builder, loc, lhs, rhs);
  else if (isa<LLVM::FDivOp>(op))
    result = arith::DivFOp::create(builder, loc, lhs, rhs);
  else if (isa<LLVM::FRemOp>(op))
    result = arith::RemFOp::create(builder, loc, lhs, rhs);
  else
    return op->emitError("unhandled float binary op");

  mapValue(op->getResult(0), result);
  return success();
}

LogicalResult FunctionConverter::convertCast(Operation *op) {
  Location loc = op->getLoc();
  Value arg = lookup(op->getOperand(0));
  if (!arg)
    return op->emitError("cast op has unmapped operand");
  Type dstTy = normalizeScalarType(ctx, op->getResult(0).getType());
  Value result;

  if (isa<LLVM::ZExtOp>(op))
    result = arith::ExtUIOp::create(builder, loc, dstTy, arg);
  else if (isa<LLVM::SExtOp>(op))
    result = arith::ExtSIOp::create(builder, loc, dstTy, arg);
  else if (isa<LLVM::TruncOp>(op))
    result = arith::TruncIOp::create(builder, loc, dstTy, arg);
  else if (isa<LLVM::FPExtOp>(op))
    result = arith::ExtFOp::create(builder, loc, dstTy, arg);
  else if (isa<LLVM::FPTruncOp>(op))
    result = arith::TruncFOp::create(builder, loc, dstTy, arg);
  else if (isa<LLVM::UIToFPOp>(op))
    result = arith::UIToFPOp::create(builder, loc, dstTy, arg);
  else if (isa<LLVM::SIToFPOp>(op))
    result = arith::SIToFPOp::create(builder, loc, dstTy, arg);
  else if (isa<LLVM::FPToSIOp>(op))
    result = arith::FPToSIOp::create(builder, loc, dstTy, arg);
  else if (isa<LLVM::FPToUIOp>(op))
    result = arith::FPToUIOp::create(builder, loc, dstTy, arg);
  else
    return op->emitError("unhandled cast op");

  mapValue(op->getResult(0), result);
  return success();
}

LogicalResult FunctionConverter::convertICmp(LLVM::ICmpOp op) {
  if (isa<LLVM::LLVMPointerType>(op.getLhs().getType()) ||
      isa<LLVM::LLVMPointerType>(op.getRhs().getType())) {
    auto isNullPointerLike = [&](Value value) -> bool {
      if (!isa<LLVM::LLVMPointerType>(value.getType()))
        return false;

      if (isa_and_nonnull<LLVM::ZeroOp>(value.getDefiningOp()))
        return true;

      if (auto constOp = dyn_cast_or_null<LLVM::ConstantOp>(value.getDefiningOp())) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
          return intAttr.getValue().isZero();
      }

      if (Value mapped = valueMap.lookupOrNull(value)) {
        if (auto constOp = mapped.getDefiningOp<arith::ConstantOp>()) {
          if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
            return intAttr.getValue().isZero();
        }
      }

      return false;
    };

    Value lhsScalar = valueMap.lookupOrNull(op.getLhs());
    Value rhsScalar = valueMap.lookupOrNull(op.getRhs());
    if (lhsScalar && rhsScalar &&
        isa<IntegerType, IndexType>(lhsScalar.getType()) &&
        isa<IntegerType, IndexType>(rhsScalar.getType())) {
      auto pred = convertICmpPredicate(op.getPredicate());
      auto result =
          arith::CmpIOp::create(builder, op.getLoc(), pred, lhsScalar, rhsScalar);
      mapValue(op.getResult(), result);
      return success();
    }

    auto lhsPtr = lookupPtr(op.getLhs());
    auto rhsPtr = lookupPtr(op.getRhs());
    if (lhsPtr.isValid() && rhsPtr.isValid()) {
      auto pred = op.getPredicate();
      if (lhsPtr.base == rhsPtr.base) {
        auto result = arith::CmpIOp::create(builder, op.getLoc(),
                                            convertICmpPredicate(pred),
                                            lhsPtr.index, rhsPtr.index);
        mapValue(op.getResult(), result);
        return success();
      }

      if (pred == LLVM::ICmpPredicate::eq || pred == LLVM::ICmpPredicate::ne) {
        bool neq = pred == LLVM::ICmpPredicate::ne;
        auto result = arith::ConstantIntOp::create(builder, op.getLoc(),
                                                   builder.getI1Type(), neq);
        mapValue(op.getResult(), result);
        return success();
      }
    }

    auto pred = op.getPredicate();
    if (pred == LLVM::ICmpPredicate::eq || pred == LLVM::ICmpPredicate::ne) {
      bool neq = pred == LLVM::ICmpPredicate::ne;
      bool lhsNull = isNullPointerLike(op.getLhs());
      bool rhsNull = isNullPointerLike(op.getRhs());

      if ((lhsPtr.isValid() && rhsNull) || (rhsPtr.isValid() && lhsNull)) {
        auto result = arith::ConstantIntOp::create(builder, op.getLoc(),
                                                   builder.getI1Type(), neq);
        mapValue(op.getResult(), result);
        return success();
      }

      if (lhsNull && rhsNull) {
        auto result = arith::ConstantIntOp::create(builder, op.getLoc(),
                                                   builder.getI1Type(), !neq);
        mapValue(op.getResult(), result);
        return success();
      }
    }
  }

  Value lhs = lookup(op.getLhs());
  Value rhs = lookup(op.getRhs());
  if (!lhs || !rhs)
    return op.emitError("icmp has unmapped operand");
  auto pred = convertICmpPredicate(op.getPredicate());
  auto result = arith::CmpIOp::create(builder, op.getLoc(), pred, lhs, rhs);
  mapValue(op.getResult(), result);
  return success();
}

LogicalResult FunctionConverter::convertFCmp(LLVM::FCmpOp op) {
  Value lhs = lookup(op.getLhs());
  Value rhs = lookup(op.getRhs());
  if (!lhs || !rhs)
    return op.emitError("fcmp has unmapped operand");
  auto pred = convertFCmpPredicate(op.getPredicate());
  auto result = arith::CmpFOp::create(builder, op.getLoc(), pred, lhs, rhs);
  mapValue(op.getResult(), result);
  return success();
}

LogicalResult FunctionConverter::convertSelect(LLVM::SelectOp op) {
  Value cond = lookup(op.getCondition());
  if (!cond)
    return op.emitError("select has unmapped condition");
  if (isa<LLVM::LLVMPointerType>(op.getType())) {
    auto truePI = lookupPtr(op.getTrueValue());
    auto falsePI = lookupPtr(op.getFalseValue());
    if (!truePI.isValid() || !falsePI.isValid())
      return op.emitError("pointer select operands not found");
    // Select on memref bases
    Value base = arith::SelectOp::create(builder, op.getLoc(), cond,
                                                  truePI.base, falsePI.base);
    Value idx = arith::SelectOp::create(builder, op.getLoc(), cond,
                                                 truePI.index, falsePI.index);
    mapPointer(op.getResult(), {base, idx, truePI.elementType});
    return success();
  }
  // Handle vector-typed selects element-wise
  if (isa<VectorType>(op.getType())) {
    auto trueElems = lookupVector(op.getTrueValue());
    auto falseElems = lookupVector(op.getFalseValue());
    if (trueElems.empty() || falseElems.empty())
      return op.emitError("vector select operands not tracked");
    SmallVector<Value, 4> resultElems;
    for (unsigned i = 0; i < trueElems.size(); ++i) {
      auto sel = arith::SelectOp::create(builder, op.getLoc(), cond,
                                          trueElems[i], falseElems[i]);
      resultElems.push_back(sel);
    }
    mapVector(op.getResult(), std::move(resultElems));
    return success();
  }
  Value trueVal = lookup(op.getTrueValue());
  Value falseVal = lookup(op.getFalseValue());
  if (!trueVal || !falseVal)
    return op.emitError("select has unmapped operand");
  auto result = arith::SelectOp::create(builder, op.getLoc(), cond,
                                                 trueVal, falseVal);
  mapValue(op.getResult(), result);
  return success();
}

LogicalResult FunctionConverter::convertConstant(LLVM::ConstantOp op) {
  // Handle vector constants by decomposing into scalar elements
  if (auto vecTy = dyn_cast<VectorType>(op.getType())) {
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue())) {
      Type elemTy = normalizeScalarType(ctx, vecTy.getElementType());
      SmallVector<Value, 4> elems;
      for (auto val : denseAttr.getValues<Attribute>()) {
        Value scalarConst;
        if (auto intAttr = dyn_cast<IntegerAttr>(val)) {
          auto intTy = dyn_cast<IntegerType>(elemTy);
          if (!intTy) return op.emitError("vector constant type mismatch");
          scalarConst = arith::ConstantIntOp::create(builder, op.getLoc(),
                                                      intTy, intAttr.getInt());
        } else if (auto fpAttr = dyn_cast<FloatAttr>(val)) {
          auto fTy = dyn_cast<FloatType>(elemTy);
          if (!fTy) return op.emitError("vector constant type mismatch");
          scalarConst = arith::ConstantFloatOp::create(builder, op.getLoc(),
                                                        fTy, fpAttr.getValue());
        } else {
          return op.emitError("unsupported vector constant element");
        }
        elems.push_back(scalarConst);
      }
      mapVector(op.getResult(), std::move(elems));
      return success();
    }
    return op.emitError("unsupported vector constant attribute type");
  }

  auto result = arith::ConstantOp::create(builder,
      op.getLoc(), op.getType(), cast<TypedAttr>(op.getValue()));
  mapValue(op.getResult(), result);
  return success();
}

LogicalResult FunctionConverter::convertIntrinsic(LLVM::CallIntrinsicOp op) {
  StringRef name = op.getIntrin();
  Location loc = op.getLoc();

  auto createMinMax = [&](arith::CmpIPredicate pred,
                          bool chooseLhsWhenTrue) -> LogicalResult {
    if (op.getNumResults() != 1 || op.getArgs().size() != 2)
      return failure();
    Value lhs = lookup(op.getArgs()[0]);
    Value rhs = lookup(op.getArgs()[1]);
    Value cond = arith::CmpIOp::create(builder, loc, pred, lhs, rhs);
    Value result = chooseLhsWhenTrue
                       ? arith::SelectOp::create(builder, loc, cond, lhs, rhs)
                       : arith::SelectOp::create(builder, loc, cond, rhs, lhs);
    mapValue(op.getResult(0), result);
    return success();
  };

  // fmuladd: a*b + c (fused multiply-add)
  if (name.starts_with("llvm.fmuladd.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 3)
      return failure();
    Value a = lookup(op.getArgs()[0]);
    Value b = lookup(op.getArgs()[1]);
    Value c = lookup(op.getArgs()[2]);
    if (!a || !b || !c)
      return op.emitError("fmuladd has unmapped operand");
    // Lower to math.fma (fused multiply-add)
    Value result = math::FmaOp::create(builder, loc, a, b, c);
    mapValue(op.getResult(0), result);
    return success();
  }

  // fabs: absolute value of float
  if (name.starts_with("llvm.fabs.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 1)
      return failure();
    Value arg = lookup(op.getArgs()[0]);
    if (!arg)
      return op.emitError("fabs has unmapped operand");
    Value result = math::AbsFOp::create(builder, loc, arg);
    mapValue(op.getResult(0), result);
    return success();
  }

  // sqrt
  if (name.starts_with("llvm.sqrt.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 1)
      return failure();
    Value arg = lookup(op.getArgs()[0]);
    if (!arg)
      return op.emitError("sqrt has unmapped operand");
    Value result = math::SqrtOp::create(builder, loc, arg);
    mapValue(op.getResult(0), result);
    return success();
  }

  // exp/log/sin/cos/pow
  if (name.starts_with("llvm.exp.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 1)
      return failure();
    Value arg = lookup(op.getArgs()[0]);
    if (!arg) return op.emitError("exp has unmapped operand");
    mapValue(op.getResult(0), math::ExpOp::create(builder, loc, arg));
    return success();
  }
  if (name.starts_with("llvm.exp2.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 1)
      return failure();
    Value arg = lookup(op.getArgs()[0]);
    if (!arg) return op.emitError("exp2 has unmapped operand");
    mapValue(op.getResult(0), math::Exp2Op::create(builder, loc, arg));
    return success();
  }
  if (name.starts_with("llvm.log.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 1)
      return failure();
    Value arg = lookup(op.getArgs()[0]);
    if (!arg) return op.emitError("log has unmapped operand");
    mapValue(op.getResult(0), math::LogOp::create(builder, loc, arg));
    return success();
  }
  if (name.starts_with("llvm.log2.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 1)
      return failure();
    Value arg = lookup(op.getArgs()[0]);
    if (!arg) return op.emitError("log2 has unmapped operand");
    mapValue(op.getResult(0), math::Log2Op::create(builder, loc, arg));
    return success();
  }
  if (name.starts_with("llvm.sin.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 1)
      return failure();
    Value arg = lookup(op.getArgs()[0]);
    if (!arg) return op.emitError("sin has unmapped operand");
    mapValue(op.getResult(0), math::SinOp::create(builder, loc, arg));
    return success();
  }
  if (name.starts_with("llvm.cos.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 1)
      return failure();
    Value arg = lookup(op.getArgs()[0]);
    if (!arg) return op.emitError("cos has unmapped operand");
    mapValue(op.getResult(0), math::CosOp::create(builder, loc, arg));
    return success();
  }
  if (name.starts_with("llvm.tanh.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 1)
      return failure();
    Value arg = lookup(op.getArgs()[0]);
    if (!arg) return op.emitError("tanh has unmapped operand");
    mapValue(op.getResult(0), math::TanhOp::create(builder, loc, arg));
    return success();
  }
  if (name.starts_with("llvm.pow.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 2)
      return failure();
    Value base = lookup(op.getArgs()[0]);
    Value exp = lookup(op.getArgs()[1]);
    if (!base || !exp) return op.emitError("pow has unmapped operand");
    mapValue(op.getResult(0), math::PowFOp::create(builder, loc, base, exp));
    return success();
  }
  if (name.starts_with("llvm.atan2.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 2)
      return failure();
    Value y = lookup(op.getArgs()[0]);
    Value x = lookup(op.getArgs()[1]);
    if (!y || !x) return op.emitError("atan2 has unmapped operand");
    mapValue(op.getResult(0), math::Atan2Op::create(builder, loc, y, x));
    return success();
  }
  if (name.starts_with("llvm.copysign.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 2)
      return failure();
    Value mag = lookup(op.getArgs()[0]);
    Value sign = lookup(op.getArgs()[1]);
    if (!mag || !sign) return op.emitError("copysign has unmapped operand");
    mapValue(op.getResult(0), math::CopySignOp::create(builder, loc, mag, sign));
    return success();
  }

  // floor/ceil/round
  if (name.starts_with("llvm.floor.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 1)
      return failure();
    Value arg = lookup(op.getArgs()[0]);
    if (!arg) return op.emitError("floor has unmapped operand");
    mapValue(op.getResult(0), math::FloorOp::create(builder, loc, arg));
    return success();
  }
  if (name.starts_with("llvm.ceil.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 1)
      return failure();
    Value arg = lookup(op.getArgs()[0]);
    if (!arg) return op.emitError("ceil has unmapped operand");
    mapValue(op.getResult(0), math::CeilOp::create(builder, loc, arg));
    return success();
  }
  if (name.starts_with("llvm.round.") ||
      name.starts_with("llvm.roundeven.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 1)
      return failure();
    Value arg = lookup(op.getArgs()[0]);
    if (!arg) return op.emitError("round has unmapped operand");
    mapValue(op.getResult(0), math::RoundEvenOp::create(builder, loc, arg));
    return success();
  }

  // minnum/maxnum (float min/max)
  if (name.starts_with("llvm.minnum.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 2)
      return failure();
    Value a = lookup(op.getArgs()[0]);
    Value b = lookup(op.getArgs()[1]);
    if (!a || !b) return op.emitError("minnum has unmapped operand");
    Value cmp = arith::CmpFOp::create(builder, loc,
        arith::CmpFPredicate::OLT, a, b);
    mapValue(op.getResult(0), arith::SelectOp::create(builder, loc, cmp, a, b));
    return success();
  }
  if (name.starts_with("llvm.maxnum.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 2)
      return failure();
    Value a = lookup(op.getArgs()[0]);
    Value b = lookup(op.getArgs()[1]);
    if (!a || !b) return op.emitError("maxnum has unmapped operand");
    Value cmp = arith::CmpFOp::create(builder, loc,
        arith::CmpFPredicate::OGT, a, b);
    mapValue(op.getResult(0), arith::SelectOp::create(builder, loc, cmp, a, b));
    return success();
  }

  if (name.starts_with("llvm.smin."))
    return createMinMax(arith::CmpIPredicate::sle, true);
  if (name.starts_with("llvm.smax."))
    return createMinMax(arith::CmpIPredicate::sge, true);
  if (name.starts_with("llvm.umin."))
    return createMinMax(arith::CmpIPredicate::ule, true);
  if (name.starts_with("llvm.umax."))
    return createMinMax(arith::CmpIPredicate::uge, true);

  // abs: |x| = (x >= 0) ? x : -x
  if (name.starts_with("llvm.abs.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 2)
      return failure();
    Value arg = lookup(op.getArgs()[0]);
    if (!arg)
      return op.emitError("abs has unmapped operand");
    auto intTy = dyn_cast<IntegerType>(arg.getType());
    if (!intTy)
      return failure();
    Value zero = arith::ConstantIntOp::create(builder, loc, intTy, 0);
    Value neg = arith::SubIOp::create(builder, loc, zero, arg);
    Value cmp = arith::CmpIOp::create(builder, loc,
                                       arith::CmpIPredicate::sge, arg, zero);
    Value result = arith::SelectOp::create(builder, loc, cmp, arg, neg);
    mapValue(op.getResult(0), result);
    return success();
  }

  // Saturating arithmetic: sadd.sat, ssub.sat, uadd.sat, usub.sat
  if (name.starts_with("llvm.sadd.sat.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 2)
      return failure();
    Value lhs = lookup(op.getArgs()[0]);
    Value rhs = lookup(op.getArgs()[1]);
    if (!lhs || !rhs)
      return op.emitError("sadd.sat has unmapped operand");
    auto intTy = dyn_cast<IntegerType>(lhs.getType());
    if (!intTy)
      return failure();
    unsigned width = intTy.getWidth();
    int64_t smax = (1LL << (width - 1)) - 1;
    int64_t smin = -(1LL << (width - 1));
    Value sum = arith::AddIOp::create(builder, loc, lhs, rhs);
    Value maxVal = arith::ConstantIntOp::create(builder, loc, intTy, smax);
    Value minVal = arith::ConstantIntOp::create(builder, loc, intTy, smin);
    // Overflow: lhs > 0 && rhs > 0 && sum <= 0 -> clamp to smax
    // Underflow: lhs < 0 && rhs < 0 && sum >= 0 -> clamp to smin
    Value zero = arith::ConstantIntOp::create(builder, loc, intTy, 0);
    Value lhsPos = arith::CmpIOp::create(builder, loc,
                                          arith::CmpIPredicate::sgt, lhs, zero);
    Value rhsPos = arith::CmpIOp::create(builder, loc,
                                          arith::CmpIPredicate::sgt, rhs, zero);
    Value sumNeg = arith::CmpIOp::create(builder, loc,
                                          arith::CmpIPredicate::sle, sum, zero);
    Value overflow = arith::AndIOp::create(builder, loc, lhsPos, rhsPos);
    overflow = arith::AndIOp::create(builder, loc, overflow, sumNeg);
    Value lhsNeg = arith::CmpIOp::create(builder, loc,
                                          arith::CmpIPredicate::slt, lhs, zero);
    Value rhsNeg = arith::CmpIOp::create(builder, loc,
                                          arith::CmpIPredicate::slt, rhs, zero);
    Value sumPos = arith::CmpIOp::create(builder, loc,
                                          arith::CmpIPredicate::sge, sum, zero);
    Value underflow = arith::AndIOp::create(builder, loc, lhsNeg, rhsNeg);
    underflow = arith::AndIOp::create(builder, loc, underflow, sumPos);
    Value result = arith::SelectOp::create(builder, loc, overflow, maxVal, sum);
    result = arith::SelectOp::create(builder, loc, underflow, minVal, result);
    mapValue(op.getResult(0), result);
    return success();
  }

  if (name.starts_with("llvm.ssub.sat.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 2)
      return failure();
    Value lhs = lookup(op.getArgs()[0]);
    Value rhs = lookup(op.getArgs()[1]);
    if (!lhs || !rhs)
      return op.emitError("ssub.sat has unmapped operand");
    auto intTy = dyn_cast<IntegerType>(lhs.getType());
    if (!intTy)
      return failure();
    unsigned width = intTy.getWidth();
    int64_t smax = (1LL << (width - 1)) - 1;
    int64_t smin = -(1LL << (width - 1));
    Value diff = arith::SubIOp::create(builder, loc, lhs, rhs);
    Value maxVal = arith::ConstantIntOp::create(builder, loc, intTy, smax);
    Value minVal = arith::ConstantIntOp::create(builder, loc, intTy, smin);
    // Overflow: lhs > 0 && rhs < 0 && diff <= 0 -> clamp to smax
    // Underflow: lhs < 0 && rhs > 0 && diff >= 0 -> clamp to smin
    Value zero = arith::ConstantIntOp::create(builder, loc, intTy, 0);
    Value lhsPos = arith::CmpIOp::create(builder, loc,
                                          arith::CmpIPredicate::sgt, lhs, zero);
    Value rhsNeg = arith::CmpIOp::create(builder, loc,
                                          arith::CmpIPredicate::slt, rhs, zero);
    Value diffNeg = arith::CmpIOp::create(builder, loc,
                                           arith::CmpIPredicate::sle, diff, zero);
    Value overflow = arith::AndIOp::create(builder, loc, lhsPos, rhsNeg);
    overflow = arith::AndIOp::create(builder, loc, overflow, diffNeg);
    Value lhsNeg = arith::CmpIOp::create(builder, loc,
                                          arith::CmpIPredicate::slt, lhs, zero);
    Value rhsPos = arith::CmpIOp::create(builder, loc,
                                          arith::CmpIPredicate::sgt, rhs, zero);
    Value diffPos = arith::CmpIOp::create(builder, loc,
                                           arith::CmpIPredicate::sge, diff, zero);
    Value underflow = arith::AndIOp::create(builder, loc, lhsNeg, rhsPos);
    underflow = arith::AndIOp::create(builder, loc, underflow, diffPos);
    Value result = arith::SelectOp::create(builder, loc, overflow, maxVal, diff);
    result = arith::SelectOp::create(builder, loc, underflow, minVal, result);
    mapValue(op.getResult(0), result);
    return success();
  }

  if (name.starts_with("llvm.uadd.sat.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 2)
      return failure();
    Value lhs = lookup(op.getArgs()[0]);
    Value rhs = lookup(op.getArgs()[1]);
    if (!lhs || !rhs)
      return op.emitError("uadd.sat has unmapped operand");
    auto intTy = dyn_cast<IntegerType>(lhs.getType());
    if (!intTy)
      return failure();
    Value sum = arith::AddIOp::create(builder, loc, lhs, rhs);
    // Unsigned overflow: sum < lhs (wrapping detected)
    Value overflow = arith::CmpIOp::create(builder, loc,
                                            arith::CmpIPredicate::ult, sum, lhs);
    Value allOnes = arith::ConstantIntOp::create(builder, loc, intTy, -1);
    Value result = arith::SelectOp::create(builder, loc, overflow, allOnes, sum);
    mapValue(op.getResult(0), result);
    return success();
  }

  if (name.starts_with("llvm.usub.sat.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 2)
      return failure();
    Value lhs = lookup(op.getArgs()[0]);
    Value rhs = lookup(op.getArgs()[1]);
    if (!lhs || !rhs)
      return op.emitError("usub.sat has unmapped operand");
    auto intTy = dyn_cast<IntegerType>(lhs.getType());
    if (!intTy)
      return failure();
    Value diff = arith::SubIOp::create(builder, loc, lhs, rhs);
    // Unsigned underflow: lhs < rhs -> clamp to 0
    Value underflow = arith::CmpIOp::create(builder, loc,
                                             arith::CmpIPredicate::ult, lhs, rhs);
    Value zero = arith::ConstantIntOp::create(builder, loc, intTy, 0);
    Value result = arith::SelectOp::create(builder, loc, underflow, zero, diff);
    mapValue(op.getResult(0), result);
    return success();
  }

  if (name.starts_with("llvm.bswap.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 1)
      return failure();
    Value result = buildByteSwap(loc, lookup(op.getArgs()[0]));
    if (!result)
      return failure();
    mapValue(op.getResult(0), result);
    return success();
  }

  if (name.starts_with("llvm.fshl.") || name.starts_with("llvm.fshr.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 3)
      return failure();
    Value lhs = lookup(op.getArgs()[0]);
    Value rhs = lookup(op.getArgs()[1]);
    Value amount = lookup(op.getArgs()[2]);
    if (!lhs || !rhs || !amount)
      return op.emitError("funnel shift has unmapped operand");
    Value result =
        buildFunnelShift(loc, lhs, rhs, amount, name.starts_with("llvm.fshl."));
    if (!result)
      return failure();
    mapValue(op.getResult(0), result);
    return success();
  }

  if (name.starts_with("llvm.bitreverse.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 1)
      return failure();
    Value value = lookup(op.getArgs()[0]);
    if (!value)
      return op.emitError("bitreverse has unmapped operand");
    Value result = buildBitReverse(loc, value);
    if (!result)
      return failure();
    mapValue(op.getResult(0), result);
    return success();
  }

  if (name.starts_with("llvm.ctpop.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 1)
      return failure();
    Value value = lookup(op.getArgs()[0]);
    if (!value)
      return op.emitError("ctpop has unmapped operand");
    Value result = buildCtPop(loc, value);
    if (!result)
      return failure();
    mapValue(op.getResult(0), result);
    return success();
  }

  if (name.starts_with("llvm.cttz.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 2)
      return failure();
    Value value = lookup(op.getArgs()[0]);
    if (!value)
      return op.emitError("cttz has unmapped operand");
    Value result = buildCountZeros(loc, value, /*leading=*/false);
    if (!result)
      return failure();
    mapValue(op.getResult(0), result);
    return success();
  }

  if (name.starts_with("llvm.ctlz.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 2)
      return failure();
    Value value = lookup(op.getArgs()[0]);
    if (!value)
      return op.emitError("ctlz has unmapped operand");
    Value result = buildCountZeros(loc, value, /*leading=*/true);
    if (!result)
      return failure();
    mapValue(op.getResult(0), result);
    return success();
  }

  return failure();
}

Value FunctionConverter::buildByteSwap(Location loc, Value value) {
  auto intTy = dyn_cast<IntegerType>(value.getType());
  if (!intTy)
    return Value();

  unsigned width = intTy.getWidth();
  if (width == 0 || (width % 8) != 0)
    return Value();

  unsigned numBytes = width / 8;
  Value result = arith::ConstantIntOp::create(builder, loc, intTy, 0);
  Value byteMask = arith::ConstantIntOp::create(builder, loc, intTy, 0xff);

  for (unsigned byteIdx = 0; byteIdx < numBytes; ++byteIdx) {
    unsigned srcShiftAmount = byteIdx * 8;
    unsigned dstShiftAmount = (numBytes - 1 - byteIdx) * 8;

    Value shifted = value;
    if (srcShiftAmount != 0) {
      Value srcShift = arith::ConstantIntOp::create(builder, loc, intTy,
                                                    srcShiftAmount);
      shifted = arith::ShRUIOp::create(builder, loc, shifted, srcShift);
    }

    Value isolated = arith::AndIOp::create(builder, loc, shifted, byteMask);
    Value dstValue = isolated;
    if (dstShiftAmount != 0) {
      Value dstShift = arith::ConstantIntOp::create(builder, loc, intTy,
                                                    dstShiftAmount);
      dstValue = arith::ShLIOp::create(builder, loc, isolated, dstShift);
    }

    result = arith::OrIOp::create(builder, loc, result, dstValue);
  }

  return result;
}

Value FunctionConverter::buildFunnelShift(Location loc, Value lhs, Value rhs,
                                          Value amount, bool isLeft) {
  auto valueTy = dyn_cast<IntegerType>(lhs.getType());
  auto rhsTy = dyn_cast<IntegerType>(rhs.getType());
  auto amountTy = dyn_cast<IntegerType>(amount.getType());
  if (!valueTy || !rhsTy || !amountTy)
    return Value();
  if (rhsTy != valueTy)
    return Value();

  unsigned width = valueTy.getWidth();
  if (width == 0)
    return Value();

  Value normalizedAmount = amount;
  if (amountTy.getWidth() < width) {
    normalizedAmount =
        arith::ExtUIOp::create(builder, loc, valueTy, normalizedAmount);
  } else if (amountTy.getWidth() > width) {
    normalizedAmount =
        arith::TruncIOp::create(builder, loc, valueTy, normalizedAmount);
  }

  Value mask = arith::ConstantIntOp::create(builder, loc, valueTy, width - 1);
  Value bitWidth = arith::ConstantIntOp::create(builder, loc, valueTy, width);
  normalizedAmount =
      arith::AndIOp::create(builder, loc, normalizedAmount, mask);
  Value reverseAmount =
      arith::SubIOp::create(builder, loc, bitWidth, normalizedAmount);
  reverseAmount = arith::AndIOp::create(builder, loc, reverseAmount, mask);

  Value lo;
  Value hi;
  if (isLeft) {
    lo = arith::ShLIOp::create(builder, loc, lhs, normalizedAmount);
    hi = arith::ShRUIOp::create(builder, loc, rhs, reverseAmount);
  } else {
    lo = arith::ShRUIOp::create(builder, loc, lhs, normalizedAmount);
    hi = arith::ShLIOp::create(builder, loc, rhs, reverseAmount);
  }
  return arith::OrIOp::create(builder, loc, lo, hi);
}

Value FunctionConverter::buildBitReverse(Location loc, Value value) {
  auto intTy = dyn_cast<IntegerType>(value.getType());
  if (!intTy)
    return Value();

  unsigned width = intTy.getWidth();
  if (width == 0)
    return Value();

  Value result = arith::ConstantIntOp::create(builder, loc, intTy, 0);
  Value one = arith::ConstantIntOp::create(builder, loc, intTy, 1);
  for (unsigned bitIdx = 0; bitIdx < width; ++bitIdx) {
    unsigned dstShiftAmount = width - 1 - bitIdx;

    Value shifted = value;
    if (bitIdx != 0) {
      Value srcShift =
          arith::ConstantIntOp::create(builder, loc, intTy, bitIdx);
      shifted = arith::ShRUIOp::create(builder, loc, shifted, srcShift);
    }

    Value isolated = arith::AndIOp::create(builder, loc, shifted, one);
    Value reversedBit = isolated;
    if (dstShiftAmount != 0) {
      Value dstShift =
          arith::ConstantIntOp::create(builder, loc, intTy, dstShiftAmount);
      reversedBit = arith::ShLIOp::create(builder, loc, isolated, dstShift);
    }
    result = arith::OrIOp::create(builder, loc, result, reversedBit);
  }
  return result;
}

Value FunctionConverter::buildCtPop(Location loc, Value value) {
  auto intTy = dyn_cast<IntegerType>(value.getType());
  if (!intTy)
    return Value();

  unsigned width = intTy.getWidth();
  if (width == 0)
    return Value();

  Value result = arith::ConstantIntOp::create(builder, loc, intTy, 0);
  Value one = arith::ConstantIntOp::create(builder, loc, intTy, 1);
  for (unsigned bitIdx = 0; bitIdx < width; ++bitIdx) {
    Value shifted = value;
    if (bitIdx != 0) {
      Value shift =
          arith::ConstantIntOp::create(builder, loc, intTy, bitIdx);
      shifted = arith::ShRUIOp::create(builder, loc, shifted, shift);
    }
    Value bit = arith::AndIOp::create(builder, loc, shifted, one);
    result = arith::AddIOp::create(builder, loc, result, bit);
  }
  return result;
}

Value FunctionConverter::buildCountZeros(Location loc, Value value,
                                         bool leading) {
  auto intTy = dyn_cast<IntegerType>(value.getType());
  if (!intTy)
    return Value();

  unsigned width = intTy.getWidth();
  if (width == 0)
    return Value();

  Value zero = arith::ConstantIntOp::create(builder, loc, intTy, 0);
  Value one = arith::ConstantIntOp::create(builder, loc, intTy, 1);
  Value result = zero;
  Value active = arith::ConstantIntOp::create(builder, loc, builder.getI1Type(),
                                              1);

  for (unsigned bitIdx = 0; bitIdx < width; ++bitIdx) {
    unsigned shiftAmount = leading ? (width - 1 - bitIdx) : bitIdx;
    Value shifted = value;
    if (shiftAmount != 0) {
      Value shift =
          arith::ConstantIntOp::create(builder, loc, intTy, shiftAmount);
      shifted = arith::ShRUIOp::create(builder, loc, shifted, shift);
    }

    Value bit = arith::AndIOp::create(builder, loc, shifted, one);
    Value isZero =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, bit, zero);
    Value shouldCount = arith::AndIOp::create(builder, loc, active, isZero);
    Value increment =
        arith::SelectOp::create(builder, loc, shouldCount, one, zero);
    result = arith::AddIOp::create(builder, loc, result, increment);
    active = arith::AndIOp::create(builder, loc, active, isZero);
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Control flow
//===----------------------------------------------------------------------===//

LogicalResult FunctionConverter::convertBr(LLVM::BrOp op) {
  Block *dst = blockMap[op.getDest()];
  if (!dst)
    return op.emitError("branch destination not mapped");
  auto args =
      materializeBranchArgs(op.getLoc(), op.getDestOperands(), dst);
  cf::BranchOp::create(builder, op.getLoc(), dst, args);
  return success();
}

LogicalResult FunctionConverter::convertCondBr(LLVM::CondBrOp op) {
  Value cond = lookup(op.getCondition());
  Block *trueDst = blockMap[op.getTrueDest()];
  Block *falseDst = blockMap[op.getFalseDest()];
  if (!trueDst || !falseDst)
    return op.emitError("branch destinations not mapped");

  auto trueArgs =
      materializeBranchArgs(op.getLoc(), op.getTrueDestOperands(), trueDst);
  auto falseArgs =
      materializeBranchArgs(op.getLoc(), op.getFalseDestOperands(), falseDst);
  cf::CondBranchOp::create(builder, op.getLoc(), cond, trueDst, trueArgs,
                                    falseDst, falseArgs);
  return success();
}

LogicalResult FunctionConverter::convertReturn(LLVM::ReturnOp op) {
  SmallVector<Value> operands;
  for (Value v : op.getOperands())
    operands.push_back(lookup(v));
  func::ReturnOp::create(builder, op.getLoc(), operands);
  return success();
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

Value FunctionConverter::lookup(Value v) {
  if (auto mapped = valueMap.lookupOrNull(v))
    return mapped;
  // If it's a pointer, return the base memref
  auto it = pointerMap.find(v);
  if (it != pointerMap.end())
    return it->second.base;
  // Should not reach here
  return nullptr;
}

PointerInfo FunctionConverter::lookupPtr(Value v) {
  auto it = pointerMap.find(v);
  if (it != pointerMap.end())
    return it->second;
  return {};
}

void FunctionConverter::mapValue(Value oldVal, Value newVal) {
  valueMap.map(oldVal, newVal);
}

void FunctionConverter::mapPointer(Value oldPtr, PointerInfo info) {
  pointerMap[oldPtr] = info;
}

void FunctionConverter::mapVector(Value oldVec, SmallVector<Value, 4> elements) {
  vectorMap[oldVec] = std::move(elements);
}

void FunctionConverter::mapAggregate(Value oldAgg,
                                     SmallVector<Value, 4> elements) {
  aggregateMap[oldAgg] = std::move(elements);
}

SmallVector<Value, 4> FunctionConverter::lookupVector(Value v) {
  auto it = vectorMap.find(v);
  if (it != vectorMap.end())
    return it->second;
  return {};
}

SmallVector<Value, 4> FunctionConverter::lookupAggregate(Value v) {
  auto it = aggregateMap.find(v);
  if (it != aggregateMap.end())
    return it->second;
  return {};
}

SmallVector<Value>
FunctionConverter::materializeBranchArgs(Location loc, OperandRange args,
                                         Block *dest) {
  SmallVector<Value> result;
  result.reserve(args.size());
  // dstArgIdx tracks the destination block argument index, which may differ
  // from the source operand index when vector operands expand into multiple args.
  unsigned dstArgIdx = 0;
  for (auto [idx, v] : llvm::enumerate(args)) {
    // Handle vector-typed branch arguments by expanding into scalar values
    if (isa<VectorType>(v.getType())) {
      auto elems = lookupVector(v);
      if (!elems.empty()) {
        for (Value elem : elems) {
          Type dstTy = dstArgIdx < dest->getNumArguments()
                           ? dest->getArgument(dstArgIdx).getType()
                           : Type();
          if (dstTy && elem.getType() != dstTy) {
            // Type mismatch: try cast
            if (isa<IndexType>(dstTy))
              elem = createIndexCast(loc, elem);
            else if (getTypeBitWidth(elem.getType()) == getTypeBitWidth(dstTy))
              elem = arith::BitcastOp::create(builder, loc, dstTy, elem);
          }
          result.push_back(elem);
          dstArgIdx++;
        }
        continue;
      }
    }

    Type dstTy = dstArgIdx < dest->getNumArguments()
                     ? dest->getArgument(dstArgIdx).getType()
                     : Type();
    dstArgIdx++;

    if (isa<LLVM::LLVMPointerType>(v.getType())) {
      auto pi = lookupPtr(v);
      if (pi.isValid()) {
        result.push_back(pi.base);
      }
    } else {
      Value mapped = lookup(v);
      if (!mapped) {
        result.push_back(mapped);
        continue;
      }
      if (!dstTy || mapped.getType() == dstTy) {
        result.push_back(mapped);
        continue;
      }

      if (isa<IndexType>(dstTy)) {
        result.push_back(createIndexCast(loc, mapped));
        continue;
      }

      if (isa<IndexType>(mapped.getType())) {
        result.push_back(arith::IndexCastOp::create(builder, loc, dstTy, mapped));
        continue;
      }

      auto srcIntTy = dyn_cast<IntegerType>(mapped.getType());
      auto dstIntTy = dyn_cast<IntegerType>(dstTy);
      if (srcIntTy && dstIntTy) {
        if (srcIntTy.getWidth() < dstIntTy.getWidth()) {
          result.push_back(
              arith::ExtUIOp::create(builder, loc, dstTy, mapped));
        } else if (srcIntTy.getWidth() > dstIntTy.getWidth()) {
          result.push_back(
              arith::TruncIOp::create(builder, loc, dstTy, mapped));
        } else {
          result.push_back(mapped);
        }
        continue;
      }

      result.push_back(mapped);
    }
  }
  return result;
}

Value FunctionConverter::createIndexCast(Location loc, Value intVal) {
  if (isa<IndexType>(intVal.getType()))
    return intVal;
  return arith::IndexCastOp::create(builder, loc, IndexType::get(ctx), intVal);
}

Value FunctionConverter::scaleIndex(Location loc, Value idx,
                                     Type fromElem, Type toElem) {
  unsigned fromBits = getTypeBitWidth(fromElem);
  unsigned toBits = getTypeBitWidth(toElem);
  if (fromBits == toBits || fromBits == 0 || toBits == 0)
    return idx;

  // scaled = idx * fromBytes / toBytes
  Value fromScale = arith::ConstantIndexOp::create(builder, loc, fromBits / 8);
  Value toScale = arith::ConstantIndexOp::create(builder, loc, toBits / 8);
  Value bytes = arith::MulIOp::create(builder, loc, idx, fromScale);
  return arith::DivUIOp::create(builder, loc, bytes, toScale);
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct ConvertLLVMToCFPassImpl
    : public PassWrapper<ConvertLLVMToCFPassImpl, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLLVMToCFPassImpl)

  StringRef getArgument() const override { return "loom-llvm-to-cf"; }
  StringRef getDescription() const override {
    return "Convert LLVM dialect to func/cf/arith/memref";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, cf::ControlFlowDialect,
                    arith::ArithDialect, memref::MemRefDialect,
                    math::MathDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    llvm::StringMap<ReadOnlyGlobalInfo> readonlyGlobals;

    // Materialize supported readonly LLVM globals as memref globals so that
    // AddressOfOp can resolve them through memref.get_global.
    SmallVector<LLVM::GlobalOp> globalsToConvert;
    module.walk([&](LLVM::GlobalOp global) {
      globalsToConvert.push_back(global);
    });
    for (LLVM::GlobalOp global : globalsToConvert) {
      ReadOnlyGlobalInfo info;
      if (failed(materializeReadOnlyGlobal(global, info)) &&
          failed(materializeReadOnlyStructGlobal(global, info)))
        continue;

      builder.setInsertionPoint(global);
      auto memrefGlobal = memref::GlobalOp::create(
          builder, global.getLoc(), global.getSymName(),
          builder.getStringAttr("private"), info.memrefType,
          info.initialValue, /*constant=*/true, IntegerAttr());
      (void)memrefGlobal;
      readonlyGlobals.insert({global.getSymName().str(), info});
      global.erase();
    }

    // Collect functions to convert (snapshot to avoid iterator invalidation)
    SmallVector<LLVM::LLVMFuncOp> funcsToConvert;
    module.walk([&](LLVM::LLVMFuncOp f) {
      if (canConvertFunction(f))
        funcsToConvert.push_back(f);
    });

    // Track existing func.func ops to distinguish newly created ones
    llvm::SmallPtrSet<Operation *, 8> existingFuncs;
    module.walk([&](func::FuncOp f) {
      existingFuncs.insert(f.getOperation());
    });

    for (auto llvmFunc : funcsToConvert) {
      FunctionConverter converter(llvmFunc, builder, readonlyGlobals);
      if (failed(converter.convert())) {
        // Graceful degradation: erase any partially created func.func
        // and skip this function
        SmallVector<func::FuncOp> toErase;
        module.walk([&](func::FuncOp f) {
          if (!existingFuncs.contains(f.getOperation()))
            toErase.push_back(f);
        });
        for (auto f : toErase)
          f->erase();
        llvm::errs() << "warning: LLVMToCF: skipping function '"
                     << llvmFunc.getSymName() << "'";
        if (converter.failedOp)
          llvm::errs() << " after failing op '"
                       << converter.failedOp->getName() << "'";
        llvm::errs() << "\n";
        continue;
      }
      existingFuncs.insert(converter.newFunc.getOperation());
      if (llvmFunc->getParentOp())
        llvmFunc->erase();
    }

    // Convert external LLVM function declarations to func.func declarations.
    // This ensures that calls to external functions (malloc, calloc, etc.)
    // have valid declarations in the output module.
    SmallVector<LLVM::LLVMFuncOp, 8> residualLLVMFuncs;
    module.walk([&](LLVM::LLVMFuncOp func) { residualLLVMFuncs.push_back(func); });
    for (LLVM::LLVMFuncOp func : residualLLVMFuncs) {
      if (!func->getParentOp())
        continue;
      if (func.isExternal() && !func.getFunctionType().isVarArg()) {
        // Create a func.func declaration for this external function,
        // but only if it doesn't have pointer-typed arguments (which
        // would need memref conversion that we can't do for externals).
        auto llvmFuncType = func.getFunctionType();
        bool hasPointerArgs = false;
        for (Type t : llvmFuncType.getParams()) {
          if (isa<LLVM::LLVMPointerType>(t)) {
            hasPointerArgs = true;
            break;
          }
        }
        if (!hasPointerArgs) {
          SmallVector<Type> argTypes;
          for (Type t : llvmFuncType.getParams())
            argTypes.push_back(normalizeScalarType(module.getContext(), t));
          SmallVector<Type> resultTypes;
          Type retTy = llvmFuncType.getReturnType();
          if (!isa<LLVM::LLVMVoidType>(retTy))
            resultTypes.push_back(normalizeScalarType(module.getContext(), retTy));
          auto funcType = FunctionType::get(module.getContext(), argTypes,
                                             resultTypes);
          OpBuilder declBuilder(module.getContext());
          declBuilder.setInsertionPoint(func);
          auto decl = func::FuncOp::create(declBuilder, func.getLoc(),
                                            func.getSymName(), funcType);
          decl.setVisibility(SymbolTable::Visibility::Private);
        }
      }
      func.erase();
    }

    // Clean up LLVM module-level ops
    module.walk([](LLVM::ModuleFlagsOp op) { op.erase(); });
  }
};

} // namespace

std::unique_ptr<Pass> loom::createConvertLLVMToCFPass() {
  return std::make_unique<ConvertLLVMToCFPassImpl>();
}
