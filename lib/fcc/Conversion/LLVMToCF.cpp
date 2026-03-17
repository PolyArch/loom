// LLVM dialect to CF-stage conversion pass.
// Converts LLVM dialect to func/cf/arith/memref/math.
// General-purpose: handles arbitrary C programs, not just vecadd.
// Functions with unconvertible ops (varargs, inline asm) are skipped.

#include "LLVMToCFTypes.h"
#include "fcc/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace fcc;

namespace {

// Per-function converter state.
struct FunctionConverter {
  LLVM::LLVMFuncOp llvmFunc;
  func::FuncOp newFunc;
  OpBuilder &builder;
  MLIRContext *ctx;

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

  FunctionConverter(LLVM::LLVMFuncOp func, OpBuilder &b)
      : llvmFunc(func), builder(b), ctx(func.getContext()) {}

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
  SmallVector<Value> materializeBranchArgs(OperandRange args, Block *dest);
  Value createIndexCast(Location loc, Value intVal);
  Value scaleIndex(Location loc, Value idx, Type fromElem, Type toElem);
  Value buildByteSwap(Location loc, Value value);
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
      Type newTy;

      if (isEntry && isa<LLVM::LLVMPointerType>(srcArg.getType())) {
        if (argMemRefTypes.count(i))
          newTy = argMemRefTypes[i];
        else
          newTy = buildStridedMemRefType(ctx, IntegerType::get(ctx, 8));
      } else if (isa<LLVM::LLVMPointerType>(srcArg.getType())) {
        // Non-entry block pointer args: infer from uses or use i8
        Type elemTy = IntegerType::get(ctx, 8);
        newTy = buildStridedMemRefType(ctx, elemTy);
      } else {
        newTy = normalizeScalarType(ctx, srcArg.getType());
      }

      auto dstArg = dstBlock->addArgument(newTy, srcArg.getLoc());

      if (isa<LLVM::LLVMPointerType>(srcArg.getType())) {
        // Pointer arg: create PointerInfo with zero index
        auto zeroIdx = OpBuilder::atBlockBegin(dstBlock)
                           .create<arith::ConstantIndexOp>(srcArg.getLoc(), 0);
        Type elemTy;
        if (isEntry && argElemTypes.count(i))
          elemTy = argElemTypes[i];
        else
          elemTy = IntegerType::get(ctx, 8);
        mapPointer(srcArg, {dstArg, zeroIdx, elemTy});
      } else {
        mapValue(srcArg, dstArg);
      }
    }
  }

  return success();
}

LogicalResult FunctionConverter::convertOps() {
  for (Block &srcBlock : llvmFunc.getBody()) {
    Block *dstBlock = blockMap[&srcBlock];
    builder.setInsertionPointToEnd(dstBlock);

    for (Operation &op : srcBlock) {
      if (failed(convertOp(&op)))
        return failure();
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
    // For now, fail - globals support would need memref::GlobalOp conversion
    return op->emitError("llvm.mlir.addressof not yet supported");
  }

  // Constants
  if (auto c = dyn_cast<LLVM::ConstantOp>(op))
    return convertConstant(c);
  if (isa<LLVM::ZeroOp>(op)) {
    Type ty = normalizeScalarType(ctx, op->getResult(0).getType());
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
    Type ty = normalizeScalarType(ctx, op->getResult(0).getType());
    if (isa<LLVM::LLVMPointerType>(op->getResult(0).getType())) {
      // Pointer undef: skip (will be handled if used)
      return success();
    }
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
    SmallVector<Value> args;
    for (Value v : call.getOperands()) {
      if (isa<LLVM::LLVMPointerType>(v.getType())) {
        auto pi = lookupPtr(v);
        if (pi.isValid())
          args.push_back(pi.base);
        else
          return call.emitError("cannot convert pointer call argument");
      } else {
        args.push_back(lookup(v));
      }
    }
    auto callee = call.getCalleeAttr();
    if (!callee)
      return call.emitError("indirect calls not supported");

    SmallVector<Type> resultTypes;
    for (Type t : call.getResultTypes()) {
      if (isa<LLVM::LLVMVoidType>(t))
        continue;
      resultTypes.push_back(normalizeScalarType(ctx, t));
    }
    auto newCall = func::CallOp::create(builder, 
        op->getLoc(), callee.getAttr(), resultTypes, args);
    for (unsigned i = 0; i < newCall->getNumResults(); ++i)
      mapValue(call->getResult(i), newCall->getResult(i));
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

  // Simple case: single dynamic index, no struct nesting
  // gep base[idx] where base is a typed pointer
  if (dynamicIndices.size() == 1 && indices.size() == 1) {
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

  // Multi-index GEP: compute byte offset, then scale to element type
  // offset_bytes = sum of (index[i] * stride[i]) for each level
  Value byteOffset = arith::ConstantIndexOp::create(builder, loc, 0);

  // Start with base's existing offset in bytes
  unsigned baseElemBytes = getTypeBitWidth(baseInfo.elementType) / 8;
  if (baseElemBytes == 0) baseElemBytes = 1;
  if (baseElemBytes != 1) {
    Value scale = arith::ConstantIndexOp::create(builder, loc, baseElemBytes);
    byteOffset = arith::MulIOp::create(builder, loc, baseInfo.index, scale);
  } else {
    byteOffset = baseInfo.index;
  }

  // Walk through GEP indices
  Type currentType = elemTy;
  unsigned dynIdx = 0;

  for (auto idxAttr : indices) {
    // Get the index value
    Value idxVal;
    if (auto constIdx = dyn_cast<IntegerAttr>(idxAttr)) {
      idxVal = arith::ConstantIndexOp::create(builder, loc,
                                                       constIdx.getInt());
    } else {
      // Dynamic index
      if (dynIdx >= dynamicIndices.size())
        return op.emitError("GEP: more indices than dynamic operands");
      idxVal = createIndexCast(loc, lookup(dynamicIndices[dynIdx++]));
    }

    // Compute stride for this level
    unsigned strideBytes = getTypeBitWidth(currentType) / 8;
    if (strideBytes == 0) strideBytes = 1;

    if (strideBytes != 1) {
      Value stride = arith::ConstantIndexOp::create(builder, loc, strideBytes);
      Value contribution = arith::MulIOp::create(builder, loc, idxVal, stride);
      byteOffset = arith::AddIOp::create(builder, loc, byteOffset, contribution);
    } else {
      byteOffset = arith::AddIOp::create(builder, loc, byteOffset, idxVal);
    }
    // TODO: handle struct field traversal (update currentType)
  }

  // Convert byte offset to element offset
  unsigned finalElemBytes = getTypeBitWidth(elemTy) / 8;
  if (finalElemBytes == 0) finalElemBytes = 1;
  Value finalIdx;
  if (finalElemBytes != 1) {
    Value divisor = arith::ConstantIndexOp::create(builder, loc, finalElemBytes);
    finalIdx = arith::DivUIOp::create(builder, loc, byteOffset, divisor);
  } else {
    finalIdx = byteOffset;
  }

  mapPointer(op.getResult(), {baseInfo.base, finalIdx, elemTy});
  return success();
}

LogicalResult FunctionConverter::convertLoad(LLVM::LoadOp op) {
  Location loc = op.getLoc();
  auto addrInfo = lookupPtr(op.getAddr());
  if (!addrInfo.isValid())
    return op.emitError("load address pointer not found");

  // If the load result is a pointer type, track as pointer
  if (isa<LLVM::LLVMPointerType>(op.getResult().getType())) {
    // Pointer load: not supported for now in general case
    // (would need to load a memref from a memref-of-memref)
    return op.emitError("pointer-typed loads not yet supported");
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

  // Bitcast if types differ but widths match
  if (result.getType() != accessType) {
    result = arith::BitcastOp::create(builder, loc, accessType, result);
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
  Type elemTy = op.getElemType();
  if (!elemTy)
    elemTy = IntegerType::get(ctx, 8);
  elemTy = normalizeScalarType(ctx, elemTy);

  Value size = lookup(op.getArraySize());
  Value sizeIdx = createIndexCast(loc, size);

  auto memrefTy = MemRefType::get({ShapedType::kDynamic}, elemTy);
  auto alloc = memref::AllocaOp::create(builder, loc, memrefTy,
                                                 ValueRange{sizeIdx});
  Value zeroIdx = arith::ConstantIndexOp::create(builder, loc, 0);
  mapPointer(op.getResult(), {alloc, zeroIdx, elemTy});
  return success();
}

//===----------------------------------------------------------------------===//
// Arithmetic operations
//===----------------------------------------------------------------------===//

LogicalResult FunctionConverter::convertBinaryIntOp(Operation *op) {
  Location loc = op->getLoc();
  Value lhs = lookup(op->getOperand(0));
  Value rhs = lookup(op->getOperand(1));
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
    auto result = arith::NegFOp::create(builder, loc, operand);
    mapValue(op->getResult(0), result);
    return success();
  }

  Value lhs = lookup(op->getOperand(0));
  Value rhs = lookup(op->getOperand(1));
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
  Value lhs = lookup(op.getLhs());
  Value rhs = lookup(op.getRhs());
  auto pred = convertICmpPredicate(op.getPredicate());
  auto result = arith::CmpIOp::create(builder, op.getLoc(), pred, lhs, rhs);
  mapValue(op.getResult(), result);
  return success();
}

LogicalResult FunctionConverter::convertFCmp(LLVM::FCmpOp op) {
  Value lhs = lookup(op.getLhs());
  Value rhs = lookup(op.getRhs());
  auto pred = convertFCmpPredicate(op.getPredicate());
  auto result = arith::CmpFOp::create(builder, op.getLoc(), pred, lhs, rhs);
  mapValue(op.getResult(), result);
  return success();
}

LogicalResult FunctionConverter::convertSelect(LLVM::SelectOp op) {
  Value cond = lookup(op.getCondition());
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
  Value trueVal = lookup(op.getTrueValue());
  Value falseVal = lookup(op.getFalseValue());
  auto result = arith::SelectOp::create(builder, op.getLoc(), cond,
                                                 trueVal, falseVal);
  mapValue(op.getResult(), result);
  return success();
}

LogicalResult FunctionConverter::convertConstant(LLVM::ConstantOp op) {
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

  if (name.starts_with("llvm.smin."))
    return createMinMax(arith::CmpIPredicate::sle, true);
  if (name.starts_with("llvm.smax."))
    return createMinMax(arith::CmpIPredicate::sge, true);
  if (name.starts_with("llvm.umin."))
    return createMinMax(arith::CmpIPredicate::ule, true);
  if (name.starts_with("llvm.umax."))
    return createMinMax(arith::CmpIPredicate::uge, true);

  if (name.starts_with("llvm.bswap.")) {
    if (op.getNumResults() != 1 || op.getArgs().size() != 1)
      return failure();
    Value result = buildByteSwap(loc, lookup(op.getArgs()[0]));
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

//===----------------------------------------------------------------------===//
// Control flow
//===----------------------------------------------------------------------===//

LogicalResult FunctionConverter::convertBr(LLVM::BrOp op) {
  Block *dst = blockMap[op.getDest()];
  if (!dst)
    return op.emitError("branch destination not mapped");
  auto args = materializeBranchArgs(op.getDestOperands(), op.getDest());
  cf::BranchOp::create(builder, op.getLoc(), dst, args);
  return success();
}

LogicalResult FunctionConverter::convertCondBr(LLVM::CondBrOp op) {
  Value cond = lookup(op.getCondition());
  Block *trueDst = blockMap[op.getTrueDest()];
  Block *falseDst = blockMap[op.getFalseDest()];
  if (!trueDst || !falseDst)
    return op.emitError("branch destinations not mapped");

  auto trueArgs = materializeBranchArgs(op.getTrueDestOperands(),
                                        op.getTrueDest());
  auto falseArgs = materializeBranchArgs(op.getFalseDestOperands(),
                                         op.getFalseDest());
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

SmallVector<Value>
FunctionConverter::materializeBranchArgs(OperandRange args, Block *dest) {
  SmallVector<Value> result;
  for (Value v : args) {
    if (isa<LLVM::LLVMPointerType>(v.getType())) {
      auto pi = lookupPtr(v);
      if (pi.isValid()) {
        // Pass the memref base as branch argument
        // TODO: handle index offset passing through block args
        result.push_back(pi.base);
      }
    } else {
      result.push_back(lookup(v));
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

  StringRef getArgument() const override { return "fcc-llvm-to-cf"; }
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
      FunctionConverter converter(llvmFunc, builder);
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
                     << llvmFunc.getSymName() << "'\n";
        continue;
      }
      existingFuncs.insert(converter.newFunc.getOperation());
      // Erase the old LLVM function after successful conversion
      llvmFunc->erase();
    }

    // Clean up LLVM module-level ops
    module.walk([](LLVM::ModuleFlagsOp op) { op.erase(); });
  }
};

} // namespace

std::unique_ptr<Pass> fcc::createConvertLLVMToCFPass() {
  return std::make_unique<ConvertLLVMToCFPassImpl>();
}
