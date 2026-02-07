//===-- LLVMToSCFSupport.cpp - LLVM to SCF helpers ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This file provides shared helper routines for the LLVM-to-SCF conversion,
// including pointer tracking, type normalization, annotation handling, and
// intrinsic conversion helpers used by the conversion implementation.
//
//===----------------------------------------------------------------------===//

#include "loom/Conversion/LLVMToSCF.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include <optional>

using namespace mlir;

namespace loom::llvm_to_scf {

void CopyLoomAnnotations(Operation *src, Operation *dst) {
  if (!src || !dst)
    return;
  auto attr = src->getAttr("loom.annotations");
  if (attr)
    dst->setAttr("loom.annotations", attr);
}

void MergeLoomAnnotationList(SmallVectorImpl<StringAttr> &dst,
                                    ArrayAttr src) {
  if (!src)
    return;
  for (Attribute attr : src) {
    auto strAttr = llvm::dyn_cast<StringAttr>(attr);
    if (!strAttr)
      continue;
    bool exists = false;
    for (StringAttr existing : dst) {
      if (existing.getValue() == strAttr.getValue()) {
        exists = true;
        break;
      }
    }
    if (!exists)
      dst.push_back(strAttr);
  }
}

ArrayAttr BuildLoomAnnotationArray(MLIRContext *context,
                                          ArrayRef<StringAttr> attrs) {
  SmallVector<Attribute, 4> storage;
  storage.reserve(attrs.size());
  for (StringAttr attr : attrs)
    storage.push_back(attr);
  return ArrayAttr::get(context, storage);
}

bool IsPointerType(Type type) {
  return llvm::isa<LLVM::LLVMPointerType>(type);
}

bool IsRawPointerCallee(StringRef callee) {
  return callee == "puts";
}

Type GetScalarType(Type type, SmallVectorImpl<int64_t> &dims) {
  while (auto arrayTy = llvm::dyn_cast<LLVM::LLVMArrayType>(type)) {
    dims.push_back(arrayTy.getNumElements());
    type = arrayTy.getElementType();
  }
  return type;
}

Type NormalizeScalarType(Type type, MLIRContext *context) {
  if (llvm::isa<IntegerType>(type) || llvm::isa<FloatType>(type))
    return type;
  if (llvm::isa<LLVM::LLVMPointerType>(type))
    return IntegerType::get(context, 8);
  return type;
}

int64_t GetByteSize(Type type) {
  if (auto intTy = llvm::dyn_cast<IntegerType>(type)) {
    unsigned width = intTy.getWidth();
    return static_cast<int64_t>((width + 7) / 8);
  }
  if (auto floatTy = llvm::dyn_cast<FloatType>(type)) {
    unsigned width = floatTy.getWidth();
    return static_cast<int64_t>((width + 7) / 8);
  }
  return 0;
}

Value BuildIndexConstant(OpBuilder &builder, Location loc, int64_t value);

Value ScaleIndexBetweenElementTypes(OpBuilder &builder, Location loc,
                                           Value index, Type fromType,
                                           Type toType) {
  if (!index || fromType == toType)
    return index;
  int64_t fromSize = GetByteSize(fromType);
  int64_t toSize = GetByteSize(toType);
  if (fromSize == 0 || toSize == 0)
    return index;
  Value byteIndex = index;
  if (fromSize != 1) {
    Value scale = BuildIndexConstant(builder, loc, fromSize);
    byteIndex = arith::MulIOp::create(builder, loc, index, scale);
  }
  if (toSize == 1)
    return byteIndex;
  Value scale = BuildIndexConstant(builder, loc, toSize);
  return arith::DivSIOp::create(builder, loc, byteIndex, scale);
}

std::optional<Value> BuildMemsetFillValue(OpBuilder &builder,
                                                 Location loc, Value fillVal,
                                                 Type elemType) {
  if (fillVal.getType() == elemType)
    return fillVal;

  auto constOp = fillVal.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return std::nullopt;
  auto intAttr = llvm::dyn_cast<IntegerAttr>(constOp.getValue());
  if (!intAttr)
    return std::nullopt;
  uint64_t byte = intAttr.getValue().getZExtValue() & 0xFFu;

  auto intTy = llvm::dyn_cast<IntegerType>(elemType);
  auto floatTy = llvm::dyn_cast<FloatType>(elemType);
  if (!intTy && !floatTy)
    return std::nullopt;

  unsigned bitWidth = intTy ? intTy.getWidth() : floatTy.getWidth();
  if (bitWidth % 8 != 0)
    return std::nullopt;

  APInt pattern(bitWidth, 0);
  APInt byteVal(8, byte);
  for (unsigned shift = 0; shift < bitWidth; shift += 8) {
    pattern |= byteVal.zext(bitWidth).shl(shift);
  }

  IntegerType patternType =
      intTy ? intTy : IntegerType::get(builder.getContext(), bitWidth);
  auto patternAttr = builder.getIntegerAttr(patternType, pattern);
  auto patternVal = arith::ConstantOp::create(builder, loc, patternAttr);
  if (intTy)
    return patternVal.getResult();

  auto bitcast =
      arith::BitcastOp::create(builder, loc, elemType, patternVal.getResult());
  return bitcast.getResult();
}

Value BuildIndexConstant(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantOp::create(builder, loc, builder.getIndexType(),
                                           builder.getIndexAttr(value));
}

Value ToIndexValue(OpBuilder &builder, Location loc, Value value) {
  if (!value)
    return nullptr;
  if (value.getType().isIndex())
    return value;
  if (llvm::isa<IntegerType>(value.getType()))
    return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                              value);
  return nullptr;
}

std::optional<Value> LookupValue(const DenseMap<Value, Value> &valueMap,
                                        Value key) {
  auto it = valueMap.find(key);
  if (it == valueMap.end())
    return std::nullopt;
  return it->second;
}

std::optional<PointerInfo>
LookupPointer(const DenseMap<Value, PointerInfo> &ptrMap, Value key) {
  auto it = ptrMap.find(key);
  if (it == ptrMap.end())
    return std::nullopt;
  return it->second;
}

SmallVector<Value, 8> *LookupVector(VectorMapT &vectorMap, Value key) {
  auto it = vectorMap.find(key);
  if (it == vectorMap.end())
    return nullptr;
  return &it->second;
}

SmallVector<Value, 8> ScalarizeDenseConstant(OpBuilder &builder, Location loc,
                                             DenseElementsAttr attr) {
  SmallVector<Value, 8> lanes;
  auto vecTy = llvm::dyn_cast<VectorType>(attr.getType());
  if (!vecTy)
    return lanes;
  Type elemTy = vecTy.getElementType();
  if (attr.isSplat()) {
    Value scalar;
    if (auto intTy = llvm::dyn_cast<IntegerType>(elemTy)) {
      auto val = attr.getSplatValue<APInt>();
      scalar = arith::ConstantOp::create(
          builder, loc, builder.getIntegerAttr(intTy, val));
    } else if (auto floatTy = llvm::dyn_cast<FloatType>(elemTy)) {
      auto val = attr.getSplatValue<APFloat>();
      scalar = arith::ConstantOp::create(
          builder, loc, builder.getFloatAttr(floatTy, val));
    }
    if (scalar) {
      int64_t n = vecTy.getNumElements();
      lanes.resize(n, scalar);
    }
    return lanes;
  }
  if (llvm::isa<IntegerType>(elemTy)) {
    for (APInt val : attr.getValues<APInt>()) {
      auto c = arith::ConstantOp::create(
          builder, loc,
          builder.getIntegerAttr(elemTy, val));
      lanes.push_back(c);
    }
  } else if (llvm::isa<FloatType>(elemTy)) {
    for (APFloat val : attr.getValues<APFloat>()) {
      auto c = arith::ConstantOp::create(
          builder, loc,
          builder.getFloatAttr(elemTy, val));
      lanes.push_back(c);
    }
  }
  return lanes;
}

std::optional<Type> GuessPointerElementTypeFromValue(Value value);

bool IsI8Type(Type type) {
  auto intTy = llvm::dyn_cast<IntegerType>(type);
  return intTy && intTy.getWidth() == 8;
}

std::optional<Type> InferPointerElementType(Value value) {
  SmallVector<int64_t, 4> dims;
  MLIRContext *context = value.getContext();
  SmallVector<Value, 8> worklist;
  llvm::DenseSet<Value> visited;
  worklist.push_back(value);
  std::optional<Type> fallback;

  auto recordCandidate = [&](Type candidate) -> std::optional<Type> {
    if (!candidate)
      return std::nullopt;
    if (!IsI8Type(candidate))
      return candidate;
    if (!fallback)
      fallback = candidate;
    return std::nullopt;
  };

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    if (auto guessed = GuessPointerElementTypeFromValue(current)) {
      if (auto strong = recordCandidate(*guessed))
        return strong;
    }

    for (Operation *user : current.getUsers()) {
      if (auto loadOp = llvm::dyn_cast<LLVM::LoadOp>(user)) {
        if (loadOp.getAddr() != current)
          continue;
        Type resultTy = loadOp.getResult().getType();
        if (IsPointerType(resultTy)) {
          worklist.push_back(loadOp.getResult());
          continue;
        }
        if (auto strong =
                recordCandidate(NormalizeScalarType(resultTy, context)))
          return strong;
        continue;
      }

      if (auto storeOp = llvm::dyn_cast<LLVM::StoreOp>(user)) {
        if (storeOp.getAddr() == current) {
          Type valueTy = storeOp.getValue().getType();
          if (!IsPointerType(valueTy)) {
            if (auto strong =
                    recordCandidate(NormalizeScalarType(valueTy, context)))
              return strong;
          }
          continue;
        }
        if (storeOp.getValue() == current) {
          Value slot = storeOp.getAddr();
          for (Operation *slotUser : slot.getUsers()) {
            auto slotLoad = llvm::dyn_cast<LLVM::LoadOp>(slotUser);
            if (!slotLoad || slotLoad.getAddr() != slot)
              continue;
            Type resultTy = slotLoad.getResult().getType();
            if (IsPointerType(resultTy)) {
              worklist.push_back(slotLoad.getResult());
              continue;
            }
            if (auto strong =
                    recordCandidate(NormalizeScalarType(resultTy, context)))
              return strong;
            continue;
          }
        }
      }

      if (auto gepOp = llvm::dyn_cast<LLVM::GEPOp>(user)) {
        if (gepOp.getBase() != current)
          continue;
        dims.clear();
        Type scalar = GetScalarType(gepOp.getElemType(), dims);
        if (auto strong =
                recordCandidate(NormalizeScalarType(scalar, context)))
          return strong;
        continue;
      }

      if (auto bitcastOp = llvm::dyn_cast<LLVM::BitcastOp>(user)) {
        if (IsPointerType(bitcastOp.getType()))
          worklist.push_back(bitcastOp.getResult());
        continue;
      }

      if (auto addrSpaceOp = llvm::dyn_cast<LLVM::AddrSpaceCastOp>(user)) {
        if (IsPointerType(addrSpaceOp.getType()))
          worklist.push_back(addrSpaceOp.getResult());
        continue;
      }

      if (auto selectOp = llvm::dyn_cast<LLVM::SelectOp>(user)) {
        if (IsPointerType(selectOp.getType()))
          worklist.push_back(selectOp.getResult());
        continue;
      }

      if (auto callOp = llvm::dyn_cast<LLVM::CallOp>(user)) {
        auto calleeAttr = callOp.getCalleeAttr();
        if (!calleeAttr)
          continue;
        auto operands = callOp.getOperands();
        auto it = llvm::find(operands, current);
        if (it == operands.end())
          continue;
        size_t operandIndex = static_cast<size_t>(it - operands.begin());
        auto module = callOp->getParentOfType<ModuleOp>();
        if (!module)
          continue;
        auto calleeFunc =
            module.lookupSymbol<LLVM::LLVMFuncOp>(calleeAttr.getValue());
        if (!calleeFunc) {
          llvm::SmallString<64> renamedName(calleeAttr.getValue());
          renamedName.append(".llvm");
          calleeFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(renamedName);
        }
        if (calleeFunc && !calleeFunc.isExternal()) {
          if (operandIndex >= calleeFunc.getNumArguments())
            continue;
          worklist.push_back(calleeFunc.getArgument(operandIndex));
          continue;
        }
        auto funcCallee =
            module.lookupSymbol<func::FuncOp>(calleeAttr.getValue());
        if (!funcCallee)
          continue;
        if (operandIndex >= funcCallee.getNumArguments())
          continue;
        Type argType = funcCallee.getArgument(operandIndex).getType();
        if (auto memrefType = llvm::dyn_cast<MemRefType>(argType)) {
          if (auto strong = recordCandidate(memrefType.getElementType()))
            return strong;
        }
        continue;
      }

      if (auto callIntrinsicOp = llvm::dyn_cast<LLVM::CallIntrinsicOp>(user)) {
        StringRef callee = callIntrinsicOp.getIntrin();
        if (callee.starts_with("llvm.memcpy") ||
            callee.starts_with("llvm.memmove")) {
          if (callIntrinsicOp.getNumOperands() < 2)
            continue;
          Value dst = callIntrinsicOp.getOperand(0);
          Value src = callIntrinsicOp.getOperand(1);
          if (current == dst && IsPointerType(src.getType()))
            worklist.push_back(src);
          if (current == src && IsPointerType(dst.getType()))
            worklist.push_back(dst);
          continue;
        }
      }

    }

    if (auto blockArg = llvm::dyn_cast<BlockArgument>(current)) {
      auto *parent = blockArg.getOwner()->getParentOp();
      auto func = llvm::dyn_cast_or_null<LLVM::LLVMFuncOp>(parent);
      if (!func)
        continue;
      if (&func.getBody().front() != blockArg.getOwner())
        continue;
      unsigned argIndex = blockArg.getArgNumber();
      auto module = func->getParentOfType<ModuleOp>();
      if (!module)
        continue;
      llvm::StringRef funcName = func.getName();
      llvm::StringRef baseName = funcName;
      llvm::StringRef kLlvmSuffix(".llvm");
      if (funcName.ends_with(kLlvmSuffix))
        baseName = funcName.drop_back(kLlvmSuffix.size());
      module.walk([&](LLVM::CallOp call) {
        auto calleeAttr = call.getCalleeAttr();
        if (!calleeAttr)
          return;
        llvm::StringRef calleeName = calleeAttr.getValue();
        if (calleeName != funcName && calleeName != baseName)
          return;
        if (argIndex >= call.getNumOperands())
          return;
        Value operand = call.getOperand(argIndex);
        worklist.push_back(operand);
      });
      std::optional<Type> callArgType;
      module.walk([&](func::CallOp call) {
        if (call.getCallee() != baseName)
          return;
        if (argIndex >= call.getNumOperands())
          return;
        Type argType = call.getOperand(argIndex).getType();
        if (auto memrefType = llvm::dyn_cast<MemRefType>(argType)) {
          callArgType = memrefType.getElementType();
        }
      });
      if (callArgType) {
        if (auto strong = recordCandidate(*callArgType))
          return strong;
      }
    }
  }
  return fallback;
}

std::optional<Type> GuessPointerElementTypeFromValue(Value value) {
  SmallVector<int64_t, 4> dims;
  MLIRContext *context = value.getContext();

  if (auto gepOp = value.getDefiningOp<LLVM::GEPOp>()) {
    Type scalar = GetScalarType(gepOp.getElemType(), dims);
    return NormalizeScalarType(scalar, context);
  }
  if (auto allocaOp = value.getDefiningOp<LLVM::AllocaOp>()) {
    Type scalar = GetScalarType(allocaOp.getElemType(), dims);
    return NormalizeScalarType(scalar, context);
  }
  if (auto bitcastOp = value.getDefiningOp<LLVM::BitcastOp>()) {
    if (IsPointerType(bitcastOp.getArg().getType()))
      return GuessPointerElementTypeFromValue(bitcastOp.getArg());
  }
  if (auto addrSpaceOp = value.getDefiningOp<LLVM::AddrSpaceCastOp>()) {
    if (IsPointerType(addrSpaceOp.getArg().getType()))
      return GuessPointerElementTypeFromValue(addrSpaceOp.getArg());
  }
  return std::nullopt;
}

std::optional<Type>
InferPointerElementTypeFromCallSites(LLVM::LLVMFuncOp func,
                                     unsigned argIndex) {
  auto module = func->getParentOfType<ModuleOp>();
  if (!module)
    return std::nullopt;
  llvm::StringRef funcName = func.getName();
  llvm::StringRef baseName = funcName;
  llvm::StringRef kLlvmSuffix(".llvm");
  if (funcName.ends_with(kLlvmSuffix))
    baseName = funcName.drop_back(kLlvmSuffix.size());
  std::optional<Type> inferredType;
  module.walk([&](LLVM::CallOp call) {
    if (inferredType)
      return;
    auto calleeAttr = call.getCalleeAttr();
    if (!calleeAttr)
      return;
    llvm::StringRef calleeName = calleeAttr.getValue();
    if (calleeName != funcName && calleeName != baseName)
      return;
    if (argIndex >= call.getNumOperands())
      return;
    Value operand = call.getOperand(argIndex);
    if (!IsPointerType(operand.getType()))
      return;
    if (auto guessed = GuessPointerElementTypeFromValue(operand)) {
      inferredType = guessed;
      return;
    }
    if (auto inferred = InferPointerElementType(operand)) {
      inferredType = inferred;
      return;
    }
  });
  if (inferredType)
    return inferredType;

  module.walk([&](func::CallOp call) {
    if (inferredType)
      return;
    if (call.getCallee() != baseName)
      return;
    if (argIndex >= call.getNumOperands())
      return;
    Type argType = call.getOperand(argIndex).getType();
    if (auto memrefType = llvm::dyn_cast<MemRefType>(argType))
      inferredType = memrefType.getElementType();
  });
  return inferredType;
}

MemRefType MakeMemRefType(Type elementType, Attribute memorySpace) {
  return MemRefType::get({ShapedType::kDynamic}, elementType,
                         MemRefLayoutAttrInterface(), memorySpace);
}

MemRefType MakeStridedMemRefType(Type elementType, Attribute memorySpace) {
  auto layout = StridedLayoutAttr::get(elementType.getContext(),
                                       ShapedType::kDynamic, {1});
  return MemRefType::get({ShapedType::kDynamic}, elementType, layout,
                         memorySpace);
}

Value StripIndexCasts(Value value) {
  while (value) {
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
      value = cast.getIn();
      continue;
    }
    if (auto cast = value.getDefiningOp<arith::ExtUIOp>()) {
      value = cast.getIn();
      continue;
    }
    if (auto cast = value.getDefiningOp<arith::ExtSIOp>()) {
      value = cast.getIn();
      continue;
    }
    if (auto cast = value.getDefiningOp<arith::TruncIOp>()) {
      value = cast.getIn();
      continue;
    }
    break;
  }
  return value;
}

std::optional<int64_t> GetConstantIntValue(Value value) {
  value = StripIndexCasts(value);
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return std::nullopt;
  auto intAttr = llvm::dyn_cast<IntegerAttr>(constOp.getValue());
  if (!intAttr)
    return std::nullopt;
  return intAttr.getInt();
}

bool IsZeroIndex(Value value) {
  if (!value)
    return true;
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return false;
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(constOp.getValue()))
    return intAttr.getValue().isZero();
  return false;
}

Value MaterializeSubview(OpBuilder &builder, Location loc, Value base,
                                Value offset, Value length) {
  SmallVector<OpFoldResult, 1> offsets{offset};
  SmallVector<OpFoldResult, 1> sizes{length};
  SmallVector<OpFoldResult, 1> strides{builder.getIndexAttr(1)};
  auto subview =
      memref::SubViewOp::create(builder, loc, base, offsets, sizes, strides);
  return subview.getResult();
}

Value MaterializeMemrefPointer(OpBuilder &builder, Location loc,
                                      const PointerInfo &info) {
  Value result = nullptr;
  if (IsZeroIndex(info.index)) {
    result = info.base;
  } else {
    Value dim = memref::DimOp::create(builder, loc, info.base, 0);
    Value size = arith::SubIOp::create(builder, loc, dim, info.index);
    result = MaterializeSubview(builder, loc, info.base, info.index, size);
  }

  auto memrefType = llvm::dyn_cast<MemRefType>(result.getType());
  if (!memrefType)
    return result;

  auto targetType =
      MakeStridedMemRefType(info.elementType, memrefType.getMemorySpace());
  if (memrefType == targetType)
    return result;
  return memref::CastOp::create(builder, loc, targetType, result);
}

Value MaterializeLLVMPointer(OpBuilder &builder, Location loc,
                                    const PointerInfo &info) {
  Value baseIndex = memref::ExtractAlignedPointerAsIndexOp::create(builder,
      loc, info.base);
  Value baseInt = arith::IndexCastOp::create(builder,
      loc, builder.getI64Type(), baseIndex);
  int64_t elemSize = GetByteSize(info.elementType);
  if (elemSize == 0)
    return LLVM::IntToPtrOp::create(builder, loc, LLVM::LLVMPointerType::get(
                                                     builder.getContext()),
                                            baseInt);
  Value offsetBytes = info.index;
  if (elemSize != 1) {
    Value scale = BuildIndexConstant(builder, loc, elemSize);
    offsetBytes = arith::MulIOp::create(builder, loc, info.index, scale);
  }
  Value offsetInt = arith::IndexCastOp::create(builder,
      loc, builder.getI64Type(), offsetBytes);
  Value total = arith::AddIOp::create(builder, loc, baseInt, offsetInt);
  return LLVM::IntToPtrOp::create(builder, loc,
                                          LLVM::LLVMPointerType::get(
                                              builder.getContext()),
                                          total);
}

std::optional<Value> ConvertLLVMConstant(OpBuilder &builder,
                                                LLVM::ConstantOp op) {
  Attribute attr = op.getValue();
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr))
    return arith::ConstantOp::create(builder, op.getLoc(), intAttr);
  if (auto floatAttr = llvm::dyn_cast<FloatAttr>(attr))
    return arith::ConstantOp::create(builder, op.getLoc(), floatAttr);
  return std::nullopt;
}

std::optional<Value>
ConvertMathCall(OpBuilder &builder, Location loc, StringRef callee,
                ValueRange operands) {
  if (operands.size() == 1) {
    if (callee == "sinf" || callee == "sin")
      return math::SinOp::create(builder, loc, operands[0]);
    if (callee == "cosf" || callee == "cos")
      return math::CosOp::create(builder, loc, operands[0]);
    if (callee == "expf" || callee == "exp")
      return math::ExpOp::create(builder, loc, operands[0]);
    if (callee == "logf" || callee == "log")
      return math::LogOp::create(builder, loc, operands[0]);
    if (callee == "log2f" || callee == "log2")
      return math::Log2Op::create(builder, loc, operands[0]);
    if (callee == "sqrtf" || callee == "sqrt")
      return math::SqrtOp::create(builder, loc, operands[0]);
    if (callee == "fabsf" || callee == "fabs")
      return math::AbsFOp::create(builder, loc, operands[0]);
  }
  if (operands.size() == 2) {
    if (callee == "powf" || callee == "pow")
      return math::PowFOp::create(builder, loc, operands[0], operands[1]);
  }
  return std::nullopt;
}

bool IsStdMinMaxName(StringRef callee, StdMinMaxKind &kind) {
  if (callee.starts_with("_ZSt3min")) {
    kind = StdMinMaxKind::Minimum;
    return true;
  }
  if (callee.starts_with("_ZSt3max")) {
    kind = StdMinMaxKind::Maximum;
    return true;
  }
  return false;
}

StdMinMaxScalarKind ParseStdMinMaxScalarKind(StringRef callee) {
  size_t templatePos = callee.find('I');
  if (templatePos == StringRef::npos || templatePos + 1 >= callee.size())
    return StdMinMaxScalarKind::UnknownKind;
  char code = callee[templatePos + 1];
  switch (code) {
  case 'h':
  case 'j':
  case 'm':
  case 't':
  case 'y':
    return StdMinMaxScalarKind::UnsignedIntKind;
  case 'a':
  case 'i':
  case 'l':
  case 's':
  case 'x':
    return StdMinMaxScalarKind::SignedIntKind;
  case 'f':
  case 'd':
  case 'e':
    return StdMinMaxScalarKind::FloatKind;
  default:
    return StdMinMaxScalarKind::UnknownKind;
  }
}

std::optional<PointerInfo> BuildPointerSelect(OpBuilder &builder,
                                                     Location loc, Value cond,
                                                     const PointerInfo &lhs,
                                                     const PointerInfo &rhs,
                                                     bool trueSelectsRhs) {
  if (lhs.elementType != rhs.elementType)
    return std::nullopt;
  auto lhsBaseType = llvm::dyn_cast<MemRefType>(lhs.base.getType());
  auto rhsBaseType = llvm::dyn_cast<MemRefType>(rhs.base.getType());
  if (!lhsBaseType || !rhsBaseType)
    return std::nullopt;
  if (lhsBaseType.getMemorySpace() != rhsBaseType.getMemorySpace())
    return std::nullopt;
  auto commonBaseType =
      MakeStridedMemRefType(lhs.elementType, lhsBaseType.getMemorySpace());
  Value lhsBase = lhs.base;
  if (lhsBaseType != commonBaseType) {
    if (!memref::CastOp::areCastCompatible(lhsBaseType, commonBaseType))
      return std::nullopt;
    lhsBase = memref::CastOp::create(builder, loc, commonBaseType, lhsBase);
  }
  Value rhsBase = rhs.base;
  if (rhsBaseType != commonBaseType) {
    if (!memref::CastOp::areCastCompatible(rhsBaseType, commonBaseType))
      return std::nullopt;
    rhsBase = memref::CastOp::create(builder, loc, commonBaseType, rhsBase);
  }

  Value trueBase = trueSelectsRhs ? rhsBase : lhsBase;
  Value falseBase = trueSelectsRhs ? lhsBase : rhsBase;
  Value trueIndex = trueSelectsRhs ? rhs.index : lhs.index;
  Value falseIndex = trueSelectsRhs ? lhs.index : rhs.index;
  Value baseSel = arith::SelectOp::create(builder, loc, cond, trueBase, falseBase);
  Value indexSel =
      arith::SelectOp::create(builder, loc, cond, trueIndex, falseIndex);
  return PointerInfo{baseSel, indexSel, lhs.elementType};
}

std::optional<Value>
ConvertStdMinMaxScalarCall(OpBuilder &builder, Location loc, StringRef callee,
                           ValueRange operands) {
  StdMinMaxKind kind = StdMinMaxKind::Minimum;
  if (!IsStdMinMaxName(callee, kind))
    return std::nullopt;
  if (operands.size() != 2)
    return std::nullopt;
  Type type = operands[0].getType();
  if (type != operands[1].getType())
    return std::nullopt;

  StdMinMaxScalarKind scalarKind = ParseStdMinMaxScalarKind(callee);
  if (auto floatTy = llvm::dyn_cast<FloatType>(type)) {
    (void)floatTy;
    if (scalarKind != StdMinMaxScalarKind::FloatKind &&
        scalarKind != StdMinMaxScalarKind::UnknownKind) {
      return std::nullopt;
    }
    if (kind == StdMinMaxKind::Minimum)
      return arith::MinimumFOp::create(builder, loc, operands[0], operands[1]);
    return arith::MaximumFOp::create(builder, loc, operands[0], operands[1]);
  }

  if (llvm::isa<IntegerType>(type)) {
    bool isUnsigned = scalarKind == StdMinMaxScalarKind::UnsignedIntKind;
    bool isSigned = scalarKind == StdMinMaxScalarKind::SignedIntKind;
    if (!isUnsigned && !isSigned)
      return std::nullopt;
    if (kind == StdMinMaxKind::Minimum) {
      if (isUnsigned)
        return arith::MinUIOp::create(builder, loc, operands[0], operands[1]);
      return arith::MinSIOp::create(builder, loc, operands[0], operands[1]);
    }
    if (isUnsigned)
      return arith::MaxUIOp::create(builder, loc, operands[0], operands[1]);
    return arith::MaxSIOp::create(builder, loc, operands[0], operands[1]);
  }

  return std::nullopt;
}

arith::CmpIPredicate ConvertICmpPredicate(LLVM::ICmpPredicate pred) {
  using LLVM::ICmpPredicate;
  switch (pred) {
  case ICmpPredicate::eq:
    return arith::CmpIPredicate::eq;
  case ICmpPredicate::ne:
    return arith::CmpIPredicate::ne;
  case ICmpPredicate::slt:
    return arith::CmpIPredicate::slt;
  case ICmpPredicate::sle:
    return arith::CmpIPredicate::sle;
  case ICmpPredicate::sgt:
    return arith::CmpIPredicate::sgt;
  case ICmpPredicate::sge:
    return arith::CmpIPredicate::sge;
  case ICmpPredicate::ult:
    return arith::CmpIPredicate::ult;
  case ICmpPredicate::ule:
    return arith::CmpIPredicate::ule;
  case ICmpPredicate::ugt:
    return arith::CmpIPredicate::ugt;
  case ICmpPredicate::uge:
    return arith::CmpIPredicate::uge;
  }
  return arith::CmpIPredicate::eq;
}

arith::CmpFPredicate ConvertFCmpPredicate(LLVM::FCmpPredicate pred) {
  using LLVM::FCmpPredicate;
  switch (pred) {
  case FCmpPredicate::oeq:
    return arith::CmpFPredicate::OEQ;
  case FCmpPredicate::one:
    return arith::CmpFPredicate::ONE;
  case FCmpPredicate::olt:
    return arith::CmpFPredicate::OLT;
  case FCmpPredicate::ole:
    return arith::CmpFPredicate::OLE;
  case FCmpPredicate::ogt:
    return arith::CmpFPredicate::OGT;
  case FCmpPredicate::oge:
    return arith::CmpFPredicate::OGE;
  case FCmpPredicate::ueq:
    return arith::CmpFPredicate::UEQ;
  case FCmpPredicate::une:
    return arith::CmpFPredicate::UNE;
  case FCmpPredicate::ult:
    return arith::CmpFPredicate::ULT;
  case FCmpPredicate::ule:
    return arith::CmpFPredicate::ULE;
  case FCmpPredicate::ugt:
    return arith::CmpFPredicate::UGT;
  case FCmpPredicate::uge:
    return arith::CmpFPredicate::UGE;
  case FCmpPredicate::ord:
    return arith::CmpFPredicate::ORD;
  case FCmpPredicate::uno:
    return arith::CmpFPredicate::UNO;
  case FCmpPredicate::_true:
    return arith::CmpFPredicate::AlwaysTrue;
  case FCmpPredicate::_false:
    return arith::CmpFPredicate::AlwaysFalse;
  }
  return arith::CmpFPredicate::OEQ;
}

} // namespace loom::llvm_to_scf
