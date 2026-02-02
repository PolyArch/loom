#include "loom/Conversion/LLVMToSCF.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>
#include <string>
#include <utility>

using namespace mlir;

namespace {

struct PointerInfo {
  Value base;
  Value index;
  Type elementType;
};

static void CopyLoomAnnotations(Operation *src, Operation *dst) {
  if (!src || !dst)
    return;
  auto attr = src->getAttr("loom.annotations");
  if (attr)
    dst->setAttr("loom.annotations", attr);
}

static bool IsPointerType(Type type) {
  return llvm::isa<LLVM::LLVMPointerType>(type);
}

static Type GetScalarType(Type type, SmallVectorImpl<int64_t> &dims) {
  while (auto arrayTy = llvm::dyn_cast<LLVM::LLVMArrayType>(type)) {
    dims.push_back(arrayTy.getNumElements());
    type = arrayTy.getElementType();
  }
  return type;
}

static Type NormalizeScalarType(Type type, MLIRContext *context) {
  if (llvm::isa<IntegerType>(type) || llvm::isa<FloatType>(type))
    return type;
  if (llvm::isa<LLVM::LLVMPointerType>(type))
    return IntegerType::get(context, 8);
  return type;
}

static int64_t GetByteSize(Type type) {
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

static Value BuildIndexConstant(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                           builder.getIndexAttr(value));
}

static Value ToIndexValue(OpBuilder &builder, Location loc, Value value) {
  if (!value)
    return nullptr;
  if (value.getType().isIndex())
    return value;
  if (llvm::isa<IntegerType>(value.getType()))
    return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                              value);
  return nullptr;
}

static std::optional<Value> LookupValue(const DenseMap<Value, Value> &valueMap,
                                        Value key) {
  auto it = valueMap.find(key);
  if (it == valueMap.end())
    return std::nullopt;
  return it->second;
}

static std::optional<PointerInfo>
LookupPointer(const DenseMap<Value, PointerInfo> &ptrMap, Value key) {
  auto it = ptrMap.find(key);
  if (it == ptrMap.end())
    return std::nullopt;
  return it->second;
}

static std::optional<Type> InferPointerElementType(Value value) {
  SmallVector<int64_t, 4> dims;
  MLIRContext *context = value.getContext();
  SmallVector<Value, 8> worklist;
  llvm::DenseSet<Value> visited;
  worklist.push_back(value);

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    for (Operation *user : current.getUsers()) {
      if (auto loadOp = llvm::dyn_cast<LLVM::LoadOp>(user)) {
        if (loadOp.getAddr() != current)
          continue;
        Type resultTy = loadOp.getResult().getType();
        if (IsPointerType(resultTy)) {
          worklist.push_back(loadOp.getResult());
          continue;
        }
        return NormalizeScalarType(resultTy, context);
      }

      if (auto storeOp = llvm::dyn_cast<LLVM::StoreOp>(user)) {
        if (storeOp.getAddr() == current) {
          Type valueTy = storeOp.getValue().getType();
          if (!IsPointerType(valueTy))
            return NormalizeScalarType(valueTy, context);
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
            return NormalizeScalarType(resultTy, context);
          }
        }
      }

      if (auto gepOp = llvm::dyn_cast<LLVM::GEPOp>(user)) {
        if (gepOp.getBase() != current)
          continue;
        dims.clear();
        Type scalar = GetScalarType(gepOp.getElemType(), dims);
        return NormalizeScalarType(scalar, context);
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
        if (auto memrefType = llvm::dyn_cast<MemRefType>(argType))
          return memrefType.getElementType();
        continue;
      }

    }
  }
  return IntegerType::get(context, 8);
}

static MemRefType MakeMemRefType(Type elementType, Attribute memorySpace = {}) {
  return MemRefType::get({ShapedType::kDynamic}, elementType,
                         MemRefLayoutAttrInterface(), memorySpace);
}

static MemRefType MakeStridedMemRefType(Type elementType,
                                        Attribute memorySpace = {}) {
  auto layout = StridedLayoutAttr::get(elementType.getContext(),
                                       ShapedType::kDynamic, {1});
  return MemRefType::get({ShapedType::kDynamic}, elementType, layout,
                         memorySpace);
}

static Value StripIndexCasts(Value value) {
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

static std::optional<int64_t> GetConstantIntValue(Value value) {
  value = StripIndexCasts(value);
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return std::nullopt;
  auto intAttr = llvm::dyn_cast<IntegerAttr>(constOp.getValue());
  if (!intAttr)
    return std::nullopt;
  return intAttr.getInt();
}

static bool IsZeroIndex(Value value) {
  if (!value)
    return true;
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return false;
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(constOp.getValue()))
    return intAttr.getValue().isZero();
  return false;
}

static Value MaterializeSubview(OpBuilder &builder, Location loc, Value base,
                                Value offset, Value length) {
  SmallVector<OpFoldResult, 1> offsets{offset};
  SmallVector<OpFoldResult, 1> sizes{length};
  SmallVector<OpFoldResult, 1> strides{builder.getIndexAttr(1)};
  auto subview =
      builder.create<memref::SubViewOp>(loc, base, offsets, sizes, strides);
  return subview.getResult();
}

static Value MaterializeMemrefPointer(OpBuilder &builder, Location loc,
                                      const PointerInfo &info) {
  Value result = nullptr;
  if (IsZeroIndex(info.index)) {
    result = info.base;
  } else {
    Value dim = builder.create<memref::DimOp>(loc, info.base, 0);
    Value size = builder.create<arith::SubIOp>(loc, dim, info.index);
    result = MaterializeSubview(builder, loc, info.base, info.index, size);
  }

  auto memrefType = llvm::dyn_cast<MemRefType>(result.getType());
  if (!memrefType)
    return result;

  auto targetType =
      MakeStridedMemRefType(info.elementType, memrefType.getMemorySpace());
  if (memrefType == targetType)
    return result;
  return builder.create<memref::CastOp>(loc, targetType, result);
}

static Value MaterializeLLVMPointer(OpBuilder &builder, Location loc,
                                    const PointerInfo &info) {
  Value baseIndex = builder.create<memref::ExtractAlignedPointerAsIndexOp>(
      loc, info.base);
  Value baseInt = builder.create<arith::IndexCastOp>(
      loc, builder.getI64Type(), baseIndex);
  int64_t elemSize = GetByteSize(info.elementType);
  if (elemSize == 0)
    return builder.create<LLVM::IntToPtrOp>(loc, LLVM::LLVMPointerType::get(
                                                     builder.getContext()),
                                            baseInt);
  Value offsetBytes = info.index;
  if (elemSize != 1) {
    Value scale = BuildIndexConstant(builder, loc, elemSize);
    offsetBytes = builder.create<arith::MulIOp>(loc, info.index, scale);
  }
  Value offsetInt = builder.create<arith::IndexCastOp>(
      loc, builder.getI64Type(), offsetBytes);
  Value total = builder.create<arith::AddIOp>(loc, baseInt, offsetInt);
  return builder.create<LLVM::IntToPtrOp>(loc,
                                          LLVM::LLVMPointerType::get(
                                              builder.getContext()),
                                          total);
}

static std::optional<Value> ConvertLLVMConstant(OpBuilder &builder,
                                                LLVM::ConstantOp op) {
  Attribute attr = op.getValue();
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr))
    return builder.create<arith::ConstantOp>(op.getLoc(), intAttr);
  if (auto floatAttr = llvm::dyn_cast<FloatAttr>(attr))
    return builder.create<arith::ConstantOp>(op.getLoc(), floatAttr);
  return std::nullopt;
}

static std::optional<Value>
ConvertMathCall(OpBuilder &builder, Location loc, StringRef callee,
                ValueRange operands) {
  if (operands.size() == 1) {
    if (callee == "sinf" || callee == "sin")
      return builder.create<math::SinOp>(loc, operands[0]);
    if (callee == "cosf" || callee == "cos")
      return builder.create<math::CosOp>(loc, operands[0]);
    if (callee == "expf" || callee == "exp")
      return builder.create<math::ExpOp>(loc, operands[0]);
    if (callee == "logf" || callee == "log")
      return builder.create<math::LogOp>(loc, operands[0]);
    if (callee == "log2f" || callee == "log2")
      return builder.create<math::Log2Op>(loc, operands[0]);
    if (callee == "sqrtf" || callee == "sqrt")
      return builder.create<math::SqrtOp>(loc, operands[0]);
    if (callee == "fabsf" || callee == "fabs")
      return builder.create<math::AbsFOp>(loc, operands[0]);
  }
  if (operands.size() == 2) {
    if (callee == "powf" || callee == "pow")
      return builder.create<math::PowFOp>(loc, operands[0], operands[1]);
  }
  return std::nullopt;
}

enum class StdMinMaxKind { Minimum, Maximum };

enum class StdMinMaxScalarKind {
  SignedIntKind,
  UnsignedIntKind,
  FloatKind,
  UnknownKind
};

static bool IsStdMinMaxName(StringRef callee, StdMinMaxKind &kind) {
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

static StdMinMaxScalarKind ParseStdMinMaxScalarKind(StringRef callee) {
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

static std::optional<PointerInfo> BuildPointerSelect(OpBuilder &builder,
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
    lhsBase = builder.create<memref::CastOp>(loc, commonBaseType, lhsBase);
  }
  Value rhsBase = rhs.base;
  if (rhsBaseType != commonBaseType) {
    if (!memref::CastOp::areCastCompatible(rhsBaseType, commonBaseType))
      return std::nullopt;
    rhsBase = builder.create<memref::CastOp>(loc, commonBaseType, rhsBase);
  }

  Value trueBase = trueSelectsRhs ? rhsBase : lhsBase;
  Value falseBase = trueSelectsRhs ? lhsBase : rhsBase;
  Value trueIndex = trueSelectsRhs ? rhs.index : lhs.index;
  Value falseIndex = trueSelectsRhs ? lhs.index : rhs.index;
  Value baseSel = builder.create<arith::SelectOp>(loc, cond, trueBase, falseBase);
  Value indexSel =
      builder.create<arith::SelectOp>(loc, cond, trueIndex, falseIndex);
  return PointerInfo{baseSel, indexSel, lhs.elementType};
}

static std::optional<Value>
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
      return builder.create<arith::MinimumFOp>(loc, operands[0], operands[1]);
    return builder.create<arith::MaximumFOp>(loc, operands[0], operands[1]);
  }

  if (llvm::isa<IntegerType>(type)) {
    bool isUnsigned = scalarKind == StdMinMaxScalarKind::UnsignedIntKind;
    bool isSigned = scalarKind == StdMinMaxScalarKind::SignedIntKind;
    if (!isUnsigned && !isSigned)
      return std::nullopt;
    if (kind == StdMinMaxKind::Minimum) {
      if (isUnsigned)
        return builder.create<arith::MinUIOp>(loc, operands[0], operands[1]);
      return builder.create<arith::MinSIOp>(loc, operands[0], operands[1]);
    }
    if (isUnsigned)
      return builder.create<arith::MaxUIOp>(loc, operands[0], operands[1]);
    return builder.create<arith::MaxSIOp>(loc, operands[0], operands[1]);
  }

  return std::nullopt;
}

static arith::CmpIPredicate ConvertICmpPredicate(LLVM::ICmpPredicate pred) {
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

static arith::CmpFPredicate ConvertFCmpPredicate(LLVM::FCmpPredicate pred) {
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

struct ConvertedGlobal {
  MemRefType type;
  LLVM::GlobalOp oldGlobal;
};

class LowerLLVMToSCFPass
    : public PassWrapper<LowerLLVMToSCFPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "loom-lower-llvm-to-scf"; }
  StringRef getDescription() const final {
    return "Lower LLVM dialect to scf-stage dialects";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    OpBuilder builder(context);

    llvm::StringSet<> varargFunctions;
    for (auto func : module.getOps<LLVM::LLVMFuncOp>()) {
      if (func.getFunctionType().isVarArg())
        varargFunctions.insert(func.getName());
    }

    llvm::StringMap<ConvertedGlobal> convertedGlobals;
    if (failed(convertGlobals(module, builder, convertedGlobals))) {
      signalPassFailure();
      return;
    }

    llvm::SmallVector<LLVM::LLVMFuncOp, 8> llvmFunctions;
    for (auto func : module.getOps<LLVM::LLVMFuncOp>())
      llvmFunctions.push_back(func);

    for (LLVM::LLVMFuncOp func : llvmFunctions) {
      if (varargFunctions.contains(func.getName()))
        continue;
      if (failed(convertFunction(module, func, builder, convertedGlobals,
                                 varargFunctions))) {
        signalPassFailure();
        return;
      }
    }

    llvm::SmallVector<LLVM::GlobalOp, 8> globalsToErase;
    for (auto &entry : convertedGlobals)
      globalsToErase.push_back(entry.second.oldGlobal);
    for (LLVM::GlobalOp global : globalsToErase)
      global.erase();
  }

private:
  LogicalResult convertGlobals(ModuleOp module, OpBuilder &builder,
                               llvm::StringMap<ConvertedGlobal> &out) {
    llvm::SmallVector<LLVM::GlobalOp, 8> globals;
    for (auto global : module.getOps<LLVM::GlobalOp>())
      globals.push_back(global);

    for (LLVM::GlobalOp global : globals) {
      if (global.getName() == "llvm.global.annotations") {
        global.erase();
        continue;
      }
      Attribute valueAttr = global.getValueAttr();
      if (!valueAttr)
        continue;

      std::string originalName = global.getName().str();
      std::string renamedName = originalName + ".llvm";
      global.setSymName(renamedName);

      SmallVector<int64_t, 4> dims;
      Type scalar = GetScalarType(global.getType(), dims);
      scalar = NormalizeScalarType(scalar, module.getContext());
      if (!llvm::isa<IntegerType>(scalar) &&
          !llvm::isa<FloatType>(scalar)) {
        global.emitError("unsupported global element type");
        return failure();
      }

      int64_t totalCount = 1;
      for (int64_t dim : dims)
        totalCount *= dim;

      MemRefType memrefType =
          MemRefType::get({totalCount}, scalar, MemRefLayoutAttrInterface(),
                          global.getAddrSpaceAttr());

      Attribute initAttr;
      if (auto strAttr = llvm::dyn_cast<StringAttr>(valueAttr)) {
        std::string data = strAttr.getValue().str();
        SmallVector<int8_t, 16> bytes;
        bytes.reserve(data.size());
        for (char ch : data)
          bytes.push_back(static_cast<int8_t>(ch));
        auto tensorType =
            RankedTensorType::get({static_cast<int64_t>(bytes.size())},
                                  IntegerType::get(module.getContext(), 8));
        llvm::ArrayRef<int8_t> byteRef(bytes.data(), bytes.size());
        initAttr = DenseElementsAttr::get(tensorType, byteRef);
      } else if (auto denseAttr =
                     llvm::dyn_cast<DenseElementsAttr>(valueAttr)) {
        initAttr = denseAttr;
      } else if (auto intAttr = llvm::dyn_cast<IntegerAttr>(valueAttr)) {
        auto tensorType = RankedTensorType::get({1}, scalar);
        initAttr = DenseElementsAttr::get(tensorType, {intAttr.getValue()});
      } else if (auto floatAttr = llvm::dyn_cast<FloatAttr>(valueAttr)) {
        auto tensorType = RankedTensorType::get({1}, scalar);
        initAttr = DenseElementsAttr::get(tensorType, {floatAttr.getValue()});
      } else {
        global.emitError("unsupported global initializer");
        return failure();
      }

      builder.setInsertionPoint(global);
      auto newGlobal = builder.create<memref::GlobalOp>(
          global.getLoc(), originalName, StringAttr(), memrefType, initAttr,
          global.getConstant(), global.getAlignmentAttr());
      out[originalName] = {memrefType, global};
      (void)newGlobal;
    }
    return success();
  }

  LogicalResult convertFunction(ModuleOp module, LLVM::LLVMFuncOp func,
                                OpBuilder &builder,
                                const llvm::StringMap<ConvertedGlobal> &globals,
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
    auto paramTypes = func.getFunctionType().getParams();
    for (size_t index = 0; index < paramTypes.size(); ++index) {
      Type paramType = paramTypes[index];
      if (IsPointerType(paramType)) {
        Type elemType = IntegerType::get(module.getContext(), 8);
        if (!func.isExternal()) {
          elemType = InferPointerElementType(func.getArgument(index))
                         .value_or(elemType);
        }
        argTypes.push_back(MakeStridedMemRefType(elemType));
        continue;
      }
      argTypes.push_back(
          NormalizeScalarType(paramType, module.getContext()));
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
        builder.create<func::FuncOp>(func.getLoc(), originalName, funcType);
    newFunc.setSymVisibilityAttr(func.getSymVisibilityAttr());
    CopyLoomAnnotations(func.getOperation(), newFunc.getOperation());
    if (func.isExternal()) {
      newFunc.setSymVisibilityAttr(builder.getStringAttr("private"));
      func.erase();
      return success();
    }

    DenseMap<Value, Value> valueMap;
    DenseMap<Value, PointerInfo> pointerMap;
    DenseMap<Value, PointerInfo> pointerSlotValues;
    llvm::DenseSet<Value> pointerSlots;
    DenseMap<Block *, Block *> blockMap;

    for (Block &oldBlock : func.getBody()) {
      Block *newBlock = new Block();
      newFunc.getBody().push_back(newBlock);
      blockMap[&oldBlock] = newBlock;
      OpBuilder blockBuilder(module.getContext());
      blockBuilder.setInsertionPointToStart(newBlock);
      Value zeroIndex =
          BuildIndexConstant(blockBuilder, func.getLoc(), 0);
      for (BlockArgument arg : oldBlock.getArguments()) {
        Type oldType = arg.getType();
        if (IsPointerType(oldType)) {
          auto elemType = InferPointerElementType(arg).value_or(
              IntegerType::get(module.getContext(), 8));
          auto memrefType = MakeStridedMemRefType(elemType);
          auto newArg = newBlock->addArgument(memrefType, arg.getLoc());
          PointerInfo info{newArg, zeroIndex, elemType};
          pointerMap[arg] = info;
          continue;
        }
        auto newArg = newBlock->addArgument(
            NormalizeScalarType(oldType, module.getContext()), arg.getLoc());
        valueMap[arg] = newArg;
      }
    }

    for (Block &oldBlock : func.getBody()) {
      Block *newBlock = blockMap[&oldBlock];
      builder.setInsertionPointToStart(newBlock);

      for (Operation &op : oldBlock.getOperations()) {
        builder.setInsertionPointToEnd(newBlock);
        Location loc = op.getLoc();

        if (auto constOp = llvm::dyn_cast<LLVM::ConstantOp>(op)) {
          auto value = ConvertLLVMConstant(builder, constOp);
          if (!value)
            return constOp.emitError("unsupported constant"), failure();
          valueMap[constOp.getResult()] = *value;
          continue;
        }

        if (auto zeroOp = llvm::dyn_cast<LLVM::ZeroOp>(op)) {
          Type type = zeroOp.getType();
          if (IsPointerType(type)) {
            auto zeroInt = builder.create<arith::ConstantOp>(
                loc, builder.getI64Type(),
                builder.getIntegerAttr(builder.getI64Type(), 0));
            auto ptr = builder.create<LLVM::IntToPtrOp>(loc, type,
                                                        zeroInt.getResult());
            valueMap[zeroOp.getResult()] = ptr.getResult();
            continue;
          }
          if (auto intTy = llvm::dyn_cast<IntegerType>(type)) {
            auto attr = builder.getIntegerAttr(intTy, 0);
            auto zero = builder.create<arith::ConstantOp>(loc, attr);
            valueMap[zeroOp.getResult()] = zero.getResult();
            continue;
          }
          if (auto floatTy = llvm::dyn_cast<FloatType>(type)) {
            auto attr = builder.getFloatAttr(floatTy, 0.0);
            auto zero = builder.create<arith::ConstantOp>(loc, attr);
            valueMap[zeroOp.getResult()] = zero.getResult();
            continue;
          }
          return zeroOp.emitError("unsupported zero type"), failure();
        }

        if (auto addrOp = llvm::dyn_cast<LLVM::AddressOfOp>(op)) {
          auto it = globals.find(addrOp.getGlobalName());
          if (it == globals.end())
            return addrOp.emitError("unknown global"), failure();
          auto memrefType = it->second.type;
          auto getGlobal = builder.create<memref::GetGlobalOp>(
              loc, memrefType, addrOp.getGlobalName());
          PointerInfo info{getGlobal, BuildIndexConstant(builder, loc, 0),
                           memrefType.getElementType()};
          pointerMap[addrOp.getResult()] = info;
          continue;
        }

        if (auto allocaOp = llvm::dyn_cast<LLVM::AllocaOp>(op)) {
          if (IsPointerType(allocaOp.getElemType())) {
            pointerSlots.insert(allocaOp.getResult());
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
            countValue = builder.create<arith::MulIOp>(loc, sizeIndex, countValue);
          }
          MemRefType memrefType;
          memref::AllocaOp alloc;
          if (staticCount && *staticCount > 0) {
            memrefType =
                MemRefType::get({*staticCount}, scalar,
                                MemRefLayoutAttrInterface(), Attribute());
            alloc = builder.create<memref::AllocaOp>(loc, memrefType);
          } else {
            memrefType = MakeMemRefType(scalar);
            alloc = builder.create<memref::AllocaOp>(loc, memrefType,
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
              offset = builder.create<arith::AddIOp>(loc, offset, term);
          };

          if (!indices.empty()) {
            Value stride0 = BuildIndexConstant(builder, loc, totalStride);
            Value term = indices[0];
            if (totalStride != 1)
              term = builder.create<arith::MulIOp>(loc, term, stride0);
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
              term = builder.create<arith::MulIOp>(loc, term, strideVal);
            }
            addTerm(term);
          }

          if (!offset)
            offset = BuildIndexConstant(builder, loc, 0);

          Value baseIndex = baseInfo->index;
          Value newIndex = baseIndex;
          if (!IsZeroIndex(offset))
            newIndex = builder.create<arith::AddIOp>(loc, baseIndex, offset);

          PointerInfo info{baseInfo->base, newIndex, scalar};
          pointerMap[gepOp.getResult()] = info;
          continue;
        }

        if (auto bitcastOp = llvm::dyn_cast<LLVM::BitcastOp>(op)) {
          if (IsPointerType(bitcastOp.getType())) {
            auto srcInfo = LookupPointer(pointerMap, bitcastOp.getArg());
            if (!srcInfo)
              return bitcastOp.emitError("missing bitcast source"), failure();
            pointerMap[bitcastOp.getResult()] = *srcInfo;
            continue;
          }
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
          Value val = builder.create<memref::LoadOp>(loc, ptrInfo->base,
                                                     ptrInfo->index);
          valueMap[loadOp.getResult()] = val;
          continue;
        }

        if (auto storeOp = llvm::dyn_cast<LLVM::StoreOp>(op)) {
          if (pointerSlots.contains(storeOp.getAddr())) {
            auto storedPtr = LookupPointer(pointerMap, storeOp.getValue());
            if (!storedPtr)
              return storeOp.emitError("missing pointer store value"), failure();
            pointerSlotValues[storeOp.getAddr()] = *storedPtr;
            continue;
          }
          auto ptrInfo = LookupPointer(pointerMap, storeOp.getAddr());
          if (!ptrInfo)
            return storeOp.emitError("missing store pointer"), failure();
          auto storedVal = LookupValue(valueMap, storeOp.getValue());
          if (!storedVal)
            return storeOp.emitError("missing store value"), failure();
          builder.create<memref::StoreOp>(loc, *storedVal, ptrInfo->base,
                                          ptrInfo->index);
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
              Value lhsVal = builder.create<memref::LoadOp>(
                  loc, lhsPtr->base, lhsPtr->index);
              Value rhsVal = builder.create<memref::LoadOp>(
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
                cmp = builder.create<arith::CmpFOp>(
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
                cmp = builder.create<arith::CmpIOp>(loc, pred, cmpLhs, cmpRhs);
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
          if (returnsPointer)
            return callOp.emitError("unsupported pointer return call"),
                   failure();
          SmallVector<Value, 8> operands;
          operands.reserve(callOp.getNumOperands());
          for (Value operand : callOp.getOperands()) {
            if (auto ptrInfo = LookupPointer(pointerMap, operand)) {
              if (isVarArg) {
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

          if (isVarArg) {
            auto newCall = builder.create<LLVM::CallOp>(loc, callOp.getResultTypes(),
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
          auto call = builder.create<func::CallOp>(loc, callee,
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
          if (callee.starts_with("llvm.stacksave")) {
            if (callOp.getNumResults() != 1)
              return callOp.emitError("unexpected stacksave results"), failure();
            auto memrefType = MakeMemRefType(builder.getI8Type());
            Value one = BuildIndexConstant(builder, loc, 1);
            auto alloc =
                builder.create<memref::AllocaOp>(loc, memrefType,
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
            Value length = builder.create<arith::DivUIOp>(loc, sizeIndex,
                                                          elemSizeVal);
            Value srcView =
                MaterializeSubview(builder, loc, srcInfo->base, srcInfo->index,
                                   length);
            Value dstView =
                MaterializeSubview(builder, loc, dstInfo->base, dstInfo->index,
                                   length);
            builder.create<memref::CopyOp>(loc, srcView, dstView);
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
            Value length = builder.create<arith::DivUIOp>(loc, sizeIndex,
                                                          elemSizeVal);
            Value fillVal = *val;
            Type elemType = dstInfo->elementType;
            if (fillVal.getType() != elemType) {
              auto constOp = fillVal.getDefiningOp<arith::ConstantOp>();
              if (!constOp)
                return callOp.emitError("unsupported memset value"), failure();
              auto intAttr = llvm::dyn_cast<IntegerAttr>(constOp.getValue());
              if (!intAttr || !intAttr.getValue().isZero())
                return callOp.emitError("unsupported memset value"), failure();
              if (auto floatTy = llvm::dyn_cast<FloatType>(elemType)) {
                fillVal = builder.create<arith::ConstantOp>(
                    loc, builder.getFloatAttr(floatTy, 0.0));
              } else if (auto intTy = llvm::dyn_cast<IntegerType>(elemType)) {
                fillVal = builder.create<arith::ConstantOp>(
                    loc, builder.getIntegerAttr(intTy, 0));
              } else {
                return callOp.emitError("unsupported memset value type"),
                       failure();
              }
            }
            Value dstView = MaterializeSubview(builder, loc, dstInfo->base,
                                               dstInfo->index, length);
            Value zero = BuildIndexConstant(builder, loc, 0);
            Value step = BuildIndexConstant(builder, loc, 1);
            auto loop = builder.create<scf::ForOp>(loc, zero, length, step);
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(loop.getBody());
            Value iv = loop.getInductionVar();
            builder.create<memref::StoreOp>(loc, fillVal, dstView, iv);
            continue;
          }
          if (callee.starts_with("llvm.fmuladd")) {
            if (callOp.getNumOperands() != 3)
              return callOp.emitError("invalid fmuladd operands"), failure();
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
            auto mul = builder.create<arith::MulFOp>(loc, *lhs, *rhs);
            auto add = builder.create<arith::AddFOp>(loc, mul, *addend);
            valueMap[callOp->getResult(0)] = add.getResult();
            continue;
          }
          if (callee.starts_with("llvm.fabs")) {
            if (callOp.getNumOperands() != 1)
              return callOp.emitError("invalid fabs operands"), failure();
            auto operand = LookupValue(valueMap, callOp.getOperand(0));
            if (!operand)
              return callOp.emitError("missing fabs operand"), failure();
            auto abs = builder.create<math::AbsFOp>(loc, *operand);
            valueMap[callOp->getResult(0)] = abs.getResult();
            continue;
          }
          return callOp.emitError("unsupported intrinsic"), failure();
        }

        if (auto addOp = llvm::dyn_cast<LLVM::AddOp>(op)) {
          auto lhs = LookupValue(valueMap, addOp.getLhs());
          auto rhs = LookupValue(valueMap, addOp.getRhs());
          if (!lhs || !rhs)
            return addOp.emitError("missing add operand"), failure();
          auto add = builder.create<arith::AddIOp>(loc, *lhs, *rhs);
          valueMap[addOp.getResult()] = add.getResult();
          continue;
        }

        if (auto subOp = llvm::dyn_cast<LLVM::SubOp>(op)) {
          auto lhs = LookupValue(valueMap, subOp.getLhs());
          auto rhs = LookupValue(valueMap, subOp.getRhs());
          if (!lhs || !rhs)
            return subOp.emitError("missing sub operand"), failure();
          auto sub = builder.create<arith::SubIOp>(loc, *lhs, *rhs);
          valueMap[subOp.getResult()] = sub.getResult();
          continue;
        }

        if (auto mulOp = llvm::dyn_cast<LLVM::MulOp>(op)) {
          auto lhs = LookupValue(valueMap, mulOp.getLhs());
          auto rhs = LookupValue(valueMap, mulOp.getRhs());
          if (!lhs || !rhs)
            return mulOp.emitError("missing mul operand"), failure();
          auto mul = builder.create<arith::MulIOp>(loc, *lhs, *rhs);
          valueMap[mulOp.getResult()] = mul.getResult();
          continue;
        }

        if (auto shlOp = llvm::dyn_cast<LLVM::ShlOp>(op)) {
          auto lhs = LookupValue(valueMap, shlOp.getLhs());
          auto rhs = LookupValue(valueMap, shlOp.getRhs());
          if (!lhs || !rhs)
            return shlOp.emitError("missing shl operand"), failure();
          auto shl = builder.create<arith::ShLIOp>(loc, *lhs, *rhs);
          valueMap[shlOp.getResult()] = shl.getResult();
          continue;
        }

        if (auto lshrOp = llvm::dyn_cast<LLVM::LShrOp>(op)) {
          auto lhs = LookupValue(valueMap, lshrOp.getLhs());
          auto rhs = LookupValue(valueMap, lshrOp.getRhs());
          if (!lhs || !rhs)
            return lshrOp.emitError("missing lshr operand"), failure();
          auto shr = builder.create<arith::ShRUIOp>(loc, *lhs, *rhs);
          valueMap[lshrOp.getResult()] = shr.getResult();
          continue;
        }

        if (auto ashrOp = llvm::dyn_cast<LLVM::AShrOp>(op)) {
          auto lhs = LookupValue(valueMap, ashrOp.getLhs());
          auto rhs = LookupValue(valueMap, ashrOp.getRhs());
          if (!lhs || !rhs)
            return ashrOp.emitError("missing ashr operand"), failure();
          auto shr = builder.create<arith::ShRSIOp>(loc, *lhs, *rhs);
          valueMap[ashrOp.getResult()] = shr.getResult();
          continue;
        }

        if (auto andOp = llvm::dyn_cast<LLVM::AndOp>(op)) {
          auto lhs = LookupValue(valueMap, andOp.getLhs());
          auto rhs = LookupValue(valueMap, andOp.getRhs());
          if (!lhs || !rhs)
            return andOp.emitError("missing and operand"), failure();
          auto andv = builder.create<arith::AndIOp>(loc, *lhs, *rhs);
          valueMap[andOp.getResult()] = andv.getResult();
          continue;
        }

        if (auto orOp = llvm::dyn_cast<LLVM::OrOp>(op)) {
          auto lhs = LookupValue(valueMap, orOp.getLhs());
          auto rhs = LookupValue(valueMap, orOp.getRhs());
          if (!lhs || !rhs)
            return orOp.emitError("missing or operand"), failure();
          auto orv = builder.create<arith::OrIOp>(loc, *lhs, *rhs);
          valueMap[orOp.getResult()] = orv.getResult();
          continue;
        }

        if (auto xorOp = llvm::dyn_cast<LLVM::XOrOp>(op)) {
          auto lhs = LookupValue(valueMap, xorOp.getLhs());
          auto rhs = LookupValue(valueMap, xorOp.getRhs());
          if (!lhs || !rhs)
            return xorOp.emitError("missing xor operand"), failure();
          auto xorv = builder.create<arith::XOrIOp>(loc, *lhs, *rhs);
          valueMap[xorOp.getResult()] = xorv.getResult();
          continue;
        }

        if (auto faddOp = llvm::dyn_cast<LLVM::FAddOp>(op)) {
          auto lhs = LookupValue(valueMap, faddOp.getLhs());
          auto rhs = LookupValue(valueMap, faddOp.getRhs());
          if (!lhs || !rhs)
            return faddOp.emitError("missing fadd operand"), failure();
          auto add = builder.create<arith::AddFOp>(loc, *lhs, *rhs);
          valueMap[faddOp.getResult()] = add.getResult();
          continue;
        }

        if (auto fnegOp = llvm::dyn_cast<LLVM::FNegOp>(op)) {
          auto operand = LookupValue(valueMap, fnegOp.getOperand());
          if (!operand)
            return fnegOp.emitError("missing fneg operand"), failure();
          auto neg = builder.create<arith::NegFOp>(loc, *operand);
          valueMap[fnegOp.getResult()] = neg.getResult();
          continue;
        }

        if (auto fsubOp = llvm::dyn_cast<LLVM::FSubOp>(op)) {
          auto lhs = LookupValue(valueMap, fsubOp.getLhs());
          auto rhs = LookupValue(valueMap, fsubOp.getRhs());
          if (!lhs || !rhs)
            return fsubOp.emitError("missing fsub operand"), failure();
          auto sub = builder.create<arith::SubFOp>(loc, *lhs, *rhs);
          valueMap[fsubOp.getResult()] = sub.getResult();
          continue;
        }

        if (auto fmulOp = llvm::dyn_cast<LLVM::FMulOp>(op)) {
          auto lhs = LookupValue(valueMap, fmulOp.getLhs());
          auto rhs = LookupValue(valueMap, fmulOp.getRhs());
          if (!lhs || !rhs)
            return fmulOp.emitError("missing fmul operand"), failure();
          if (!llvm::isa<FloatType>(lhs->getType()) ||
              !llvm::isa<FloatType>(rhs->getType()))
            return fmulOp.emitError("fmul expects float operands"), failure();
          auto mul = builder.create<arith::MulFOp>(loc, *lhs, *rhs);
          valueMap[fmulOp.getResult()] = mul.getResult();
          continue;
        }

        if (auto fdivOp = llvm::dyn_cast<LLVM::FDivOp>(op)) {
          auto lhs = LookupValue(valueMap, fdivOp.getLhs());
          auto rhs = LookupValue(valueMap, fdivOp.getRhs());
          if (!lhs || !rhs)
            return fdivOp.emitError("missing fdiv operand"), failure();
          auto div = builder.create<arith::DivFOp>(loc, *lhs, *rhs);
          valueMap[fdivOp.getResult()] = div.getResult();
          continue;
        }

        if (auto sdivOp = llvm::dyn_cast<LLVM::SDivOp>(op)) {
          auto lhs = LookupValue(valueMap, sdivOp.getLhs());
          auto rhs = LookupValue(valueMap, sdivOp.getRhs());
          if (!lhs || !rhs)
            return sdivOp.emitError("missing sdiv operand"), failure();
          auto div = builder.create<arith::DivSIOp>(loc, *lhs, *rhs);
          valueMap[sdivOp.getResult()] = div.getResult();
          continue;
        }

        if (auto udivOp = llvm::dyn_cast<LLVM::UDivOp>(op)) {
          auto lhs = LookupValue(valueMap, udivOp.getLhs());
          auto rhs = LookupValue(valueMap, udivOp.getRhs());
          if (!lhs || !rhs)
            return udivOp.emitError("missing udiv operand"), failure();
          auto div = builder.create<arith::DivUIOp>(loc, *lhs, *rhs);
          valueMap[udivOp.getResult()] = div.getResult();
          continue;
        }

        if (auto uremOp = llvm::dyn_cast<LLVM::URemOp>(op)) {
          auto lhs = LookupValue(valueMap, uremOp.getLhs());
          auto rhs = LookupValue(valueMap, uremOp.getRhs());
          if (!lhs || !rhs)
            return uremOp.emitError("missing urem operand"), failure();
          auto rem = builder.create<arith::RemUIOp>(loc, *lhs, *rhs);
          valueMap[uremOp.getResult()] = rem.getResult();
          continue;
        }

        if (auto icmpOp = llvm::dyn_cast<LLVM::ICmpOp>(op)) {
          auto lhs = LookupValue(valueMap, icmpOp.getLhs());
          auto rhs = LookupValue(valueMap, icmpOp.getRhs());
          if (!lhs || !rhs)
            return icmpOp.emitError("missing icmp operand"), failure();
          auto pred = ConvertICmpPredicate(icmpOp.getPredicate());
          auto cmp = builder.create<arith::CmpIOp>(loc, pred, *lhs, *rhs);
          valueMap[icmpOp.getResult()] = cmp.getResult();
          continue;
        }

        if (auto fcmpOp = llvm::dyn_cast<LLVM::FCmpOp>(op)) {
          auto lhs = LookupValue(valueMap, fcmpOp.getLhs());
          auto rhs = LookupValue(valueMap, fcmpOp.getRhs());
          if (!lhs || !rhs)
            return fcmpOp.emitError("missing fcmp operand"), failure();
          auto pred = ConvertFCmpPredicate(fcmpOp.getPredicate());
          auto cmp = builder.create<arith::CmpFOp>(loc, pred, *lhs, *rhs);
          valueMap[fcmpOp.getResult()] = cmp.getResult();
          continue;
        }

        if (auto selectOp = llvm::dyn_cast<LLVM::SelectOp>(op)) {
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
              lhsBase = builder.create<memref::CastOp>(loc, commonBaseType,
                                                       lhsBase);
            }
            Value rhsBase = rhsPtr->base;
            if (rhsBaseType != commonBaseType) {
              if (!memref::CastOp::areCastCompatible(
                      rhsBaseType, commonBaseType))
                return selectOp.emitError("select pointer base mismatch"),
                       failure();
              rhsBase = builder.create<memref::CastOp>(loc, commonBaseType,
                                                       rhsBase);
            }
            Value baseSel =
                builder.create<arith::SelectOp>(loc, *cond, lhsBase, rhsBase);
            Value indexSel =
                builder.create<arith::SelectOp>(loc, *cond, lhsPtr->index,
                                                rhsPtr->index);
            pointerMap[selectOp.getResult()] =
                PointerInfo{baseSel, indexSel, lhsPtr->elementType};
            continue;
          }

          auto lhs = LookupValue(valueMap, selectOp.getTrueValue());
          auto rhs = LookupValue(valueMap, selectOp.getFalseValue());
          if (!lhs || !rhs)
            return selectOp.emitError("missing select operand"), failure();
          auto sel = builder.create<arith::SelectOp>(loc, *cond, *lhs, *rhs);
          valueMap[selectOp.getResult()] = sel.getResult();
          continue;
        }

        if (auto zextOp = llvm::dyn_cast<LLVM::ZExtOp>(op)) {
          auto src = LookupValue(valueMap, zextOp.getArg());
          if (!src)
            return zextOp.emitError("missing zext operand"), failure();
          auto dstType = NormalizeScalarType(zextOp.getType(), module.getContext());
          auto ext = builder.create<arith::ExtUIOp>(loc, dstType, *src);
          valueMap[zextOp.getResult()] = ext.getResult();
          continue;
        }

        if (auto sextOp = llvm::dyn_cast<LLVM::SExtOp>(op)) {
          auto src = LookupValue(valueMap, sextOp.getArg());
          if (!src)
            return sextOp.emitError("missing sext operand"), failure();
          auto dstType = NormalizeScalarType(sextOp.getType(), module.getContext());
          auto ext = builder.create<arith::ExtSIOp>(loc, dstType, *src);
          valueMap[sextOp.getResult()] = ext.getResult();
          continue;
        }

        if (auto truncOp = llvm::dyn_cast<LLVM::TruncOp>(op)) {
          auto src = LookupValue(valueMap, truncOp.getArg());
          if (!src)
            return truncOp.emitError("missing trunc operand"), failure();
          auto dstType = NormalizeScalarType(truncOp.getType(), module.getContext());
          auto trunc = builder.create<arith::TruncIOp>(loc, dstType, *src);
          valueMap[truncOp.getResult()] = trunc.getResult();
          continue;
        }

        if (auto fpextOp = llvm::dyn_cast<LLVM::FPExtOp>(op)) {
          auto src = LookupValue(valueMap, fpextOp.getArg());
          if (!src)
            return fpextOp.emitError("missing fpext operand"), failure();
          auto dstType = NormalizeScalarType(fpextOp.getType(), module.getContext());
          auto ext = builder.create<arith::ExtFOp>(loc, dstType, *src);
          valueMap[fpextOp.getResult()] = ext.getResult();
          continue;
        }

        if (auto fptruncOp = llvm::dyn_cast<LLVM::FPTruncOp>(op)) {
          auto src = LookupValue(valueMap, fptruncOp.getArg());
          if (!src)
            return fptruncOp.emitError("missing fptrunc operand"), failure();
          auto dstType = NormalizeScalarType(fptruncOp.getType(), module.getContext());
          auto trunc = builder.create<arith::TruncFOp>(loc, dstType, *src);
          valueMap[fptruncOp.getResult()] = trunc.getResult();
          continue;
        }

        if (auto uitofpOp = llvm::dyn_cast<LLVM::UIToFPOp>(op)) {
          auto src = LookupValue(valueMap, uitofpOp.getArg());
          if (!src)
            return uitofpOp.emitError("missing uitofp operand"), failure();
          auto dstType = NormalizeScalarType(uitofpOp.getType(), module.getContext());
          auto cast = builder.create<arith::UIToFPOp>(loc, dstType, *src);
          valueMap[uitofpOp.getResult()] = cast.getResult();
          continue;
        }

        if (auto sitofpOp = llvm::dyn_cast<LLVM::SIToFPOp>(op)) {
          auto src = LookupValue(valueMap, sitofpOp.getArg());
          if (!src)
            return sitofpOp.emitError("missing sitofp operand"), failure();
          auto dstType = NormalizeScalarType(sitofpOp.getType(), module.getContext());
          auto cast = builder.create<arith::SIToFPOp>(loc, dstType, *src);
          valueMap[sitofpOp.getResult()] = cast.getResult();
          continue;
        }

        if (auto fptosiOp = llvm::dyn_cast<LLVM::FPToSIOp>(op)) {
          auto src = LookupValue(valueMap, fptosiOp.getArg());
          if (!src)
            return fptosiOp.emitError("missing fptosi operand"), failure();
          auto dstType = NormalizeScalarType(fptosiOp.getType(), module.getContext());
          auto cast = builder.create<arith::FPToSIOp>(loc, dstType, *src);
          valueMap[fptosiOp.getResult()] = cast.getResult();
          continue;
        }

        if (auto fptouiOp = llvm::dyn_cast<LLVM::FPToUIOp>(op)) {
          auto src = LookupValue(valueMap, fptouiOp.getArg());
          if (!src)
            return fptouiOp.emitError("missing fptoui operand"), failure();
          auto dstType = NormalizeScalarType(fptouiOp.getType(), module.getContext());
          auto cast = builder.create<arith::FPToUIOp>(loc, dstType, *src);
          valueMap[fptouiOp.getResult()] = cast.getResult();
          continue;
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
          builder.create<cf::BranchOp>(loc, dest, operands);
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

          builder.create<cf::CondBranchOp>(loc, *cond,
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
          builder.create<func::ReturnOp>(loc, results);
          break;
        }

        if (llvm::isa<LLVM::InlineAsmOp>(op))
          continue;
        if (llvm::isa<LLVM::DbgDeclareOp, LLVM::DbgValueOp,
                      LLVM::DbgLabelOp>(op))
          continue;

        op.emitError("unsupported LLVM operation in scf lowering: ")
            << op.getName();
        return failure();
      }
    }

    func.erase();
    return success();
  }
};

} // namespace

namespace loom {

std::unique_ptr<mlir::Pass> createLowerLLVMToSCFPass() {
  return std::make_unique<LowerLLVMToSCFPass>();
}

} // namespace loom
