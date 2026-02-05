//===-- SCFToHandshake.cpp - SCF to Handshake pass entry --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SCF-to-Handshake dataflow conversion pass. It
// identifies accelerator-marked functions, inlines callees, invokes the
// HandshakeLowering class to convert them to Handshake dialect, and rewrites
// host-side calls to use ESI channel-based accelerator instances.
//
//===----------------------------------------------------------------------===//

#include "loom/Conversion/SCFToHandshake.h"
#include "loom/Conversion/SCFToHandshakeImpl.h"
#include "loom/Dialect/Dataflow/IR/DataflowDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>

namespace {

using namespace mlir;

using loom::detail::HandshakeLowering;
using loom::detail::inlineCallsInAccel;
using loom::detail::isAccelFunc;
using loom::detail::getMemTargetHint;
using loom::detail::getStaticMemrefByteSize;
using loom::detail::MemTargetHint;

constexpr int64_t kOnChipMemThresholdBytes = 4096;

struct LiftedMemref {
  enum class Kind { Alloc, Global };
  Kind kind = Kind::Alloc;
  mlir::Operation *op = nullptr;
  mlir::MemRefType type;
  llvm::SmallVector<mlir::Value, 4> dynSizes;
  mlir::IntegerAttr alignment;
  std::string globalName;
  mlir::Value result;
};

struct DimKey {
  unsigned argIndex = 0;
  int64_t dim = 0;
  bool operator==(const DimKey &other) const {
    return argIndex == other.argIndex && dim == other.dim;
  }
};

struct DimKeyInfo {
  static DimKey getEmptyKey() { return DimKey{~0u, -1}; }
  static DimKey getTombstoneKey() { return DimKey{~0u - 1, -2}; }
  static unsigned getHashValue(const DimKey &key) {
    return llvm::hash_combine(key.argIndex, key.dim);
  }
  static bool isEqual(const DimKey &lhs, const DimKey &rhs) {
    return lhs == rhs;
  }
};

struct MergeGroup {
  llvm::SmallVector<unsigned, 4> memberIndices;
  mlir::MemRefType commonType;
  mlir::Type elemType;
  mlir::Attribute memorySpace;
};

static std::optional<int64_t> getConstantIndexValue(mlir::Value value) {
  if (!value)
    return std::nullopt;
  if (auto cst = value.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
      return intAttr.getInt();
  }
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>())
    return getConstantIndexValue(cast.getIn());
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastUIOp>())
    return getConstantIndexValue(cast.getIn());
  return std::nullopt;
}

static int64_t getElementByteSize(mlir::Type type) {
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
    return (intType.getWidth() + 7) / 8;
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type))
    return (floatType.getWidth() + 7) / 8;
  return 0;
}

static mlir::Value buildIndexConstant(mlir::OpBuilder &builder,
                                      mlir::Location loc, int64_t value) {
  return builder.create<mlir::arith::ConstantOp>(
      loc, builder.getIndexType(), builder.getIndexAttr(value));
}

static MemTargetHint getMemTargetHintForOp(mlir::Operation *op,
                                           mlir::ModuleOp module) {
  if (!op)
    return MemTargetHint::None;
  MemTargetHint hint = getMemTargetHint(op);
  if (hint != MemTargetHint::None)
    return hint;
  if (auto getGlobal = mlir::dyn_cast<mlir::memref::GetGlobalOp>(op)) {
    auto global = module.lookupSymbol<mlir::memref::GlobalOp>(
        getGlobal.getName());
    if (global)
      return getMemTargetHint(global);
  }
  return MemTargetHint::None;
}

static bool isCloneableOp(mlir::Operation *op) {
  if (!op)
    return false;
  if (op->getNumRegions() != 0)
    return false;
  if (mlir::isa<mlir::func::CallOp>(op))
    return false;
  if (auto iface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op))
    return iface.hasNoEffect();
  return mlir::isPure(op);
}

static mlir::Value cloneValueForCall(mlir::Value value, mlir::func::FuncOp func,
                                     mlir::func::CallOp call,
                                     llvm::DenseMap<mlir::Value, mlir::Value>
                                         &cloned,
                                     mlir::OpBuilder &builder) {
  if (!value)
    return nullptr;
  auto it = cloned.find(value);
  if (it != cloned.end())
    return it->second;
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    if (arg.getOwner() == &func.getBody().front()) {
      mlir::Value mapped = call.getOperand(arg.getArgNumber());
      cloned[value] = mapped;
      return mapped;
    }
  }
  if (auto cst = value.getDefiningOp<mlir::arith::ConstantOp>()) {
    mlir::Value created = builder.create<mlir::arith::ConstantOp>(
        cst.getLoc(), cst.getValue());
    cloned[value] = created;
    return created;
  }
  mlir::Operation *def = value.getDefiningOp();
  if (!def || !isCloneableOp(def))
    return nullptr;
  mlir::IRMapping mapping;
  for (mlir::Value operand : def->getOperands()) {
    mlir::Value mapped =
        cloneValueForCall(operand, func, call, cloned, builder);
    if (!mapped)
      return nullptr;
    mapping.map(operand, mapped);
  }
  mlir::Operation *clone = builder.clone(*def, mapping);
  for (auto [orig, mapped] :
       llvm::zip(def->getResults(), clone->getResults())) {
    cloned[orig] = mapped;
  }
  auto resultIt = cloned.find(value);
  return resultIt != cloned.end() ? resultIt->second : nullptr;
}

static void rewriteHostCalls(ModuleOp module,
                             llvm::DenseSet<StringRef> accelNames) {
  llvm::DenseMap<func::FuncOp, Value> clkCache;
  llvm::DenseMap<func::FuncOp, Value> rstCache;
  llvm::DenseMap<func::FuncOp, Value> trueCache;
  llvm::DenseMap<func::FuncOp, Value> falseCache;
  llvm::DenseMap<func::FuncOp, unsigned> instCount;

  auto getOrCreateEsiWrapper =
      [&](circt::handshake::FuncOp target) -> circt::handshake::FuncOp {
    if (!target)
      return target;
    auto targetType = target.getFunctionType();
    if (targetType.getNumInputs() == 0 || targetType.getNumResults() == 0)
      return target;
    auto lastInput = targetType.getInput(targetType.getNumInputs() - 1);
    auto lastResult = targetType.getResult(targetType.getNumResults() - 1);
    if (!mlir::isa<mlir::NoneType>(lastInput) ||
        !mlir::isa<mlir::NoneType>(lastResult))
      return target;

    std::string wrapperName = target.getName().str() + "_esi";
    if (auto existing =
            module.lookupSymbol<circt::handshake::FuncOp>(wrapperName))
      return existing;

    OpBuilder builder(module.getContext());
    builder.setInsertionPoint(target);

    llvm::SmallVector<Type, 8> inputs(targetType.getInputs().begin(),
                                      targetType.getInputs().end());
    llvm::SmallVector<Type, 4> results(targetType.getResults().begin(),
                                       targetType.getResults().end());
    inputs.back() = builder.getI1Type();
    results.back() = builder.getI1Type();
    auto wrapperType = builder.getFunctionType(inputs, results);

    auto wrapper =
        builder.create<circt::handshake::FuncOp>(target.getLoc(), wrapperName,
                                                 wrapperType);
    wrapper.resolveArgAndResNames();
    if (auto visibility =
            target->getAttrOfType<mlir::StringAttr>(
                mlir::SymbolTable::getVisibilityAttrName()))
      wrapper->setAttr(mlir::SymbolTable::getVisibilityAttrName(), visibility);
    if (auto argNames = target->getAttrOfType<ArrayAttr>("argNames"))
      wrapper->setAttr("argNames", argNames);
    if (auto resNames = target->getAttrOfType<ArrayAttr>("resNames"))
      wrapper->setAttr("resNames", resNames);
    if (auto annotations = target->getAttrOfType<ArrayAttr>("loom.annotations"))
      wrapper->setAttr("loom.annotations", annotations);

    Block *entry = new Block();
    wrapper.getBody().push_back(entry);
    for (Type inputType : wrapperType.getInputs())
      entry->addArgument(inputType, target.getLoc());
    builder.setInsertionPointToStart(entry);

    auto args = entry->getArguments();
    Value startI1 = args.back();
    auto ctrlNone =
        builder.create<circt::handshake::JoinOp>(target.getLoc(),
                                                 mlir::ValueRange{startI1})
            .getResult();

    mlir::IRMapping mapping;
    Block &origBlock = target.getBody().front();
    for (unsigned i = 0, e = origBlock.getNumArguments(); i < e; ++i) {
      if (i + 1 == e) {
        mapping.map(origBlock.getArgument(i), ctrlNone);
      } else {
        mapping.map(origBlock.getArgument(i), args[i]);
      }
    }

    llvm::SmallVector<mlir::Operation *, 64> clonedOps;
    for (mlir::Operation &op : origBlock) {
      if (auto ret = mlir::dyn_cast<circt::handshake::ReturnOp>(op)) {
        llvm::SmallVector<Value, 8> retOperands;
        for (Value operand : ret.getOperands().drop_back(1))
          retOperands.push_back(mapping.lookup(operand));
        Value doneNone = mapping.lookup(ret.getOperands().back());
        mlir::OperationState doneState(
            target.getLoc(), circt::handshake::ConstantOp::getOperationName());
        doneState.addOperands(doneNone);
        doneState.addTypes(builder.getI1Type());
        doneState.addAttribute("value", builder.getBoolAttr(true));
        auto doneOp = builder.create(doneState);
        retOperands.push_back(doneOp->getResult(0));
        builder.create<circt::handshake::ReturnOp>(target.getLoc(),
                                                   retOperands);
        continue;
      }
      clonedOps.push_back(builder.clone(op, mapping));
    }

    for (mlir::Operation *cloned : clonedOps) {
      for (mlir::OpOperand &operand : cloned->getOpOperands()) {
        if (mlir::Value mapped = mapping.lookupOrNull(operand.get()))
          operand.set(mapped);
      }
    }
    return wrapper;
  };

  auto getBoolConst = [&](func::FuncOp func, bool value) -> Value {
    auto &cache = value ? trueCache : falseCache;
    auto it = cache.find(func);
    if (it != cache.end())
      return it->second;
    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(&func.front());
    Value cst = builder.create<arith::ConstantOp>(
        func.getLoc(), builder.getI1Type(), builder.getBoolAttr(value));
    cache[func] = cst;
    return cst;
  };

  auto getClockConst = [&](func::FuncOp func) -> Value {
    auto it = clkCache.find(func);
    if (it != clkCache.end())
      return it->second;
    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(&func.front());
    auto attr = circt::seq::ClockConstAttr::get(
        func.getContext(), circt::seq::ClockConst::Low);
    Value clk = builder.create<circt::seq::ConstClockOp>(func.getLoc(), attr)
                    .getResult();
    clkCache[func] = clk;
    return clk;
  };

  auto getResetConst = [&](func::FuncOp func) -> Value {
    auto it = rstCache.find(func);
    if (it != rstCache.end())
      return it->second;
    Value rst = getBoolConst(func, false);
    rstCache[func] = rst;
    return rst;
  };

  module.walk([&](func::FuncOp func) {
    llvm::SmallVector<func::CallOp, 8> calls;
    func.walk([&](func::CallOp call) {
      if (!accelNames.contains(call.getCallee()))
        return;
      calls.push_back(call);
    });

    for (func::CallOp call : calls) {
      OpBuilder builder(call);
      Location loc = call.getLoc();
      Value clk = getClockConst(func);
      Value rst = getResetConst(func);
      Value valid = getBoolConst(func, true);
      Value ready = getBoolConst(func, true);

      auto target = module.lookupSymbol<circt::handshake::FuncOp>(
          call.getCallee());
      auto esiTarget = getOrCreateEsiWrapper(target);

      llvm::SmallVector<Value, 8> chanOperands;
      chanOperands.reserve(call.getNumOperands() + 1);
      for (Value operand : call.getOperands()) {
        auto wrap = builder.create<circt::esi::WrapValidReadyOp>(
            loc, operand, valid);
        chanOperands.push_back(wrap.getResult(0));
      }
      auto entryWrap =
          builder.create<circt::esi::WrapValidReadyOp>(loc, valid, valid);
      chanOperands.push_back(entryWrap.getResult(0));

      llvm::SmallVector<Type, 4> resultTypes;
      resultTypes.reserve(call.getNumResults() + 1);
      for (Type type : call.getResultTypes())
        resultTypes.push_back(
            circt::esi::ChannelType::get(call.getContext(), type));
      resultTypes.push_back(circt::esi::ChannelType::get(
          call.getContext(), builder.getI1Type()));

      unsigned count = instCount[func]++;
      std::string instName = call.getCallee().str() + "_inst" +
                             std::to_string(count);

      auto esiInst = builder.create<circt::handshake::ESIInstanceOp>(
          loc, resultTypes,
          esiTarget ? esiTarget.getName() : call.getCallee(), instName, clk, rst,
          chanOperands);

      for (unsigned i = 0, e = call.getNumResults(); i < e; ++i) {
        auto unwrap = builder.create<circt::esi::UnwrapValidReadyOp>(
            loc, esiInst.getResult(i), ready);
        call.getResult(i).replaceAllUsesWith(unwrap.getResult(0));
      }
      if (call.getNumResults() < esiInst.getNumResults()) {
        auto doneUnwrap = builder.create<circt::esi::UnwrapValidReadyOp>(
            loc, esiInst.getResult(call.getNumResults()), ready);
        (void)doneUnwrap;
      }

      call.erase();
    }
  });
}

static void collectCallSites(ModuleOp module, StringRef callee,
                             llvm::SmallVector<func::CallOp, 8> &calls) {
  calls.clear();
  module.walk([&](func::CallOp call) {
    if (call.getCallee() == callee)
      calls.push_back(call);
  });
}

static bool shouldExternalizeAlloc(mlir::MemRefType type,
                                   MemTargetHint hint) {
  if (hint == MemTargetHint::Extmemory)
    return true;
  if (!type || !type.hasStaticShape())
    return true;
  auto sizeBytes = getStaticMemrefByteSize(type);
  if (!sizeBytes)
    return true;
  return *sizeBytes > kOnChipMemThresholdBytes;
}

static bool shouldExternalizeGlobal(mlir::memref::GlobalOp global,
                                    MemTargetHint hint) {
  if (!global)
    return true;
  if (hint == MemTargetHint::Extmemory)
    return true;
  bool isConst = global.getConstant();
  if (!isConst)
    return true;
  if (hint == MemTargetHint::Rom)
    return false;
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(global.getType());
  auto sizeBytes = memrefType ? getStaticMemrefByteSize(memrefType)
                              : std::nullopt;
  if (!sizeBytes)
    return true;
  return *sizeBytes > kOnChipMemThresholdBytes;
}

static mlir::LogicalResult liftExternalMemrefs(
    ModuleOp module, func::FuncOp func,
    llvm::SmallVector<func::CallOp, 8> &calls) {
  llvm::SmallVector<LiftedMemref, 8> lifted;
  bool hadError = false;

  func.walk([&](mlir::Operation *op) {
    if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
      MemTargetHint hint = getMemTargetHintForOp(op, module);
      if (hint == MemTargetHint::Rom) {
        allocOp.emitError("loom.target=rom is not valid for memref.alloc");
        hadError = true;
        return;
      }
      if (!shouldExternalizeAlloc(allocOp.getType(), hint))
        return;
      if (allocOp->getBlock() != &func.getBody().front()) {
        allocOp.emitError("externalized alloc must be in entry block");
        hadError = true;
        return;
      }
      LiftedMemref item;
      item.kind = LiftedMemref::Kind::Alloc;
      item.op = op;
      item.type = allocOp.getType();
      item.result = allocOp.getResult();
      for (mlir::Value size : allocOp.getDynamicSizes())
        item.dynSizes.push_back(size);
      item.alignment = allocOp.getAlignmentAttr();
      lifted.push_back(item);
      return;
    }
    if (auto allocaOp = mlir::dyn_cast<mlir::memref::AllocaOp>(op)) {
      MemTargetHint hint = getMemTargetHintForOp(op, module);
      if (hint == MemTargetHint::Rom) {
        allocaOp.emitError("loom.target=rom is not valid for memref.alloca");
        hadError = true;
        return;
      }
      if (!shouldExternalizeAlloc(allocaOp.getType(), hint))
        return;
      if (allocaOp->getBlock() != &func.getBody().front()) {
        allocaOp.emitError("externalized alloca must be in entry block");
        hadError = true;
        return;
      }
      LiftedMemref item;
      item.kind = LiftedMemref::Kind::Alloc;
      item.op = op;
      item.type = allocaOp.getType();
      item.result = allocaOp.getResult();
      for (mlir::Value size : allocaOp.getDynamicSizes())
        item.dynSizes.push_back(size);
      item.alignment = allocaOp.getAlignmentAttr();
      lifted.push_back(item);
      return;
    }
    if (auto getGlobal = mlir::dyn_cast<mlir::memref::GetGlobalOp>(op)) {
      auto global =
          module.lookupSymbol<mlir::memref::GlobalOp>(getGlobal.getName());
      if (!global) {
        getGlobal.emitError("missing memref.global for get_global");
        hadError = true;
        return;
      }
      auto memrefType = mlir::dyn_cast<mlir::MemRefType>(global.getType());
      if (!memrefType || !memrefType.hasStaticShape()) {
        getGlobal.emitError("memref.global must have static shape");
        hadError = true;
        return;
      }
      MemTargetHint hint = getMemTargetHintForOp(op, module);
      if (hint == MemTargetHint::Rom && !global.getConstant()) {
        getGlobal.emitError("loom.target=rom requires constant global");
        hadError = true;
        return;
      }
      if (!shouldExternalizeGlobal(global, hint))
        return;
      LiftedMemref item;
      item.kind = LiftedMemref::Kind::Global;
      item.op = op;
      item.type = memrefType;
      item.result = getGlobal.getResult();
      item.globalName = getGlobal.getName().str();
      lifted.push_back(item);
      return;
    }
  });

  if (hadError)
    return mlir::failure();

  if (lifted.empty())
    return mlir::success();

  mlir::OpBuilder builder(func.getContext());
  auto funcType = func.getFunctionType();
  unsigned oldArgCount = funcType.getNumInputs();
  llvm::SmallVector<mlir::Type, 8> inputTypes(funcType.getInputs().begin(),
                                              funcType.getInputs().end());
  for (const LiftedMemref &item : lifted)
    inputTypes.push_back(item.type);
  func.setFunctionType(
      builder.getFunctionType(inputTypes, funcType.getResults()));

  llvm::SmallVector<mlir::DictionaryAttr, 8> argAttrs;
  argAttrs.reserve(oldArgCount + lifted.size());
  for (unsigned i = 0; i < oldArgCount; ++i)
    argAttrs.push_back(func.getArgAttrDict(i));
  for (size_t i = 0; i < lifted.size(); ++i)
    argAttrs.push_back(builder.getDictionaryAttr({}));
  func.setAllArgAttrs(argAttrs);

  mlir::Block &entry = func.getBody().front();
  llvm::SmallVector<mlir::Value, 8> newArgs;
  newArgs.reserve(lifted.size());
  for (const LiftedMemref &item : lifted) {
    mlir::Location loc = item.op ? item.op->getLoc() : func.getLoc();
    mlir::Value arg = entry.addArgument(item.type, loc);
    newArgs.push_back(arg);
  }

  for (auto [item, arg] : llvm::zip(lifted, newArgs)) {
    if (item.result)
      item.result.replaceAllUsesWith(arg);
    if (item.op)
      item.op->erase();
  }

  llvm::SmallVector<mlir::Operation *, 8> deallocs;
  func.walk([&](mlir::memref::DeallocOp dealloc) {
    deallocs.push_back(dealloc.getOperation());
  });
  for (mlir::Operation *op : deallocs)
    op->erase();

  for (func::CallOp call : calls) {
    mlir::OpBuilder callBuilder(call);
    llvm::SmallVector<mlir::Value, 8> newOperands;
    llvm::SmallVector<mlir::Value, 4> toDealloc;
    for (const LiftedMemref &item : lifted) {
      if (item.kind == LiftedMemref::Kind::Global) {
        auto getGlobal = callBuilder.create<mlir::memref::GetGlobalOp>(
            call.getLoc(), item.type, item.globalName);
        newOperands.push_back(getGlobal.getResult());
        continue;
      }
      llvm::DenseMap<mlir::Value, mlir::Value> cloned;
      llvm::SmallVector<mlir::Value, 4> dynSizes;
      dynSizes.reserve(item.dynSizes.size());
      for (mlir::Value size : item.dynSizes) {
        mlir::Value mapped =
            cloneValueForCall(size, func, call, cloned, callBuilder);
        if (!mapped)
          return call.emitError(
              "dynamic alloc size must derive from func args");
        dynSizes.push_back(mapped);
      }
      auto alloc = callBuilder.create<mlir::memref::AllocOp>(
          call.getLoc(), item.type, dynSizes, item.alignment);
      newOperands.push_back(alloc.getResult());
      toDealloc.push_back(alloc.getResult());
    }

    if (!newOperands.empty())
      call->insertOperands(call.getNumOperands(), newOperands);

    if (!toDealloc.empty()) {
      mlir::OpBuilder afterBuilder(call->getBlock(),
                                   std::next(call->getIterator()));
      for (mlir::Value memref : toDealloc)
        afterBuilder.create<mlir::memref::DeallocOp>(call.getLoc(), memref);
    }
  }

  return mlir::success();
}

struct DimReplacement {
  enum class Kind { Const, Arg };
  Kind kind = Kind::Const;
  int64_t constValue = 0;
  DimKey key;
};

static mlir::LogicalResult lowerMemrefDims(
    func::FuncOp func, llvm::SmallVector<func::CallOp, 8> &calls) {
  llvm::SmallVector<mlir::memref::DimOp, 8> dimOps;
  llvm::DenseMap<mlir::Operation *, DimReplacement> replacements;
  llvm::DenseSet<DimKey, DimKeyInfo> needed;

  func.walk([&](mlir::memref::DimOp dimOp) {
    dimOps.push_back(dimOp);
  });

  if (dimOps.empty())
    return mlir::success();

  for (mlir::memref::DimOp dimOp : dimOps) {
    mlir::Value root = loom::detail::getMemrefRoot(dimOp.getSource());
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(root.getType());
    if (!memrefType) {
      return dimOp.emitError("memref.dim requires memref operand");
    }
    auto dimIndex = getConstantIndexValue(dimOp.getIndex());
    if (!dimIndex) {
      return dimOp.emitError("memref.dim requires constant dimension index");
    }
    int64_t dim = *dimIndex;
    if (dim < 0 || dim >= memrefType.getRank()) {
      return dimOp.emitError("memref.dim index out of range");
    }
    int64_t shapeValue = memrefType.getShape()[dim];
    if (shapeValue != mlir::ShapedType::kDynamic) {
      DimReplacement rep;
      rep.kind = DimReplacement::Kind::Const;
      rep.constValue = shapeValue;
      replacements[dimOp.getOperation()] = rep;
      continue;
    }
    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(root)) {
      DimKey key{arg.getArgNumber(), dim};
      needed.insert(key);
      DimReplacement rep;
      rep.kind = DimReplacement::Kind::Arg;
      rep.key = key;
      replacements[dimOp.getOperation()] = rep;
      continue;
    }
    return dimOp.emitError(
        "dynamic memref.dim requires memref argument");
  }

  llvm::SmallVector<DimKey, 8> keys;
  keys.reserve(needed.size());
  for (const DimKey &key : needed)
    keys.push_back(key);
  llvm::sort(keys, [](const DimKey &lhs, const DimKey &rhs) {
    if (lhs.argIndex != rhs.argIndex)
      return lhs.argIndex < rhs.argIndex;
    return lhs.dim < rhs.dim;
  });

  llvm::DenseMap<DimKey, mlir::Value, DimKeyInfo> keyToArg;
  if (!keys.empty()) {
    mlir::OpBuilder builder(func.getContext());
    auto funcType = func.getFunctionType();
    unsigned oldArgCount = funcType.getNumInputs();
    llvm::SmallVector<mlir::Type, 8> inputTypes(funcType.getInputs().begin(),
                                                funcType.getInputs().end());
    for (size_t i = 0; i < keys.size(); ++i)
      inputTypes.push_back(builder.getIndexType());
    func.setFunctionType(
        builder.getFunctionType(inputTypes, funcType.getResults()));
    llvm::SmallVector<mlir::DictionaryAttr, 8> argAttrs;
    argAttrs.reserve(oldArgCount + keys.size());
    for (unsigned i = 0; i < oldArgCount; ++i)
      argAttrs.push_back(func.getArgAttrDict(i));
    for (size_t i = 0; i < keys.size(); ++i)
      argAttrs.push_back(builder.getDictionaryAttr({}));
    func.setAllArgAttrs(argAttrs);
    mlir::Block &entry = func.getBody().front();
    for (const DimKey &key : keys) {
      mlir::Value arg = entry.addArgument(builder.getIndexType(), func.getLoc());
      keyToArg[key] = arg;
    }

    for (func::CallOp call : calls) {
      mlir::OpBuilder callBuilder(call);
      llvm::SmallVector<mlir::Value, 8> newOperands;
      newOperands.reserve(keys.size());
      for (const DimKey &key : keys) {
        mlir::Value mem = call.getOperand(key.argIndex);
        mlir::Value dim =
            buildIndexConstant(callBuilder, call.getLoc(), key.dim);
        auto dimOp =
            callBuilder.create<mlir::memref::DimOp>(call.getLoc(), mem, dim);
        newOperands.push_back(dimOp.getResult());
      }
      if (!newOperands.empty())
        call->insertOperands(call.getNumOperands(), newOperands);
    }
  }

  for (mlir::memref::DimOp dimOp : dimOps) {
    auto it = replacements.find(dimOp.getOperation());
    if (it == replacements.end())
      continue;
    DimReplacement rep = it->second;
    mlir::OpBuilder builder(dimOp);
    mlir::Value replacement;
    if (rep.kind == DimReplacement::Kind::Const) {
      replacement = buildIndexConstant(builder, dimOp.getLoc(), rep.constValue);
    } else {
      auto argIt = keyToArg.find(rep.key);
      if (argIt == keyToArg.end())
        return dimOp.emitError("missing memref.dim size argument");
      replacement = argIt->second;
    }
    dimOp.getResult().replaceAllUsesWith(replacement);
    dimOp.erase();
  }

  return mlir::success();
}

static bool hasNoAliasAttr(func::FuncOp func, unsigned argIndex) {
  return func.getArgAttr(argIndex, "loom.noalias") != nullptr;
}

static mlir::LogicalResult rewriteLoadsForGroup(
    mlir::Block &block,
    const llvm::DenseMap<mlir::Value, std::pair<mlir::Value, mlir::Value>>
        &mergedArgs) {
  for (mlir::Operation &op : block) {
    if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
      mlir::Value root = loom::detail::getMemrefRoot(load.getMemref());
      auto it = mergedArgs.find(root);
      if (it == mergedArgs.end())
        continue;
      if (load.getIndices().size() != 1) {
        return load.emitError(
            "alias-merged memrefs require rank-1 indices");
      }
      mlir::OpBuilder builder(load);
      mlir::Value index = load.getIndices().front();
      mlir::Value offset = it->second.second;
      mlir::Value newIndex =
          builder.create<mlir::arith::AddIOp>(load.getLoc(), index, offset);
      load->setOperand(0, it->second.first);
      load->setOperand(1, newIndex);
      continue;
    }
    if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
      mlir::Value root = loom::detail::getMemrefRoot(store.getMemref());
      auto it = mergedArgs.find(root);
      if (it == mergedArgs.end())
        continue;
      if (store.getIndices().size() != 1) {
        return store.emitError(
            "alias-merged memrefs require rank-1 indices");
      }
      mlir::OpBuilder builder(store);
      mlir::Value index = store.getIndices().front();
      mlir::Value offset = it->second.second;
      mlir::Value newIndex =
          builder.create<mlir::arith::AddIOp>(store.getLoc(), index, offset);
      store->setOperand(1, it->second.first);
      store->setOperand(2, newIndex);
      continue;
    }
  }
  return mlir::success();
}

struct PackedCallValues {
  mlir::Value packedMemref;
  llvm::SmallVector<mlir::Value, 4> offsets;
};

static PackedCallValues buildPackedCallValues(
    func::CallOp call, const MergeGroup &group,
    llvm::ArrayRef<mlir::Value> oldOperands) {
  mlir::OpBuilder builder(call);
  mlir::Location loc = call.getLoc();
  int64_t elemBytes = getElementByteSize(group.elemType);
  mlir::Value elemSize = buildIndexConstant(builder, loc, elemBytes);

  llvm::SmallVector<mlir::Value, 4> addrElems;
  llvm::SmallVector<mlir::Value, 4> endElems;
  llvm::SmallVector<mlir::Value, 4> castedMemrefs;
  addrElems.reserve(group.memberIndices.size());
  endElems.reserve(group.memberIndices.size());
  castedMemrefs.reserve(group.memberIndices.size());

  for (unsigned index : group.memberIndices) {
    mlir::Value memref = oldOperands[index];
    auto cast =
        builder.create<mlir::memref::CastOp>(loc, group.commonType, memref);
    castedMemrefs.push_back(cast.getResult());

    auto basePtr =
        builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
            loc, memref);
    mlir::Value baseElem = basePtr.getResult();
    if (elemBytes != 1) {
      baseElem =
          builder.create<mlir::arith::DivUIOp>(loc, baseElem, elemSize);
    }
    addrElems.push_back(baseElem);

    mlir::Value dimIndex = buildIndexConstant(builder, loc, 0);
    auto dimOp =
        builder.create<mlir::memref::DimOp>(loc, memref, dimIndex);
    auto end =
        builder.create<mlir::arith::AddIOp>(loc, baseElem, dimOp.getResult());
    endElems.push_back(end);
  }

  mlir::Value baseAddr = addrElems.front();
  mlir::Value baseMem = castedMemrefs.front();
  for (size_t i = 1; i < addrElems.size(); ++i) {
    auto cmp = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ult, addrElems[i], baseAddr);
    baseAddr = builder.create<mlir::arith::SelectOp>(loc, cmp, addrElems[i],
                                                     baseAddr);
    baseMem = builder.create<mlir::arith::SelectOp>(loc, cmp, castedMemrefs[i],
                                                    baseMem);
  }

  mlir::Value maxEnd = endElems.front();
  for (size_t i = 1; i < endElems.size(); ++i) {
    auto cmp = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ugt, endElems[i], maxEnd);
    maxEnd =
        builder.create<mlir::arith::SelectOp>(loc, cmp, endElems[i], maxEnd);
  }

  mlir::Value span =
      builder.create<mlir::arith::SubIOp>(loc, maxEnd, baseAddr);

  mlir::Value zero = buildIndexConstant(builder, loc, 0);
  mlir::Value one = buildIndexConstant(builder, loc, 1);
  auto packed = builder.create<mlir::memref::ReinterpretCastOp>(
      loc, group.commonType, baseMem, zero, mlir::ValueRange{span},
      mlir::ValueRange{one});

  PackedCallValues values;
  values.packedMemref = packed.getResult();
  values.offsets.reserve(addrElems.size());
  for (mlir::Value addr : addrElems) {
    mlir::Value offset =
        builder.create<mlir::arith::SubIOp>(loc, addr, baseAddr);
    values.offsets.push_back(offset);
  }
  return values;
}

static mlir::LogicalResult mergeAliasGroups(
    func::FuncOp func, llvm::SmallVector<func::CallOp, 8> &calls) {
  llvm::SmallVector<unsigned, 8> memrefArgs;
  unsigned oldArgCount = func.getNumArguments();
  for (unsigned i = 0, e = oldArgCount; i < e; ++i) {
    if (mlir::isa<mlir::MemRefType>(func.getArgument(i).getType()))
      memrefArgs.push_back(i);
  }

  if (memrefArgs.size() < 2)
    return mlir::success();

  struct DSU {
    llvm::SmallVector<int, 8> parent;
    explicit DSU(size_t n) : parent(n) {
      for (size_t i = 0; i < n; ++i)
        parent[i] = static_cast<int>(i);
    }
    int find(int x) {
      if (parent[x] == x)
        return x;
      parent[x] = find(parent[x]);
      return parent[x];
    }
    void unite(int a, int b) {
      a = find(a);
      b = find(b);
      if (a != b)
        parent[b] = a;
    }
  };

  DSU dsu(memrefArgs.size());
  mlir::AliasAnalysis aliasAnalysis(func);

  for (size_t i = 0; i < memrefArgs.size(); ++i) {
    unsigned argI = memrefArgs[i];
    if (hasNoAliasAttr(func, argI))
      continue;
    for (size_t j = i + 1; j < memrefArgs.size(); ++j) {
      unsigned argJ = memrefArgs[j];
      if (hasNoAliasAttr(func, argJ))
        continue;
      auto alias = aliasAnalysis.alias(func.getArgument(argI),
                                       func.getArgument(argJ));
      if (alias == mlir::AliasResult::NoAlias)
        continue;
      dsu.unite(static_cast<int>(i), static_cast<int>(j));
    }
  }

  llvm::DenseMap<int, MergeGroup> groups;
  for (size_t i = 0; i < memrefArgs.size(); ++i) {
    int root = dsu.find(static_cast<int>(i));
    groups[root].memberIndices.push_back(memrefArgs[i]);
  }

  llvm::SmallVector<MergeGroup, 4> mergeGroups;
  for (auto &entry : groups) {
    MergeGroup group = entry.second;
    if (group.memberIndices.size() <= 1)
      continue;
    llvm::sort(group.memberIndices);
    mlir::MemRefType firstType =
        mlir::dyn_cast<mlir::MemRefType>(
            func.getArgument(group.memberIndices.front()).getType());
    if (!firstType || firstType.getRank() != 1)
      return func.emitError(
          "alias merge requires rank-1 memref arguments");
    group.elemType = firstType.getElementType();
    if (getElementByteSize(group.elemType) == 0)
      return func.emitError(
          "alias merge requires integer or float element types");
    group.memorySpace = firstType.getMemorySpace();
    for (unsigned idx : group.memberIndices) {
      auto type =
          mlir::dyn_cast<mlir::MemRefType>(func.getArgument(idx).getType());
      if (!type || type.getRank() != 1)
        return func.emitError(
            "alias merge requires rank-1 memref arguments");
      if (type.getElementType() != group.elemType)
        return func.emitError(
            "alias merge requires identical element types");
      if (type.getMemorySpace() != group.memorySpace)
        return func.emitError(
            "alias merge requires identical memory spaces");
    }
    group.commonType = mlir::MemRefType::get(
        {mlir::ShapedType::kDynamic}, group.elemType,
        mlir::MemRefLayoutAttrInterface(), group.memorySpace);
    mergeGroups.push_back(group);
  }

  if (mergeGroups.empty())
    return mlir::success();

  llvm::sort(mergeGroups, [](const MergeGroup &lhs, const MergeGroup &rhs) {
    return lhs.memberIndices.front() < rhs.memberIndices.front();
  });

  llvm::DenseSet<unsigned> removedArgs;
  for (const MergeGroup &group : mergeGroups) {
    for (unsigned idx : group.memberIndices)
      removedArgs.insert(idx);
  }

  llvm::SmallVector<mlir::Type, 16> newArgTypes;
  llvm::SmallVector<mlir::DictionaryAttr, 16> newArgAttrs;
  llvm::DenseMap<unsigned, unsigned> oldToNew;
  for (unsigned i = 0, e = oldArgCount; i < e; ++i) {
    if (removedArgs.contains(i))
      continue;
    oldToNew[i] = newArgTypes.size();
    newArgTypes.push_back(func.getArgument(i).getType());
    newArgAttrs.push_back(func.getArgAttrDict(i));
  }

  struct GroupArgLayout {
    MergeGroup group;
    unsigned packedIndex = 0;
    llvm::SmallVector<unsigned, 4> offsetIndices;
  };
  llvm::SmallVector<GroupArgLayout, 4> groupLayouts;

  mlir::Builder builder(func.getContext());
  for (const MergeGroup &group : mergeGroups) {
    GroupArgLayout layout;
    layout.group = group;
    layout.packedIndex = newArgTypes.size();
    newArgTypes.push_back(group.commonType);
    newArgAttrs.push_back(builder.getDictionaryAttr({}));
    for (size_t i = 0; i < group.memberIndices.size(); ++i) {
      layout.offsetIndices.push_back(newArgTypes.size());
      newArgTypes.push_back(builder.getIndexType());
      newArgAttrs.push_back(builder.getDictionaryAttr({}));
    }
    groupLayouts.push_back(std::move(layout));
  }

  auto funcType = builder.getFunctionType(newArgTypes, func.getResultTypes());
  func.setFunctionType(funcType);
  func.setAllArgAttrs(newArgAttrs);

  mlir::Block &oldBlock = func.getBody().front();
  mlir::Block *newBlock = new mlir::Block();
  func.getBody().push_back(newBlock);
  for (mlir::Type type : newArgTypes)
    newBlock->addArgument(type, func.getLoc());

  newBlock->getOperations().splice(newBlock->end(), oldBlock.getOperations());

  for (unsigned i = 0, e = oldArgCount; i < e; ++i) {
    if (removedArgs.contains(i))
      continue;
    auto it = oldToNew.find(i);
    if (it == oldToNew.end())
      continue;
    oldBlock.getArgument(i).replaceAllUsesWith(
        newBlock->getArgument(it->second));
  }

  llvm::DenseMap<mlir::Value, std::pair<mlir::Value, mlir::Value>> mergedArgs;
  for (const GroupArgLayout &layout : groupLayouts) {
    mlir::Value packed = newBlock->getArgument(layout.packedIndex);
    for (size_t i = 0; i < layout.group.memberIndices.size(); ++i) {
      unsigned oldIndex = layout.group.memberIndices[i];
      mlir::Value offset = newBlock->getArgument(layout.offsetIndices[i]);
      mergedArgs[oldBlock.getArgument(oldIndex)] = {packed, offset};
    }
  }

  if (mlir::failed(rewriteLoadsForGroup(*newBlock, mergedArgs)))
    return mlir::failure();

  for (unsigned idx : removedArgs) {
    if (!oldBlock.getArgument(idx).use_empty()) {
      return func.emitError(
          "memref argument still used after alias merge");
    }
  }

  oldBlock.erase();

  for (func::CallOp call : calls) {
    llvm::SmallVector<mlir::Value, 16> oldOperands(call.getOperands().begin(),
                                                   call.getOperands().end());
    llvm::SmallVector<mlir::Value, 16> newOperands;
    for (unsigned i = 0, e = oldOperands.size(); i < e; ++i) {
      if (removedArgs.contains(i))
        continue;
      newOperands.push_back(oldOperands[i]);
    }
    for (const GroupArgLayout &layout : groupLayouts) {
      PackedCallValues values =
          buildPackedCallValues(call, layout.group, oldOperands);
      newOperands.push_back(values.packedMemref);
      for (mlir::Value offset : values.offsets)
        newOperands.push_back(offset);
    }
    call->setOperands(newOperands);
  }

  return mlir::success();
}

static mlir::LogicalResult prepareAccelMemrefs(ModuleOp module,
                                               func::FuncOp func) {
  llvm::SmallVector<func::CallOp, 8> calls;
  collectCallSites(module, func.getName(), calls);

  if (mlir::failed(liftExternalMemrefs(module, func, calls)))
    return mlir::failure();
  if (mlir::failed(lowerMemrefDims(func, calls)))
    return mlir::failure();
  if (mlir::failed(mergeAliasGroups(func, calls)))
    return mlir::failure();
  return mlir::success();
}

struct SCFToHandshakeDataflowPass
    : public PassWrapper<SCFToHandshakeDataflowPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<loom::dataflow::DataflowDialect,
                    circt::handshake::HandshakeDialect, circt::esi::ESIDialect,
                    circt::seq::SeqDialect, func::FuncDialect,
                    scf::SCFDialect, memref::MemRefDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::SmallVector<func::FuncOp, 8> accelFuncs;
    llvm::DenseSet<StringRef> accelNames;

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (!isAccelFunc(func))
        continue;
      accelFuncs.push_back(func);
      accelNames.insert(func.getName());
    }

    for (func::FuncOp func : accelFuncs) {
      SymbolTable symbols(module);
      if (failed(inlineCallsInAccel(func, symbols))) {
        signalPassFailure();
        return;
      }
    }

    for (func::FuncOp func : accelFuncs) {
      if (failed(prepareAccelMemrefs(module, func))) {
        signalPassFailure();
        return;
      }
    }

    for (func::FuncOp func : accelFuncs) {
      AliasAnalysis aliasAnalysis(func);
      HandshakeLowering lowering(func, aliasAnalysis);
      if (failed(lowering.run())) {
        signalPassFailure();
        return;
      }
    }

    rewriteHostCalls(module, accelNames);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::createSCFToHandshakeDataflowPass() {
  return std::make_unique<SCFToHandshakeDataflowPass>();
}
