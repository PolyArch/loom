//===-- SCFToHandshakeConvert.cpp - Class method conversion ----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This file implements the HandshakeConversion class methods that convert SCF
// operations (for, while, if, index_switch) into Handshake dataflow operations.
// Analysis helpers are in SCFToHandshakeAnalysis.cpp.
//
//===----------------------------------------------------------------------===//

#include "loom/Conversion/HandshakeOptimize.h"
#include "loom/Conversion/SCFToHandshakeImpl.h"
#include "loom/Dialect/Dataflow/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include <string>

namespace loom {
namespace detail {

using loom::dataflow::CarryOp;
using loom::dataflow::GateOp;
using loom::dataflow::InvariantOp;
using loom::dataflow::StreamOp;

struct HandshakeConversion::RegionState {
  mlir::Region *region = nullptr;
  RegionState *parent = nullptr;
  llvm::DenseMap<mlir::Value, mlir::Value> valueMap;
  mlir::Value invariantCond;
  bool pendingCond = false;
  llvm::SmallVector<InvariantOp, 4> pendingInvariants;
  mlir::Value controlToken;
  bool controlPending = false;
  InvariantOp controlInvariant;
};

HandshakeConversion::HandshakeConversion(mlir::func::FuncOp func,
                                     mlir::AliasAnalysis &aa)
    : func(func), aliasAnalysis(aa), builder(func.getContext()),
      returnLoc(func.getLoc()) {}

mlir::Value HandshakeConversion::makeConstant(mlir::Location loc,
                                            mlir::Attribute value,
                                            mlir::Type type,
                                            mlir::Value ctrlToken) {
  mlir::Value ctrl = ctrlToken ? ctrlToken : getEntryToken(loc);
  mlir::OperationState constState(
      loc, circt::handshake::ConstantOp::getOperationName());
  constState.addOperands(ctrl);
  constState.addTypes(type);
  constState.addAttribute("value", value);
  mlir::Operation *created = builder.create(constState);
  return created->getResult(0);
}

mlir::Value HandshakeConversion::makeBool(mlir::Location loc, bool value) {
  return makeConstant(loc, builder.getBoolAttr(value), builder.getI1Type(),
                      getEntryToken(loc));
}

mlir::Value HandshakeConversion::makeDummyData(mlir::Location loc,
                                             mlir::Type type) {
  return circt::handshake::SourceOp::create(builder, loc, type).getResult();
}

mlir::Value HandshakeConversion::getEntryToken(mlir::Location loc) {
  if (!entryToken)
    entryToken =
        circt::handshake::SourceOp::create(builder, loc, builder.getNoneType())
            .getResult();
  return entryToken;
}

void HandshakeConversion::assignHandshakeNames() {
  unsigned argCount = func.getFunctionType().getNumInputs();
  unsigned resCount = func.getFunctionType().getNumResults();

  llvm::SmallVector<std::string, 8> argNames;
  llvm::SmallVector<std::string, 4> resNames;
  bool parsed = false;

  auto path = resolveSourcePath(func.getLoc());
  if (path) {
    std::string content;
    if (readFile(*path, content)) {
      std::string params;
      std::string body;
      std::string funcName = demangleBaseName(func.getName());
      if (extractFunctionSource(content, funcName, params, body)) {
        argNames = extractParamNames(params);
        if (resCount == 1) {
          if (auto retName = extractReturnName(body))
            resNames.push_back(*retName);
        }
        parsed = true;
      }
    }
  }

  if (!parsed || argNames.size() != argCount) {
    argNames.clear();
    for (unsigned i = 0; i < argCount; ++i)
      argNames.push_back("in" + std::to_string(i));
  }

  if (resCount == 1 && resNames.empty())
    resNames.push_back("out0");
  if (resCount > 1) {
    resNames.clear();
    for (unsigned i = 0; i < resCount; ++i)
      resNames.push_back("out" + std::to_string(i));
  }

  llvm::StringSet<> usedArgs;
  auto makeUnique = [](llvm::StringSet<> &used, llvm::StringRef base) {
    std::string name = base.str();
    unsigned suffix = 0;
    while (used.contains(name)) {
      name = base.str() + "_" + std::to_string(suffix++);
    }
    used.insert(name);
    return name;
  };

  llvm::SmallVector<mlir::Attribute, 8> argAttrs;
  argAttrs.reserve(argNames.size() + 1);
  for (const std::string &name : argNames) {
    std::string unique = makeUnique(usedArgs, name);
    argAttrs.push_back(builder.getStringAttr(unique));
  }
  std::string startName = makeUnique(usedArgs, "start_token");
  argAttrs.push_back(builder.getStringAttr(startName));

  llvm::StringSet<> usedRes;
  llvm::SmallVector<mlir::Attribute, 4> resAttrs;
  resAttrs.reserve(resNames.size() + 1);
  for (const std::string &name : resNames) {
    std::string unique = makeUnique(usedRes, name);
    resAttrs.push_back(builder.getStringAttr(unique));
  }
  std::string doneName = makeUnique(usedRes, "done_token");
  resAttrs.push_back(builder.getStringAttr(doneName));

  handshakeFunc->setAttr("argNames", builder.getArrayAttr(argAttrs));
  handshakeFunc->setAttr("resNames", builder.getArrayAttr(resAttrs));
}

mlir::Value HandshakeConversion::mapValue(mlir::Value value, RegionState &state,
                                        mlir::Location loc) {
  if (!value)
    return value;
  auto it = state.valueMap.find(value);
  if (it != state.valueMap.end())
    return it->second;

  if (state.parent) {
    mlir::Value outer = mapValue(value, *state.parent, loc);
    if (!state.invariantCond || !isLocalToRegion(value, state.parent->region))
      return outer;
    if (mlir::isa<mlir::BaseMemRefType>(outer.getType()))
      return outer;

    auto inv = InvariantOp::create(builder, loc, outer.getType(),
                                           state.invariantCond, outer);
    if (state.pendingCond) {
      state.pendingInvariants.push_back(inv);
    } else {
      state.pendingCond = true;
    }
    state.valueMap[value] = inv.getO();
    return inv.getO();
  }

  return value;
}

void HandshakeConversion::updateInvariantCond(RegionState &state,
                                            mlir::Value cond) {
  if (!state.pendingCond)
    return;
  for (InvariantOp inv : state.pendingInvariants)
    inv->setOperand(0, cond);
  state.pendingInvariants.clear();
  state.pendingCond = false;
}

mlir::LogicalResult HandshakeConversion::convertReturn(mlir::func::ReturnOp op,
                                                   RegionState &state) {
  if (sawReturn)
    return op.emitError("multiple func.return in accel function");
  sawReturn = true;
  for (mlir::Value operand : op.getOperands())
    pendingReturnValues.push_back(mapValue(operand, state, op.getLoc()));
  returnLoc = op.getLoc();
  return mlir::success();
}

mlir::LogicalResult HandshakeConversion::convertLoad(mlir::memref::LoadOp op,
                                                 RegionState &state) {
  mlir::Location loc = op.getLoc();
  llvm::SmallVector<mlir::Value, 4> addrOperands;
  addrOperands.reserve(op.getIndices().size());
  for (mlir::Value index : op.getIndices())
    addrOperands.push_back(mapValue(index, state, loc));

  auto emitLoad = [&](mlir::Value origMemref, mlir::Value mappedMemref,
                      mlir::Value ctrlToken) {
    mlir::Value rootMemref = getMemrefRoot(mappedMemref);
    mlir::Value dummyCtrl = makeDummyData(loc, builder.getNoneType());

    llvm::SmallVector<mlir::Value, 6> operands(addrOperands.begin(),
                                               addrOperands.end());
    operands.push_back(dummyCtrl);
    operands.push_back(dummyCtrl);

    llvm::SmallVector<mlir::Type, 4> resultTypes;
    resultTypes.push_back(op.getType());
    for (mlir::Value addr : addrOperands)
      resultTypes.push_back(addr.getType());

    mlir::OperationState loadState(
        loc, circt::handshake::LoadOp::getOperationName());
    loadState.addOperands(operands);
    loadState.addTypes(resultTypes);
    auto load = mlir::cast<circt::handshake::LoadOp>(builder.create(loadState));
    copyLoomAnnotations(op, load);

    MemAccess access;
    access.origOp = op;
    access.origMemref = origMemref;
    access.memref = rootMemref;
    access.kind = AccessKind::Load;
    access.order = orderCounter++;
    access.path = computeScfPath(op);
    access.loadOp = load;
    access.controlToken = ctrlToken ? ctrlToken : getEntryToken(loc);
    memAccesses.push_back(access);

    return load.getResult(0);
  };

  if (auto selectOp = op.getMemref().getDefiningOp<mlir::arith::SelectOp>()) {
    if (mlir::isa<mlir::BaseMemRefType>(selectOp.getTrueValue().getType()) &&
        mlir::isa<mlir::BaseMemRefType>(selectOp.getFalseValue().getType())) {
      mlir::Value cond = mapValue(selectOp.getCondition(), state, loc);
      mlir::Value baseCtrl = state.controlToken ? state.controlToken
                                                : getEntryToken(loc);
      auto branch = circt::handshake::ConditionalBranchOp::create(builder,
          loc, cond, baseCtrl);
      mlir::Value trueData =
          emitLoad(selectOp.getTrueValue(),
                   mapValue(selectOp.getTrueValue(), state, loc),
                   branch.getTrueResult());
      mlir::Value falseData =
          emitLoad(selectOp.getFalseValue(),
                   mapValue(selectOp.getFalseValue(), state, loc),
                   branch.getFalseResult());
      mlir::Value zero = makeConstant(
          loc, builder.getIndexAttr(0), builder.getIndexType(), baseCtrl);
      mlir::Value one = makeConstant(
          loc, builder.getIndexAttr(1), builder.getIndexType(), baseCtrl);
      mlir::Value select =
          mlir::arith::SelectOp::create(builder, loc, cond, one, zero);
      auto mux = circt::handshake::MuxOp::create(builder,
          loc, select, mlir::ValueRange{falseData, trueData});
      state.valueMap[op.getResult()] = mux.getResult();
      return mlir::success();
    }
  }

  mlir::Value mappedMemref = mapValue(op.getMemref(), state, loc);
  mlir::Value data =
      emitLoad(op.getMemref(), mappedMemref,
               state.controlToken ? state.controlToken : getEntryToken(loc));
  state.valueMap[op.getResult()] = data;
  return mlir::success();
}

mlir::LogicalResult HandshakeConversion::convertStore(mlir::memref::StoreOp op,
                                                  RegionState &state) {
  mlir::Location loc = op.getLoc();
  llvm::SmallVector<mlir::Value, 4> addrOperands;
  addrOperands.reserve(op.getIndices().size());
  for (mlir::Value index : op.getIndices())
    addrOperands.push_back(mapValue(index, state, loc));

  mlir::Value dataValue = mapValue(op.getValue(), state, loc);

  auto emitStore = [&](mlir::Value origMemref, mlir::Value mappedMemref,
                       mlir::Value ctrlToken) {
    mlir::Value rootMemref = getMemrefRoot(mappedMemref);
    mlir::Value dummyCtrl = getEntryToken(loc);

    llvm::SmallVector<mlir::Value, 6> operands(addrOperands.begin(),
                                               addrOperands.end());
    operands.push_back(dataValue);
    operands.push_back(dummyCtrl);

    llvm::SmallVector<mlir::Type, 4> resultTypes;
    resultTypes.push_back(dataValue.getType());
    for (mlir::Value addr : addrOperands)
      resultTypes.push_back(addr.getType());

    mlir::OperationState storeState(
        loc, circt::handshake::StoreOp::getOperationName());
    storeState.addOperands(operands);
    storeState.addTypes(resultTypes);
    auto store =
        mlir::cast<circt::handshake::StoreOp>(builder.create(storeState));
    copyLoomAnnotations(op, store);

    MemAccess access;
    access.origOp = op;
    access.origMemref = origMemref;
    access.memref = rootMemref;
    access.kind = AccessKind::Store;
    access.order = orderCounter++;
    access.path = computeScfPath(op);
    access.storeOp = store;
    access.controlToken = ctrlToken ? ctrlToken : getEntryToken(loc);
    memAccesses.push_back(access);
  };

  if (auto selectOp = op.getMemref().getDefiningOp<mlir::arith::SelectOp>()) {
    if (mlir::isa<mlir::BaseMemRefType>(selectOp.getTrueValue().getType()) &&
        mlir::isa<mlir::BaseMemRefType>(selectOp.getFalseValue().getType())) {
      mlir::Value cond = mapValue(selectOp.getCondition(), state, loc);
      mlir::Value baseCtrl = state.controlToken ? state.controlToken
                                                : getEntryToken(loc);
      auto branch = circt::handshake::ConditionalBranchOp::create(builder,
          loc, cond, baseCtrl);
      emitStore(selectOp.getTrueValue(),
                mapValue(selectOp.getTrueValue(), state, loc),
                branch.getTrueResult());
      emitStore(selectOp.getFalseValue(),
                mapValue(selectOp.getFalseValue(), state, loc),
                branch.getFalseResult());
      return mlir::success();
    }
  }

  mlir::Value mappedMemref = mapValue(op.getMemref(), state, loc);
  emitStore(op.getMemref(), mappedMemref,
            state.controlToken ? state.controlToken : getEntryToken(loc));
  return mlir::success();
}

mlir::LogicalResult HandshakeConversion::convertFor(mlir::scf::ForOp op,
                                                RegionState &state) {
  mlir::Location loc = op.getLoc();
  mlir::Value lower = mapValue(op.getLowerBound(), state, loc);
  mlir::Value upper = mapValue(op.getUpperBound(), state, loc);
  mlir::Value step = mapValue(op.getStep(), state, loc);

  auto stream = StreamOp::create(builder, loc, lower, step, upper);
  copyLoomAnnotations(op, stream);
  mlir::Value rawIndex = stream.getIndex();
  mlir::Value rawCond = stream.getWillContinue();
  forConds[op] = rawCond;

  auto gate = GateOp::create(builder,
      loc, mlir::TypeRange{rawIndex.getType(), builder.getI1Type()}, rawIndex,
      rawCond);
  mlir::Value bodyIndex = gate.getAfterValue();
  mlir::Value bodyCond = gate.getAfterCond();

  llvm::SmallVector<CarryOp, 4> carries;
  llvm::SmallVector<mlir::Value, 4> bodyIterValues;
  llvm::SmallVector<mlir::Value, 4> loopResults;

  auto iterOperands = op.getInitArgs();
  for (mlir::Value init : iterOperands) {
    mlir::Value initValue = mapValue(init, state, loc);
    auto carry = CarryOp::create(builder, loc, initValue.getType(), rawCond,
                                         initValue, initValue);
    carries.push_back(carry);
    auto iterGate = GateOp::create(builder,
        loc, mlir::TypeRange{carry.getO().getType(), builder.getI1Type()},
        carry.getO(), rawCond);
    bodyIterValues.push_back(iterGate.getAfterValue());
    auto branch = circt::handshake::ConditionalBranchOp::create(builder,
        loc, rawCond, carry.getO());
    loopResults.push_back(branch.getFalseResult());
  }

  mlir::Block *bodyBlock = op.getBody();
  mlir::Region *bodyRegion = bodyBlock->getParent();
  if (!bodyRegion || !bodyRegion->hasOneBlock())
    return op.emitError("expected single-block scf.for body");

  RegionState bodyState;
  bodyState.region = bodyRegion;
  bodyState.parent = &state;
  bodyState.invariantCond = bodyCond;
  bodyState.pendingCond = false;
  mlir::Value parentCtrl =
      state.controlToken ? state.controlToken : getEntryToken(loc);
  bodyState.controlToken =
      InvariantOp::create(builder, loc, parentCtrl.getType(), bodyCond,
                          parentCtrl)
          .getO();

  bodyState.valueMap[bodyBlock->getArgument(0)] = bodyIndex;
  for (unsigned i = 0, e = bodyIterValues.size(); i < e; ++i)
    bodyState.valueMap[bodyBlock->getArgument(i + 1)] = bodyIterValues[i];

  llvm::SmallVector<mlir::Value, 4> yieldValues;
  for (mlir::Operation &nested : *bodyBlock) {
    if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(nested)) {
      for (mlir::Value operand : yield.getOperands())
        yieldValues.push_back(mapValue(operand, bodyState, yield.getLoc()));
      break;
    }
    if (mlir::failed(convertOp(&nested, bodyState)))
      return mlir::failure();
  }

  if (yieldValues.size() != carries.size())
    return op.emitError("scf.for yield arity mismatch");

  for (unsigned i = 0, e = carries.size(); i < e; ++i)
    carries[i]->setOperand(2, yieldValues[i]);

  for (unsigned i = 0, e = loopResults.size(); i < e; ++i)
    state.valueMap[op.getResult(i)] = loopResults[i];

  updateInvariantCond(bodyState, bodyCond);
  return mlir::success();
}

mlir::LogicalResult HandshakeConversion::convertWhile(mlir::scf::WhileOp op,
                                                  RegionState &state) {
  mlir::Location loc = op.getLoc();

  if (auto streamAttr = getStreamWhileAttr(op)) {
    StreamWhileOperands operands;
    mlir::ScopedDiagnosticHandler handler(
        op.getContext(),
        [&](mlir::Diagnostic &) { return mlir::success(); });
    if (succeeded(analyzeStreamableWhile(op, *streamAttr, operands))) {
      mlir::Value startValue = mapValue(operands.init, state, loc);
      mlir::Value boundValue = mapValue(operands.bound, state, loc);
      mlir::Value stepValue;
      if (operands.stepIsConst) {
        mlir::Value ctrl =
            state.controlToken ? state.controlToken : getEntryToken(loc);
        stepValue =
            makeConstant(loc, builder.getIndexAttr(operands.stepConst),
                         builder.getIndexType(), ctrl);
      } else {
        stepValue = mapValue(operands.step, state, loc);
      }

      mlir::Value startIndex = castToIndex(builder, loc, startValue);
      mlir::Value boundIndex = castToIndex(builder, loc, boundValue);
      mlir::Value stepIndex = castToIndex(builder, loc, stepValue);
      if (!startIndex || !boundIndex || !stepIndex)
        return op.emitError("failed to cast stream operands to index");

      auto stream = StreamOp::create(builder, loc, startIndex, stepIndex,
                                             boundIndex);
      stream->setAttr("step_op", builder.getStringAttr(streamAttr->stepOp));
      stream->setAttr("stop_cond", builder.getStringAttr(streamAttr->stopCond));
      copyLoomAnnotations(op, stream);

      mlir::Value rawIndex = stream.getIndex();
      mlir::Value rawCond = stream.getWillContinue();
      whileConds[op] = rawCond;

      auto gate = GateOp::create(builder,
          loc, mlir::TypeRange{rawIndex.getType(), builder.getI1Type()},
          rawIndex, rawCond);
      mlir::Value bodyIndex = gate.getAfterValue();
      mlir::Value bodyCond = gate.getAfterCond();

      llvm::SmallVector<CarryOp, 4> carries;
      llvm::SmallVector<mlir::Value, 4> bodyIterValues;
      llvm::SmallVector<mlir::Value, 4> loopResults;

      auto iterOperands = op.getOperands();
      for (unsigned i = 0, e = iterOperands.size(); i < e; ++i) {
        if (static_cast<int64_t>(i) == streamAttr->ivIndex)
          continue;
        mlir::Value initValue = mapValue(iterOperands[i], state, loc);
        auto carry = CarryOp::create(builder, loc, initValue.getType(), rawCond,
                                             initValue, initValue);
        carries.push_back(carry);
        auto iterGate = GateOp::create(builder,
            loc, mlir::TypeRange{carry.getO().getType(), builder.getI1Type()},
            carry.getO(), rawCond);
        bodyIterValues.push_back(iterGate.getAfterValue());
        auto branch = circt::handshake::ConditionalBranchOp::create(builder,
            loc, rawCond, carry.getO());
        loopResults.push_back(branch.getFalseResult());
      }

      bool bodyInBefore = operands.bodyInBefore;
      mlir::Region *bodyRegion =
          bodyInBefore ? &op.getBefore() : &op.getAfter();
      if (!bodyRegion || !bodyRegion->hasOneBlock())
        return op.emitError("expected single-block scf.while body");
      mlir::Block *bodyBlock = &bodyRegion->front();

      RegionState bodyState;
      bodyState.region = bodyRegion;
      bodyState.parent = &state;
      bodyState.invariantCond = bodyCond;
      bodyState.pendingCond = false;
      mlir::Value parentCtrl =
          state.controlToken ? state.controlToken : getEntryToken(loc);
      bodyState.controlToken =
          InvariantOp::create(builder, loc, parentCtrl.getType(), bodyCond,
                              parentCtrl)
              .getO();

      unsigned iterIndex = 0;
      for (unsigned i = 0, e = bodyBlock->getNumArguments(); i < e; ++i) {
        if (static_cast<int64_t>(i) == streamAttr->ivIndex) {
          mlir::Value casted =
              castIndexToType(builder, loc, bodyIndex,
                              bodyBlock->getArgument(i).getType());
          if (!casted)
            return op.emitError("failed to cast stream index to iv type");
          bodyState.valueMap[bodyBlock->getArgument(i)] = casted;
        } else {
          if (iterIndex >= bodyIterValues.size())
            return op.emitError("scf.while iter arg mismatch");
          bodyState.valueMap[bodyBlock->getArgument(i)] =
              bodyIterValues[iterIndex++];
        }
      }

      llvm::SmallVector<mlir::Value, 4> yieldValues;
      for (mlir::Operation &nested : *bodyBlock) {
        if (bodyInBefore) {
          if (auto condition =
                  mlir::dyn_cast<mlir::scf::ConditionOp>(nested)) {
            for (mlir::Value operand : condition.getArgs())
              yieldValues.push_back(
                  mapValue(operand, bodyState, condition.getLoc()));
            break;
          }
        } else if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(nested)) {
          for (mlir::Value operand : yield.getOperands())
            yieldValues.push_back(mapValue(operand, bodyState, yield.getLoc()));
          break;
        }
        if (mlir::failed(convertOp(&nested, bodyState)))
          return mlir::failure();
      }

      if (yieldValues.size() != op.getNumOperands())
        return op.emitError("scf.while yield arity mismatch");

      unsigned carryIndex = 0;
      for (unsigned i = 0, e = yieldValues.size(); i < e; ++i) {
        if (static_cast<int64_t>(i) == streamAttr->ivIndex)
          continue;
        if (carryIndex >= carries.size())
          return op.emitError("scf.while carry mismatch");
        carries[carryIndex++]->setOperand(2, yieldValues[i]);
      }

      unsigned resultIndex = 0;
      for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
        if (static_cast<int64_t>(i) == streamAttr->ivIndex)
          continue;
        if (resultIndex >= loopResults.size())
          return op.emitError("scf.while result mismatch");
        state.valueMap[op.getResult(i)] = loopResults[resultIndex++];
      }

      updateInvariantCond(bodyState, bodyCond);
      return mlir::success();
    }
    op->removeAttr("loom.stream");
  }

  llvm::SmallVector<mlir::Value, 4> initValues;
  initValues.reserve(op.getNumOperands());
  for (mlir::Value operand : op.getOperands())
    initValues.push_back(mapValue(operand, state, loc));

  llvm::SmallVector<CarryOp, 4> carries;
  for (mlir::Value initValue : initValues) {
    mlir::Value placeholderCond = makeBool(loc, true);
    auto carry = CarryOp::create(builder, loc, initValue.getType(),
                                         placeholderCond, initValue, initValue);
    carries.push_back(carry);
  }

  mlir::Block &beforeBlock = op.getBefore().front();
  if (beforeBlock.getNumArguments() != carries.size())
    return op.emitError("scf.while before arity mismatch");

  RegionState beforeState;
  beforeState.region = &op.getBefore();
  beforeState.parent = &state;
  beforeState.pendingCond = true;
  mlir::Value parentCtrl =
      state.controlToken ? state.controlToken : getEntryToken(loc);
  mlir::Value placeholderCond = makeBool(loc, true);
  beforeState.controlInvariant = InvariantOp::create(builder,
      loc, parentCtrl.getType(), placeholderCond, parentCtrl);
  beforeState.controlToken = beforeState.controlInvariant.getO();

  for (unsigned i = 0, e = carries.size(); i < e; ++i)
    beforeState.valueMap[beforeBlock.getArgument(i)] = carries[i].getO();

  mlir::Value condValue;
  llvm::SmallVector<mlir::Value, 4> condArgs;
  for (mlir::Operation &nested : beforeBlock) {
    if (auto condition = mlir::dyn_cast<mlir::scf::ConditionOp>(nested)) {
      condValue = mapValue(condition.getCondition(), beforeState,
                           condition.getLoc());
      for (mlir::Value operand : condition.getArgs())
        condArgs.push_back(mapValue(operand, beforeState, condition.getLoc()));
      break;
    }
    if (mlir::failed(convertOp(&nested, beforeState)))
      return mlir::failure();
  }

  if (!condValue)
    return op.emitError("scf.while missing condition");

  whileConds[op] = condValue;

  if (condArgs.size() != op.getNumResults())
    return op.emitError("scf.while result arity mismatch");

  updateInvariantCond(beforeState, condValue);

  llvm::SmallVector<mlir::Value, 4> afterArgs;
  llvm::SmallVector<mlir::Value, 4> exitValues;
  afterArgs.reserve(condArgs.size());
  exitValues.reserve(condArgs.size());
  for (mlir::Value value : condArgs) {
    auto branch = circt::handshake::ConditionalBranchOp::create(builder,
        loc, condValue, value);
    afterArgs.push_back(branch.getTrueResult());
    exitValues.push_back(branch.getFalseResult());
  }

  auto gate = GateOp::create(builder,
      loc, mlir::TypeRange{condValue.getType(), builder.getI1Type()}, condValue,
      condValue);
  mlir::Value bodyCond = gate.getAfterCond();

  mlir::Block &afterBlock = op.getAfter().front();
  if (afterBlock.getNumArguments() != afterArgs.size())
    return op.emitError("scf.while after arity mismatch");

  RegionState afterState;
  afterState.region = &op.getAfter();
  afterState.parent = &state;
  afterState.invariantCond = bodyCond;
  afterState.pendingCond = false;
  afterState.controlToken =
      InvariantOp::create(builder, loc, beforeState.controlToken.getType(),
                          bodyCond, beforeState.controlToken)
          .getO();

  for (unsigned i = 0, e = afterArgs.size(); i < e; ++i)
    afterState.valueMap[afterBlock.getArgument(i)] = afterArgs[i];

  llvm::SmallVector<mlir::Value, 4> yieldValues;
  for (mlir::Operation &nested : afterBlock) {
    if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(nested)) {
      for (mlir::Value operand : yield.getOperands())
        yieldValues.push_back(mapValue(operand, afterState, yield.getLoc()));
      break;
    }
    if (mlir::failed(convertOp(&nested, afterState)))
      return mlir::failure();
  }

  if (yieldValues.size() != carries.size())
    return op.emitError("scf.while yield arity mismatch");

  for (unsigned i = 0, e = carries.size(); i < e; ++i)
    carries[i]->setOperand(2, yieldValues[i]);

  for (unsigned i = 0, e = op.getNumResults(); i < e; ++i)
    state.valueMap[op.getResult(i)] = exitValues[i];

  updateInvariantCond(afterState, bodyCond);
  return mlir::success();
}

mlir::LogicalResult HandshakeConversion::convertIf(mlir::scf::IfOp op,
                                               RegionState &state) {
  mlir::Location loc = op.getLoc();
  mlir::Value condValue = mapValue(op.getCondition(), state, loc);
  ifConds[op] = condValue;
  mlir::Value ctrlToken = state.controlToken ? state.controlToken
                                             : getEntryToken(loc);

  auto branch = circt::handshake::ConditionalBranchOp::create(builder,
      loc, condValue, ctrlToken);
  mlir::Value thenCtrl = branch.getTrueResult();
  mlir::Value elseCtrl = branch.getFalseResult();

  mlir::Region &thenRegion = op.getThenRegion();
  RegionState thenState;
  thenState.region = &thenRegion;
  thenState.parent = &state;
  thenState.controlToken = thenCtrl;

  llvm::SmallVector<mlir::Value, 4> thenValues;
  if (!thenRegion.hasOneBlock())
    return op.emitError("expected single-block scf.if then region");
  for (mlir::Operation &nested : thenRegion.front()) {
    if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(nested)) {
      for (mlir::Value operand : yield.getOperands())
        thenValues.push_back(mapValue(operand, thenState, yield.getLoc()));
      break;
    }
    if (mlir::failed(convertOp(&nested, thenState)))
      return mlir::failure();
  }

  llvm::SmallVector<mlir::Value, 4> elseValues;
  bool hasElse = !op.getElseRegion().empty();
  if (hasElse) {
    mlir::Region &elseRegion = op.getElseRegion();
    RegionState elseState;
    elseState.region = &elseRegion;
    elseState.parent = &state;
    elseState.controlToken = elseCtrl;
    if (!elseRegion.hasOneBlock())
      return op.emitError("expected single-block scf.if else region");
    for (mlir::Operation &nested : elseRegion.front()) {
      if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(nested)) {
        for (mlir::Value operand : yield.getOperands())
          elseValues.push_back(mapValue(operand, elseState, yield.getLoc()));
        break;
      }
      if (mlir::failed(convertOp(&nested, elseState)))
        return mlir::failure();
    }
  }

  if (op.getNumResults() == 0)
    return mlir::success();

  if (!hasElse)
    return op.emitError("scf.if without else cannot return values");

  mlir::Value zero = makeConstant(
      loc, builder.getIndexAttr(0), builder.getIndexType(), ctrlToken);
  mlir::Value one = makeConstant(
      loc, builder.getIndexAttr(1), builder.getIndexType(), ctrlToken);
  mlir::Value select =
      mlir::arith::SelectOp::create(builder, loc, condValue, one, zero);

  for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
    auto mux = circt::handshake::MuxOp::create(builder,
        loc, select, mlir::ValueRange{elseValues[i], thenValues[i]});
    state.valueMap[op.getResult(i)] = mux.getResult();
  }

  return mlir::success();
}

mlir::LogicalResult HandshakeConversion::convertIndexSwitch(
    mlir::scf::IndexSwitchOp op, RegionState &state) {
  mlir::Location loc = op.getLoc();
  mlir::Value indexValue = mapValue(op.getArg(), state, loc);
  switchConds[op] = indexValue;
  mlir::Value ctrlToken = state.controlToken ? state.controlToken
                                             : getEntryToken(loc);

  llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4> regionResults;
  llvm::SmallVector<mlir::Value, 4> caseConds;

  mlir::Value chainCtrl = ctrlToken;
  auto cases = op.getCases();
  auto caseRegions = op.getCaseRegions();

  for (auto [caseValue, caseRegion] : llvm::zip(cases, caseRegions)) {
    mlir::Value caseConst = makeConstant(
        loc, builder.getIndexAttr(caseValue), builder.getIndexType(), ctrlToken);
    mlir::Value caseCond = mlir::arith::CmpIOp::create(builder,
        loc, mlir::arith::CmpIPredicate::eq, indexValue, caseConst);
    caseConds.push_back(caseCond);

    auto branch = circt::handshake::ConditionalBranchOp::create(builder,
        loc, caseCond, chainCtrl);
    mlir::Value caseCtrl = branch.getTrueResult();
    chainCtrl = branch.getFalseResult();

    RegionState caseState;
    caseState.region = &caseRegion;
    caseState.parent = &state;
    caseState.controlToken = caseCtrl;

    if (!caseRegion.hasOneBlock())
      return op.emitError("expected single-block scf.index_switch case region");

    llvm::SmallVector<mlir::Value, 4> caseValues;
    for (mlir::Operation &nested : caseRegion.front()) {
      if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(nested)) {
        for (mlir::Value operand : yield.getOperands())
          caseValues.push_back(mapValue(operand, caseState, yield.getLoc()));
        break;
      }
      if (mlir::failed(convertOp(&nested, caseState)))
        return mlir::failure();
    }

    regionResults.push_back(std::move(caseValues));
  }

  mlir::Region &defaultRegion = op.getDefaultRegion();
  RegionState defaultState;
  defaultState.region = &defaultRegion;
  defaultState.parent = &state;
  defaultState.controlToken = chainCtrl;

  if (!defaultRegion.hasOneBlock())
    return op.emitError("expected single-block scf.index_switch default region");

  llvm::SmallVector<mlir::Value, 4> defaultValues;
  for (mlir::Operation &nested : defaultRegion.front()) {
    if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(nested)) {
      for (mlir::Value operand : yield.getOperands())
        defaultValues.push_back(mapValue(operand, defaultState, yield.getLoc()));
      break;
    }
    if (mlir::failed(convertOp(&nested, defaultState)))
      return mlir::failure();
  }

  regionResults.push_back(std::move(defaultValues));

  if (op.getNumResults() == 0)
    return mlir::success();

  mlir::Value select = makeConstant(
      loc, builder.getIndexAttr(cases.size()), builder.getIndexType(), ctrlToken);
  for (int64_t i = static_cast<int64_t>(caseConds.size()) - 1; i >= 0; --i) {
    mlir::Value caseIndex = makeConstant(
        loc, builder.getIndexAttr(i), builder.getIndexType(), ctrlToken);
    select = mlir::arith::SelectOp::create(builder,
        loc, caseConds[static_cast<size_t>(i)], caseIndex, select);
  }

  for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
    llvm::SmallVector<mlir::Value, 4> values;
    values.reserve(regionResults.size());
    for (auto &caseValues : regionResults) {
      if (caseValues.size() != e)
        return op.emitError("scf.index_switch yield arity mismatch");
      values.push_back(caseValues[i]);
    }
    auto mux = circt::handshake::MuxOp::create(builder, loc, select, values);
    state.valueMap[op.getResult(i)] = mux.getResult();
  }

  return mlir::success();
}

mlir::LogicalResult HandshakeConversion::convertOp(mlir::Operation *op,
                                               RegionState &state) {
  if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op))
    return convertFor(forOp, state);
  if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(op))
    return convertWhile(whileOp, state);
  if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op))
    return convertIf(ifOp, state);
  if (auto switchOp = mlir::dyn_cast<mlir::scf::IndexSwitchOp>(op))
    return convertIndexSwitch(switchOp, state);
  if (auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(op))
    return convertReturn(ret, state);
  if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op))
    return convertLoad(load, state);
  if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op))
    return convertStore(store, state);
  if (auto castOp = mlir::dyn_cast<mlir::memref::CastOp>(op)) {
    mlir::Value mapped = mapValue(castOp.getSource(), state, op->getLoc());
    state.valueMap[castOp.getResult()] = mapped;
    return mlir::success();
  }
  if (auto viewOp = mlir::dyn_cast<mlir::memref::ViewOp>(op)) {
    mlir::Value mapped = mapValue(viewOp.getSource(), state, op->getLoc());
    state.valueMap[viewOp.getResult()] = mapped;
    return mlir::success();
  }
  if (auto reinterpretOp =
          mlir::dyn_cast<mlir::memref::ReinterpretCastOp>(op)) {
    mlir::Value mapped = mapValue(reinterpretOp.getSource(), state, op->getLoc());
    state.valueMap[reinterpretOp.getResult()] = mapped;
    return mlir::success();
  }
  if (auto subviewOp = mlir::dyn_cast<mlir::memref::SubViewOp>(op)) {
    mlir::Value mapped = mapValue(subviewOp.getSource(), state, op->getLoc());
    state.valueMap[subviewOp.getResult()] = mapped;
    return mlir::success();
  }
  if (auto collapseOp =
          mlir::dyn_cast<mlir::memref::CollapseShapeOp>(op)) {
    mlir::Value mapped = mapValue(collapseOp.getSrc(), state, op->getLoc());
    state.valueMap[collapseOp.getResult()] = mapped;
    return mlir::success();
  }
  if (auto expandOp = mlir::dyn_cast<mlir::memref::ExpandShapeOp>(op)) {
    mlir::Value mapped = mapValue(expandOp.getSrc(), state, op->getLoc());
    state.valueMap[expandOp.getResult()] = mapped;
    return mlir::success();
  }
  if (auto getGlobalOp = mlir::dyn_cast<mlir::memref::GetGlobalOp>(op)) {
    state.valueMap[getGlobalOp.getResult()] = getGlobalOp.getResult();
    return mlir::success();
  }
  if (auto allocaOp = mlir::dyn_cast<mlir::memref::AllocaOp>(op)) {
    state.valueMap[allocaOp.getResult()] = allocaOp.getResult();
    return mlir::success();
  }
  if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
    state.valueMap[allocOp.getResult()] = allocOp.getResult();
    return mlir::success();
  }
  if (auto deallocOp = mlir::dyn_cast<mlir::memref::DeallocOp>(op)) {
    (void)deallocOp;
    return mlir::success();
  }
  if (auto dimOp = mlir::dyn_cast<mlir::memref::DimOp>(op)) {
    return dimOp.emitError("memref.dim must be converted before handshake");
  }
  if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op)) {
    llvm::SmallVector<mlir::Value, 4> args;
    for (mlir::Value operand : callOp.getOperands())
      args.push_back(mapValue(operand, state, op->getLoc()));
    auto newCall = mlir::func::CallOp::create(builder,
        op->getLoc(), callOp.getCallee(), callOp.getResultTypes(), args);
    for (unsigned i = 0, e = callOp.getNumResults(); i < e; ++i)
      state.valueMap[callOp.getResult(i)] = newCall.getResult(i);
    return mlir::success();
  }

  if (auto constantOp = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
    mlir::Location loc = op->getLoc();
    mlir::Value ctrlToken =
        state.controlToken ? state.controlToken : getEntryToken(loc);
    mlir::Value result = makeConstant(loc, constantOp.getValue(),
                                      constantOp.getType(), ctrlToken);
    if (mlir::Operation *def = result.getDefiningOp())
      copyLoomAnnotations(op, def);
    state.valueMap[constantOp.getResult()] = result;
    return mlir::success();
  }

  if (op->getNumRegions() == 0) {
    mlir::IRMapping mapping;
    for (mlir::Value operand : op->getOperands())
      mapping.map(operand, mapValue(operand, state, op->getLoc()));
    mlir::Operation *clone = builder.clone(*op, mapping);
    copyLoomAnnotations(op, clone);
    for (unsigned i = 0, e = op->getNumResults(); i < e; ++i)
      state.valueMap[op->getResult(i)] = clone->getResult(i);
    return mlir::success();
  }

  if (auto *dialect = op->getDialect()) {
    if (dialect->getNamespace() == "memref")
      return op->emitError("memref op must be converted before handshake");
  }
  op->emitError("unsupported op in SCF to Handshake conversion");
  return mlir::failure();
}

void HandshakeConversion::insertForks() {
  mlir::Block *block = handshakeFunc.getBodyBlock();
  llvm::SmallVector<mlir::Value, 16> values;
  for (mlir::BlockArgument arg : block->getArguments())
    values.push_back(arg);
  for (mlir::Operation &op : *block) {
    for (mlir::Value res : op.getResults())
      values.push_back(res);
  }

  for (mlir::Value value : values) {
    if (!value)
      continue;
    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      if (mlir::isa<mlir::BaseMemRefType>(arg.getType()))
        continue;
    }
    if (value.use_empty() || value.hasOneUse())
      continue;

    llvm::SmallVector<mlir::OpOperand *, 4> uses;
    for (mlir::OpOperand &use : value.getUses())
      uses.push_back(&use);

    mlir::OpBuilder::InsertionGuard guard(builder);
    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value))
      builder.setInsertionPointToStart(block);
    else
      builder.setInsertionPointAfter(value.getDefiningOp());

    auto fork = circt::handshake::ForkOp::create(builder,
        value.getLoc(), value, static_cast<unsigned>(uses.size()));
    for (size_t i = 0; i < uses.size(); ++i)
      uses[i]->set(fork.getResults()[i]);
  }
}

mlir::LogicalResult HandshakeConversion::run() {
  builder.setInsertionPointAfter(func);
  auto originalType = func.getFunctionType();
  llvm::SmallVector<mlir::Type, 8> inputTypes;
  llvm::SmallVector<mlir::Type, 4> resultTypes;
  for (mlir::Type type : originalType.getInputs())
    inputTypes.push_back(type);
  for (mlir::Type type : originalType.getResults())
    resultTypes.push_back(type);
  inputTypes.push_back(builder.getNoneType());
  resultTypes.push_back(builder.getNoneType());
  auto handshakeType =
      builder.getFunctionType(inputTypes, resultTypes);

  handshakeFunc = circt::handshake::FuncOp::create(builder,
      func.getLoc(), func.getName(), handshakeType);
  handshakeFunc.resolveArgAndResNames();
  if (auto visibility = func.getSymVisibilityAttr())
    handshakeFunc->setAttr(mlir::SymbolTable::getVisibilityAttrName(),
                           visibility);
  copyLoomAnnotations(func, handshakeFunc);
  assignHandshakeNames();

  mlir::Block *entry = new mlir::Block();
  handshakeFunc.getBody().push_back(entry);
  for (mlir::Type inputType : handshakeType.getInputs())
    entry->addArgument(inputType, func.getLoc());
  builder.setInsertionPointToStart(entry);

  if (!func.getBody().hasOneBlock())
    return func.emitError("expected single-block function body");

  RegionState state;
  state.region = &func.getBody();
  state.parent = nullptr;
  entrySignal = handshakeFunc.getArguments().back();
  entryToken =
      circt::handshake::JoinOp::create(builder, func.getLoc(),
                                       mlir::ValueRange{entrySignal})
          .getResult();
  state.controlToken = entryToken;

  auto newArgs = handshakeFunc.getArguments().drop_back(1);
  for (auto [oldArg, newArg] : llvm::zip(func.getArguments(), newArgs)) {
    state.valueMap[oldArg] = newArg;
  }

  mlir::Block &bodyBlock = func.getBody().front();
  for (mlir::Operation &op : bodyBlock) {
    if (mlir::failed(convertOp(&op, state)))
      return mlir::failure();
  }

  if (!sawReturn)
    return func.emitError("missing func.return in accel function");

  finalizeMemory();
  if (mlir::failed(buildMemoryControl()))
    return mlir::failure();
  if (mlir::failed(verifyMemoryControl()))
    return mlir::failure();

  mlir::Value doneCtrl = memoryDoneToken ? memoryDoneToken : entryToken;
  if (!doneCtrl)
    doneCtrl = getEntryToken(func.getLoc());
  doneSignal = doneCtrl;

  llvm::SmallVector<mlir::Value, 4> returnOperands(pendingReturnValues.begin(),
                                                   pendingReturnValues.end());
  returnOperands.push_back(doneSignal);
  circt::handshake::ReturnOp::create(builder, returnLoc, returnOperands);

  insertForks();
  if (mlir::failed(loom::runHandshakeCleanup(handshakeFunc, builder)))
    return mlir::failure();

  bool hasMemrefOp = false;
  handshakeFunc.walk([&](mlir::Operation *op) {
    if (auto *dialect = op->getDialect()) {
      if (dialect->getNamespace() == "memref") {
        op->emitError("memref ops are not allowed in handshake.func");
        hasMemrefOp = true;
        return mlir::WalkResult::interrupt();
      }
    }
    return mlir::WalkResult::advance();
  });
  if (hasMemrefOp)
    return mlir::failure();

  func.erase();
  return mlir::success();
}

} // namespace detail
} // namespace loom
