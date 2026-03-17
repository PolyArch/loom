//===-- SCFToDFG.cpp - SCF to DFG conversion pass ---------------*- C++ -*-===//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//
//
// Converts SCF-level functions marked with fcc.dfg_candidate to
// handshake.func with dataflow.stream/gate/carry/invariant for loops,
// and handshake.load/store/extmemory for memory operations.
//
// The conversion handles:
//   - scf.if (zero-trip guard) -> handshake.cond_br + handshake.mux
//   - scf.for -> dataflow.stream + dataflow.gate
//   - scf.while (streamable pattern) -> dataflow.stream + dataflow.gate
//   - memref.load/store -> handshake.load/store + handshake.extmemory
//   - ctrl-done chains with dataflow.carry for memory ordering
//   - loop-carried values with dataflow.carry
//   - loop-invariant values with dataflow.invariant
//
// Non-candidate functions are left unchanged.
//
//===----------------------------------------------------------------------===//

#include "fcc/Conversion/Passes.h"
#include "fcc/Dialect/Dataflow/DataflowOps.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "fcc-scf-to-dfg"

using namespace mlir;

namespace {

using fcc::dataflow::CarryOp;
using fcc::dataflow::GateOp;
using fcc::dataflow::InvariantOp;
using fcc::dataflow::StreamOp;

// Access kind for memory operations
enum class AccessKind { Load, Store };

// SCF path entry for tracking nesting context of memory accesses
struct PathEntry {
  Operation *op = nullptr;
  unsigned region = 0;
};
using ScfPath = SmallVector<PathEntry, 4>;

// Tracked memory access information
struct MemAccess {
  Operation *origOp = nullptr;
  Value origMemref;
  Value memref;
  AccessKind kind;
  unsigned order = 0;
  ScfPath path;
  circt::handshake::LoadOp loadOp;
  circt::handshake::StoreOp storeOp;
  Value controlToken;
  Value doneToken;
};

// State for a region being converted (function body, loop body, if branch)
struct RegionState {
  Region *region = nullptr;
  RegionState *parent = nullptr;
  DenseMap<Value, Value> valueMap;
  Value invariantCond;
  bool pendingCond = false;
  SmallVector<InvariantOp, 4> pendingInvariants;
  Value controlToken;
};

// Get the root memref value (tracing through cast/subview/etc.)
static Value getMemrefRoot(Value v) {
  while (v) {
    if (auto cast = v.getDefiningOp<memref::CastOp>()) {
      v = cast.getSource();
      continue;
    }
    if (auto subview = v.getDefiningOp<memref::SubViewOp>()) {
      v = subview.getSource();
      continue;
    }
    if (auto reinterpret = v.getDefiningOp<memref::ReinterpretCastOp>()) {
      v = reinterpret.getSource();
      continue;
    }
    if (auto collapse = v.getDefiningOp<memref::CollapseShapeOp>()) {
      v = collapse.getSrc();
      continue;
    }
    if (auto expand = v.getDefiningOp<memref::ExpandShapeOp>()) {
      v = expand.getSrc();
      continue;
    }
    break;
  }
  return v;
}

// Compute the SCF nesting path for an operation
static ScfPath computeScfPath(Operation *op) {
  ScfPath path;
  Operation *current = op->getParentOp();
  while (current) {
    if (isa<scf::ForOp, scf::WhileOp, scf::IfOp, scf::IndexSwitchOp>(
            current)) {
      Region *parentRegion = op->getParentRegion();
      unsigned regionIdx = 0;
      for (Region &r : current->getRegions()) {
        if (&r == parentRegion)
          break;
        ++regionIdx;
      }
      path.push_back(PathEntry{current, regionIdx});
      op = current;
    }
    current = current->getParentOp();
  }
  std::reverse(path.begin(), path.end());
  return path;
}

static bool pathMatchesPrefix(const ScfPath &path, const ScfPath &prefix) {
  if (path.size() < prefix.size())
    return false;
  for (size_t i = 0; i < prefix.size(); ++i) {
    if (path[i].op != prefix[i].op || path[i].region != prefix[i].region)
      return false;
  }
  return true;
}

// Check if a value is defined locally within a region
static bool isLocalToRegion(Value value, Region *region) {
  if (!value || !region)
    return false;
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return blockArg.getOwner()->getParent() == region;
  if (Operation *def = value.getDefiningOp())
    return region->isAncestor(def->getParentRegion());
  return false;
}

static bool isValueDefinedInsideRoot(Operation *root, Value value) {
  if (!value || !root)
    return false;
  if (Operation *def = value.getDefiningOp())
    return root->isAncestor(def);
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Operation *owner = blockArg.getOwner()->getParentOp();
    return owner && root->isAncestor(owner);
  }
  return false;
}

static bool isSideEffectFreeOp(Operation *op) {
  if (!op || op->getNumRegions() != 0)
    return false;
  if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op))
    return memEffect.hasNoEffect();
  return false;
}

static bool isLiveOutOfRoot(Operation *root, Value value) {
  if (!value || !root)
    return false;
  for (OpOperand &use : value.getUses()) {
    if (!root->isAncestor(use.getOwner()))
      return true;
  }
  return false;
}

static void collectCandidateInputs(Value value, Operation *root,
                                   SmallVectorImpl<Value> &inputs,
                                   DenseSet<Value> &seenInputs,
                                   DenseSet<Operation *> &visitedPureDefs) {
  if (!value || isValueDefinedInsideRoot(root, value))
    return;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (seenInputs.insert(value).second)
      inputs.push_back(value);
    return;
  }

  Operation *def = value.getDefiningOp();
  if (!def || !isSideEffectFreeOp(def) || root->isAncestor(def)) {
    if (seenInputs.insert(value).second)
      inputs.push_back(value);
    return;
  }

  if (!visitedPureDefs.insert(def).second)
    return;
  for (Value operand : def->getOperands())
    collectCandidateInputs(operand, root, inputs, seenInputs, visitedPureDefs);
}

static Operation *getExtractionRoot(Operation *selectedRoot) {
  Operation *root = selectedRoot;
  while (auto parentIf = dyn_cast_or_null<scf::IfOp>(root->getParentOp())) {
    if (!parentIf.getElseRegion().empty())
      break;
    if (!parentIf.getThenRegion().hasOneBlock())
      break;

    Block &thenBlock = parentIf.getThenRegion().front();
    Operation *soleOp = nullptr;
    for (Operation &op : thenBlock) {
      if (isa<scf::YieldOp>(op))
        continue;
      if (soleOp) {
        soleOp = nullptr;
        break;
      }
      soleOp = &op;
    }
    if (soleOp != root)
      break;
    root = parentIf.getOperation();
  }
  return root;
}

static FailureOr<func::FuncOp> extractCandidateFunc(func::FuncOp sourceFunc,
                                                    Operation *selectedRoot) {
  ModuleOp module = sourceFunc->getParentOfType<ModuleOp>();
  if (!module)
    return failure();

  Operation *root = getExtractionRoot(selectedRoot);
  SmallVector<Value, 8> inputValues;
  DenseSet<Value> seenInputs;
  DenseSet<Operation *> visitedPureDefs;
  root->walk([&](Operation *op) {
    for (Value operand : op->getOperands())
      collectCandidateInputs(operand, root, inputValues, seenInputs,
                             visitedPureDefs);
  });

  SmallVector<Value, 4> liveOuts;
  for (Value result : root->getResults()) {
    if (isLiveOutOfRoot(root, result))
      liveOuts.push_back(result);
  }

  OpBuilder moduleBuilder(sourceFunc.getContext());
  moduleBuilder.setInsertionPoint(sourceFunc);

  SmallVector<Type, 8> inputTypes;
  for (Value input : inputValues)
    inputTypes.push_back(input.getType());
  SmallVector<Type, 4> resultTypes;
  for (Value liveOut : liveOuts)
    resultTypes.push_back(liveOut.getType());

  std::string tempName =
      "__fcc_dfg_candidate_" + sourceFunc.getName().str();
  auto tempFunc = moduleBuilder.create<func::FuncOp>(
      root->getLoc(), tempName,
      moduleBuilder.getFunctionType(inputTypes, resultTypes));
  if (auto visibility = sourceFunc.getSymVisibilityAttr())
    tempFunc->setAttr(SymbolTable::getVisibilityAttrName(), visibility);

  Block *entry = tempFunc.addEntryBlock();
  OpBuilder bodyBuilder(sourceFunc.getContext());
  bodyBuilder.setInsertionPointToStart(entry);

  DenseMap<Value, unsigned> inputIndex;
  for (auto [idx, value] : llvm::enumerate(inputValues))
    inputIndex[value] = idx;

  IRMapping mapping;
  for (auto [idx, value] : llvm::enumerate(inputValues))
    mapping.map(value, entry->getArgument(idx));

  std::function<LogicalResult(Value)> materialize =
      [&](Value value) -> LogicalResult {
    if (!value || isValueDefinedInsideRoot(root, value))
      return success();
    if (mapping.lookupOrNull(value))
      return success();

    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      auto it = inputIndex.find(blockArg);
      if (it == inputIndex.end()) {
        sourceFunc.emitError("missing extracted candidate input");
        return failure();
      }
      mapping.map(value, entry->getArgument(it->second));
      return success();
    }

    Operation *def = value.getDefiningOp();
    if (!def || !isSideEffectFreeOp(def) || root->isAncestor(def)) {
      auto it = inputIndex.find(value);
      if (it == inputIndex.end()) {
        sourceFunc.emitError("missing extracted candidate input");
        return failure();
      }
      mapping.map(value, entry->getArgument(it->second));
      return success();
    }

    for (Value operand : def->getOperands()) {
      if (failed(materialize(operand)))
        return failure();
    }

    Operation *clone = bodyBuilder.clone(*def, mapping);
    for (auto [orig, repl] : llvm::zip(def->getResults(), clone->getResults()))
      mapping.map(orig, repl);
    return success();
  };

  bool failedMaterialize = false;
  root->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if (!isValueDefinedInsideRoot(root, operand) &&
          failed(materialize(operand))) {
        failedMaterialize = true;
        return;
      }
    }
  });
  if (failedMaterialize) {
    tempFunc.erase();
    return failure();
  }

  Operation *rootClone = bodyBuilder.clone(*root, mapping);
  for (auto [orig, repl] : llvm::zip(root->getResults(), rootClone->getResults()))
    mapping.map(orig, repl);

  SmallVector<Value, 4> returnOperands;
  for (Value liveOut : liveOuts) {
    Value mapped = mapping.lookupOrNull(liveOut);
    if (!mapped) {
      tempFunc.erase();
      sourceFunc.emitError("missing extracted candidate live-out");
      return failure();
    }
    returnOperands.push_back(mapped);
  }
  bodyBuilder.create<func::ReturnOp>(root->getLoc(), returnOperands);
  return tempFunc;
}

//===----------------------------------------------------------------------===//
// DFGConverter: converts a single func.func to handshake.func
//===----------------------------------------------------------------------===//

class DFGConverter {
public:
  DFGConverter(func::FuncOp func)
      : func(func), builder(func.getContext()), returnLoc(func.getLoc()) {}

  LogicalResult run();
  circt::handshake::FuncOp getHandshakeFunc() const { return handshakeFunc; }

private:
  // Create handshake constants
  Value makeConstant(Location loc, Attribute value, Type type, Value ctrl);
  Value makeDummyData(Location loc, Type type);

  // Value mapping across regions
  Value mapValue(Value value, RegionState &state, Location loc);

  // Update pending invariant conditions
  void updateInvariantCond(RegionState &state, Value cond);

  // Convert individual operations
  LogicalResult convertOp(Operation *op, RegionState &state);
  LogicalResult convertReturn(func::ReturnOp op, RegionState &state);
  LogicalResult convertLoad(memref::LoadOp op, RegionState &state);
  LogicalResult convertStore(memref::StoreOp op, RegionState &state);
  LogicalResult convertFor(scf::ForOp op, RegionState &state);
  LogicalResult convertWhile(scf::WhileOp op, RegionState &state);
  LogicalResult convertIf(scf::IfOp op, RegionState &state);

  // Memory finalization: connect loads/stores to extmemory ops
  LogicalResult finalizeMemory();

  // Memory control: build ctrl-done chains for ordering
  LogicalResult buildMemoryControl();

  func::FuncOp func;
  OpBuilder builder;

  // The created handshake function
  circt::handshake::FuncOp handshakeFunc;

  // Entry token from the handshake function's start argument
  Value entryToken;
  Value entrySignal;

  // Memory tracking
  SmallVector<MemAccess, 8> memAccesses;
  unsigned orderCounter = 0;
  unsigned memoryId = 0;

  // SCF condition maps (for memory control building)
  DenseMap<Operation *, Value> forConds;
  DenseMap<Operation *, Value> ifConds;

  // Return values to finalize
  SmallVector<Value, 4> pendingReturnValues;
  Location returnLoc;
  bool sawReturn = false;

  // Done token from memory control
  Value memoryDoneToken;
};

//===----------------------------------------------------------------------===//
// DFGConverter implementation
//===----------------------------------------------------------------------===//

Value DFGConverter::makeConstant(Location loc, Attribute value, Type type,
                                 Value ctrl) {
  auto typedValue = dyn_cast<TypedAttr>(value);
  if (!typedValue)
    typedValue = IntegerAttr::get(type, cast<IntegerAttr>(value).getInt());
  return circt::handshake::ConstantOp::create(builder, loc, type, typedValue,
                                              ctrl)
      .getResult();
}

Value DFGConverter::makeDummyData(Location loc, Type type) {
  return circt::handshake::SourceOp::create(builder, loc, type).getResult();
}

Value DFGConverter::mapValue(Value value, RegionState &state, Location loc) {
  if (!value)
    return value;

  // Check local mapping
  auto it = state.valueMap.find(value);
  if (it != state.valueMap.end())
    return it->second;

  // Try parent region
  if (state.parent) {
    Value outer = mapValue(value, *state.parent, loc);
    // If we have a loop condition and the value comes from outside, wrap
    // it in a dataflow.invariant
    if (!state.invariantCond || !isLocalToRegion(value, state.parent->region))
      return outer;
    // Memrefs are consumed by extmemory directly, not via invariant
    if (isa<BaseMemRefType>(outer.getType()))
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

void DFGConverter::updateInvariantCond(RegionState &state, Value cond) {
  if (!state.pendingCond)
    return;
  for (InvariantOp inv : state.pendingInvariants)
    inv->setOperand(0, cond);
  state.pendingInvariants.clear();
  state.pendingCond = false;
}

LogicalResult DFGConverter::convertReturn(func::ReturnOp op,
                                          RegionState &state) {
  if (sawReturn)
    return op.emitError("multiple func.return in candidate function");
  sawReturn = true;
  for (Value operand : op.getOperands())
    pendingReturnValues.push_back(mapValue(operand, state, op.getLoc()));
  returnLoc = op.getLoc();
  return success();
}

LogicalResult DFGConverter::convertLoad(memref::LoadOp op,
                                        RegionState &state) {
  Location loc = op.getLoc();
  SmallVector<Value, 4> addrOperands;
  for (Value index : op.getIndices())
    addrOperands.push_back(mapValue(index, state, loc));

  Value mappedMemref = mapValue(op.getMemref(), state, loc);
  Value rootMemref = getMemrefRoot(mappedMemref);
  Value dummyData = makeDummyData(loc, op.getType());
  Value dummyCtrl = makeDummyData(loc, builder.getNoneType());

  // Build LoadOp via OperationState: operands = [addr..., data, ctrl]
  // Results = [dataResult, addrToMem...]
  SmallVector<Value, 4> operands(addrOperands.begin(), addrOperands.end());
  operands.push_back(dummyData);
  operands.push_back(dummyCtrl);

  SmallVector<Type, 4> resultTypes;
  resultTypes.push_back(op.getType());
  for (Value addr : addrOperands)
    resultTypes.push_back(addr.getType());

  OperationState loadState(loc,
                           circt::handshake::LoadOp::getOperationName());
  loadState.addOperands(operands);
  loadState.addTypes(resultTypes);
  auto load = cast<circt::handshake::LoadOp>(builder.create(loadState));

  MemAccess access;
  access.origOp = op;
  access.origMemref = op.getMemref();
  access.memref = rootMemref;
  access.kind = AccessKind::Load;
  access.order = orderCounter++;
  access.path = computeScfPath(op);
  access.loadOp = load;
  access.controlToken =
      state.controlToken ? state.controlToken : entryToken;
  memAccesses.push_back(access);

  state.valueMap[op.getResult()] = load.getResult(0);
  return success();
}

LogicalResult DFGConverter::convertStore(memref::StoreOp op,
                                         RegionState &state) {
  Location loc = op.getLoc();
  SmallVector<Value, 4> addrOperands;
  for (Value index : op.getIndices())
    addrOperands.push_back(mapValue(index, state, loc));

  Value dataValue = mapValue(op.getValue(), state, loc);
  Value mappedMemref = mapValue(op.getMemref(), state, loc);
  Value rootMemref = getMemrefRoot(mappedMemref);
  Value dummyCtrl = makeDummyData(loc, builder.getNoneType());

  // Build StoreOp via OperationState: operands = [addr..., data, ctrl]
  // Results = [dataToMem, addrToMem...]
  SmallVector<Value, 4> operands(addrOperands.begin(), addrOperands.end());
  operands.push_back(dataValue);
  operands.push_back(dummyCtrl);

  SmallVector<Type, 4> resultTypes;
  resultTypes.push_back(dataValue.getType());
  for (Value addr : addrOperands)
    resultTypes.push_back(addr.getType());

  OperationState storeState(loc,
                            circt::handshake::StoreOp::getOperationName());
  storeState.addOperands(operands);
  storeState.addTypes(resultTypes);
  auto store = cast<circt::handshake::StoreOp>(builder.create(storeState));

  MemAccess access;
  access.origOp = op;
  access.origMemref = op.getMemref();
  access.memref = rootMemref;
  access.kind = AccessKind::Store;
  access.order = orderCounter++;
  access.path = computeScfPath(op);
  access.storeOp = store;
  access.controlToken =
      state.controlToken ? state.controlToken : entryToken;
  memAccesses.push_back(access);

  return success();
}

LogicalResult DFGConverter::convertFor(scf::ForOp op, RegionState &state) {
  Location loc = op.getLoc();
  Value lower = mapValue(op.getLowerBound(), state, loc);
  Value upper = mapValue(op.getUpperBound(), state, loc);
  Value step = mapValue(op.getStep(), state, loc);

  // Create stream: generates index + will_continue streams
  auto stream = StreamOp::create(builder, loc, builder.getIndexType(),
                                 builder.getI1Type(), lower, step, upper,
                                 "+=", "<");
  Value rawIndex = stream.getIndex();
  Value rawCond = stream.getWillContinue();
  forConds[op] = rawCond;

  // Gate: separates before-region (N+1) from body (N)
  auto gate = GateOp::create(builder, loc, rawIndex.getType(),
                             builder.getI1Type(), rawIndex, rawCond);
  Value bodyIndex = gate.getAfterValue();
  Value bodyCond = gate.getAfterCond();

  // Handle loop-carried values
  SmallVector<CarryOp, 4> carries;
  SmallVector<Value, 4> bodyIterValues;
  SmallVector<Value, 4> loopResults;

  for (Value init : op.getInitArgs()) {
    Value initValue = mapValue(init, state, loc);
    auto carry = CarryOp::create(builder, loc, initValue.getType(), rawCond,
                                 initValue, initValue);
    carries.push_back(carry);
    auto iterGate = GateOp::create(
        builder, loc, carry.getO().getType(), builder.getI1Type(),
        carry.getO(), rawCond);
    bodyIterValues.push_back(iterGate.getAfterValue());
    // Exit value: when rawCond is false, take the carry output
    auto branch = circt::handshake::ConditionalBranchOp::create(
        builder, loc, rawCond, carry.getO());
    loopResults.push_back(branch.getFalseResult());
  }

  // Set up body region state
  Block *bodyBlock = op.getBody();
  Region *bodyRegion = bodyBlock->getParent();
  if (!bodyRegion || !bodyRegion->hasOneBlock())
    return op.emitError("expected single-block scf.for body");

  RegionState bodyState;
  bodyState.region = bodyRegion;
  bodyState.parent = &state;
  bodyState.invariantCond = bodyCond;
  bodyState.pendingCond = false;
  Value parentCtrl = state.controlToken ? state.controlToken : entryToken;
  bodyState.controlToken =
      InvariantOp::create(builder, loc, parentCtrl.getType(), bodyCond,
                          parentCtrl)
          .getO();

  // Map block arguments: arg0 = induction variable, arg1.. = iter args
  bodyState.valueMap[bodyBlock->getArgument(0)] = bodyIndex;
  for (unsigned i = 0, e = bodyIterValues.size(); i < e; ++i)
    bodyState.valueMap[bodyBlock->getArgument(i + 1)] = bodyIterValues[i];

  // Convert body operations
  SmallVector<Value, 4> yieldValues;
  for (Operation &nested : *bodyBlock) {
    if (auto yield = dyn_cast<scf::YieldOp>(nested)) {
      for (Value operand : yield.getOperands())
        yieldValues.push_back(mapValue(operand, bodyState, yield.getLoc()));
      break;
    }
    if (failed(convertOp(&nested, bodyState)))
      return failure();
  }

  // Connect yield values back to carries
  if (yieldValues.size() != carries.size())
    return op.emitError("scf.for yield arity mismatch");

  for (unsigned i = 0, e = carries.size(); i < e; ++i)
    carries[i]->setOperand(2, yieldValues[i]);

  // Map loop results
  for (unsigned i = 0, e = loopResults.size(); i < e; ++i)
    state.valueMap[op.getResult(i)] = loopResults[i];

  updateInvariantCond(bodyState, bodyCond);

  // Clean up the control-token invariant if nothing in the body used it.
  // In handshake semantics, an invariant with no consumers is dead and its
  // removal does not affect upstream producers (the cond_br that feeds it
  // still has its other result consumed).
  if (bodyState.controlToken) {
    if (auto invOp = bodyState.controlToken.getDefiningOp<InvariantOp>()) {
      if (invOp.getO().use_empty())
        invOp->erase();
    }
  }

  return success();
}

LogicalResult DFGConverter::convertWhile(scf::WhileOp op,
                                         RegionState &state) {
  Location loc = op.getLoc();

  // For now, handle general while loops via carry-based approach
  // (streamable while detection can be added later)
  SmallVector<Value, 4> initValues;
  initValues.reserve(op.getNumOperands());
  for (Value operand : op.getOperands())
    initValues.push_back(mapValue(operand, state, loc));

  // Create carries for each iter arg with placeholder condition
  Value placeholderCond =
      makeConstant(loc, builder.getBoolAttr(true), builder.getI1Type(),
                   state.controlToken ? state.controlToken : entryToken);
  SmallVector<CarryOp, 4> carries;
  for (Value initValue : initValues) {
    auto carry = CarryOp::create(builder, loc, initValue.getType(),
                                 placeholderCond, initValue, initValue);
    carries.push_back(carry);
  }

  // Convert before region
  Block &beforeBlock = op.getBefore().front();
  if (beforeBlock.getNumArguments() != carries.size())
    return op.emitError("scf.while before arity mismatch");

  RegionState beforeState;
  beforeState.region = &op.getBefore();
  beforeState.parent = &state;
  beforeState.pendingCond = true;
  Value parentCtrl = state.controlToken ? state.controlToken : entryToken;
  auto beforeCtrlInv = InvariantOp::create(builder, loc, parentCtrl.getType(),
                                           placeholderCond, parentCtrl);
  beforeState.controlToken = beforeCtrlInv.getO();

  for (unsigned i = 0, e = carries.size(); i < e; ++i)
    beforeState.valueMap[beforeBlock.getArgument(i)] = carries[i].getO();

  Value condValue;
  SmallVector<Value, 4> condArgs;
  for (Operation &nested : beforeBlock) {
    if (auto condition = dyn_cast<scf::ConditionOp>(nested)) {
      condValue = mapValue(condition.getCondition(), beforeState,
                           condition.getLoc());
      for (Value operand : condition.getArgs())
        condArgs.push_back(mapValue(operand, beforeState, condition.getLoc()));
      break;
    }
    if (failed(convertOp(&nested, beforeState)))
      return failure();
  }

  if (!condValue)
    return op.emitError("scf.while missing condition");

  // Patch carries and invariants to use the real condition
  for (auto &carry : carries)
    carry->setOperand(0, condValue);
  beforeCtrlInv->setOperand(0, condValue);
  updateInvariantCond(beforeState, condValue);

  // Split condArgs by condition: true -> after region, false -> exit
  SmallVector<Value, 4> afterArgs;
  SmallVector<Value, 4> exitValues;
  for (Value value : condArgs) {
    auto branch = circt::handshake::ConditionalBranchOp::create(
        builder, loc, condValue, value);
    afterArgs.push_back(branch.getTrueResult());
    exitValues.push_back(branch.getFalseResult());
  }

  // Gate for body condition
  auto gate = GateOp::create(builder, loc, condValue.getType(),
                             builder.getI1Type(), condValue, condValue);
  Value bodyCond = gate.getAfterCond();

  // Convert after region
  Block &afterBlock = op.getAfter().front();
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

  SmallVector<Value, 4> yieldValues;
  for (Operation &nested : afterBlock) {
    if (auto yield = dyn_cast<scf::YieldOp>(nested)) {
      for (Value operand : yield.getOperands())
        yieldValues.push_back(mapValue(operand, afterState, yield.getLoc()));
      break;
    }
    if (failed(convertOp(&nested, afterState)))
      return failure();
  }

  if (yieldValues.size() != carries.size())
    return op.emitError("scf.while yield arity mismatch");

  for (unsigned i = 0, e = carries.size(); i < e; ++i)
    carries[i]->setOperand(2, yieldValues[i]);

  for (unsigned i = 0, e = op.getNumResults(); i < e; ++i)
    state.valueMap[op.getResult(i)] = exitValues[i];

  updateInvariantCond(afterState, bodyCond);

  // Clean up unused control-token invariants in before and after regions.
  if (beforeState.controlToken) {
    if (auto invOp = beforeState.controlToken.getDefiningOp<InvariantOp>()) {
      if (invOp.getO().use_empty())
        invOp->erase();
    }
  }
  if (afterState.controlToken) {
    if (auto invOp = afterState.controlToken.getDefiningOp<InvariantOp>()) {
      if (invOp.getO().use_empty())
        invOp->erase();
    }
  }

  return success();
}

LogicalResult DFGConverter::convertIf(scf::IfOp op, RegionState &state) {
  Location loc = op.getLoc();
  Value condValue = mapValue(op.getCondition(), state, loc);
  ifConds[op] = condValue;
  Value ctrlToken = state.controlToken ? state.controlToken : entryToken;

  auto branch = circt::handshake::ConditionalBranchOp::create(
      builder, loc, condValue, ctrlToken);
  Value thenCtrl = branch.getTrueResult();
  Value elseCtrl = branch.getFalseResult();

  // Convert then region
  Region &thenRegion = op.getThenRegion();
  RegionState thenState;
  thenState.region = &thenRegion;
  thenState.parent = &state;
  thenState.controlToken = thenCtrl;

  SmallVector<Value, 4> thenValues;
  if (!thenRegion.hasOneBlock())
    return op.emitError("expected single-block scf.if then region");
  for (Operation &nested : thenRegion.front()) {
    if (auto yield = dyn_cast<scf::YieldOp>(nested)) {
      for (Value operand : yield.getOperands())
        thenValues.push_back(mapValue(operand, thenState, yield.getLoc()));
      break;
    }
    if (failed(convertOp(&nested, thenState)))
      return failure();
  }

  // Convert else region
  SmallVector<Value, 4> elseValues;
  bool hasElse = !op.getElseRegion().empty();
  if (hasElse) {
    Region &elseRegion = op.getElseRegion();
    RegionState elseState;
    elseState.region = &elseRegion;
    elseState.parent = &state;
    elseState.controlToken = elseCtrl;
    if (!elseRegion.hasOneBlock())
      return op.emitError("expected single-block scf.if else region");
    for (Operation &nested : elseRegion.front()) {
      if (auto yield = dyn_cast<scf::YieldOp>(nested)) {
        for (Value operand : yield.getOperands())
          elseValues.push_back(mapValue(operand, elseState, yield.getLoc()));
        break;
      }
      if (failed(convertOp(&nested, elseState)))
        return failure();
    }
  }

  if (op.getNumResults() == 0)
    return success();

  if (!hasElse)
    return op.emitError("scf.if without else cannot return values");

  // Create mux to select between then/else results
  Value zero =
      makeConstant(loc, builder.getIndexAttr(0), builder.getIndexType(),
                   ctrlToken);
  Value one =
      makeConstant(loc, builder.getIndexAttr(1), builder.getIndexType(),
                   ctrlToken);
  Value select =
      arith::SelectOp::create(builder, loc, condValue, one, zero);

  for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
    auto mux = circt::handshake::MuxOp::create(
        builder, loc, select, ValueRange{elseValues[i], thenValues[i]});
    state.valueMap[op.getResult(i)] = mux.getResult();
  }

  return success();
}

LogicalResult DFGConverter::convertOp(Operation *op, RegionState &state) {
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return convertFor(forOp, state);
  if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    return convertWhile(whileOp, state);
  if (auto ifOp = dyn_cast<scf::IfOp>(op))
    return convertIf(ifOp, state);
  if (auto ret = dyn_cast<func::ReturnOp>(op))
    return convertReturn(ret, state);
  if (auto load = dyn_cast<memref::LoadOp>(op))
    return convertLoad(load, state);
  if (auto store = dyn_cast<memref::StoreOp>(op))
    return convertStore(store, state);

  // Memref shape operations: pass through the source memref
  if (auto castOp = dyn_cast<memref::CastOp>(op)) {
    state.valueMap[castOp.getResult()] =
        mapValue(castOp.getSource(), state, op->getLoc());
    return success();
  }
  if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
    state.valueMap[subviewOp.getResult()] =
        mapValue(subviewOp.getSource(), state, op->getLoc());
    return success();
  }
  if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(op)) {
    state.valueMap[reinterpret.getResult()] =
        mapValue(reinterpret.getSource(), state, op->getLoc());
    return success();
  }
  if (auto collapse = dyn_cast<memref::CollapseShapeOp>(op)) {
    state.valueMap[collapse.getResult()] =
        mapValue(collapse.getSrc(), state, op->getLoc());
    return success();
  }
  if (auto expand = dyn_cast<memref::ExpandShapeOp>(op)) {
    state.valueMap[expand.getResult()] =
        mapValue(expand.getSrc(), state, op->getLoc());
    return success();
  }
  if (isa<memref::DeallocOp>(op))
    return success();

  // arith.constant -> handshake.constant
  if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
    Location loc = op->getLoc();
    Value ctrlToken =
        state.controlToken ? state.controlToken : entryToken;
    Value result =
        makeConstant(loc, constantOp.getValue(), constantOp.getType(),
                     ctrlToken);
    state.valueMap[constantOp.getResult()] = result;
    return success();
  }

  // Generic regionless ops: clone with mapped operands
  if (op->getNumRegions() == 0) {
    IRMapping mapping;
    for (Value operand : op->getOperands())
      mapping.map(operand, mapValue(operand, state, op->getLoc()));
    Operation *clone = builder.clone(*op, mapping);
    for (unsigned i = 0, e = op->getNumResults(); i < e; ++i)
      state.valueMap[op->getResult(i)] = clone->getResult(i);
    return success();
  }

  op->emitError("unsupported op in SCF to DFG conversion: ")
      << op->getName();
  return failure();
}

//===----------------------------------------------------------------------===//
// Memory finalization: connect loads/stores to extmemory ops
//===----------------------------------------------------------------------===//

LogicalResult DFGConverter::finalizeMemory() {
  OpBuilder::InsertionGuard guard(builder);
  // Insert extmemory ops before the return
  Operation *returnOp = nullptr;
  handshakeFunc.walk([&](circt::handshake::ReturnOp ret) {
    returnOp = ret.getOperation();
    return WalkResult::interrupt();
  });
  if (returnOp)
    builder.setInsertionPoint(returnOp);
  else
    builder.setInsertionPointToEnd(handshakeFunc.getBodyBlock());

  // Group accesses by root memref
  DenseMap<Value, SmallVector<MemAccess *, 4>> loadsByMemref;
  DenseMap<Value, SmallVector<MemAccess *, 4>> storesByMemref;

  for (MemAccess &access : memAccesses) {
    if (!access.memref)
      continue;
    if (access.kind == AccessKind::Load)
      loadsByMemref[access.memref].push_back(&access);
    else
      storesByMemref[access.memref].push_back(&access);
  }

  // Collect all memrefs (preserve order of first appearance)
  SmallVector<Value, 8> memrefs;
  DenseSet<Value> memrefSet;
  for (auto &entry : loadsByMemref) {
    if (memrefSet.insert(entry.first).second)
      memrefs.push_back(entry.first);
  }
  for (auto &entry : storesByMemref) {
    if (memrefSet.insert(entry.first).second)
      memrefs.push_back(entry.first);
  }
  // Include memref args that might not have been accessed
  for (BlockArgument arg : handshakeFunc.getArguments()) {
    if (!isa<MemRefType>(arg.getType()))
      continue;
    if (memrefSet.insert(arg).second)
      memrefs.push_back(arg);
  }

  for (Value memrefValue : memrefs) {
    auto &loads = loadsByMemref[memrefValue];
    auto &stores = storesByMemref[memrefValue];

    if (loads.empty() && stores.empty())
      continue;

    // Build extmemory operands:
    // stores first (interleaved data+addr pairs), then load addresses
    SmallVector<Value, 8> operands;
    for (MemAccess *access : stores) {
      operands.push_back(access->storeOp.getDataResult());
      for (Value addr : access->storeOp.getAddressResult())
        operands.push_back(addr);
    }
    for (MemAccess *access : loads) {
      for (Value addr : access->loadOp.getAddressResults())
        operands.push_back(addr);
    }

    unsigned ldCount = loads.size();
    unsigned stCount = stores.size();
    Location loc = memrefValue.getLoc();

    auto extMem = circt::handshake::ExternalMemoryOp::create(
        builder, loc, memrefValue, operands, ldCount, stCount, memoryId++);

    auto memResults = extMem->getResults();

    // Result order: ldData..., stDone..., ldDone...
    for (unsigned i = 0; i < loads.size(); ++i) {
      MemAccess *access = loads[i];
      auto load = access->loadOp;
      unsigned addrCount = load.getAddresses().size();
      // Connect data-from-mem to load's data operand
      load->setOperand(addrCount, memResults[i]);
      // Done token
      access->doneToken = memResults[ldCount + stCount + i];
    }

    for (unsigned i = 0; i < stores.size(); ++i) {
      stores[i]->doneToken = memResults[ldCount + i];
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Memory control: build ctrl-done chains for ordering
//===----------------------------------------------------------------------===//

// Set the ctrl operand on a load or store op
static void setAccessCtrl(MemAccess &access, Value ctrl) {
  if (access.kind == AccessKind::Load) {
    auto load = access.loadOp;
    unsigned addrCount = load.getAddresses().size();
    load->setOperand(addrCount + 1, ctrl);
  } else {
    auto store = access.storeOp;
    unsigned addrCount = store.getAddresses().size();
    store->setOperand(addrCount + 1, ctrl);
  }
}

// Recursive memory control builder that walks the SCF hierarchy
class MemoryCtrlBuilder {
public:
  MemoryCtrlBuilder(OpBuilder &builder,
                    ArrayRef<MemAccess *> accesses,
                    DenseMap<Operation *, Value> &forConds,
                    DenseMap<Operation *, Value> &ifConds,
                    Value entryControl)
      : builder(builder), forConds(forConds), ifConds(ifConds),
        entryControl(entryControl) {
    sortedAccesses.append(accesses.begin(), accesses.end());
    llvm::sort(sortedAccesses,
               [](MemAccess *a, MemAccess *b) { return a->order < b->order; });
  }

  LogicalResult run() {
    if (sortedAccesses.empty())
      return success();
    cursor = 0;
    doneToken = processLevel(ScfPath{}, entryControl);
    if (failed && cursor != sortedAccesses.size())
      return failure();
    if (!failed && cursor != sortedAccesses.size()) {
      sortedAccesses[cursor]->origOp->emitError(
          "memory control did not consume all accesses");
      return failure();
    }
    return failed ? failure() : success();
  }

  Value getDoneToken() const { return doneToken; }

private:
  bool isAtLevel(const MemAccess *access, const ScfPath &path) const {
    return access && access->path.size() == path.size() &&
           pathMatchesPrefix(access->path, path);
  }

  Value processLevel(const ScfPath &path, Value ctrl) {
    while (cursor < sortedAccesses.size()) {
      MemAccess *access = sortedAccesses[cursor];
      if (!pathMatchesPrefix(access->path, path))
        break;

      // If the access is deeper than current level, process the child SCF op
      if (access->path.size() > path.size()) {
        PathEntry child = access->path[path.size()];
        size_t beforeCursor = cursor;
        ctrl = processChild(path, child, ctrl);
        if (cursor == beforeCursor) {
          if (!failed) {
            access->origOp->emitError(
                "memory control did not advance cursor");
            failed = true;
          }
          break;
        }
        continue;
      }

      // Access is at this level: set its ctrl and advance
      setAccessCtrl(*access, ctrl);
      ctrl = access->doneToken ? access->doneToken : ctrl;
      ++cursor;
    }
    return ctrl;
  }

  Value processChild(const ScfPath &parentPath, const PathEntry &child,
                     Value ctrl) {
    if (auto forOp = dyn_cast<scf::ForOp>(child.op))
      return processFor(forOp, parentPath, ctrl);
    if (auto ifOp = dyn_cast<scf::IfOp>(child.op))
      return processIf(ifOp, parentPath, ctrl);
    child.op->emitError("unsupported SCF op in memory control");
    failed = true;
    return ctrl;
  }

  Value processFor(scf::ForOp op, const ScfPath &parentPath, Value ctrl) {
    auto condIt = forConds.find(op.getOperation());
    if (condIt == forConds.end()) {
      op.emitError("missing will_continue for scf.for in memory control");
      failed = true;
      return ctrl;
    }
    Value wc = condIt->second;
    Location loc = op.getLoc();

    // Carry wraps ctrl for loop iteration
    auto carry = CarryOp::create(builder, loc, ctrl.getType(), wc, ctrl, ctrl);
    Value loopCtrl = carry.getO();

    // Process body accesses
    ScfPath bodyPath = parentPath;
    bodyPath.push_back(PathEntry{op, 0});
    Value bodyDone = processLevel(bodyPath, loopCtrl);

    // Route done: true -> loopback, false -> exit
    auto doneBranch = circt::handshake::ConditionalBranchOp::create(
        builder, loc, wc, bodyDone);
    carry->setOperand(2, doneBranch.getTrueResult());
    return doneBranch.getFalseResult();
  }

  Value processIf(scf::IfOp op, const ScfPath &parentPath, Value ctrl) {
    auto condIt = ifConds.find(op.getOperation());
    if (condIt == ifConds.end()) {
      op.emitError("missing condition for scf.if in memory control");
      failed = true;
      return ctrl;
    }
    Value cond = condIt->second;
    Location loc = op.getLoc();

    // Split ctrl into then/else paths
    auto branch = circt::handshake::ConditionalBranchOp::create(
        builder, loc, cond, ctrl);
    Value thenCtrl = branch.getTrueResult();
    Value elseCtrl = branch.getFalseResult();

    // Process then-region accesses
    ScfPath thenPath = parentPath;
    thenPath.push_back(PathEntry{op, 0});
    Value thenDone = processLevel(thenPath, thenCtrl);

    // Process else-region accesses (if any)
    ScfPath elsePath = parentPath;
    elsePath.push_back(PathEntry{op, 1});
    Value elseDone = processLevel(elsePath, elseCtrl);

    // Mux to reconverge the two paths
    Value zero =
        makeConstant(loc, builder.getIndexAttr(0), builder.getIndexType(),
                     ctrl);
    Value one =
        makeConstant(loc, builder.getIndexAttr(1), builder.getIndexType(),
                     ctrl);
    Value sel = arith::SelectOp::create(builder, loc, cond, one, zero);
    auto mux = circt::handshake::MuxOp::create(
        builder, loc, sel, ValueRange{elseDone, thenDone});
    return mux.getResult();
  }

  Value makeConstant(Location loc, Attribute value, Type type, Value ctrl) {
    auto typedValue = dyn_cast<TypedAttr>(value);
    if (!typedValue)
      typedValue = IntegerAttr::get(type, cast<IntegerAttr>(value).getInt());
    return circt::handshake::ConstantOp::create(builder, loc, type, typedValue,
                                                ctrl)
        .getResult();
  }

  OpBuilder &builder;
  DenseMap<Operation *, Value> &forConds;
  DenseMap<Operation *, Value> &ifConds;
  SmallVector<MemAccess *, 16> sortedAccesses;
  Value entryControl;
  Value doneToken;
  size_t cursor = 0;
  bool failed = false;
};

LogicalResult DFGConverter::buildMemoryControl() {
  memoryDoneToken = entryToken;
  if (memAccesses.empty())
    return success();

  // Verify all accesses have done tokens
  for (MemAccess &access : memAccesses) {
    if (!access.doneToken) {
      access.origOp->emitError("missing memory done token after finalization");
      return failure();
    }
  }

  // Group accesses by root memref for independent ctrl-done chains
  DenseMap<Value, SmallVector<MemAccess *, 8>> groups;
  for (MemAccess &access : memAccesses) {
    Value root = getMemrefRoot(access.memref);
    groups[root].push_back(&access);
  }

  // For each memory group, build a recursive ctrl-done chain
  SmallVector<Value, 4> doneTokens;
  for (auto &[memref, accesses] : groups) {
    MemoryCtrlBuilder ctrlBuilder(builder, accesses, forConds, ifConds,
                                  entryToken);
    if (failed(ctrlBuilder.run()))
      return failure();
    if (Value done = ctrlBuilder.getDoneToken())
      doneTokens.push_back(done);
  }

  // Join all group done tokens
  if (doneTokens.empty()) {
    memoryDoneToken = entryToken;
  } else if (doneTokens.size() == 1) {
    memoryDoneToken = doneTokens.front();
  } else {
    auto join = circt::handshake::JoinOp::create(builder, func.getLoc(),
                                                 builder.getNoneType(),
                                                 doneTokens);
    memoryDoneToken = join.getResult();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Main conversion entry point
//===----------------------------------------------------------------------===//

LogicalResult DFGConverter::run() {
  returnLoc = func.getLoc();

  // Build handshake function type:
  // inputs = [original args..., none (start token)]
  // results = [original results..., none (done token)]
  builder.setInsertionPointAfter(func);
  auto originalType = func.getFunctionType();
  SmallVector<Type, 8> inputTypes;
  SmallVector<Type, 4> resultTypes;
  for (Type type : originalType.getInputs())
    inputTypes.push_back(type);
  for (Type type : originalType.getResults())
    resultTypes.push_back(type);
  inputTypes.push_back(builder.getNoneType());
  resultTypes.push_back(builder.getNoneType());
  auto handshakeType = builder.getFunctionType(inputTypes, resultTypes);

  // Create handshake.func
  handshakeFunc = circt::handshake::FuncOp::create(builder, func.getLoc(),
                                                   func.getName(),
                                                   handshakeType);
  handshakeFunc.resolveArgAndResNames();
  if (auto visibility = func.getSymVisibilityAttr())
    handshakeFunc->setAttr(SymbolTable::getVisibilityAttrName(), visibility);

  // Assign arg/res names
  SmallVector<Attribute, 8> argNames;
  for (unsigned i = 0, e = originalType.getNumInputs(); i < e; ++i)
    argNames.push_back(builder.getStringAttr("in" + std::to_string(i)));
  argNames.push_back(builder.getStringAttr("start_token"));
  handshakeFunc->setAttr("argNames", builder.getArrayAttr(argNames));

  SmallVector<Attribute, 4> resNames;
  for (unsigned i = 0, e = originalType.getNumResults(); i < e; ++i)
    resNames.push_back(builder.getStringAttr("out" + std::to_string(i)));
  resNames.push_back(builder.getStringAttr("done_token"));
  handshakeFunc->setAttr("resNames", builder.getArrayAttr(resNames));

  // Create entry block with matching args
  Block *entry = new Block();
  handshakeFunc.getBody().push_back(entry);
  for (Type inputType : handshakeType.getInputs())
    entry->addArgument(inputType, func.getLoc());
  builder.setInsertionPointToStart(entry);

  if (!func.getBody().hasOneBlock())
    return func.emitError("expected single-block function body");

  // Set up initial state
  RegionState state;
  state.region = &func.getBody();
  state.parent = nullptr;
  entrySignal = handshakeFunc.getArguments().back();
  entryToken = circt::handshake::JoinOp::create(
                   builder, func.getLoc(), builder.getNoneType(),
                   ValueRange{entrySignal})
                   .getResult();
  state.controlToken = entryToken;

  // Map original function arguments to handshake function arguments
  auto newArgs = handshakeFunc.getArguments().drop_back(1);
  for (auto [oldArg, newArg] : llvm::zip(func.getArguments(), newArgs))
    state.valueMap[oldArg] = newArg;

  // Convert all operations in the function body
  Block &bodyBlock = func.getBody().front();
  for (Operation &op : bodyBlock) {
    if (failed(convertOp(&op, state)))
      return failure();
  }

  if (!sawReturn)
    return func.emitError("missing func.return in candidate function");

  // Finalize memory: connect loads/stores to extmemory
  if (failed(finalizeMemory()))
    return failure();

  // Build memory control chains
  if (failed(buildMemoryControl()))
    return failure();

  // Build the handshake.return
  Value doneCtrl = memoryDoneToken ? memoryDoneToken : entryToken;
  SmallVector<Value, 4> returnOperands(pendingReturnValues.begin(),
                                       pendingReturnValues.end());
  returnOperands.push_back(doneCtrl);
  circt::handshake::ReturnOp::create(builder, returnLoc, returnOperands);

  // Erase the original function
  func.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct ConvertSCFToDFGPass
    : public PassWrapper<ConvertSCFToDFGPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertSCFToDFGPass)

  StringRef getArgument() const override { return "fcc-scf-to-dfg"; }
  StringRef getDescription() const override {
    return "Convert SCF to handshake+dataflow DFG IR";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<circt::handshake::HandshakeDialect>();
    registry.insert<fcc::dataflow::DataflowDialect>();
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();

    SmallVector<func::FuncOp, 4> candidates;
    module.walk([&](func::FuncOp func) {
      if (func->hasAttr("fcc.dfg_candidate"))
        candidates.push_back(func);
    });

    for (func::FuncOp func : candidates) {
      std::string accelName = func.getName().str();
      Operation *selectedRoot = func.getOperation();
      func.walk([&](Operation *op) {
        if (op != func.getOperation() && op->hasAttr("fcc.selected_dfg_root"))
          selectedRoot = op;
      });

      func::FuncOp sourceFunc = func;
      if (selectedRoot != func.getOperation()) {
        auto extracted = extractCandidateFunc(func, selectedRoot);
        if (failed(extracted)) {
          func.emitError("failed to extract selected DFG candidate region");
          signalPassFailure();
          return;
        }
        sourceFunc = *extracted;
      }

      DFGConverter converter(sourceFunc);
      if (failed(converter.run())) {
        func.emitError("failed to convert to DFG");
        signalPassFailure();
        return;
      }

      if (sourceFunc != func) {
        auto hsFunc = converter.getHandshakeFunc();
        Attribute regionIdAttr = func->getAttr("fcc.dfg_region_id");
        Attribute estimatedPEsAttr = func->getAttr("fcc.dfg_estimated_pes");
        Attribute estimatedMemAttr = func->getAttr("fcc.dfg_estimated_mem");
        func.erase();
        hsFunc->setAttr(SymbolTable::getSymbolAttrName(),
                        StringAttr::get(module.getContext(), accelName));
        if (regionIdAttr)
          hsFunc->setAttr("fcc.dfg_region_id", regionIdAttr);
        if (estimatedPEsAttr)
          hsFunc->setAttr("fcc.dfg_estimated_pes", estimatedPEsAttr);
        if (estimatedMemAttr)
          hsFunc->setAttr("fcc.dfg_estimated_mem", estimatedMemAttr);
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> fcc::createConvertSCFToDFGPass() {
  return std::make_unique<ConvertSCFToDFGPass>();
}
