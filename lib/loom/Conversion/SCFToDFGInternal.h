//===-- SCFToDFGInternal.h - SCF-to-DFG internal types ----------*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// Internal header for the SCF-to-DFG conversion pass. Contains shared types,
// helper functions, the MemoryCtrlBuilder class, and candidate extraction
// utilities. The DFGConverter class is declared here and defined in
// SCFToDFG.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_CONVERSION_SCFTODFG_INTERNAL_H
#define LOOM_CONVERSION_SCFTODFG_INTERNAL_H

#include "loom/Conversion/Passes.h"
#include "loom/Dialect/Dataflow/DataflowOps.h"

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

namespace loom {
namespace detail {

using mlir::Attribute;
using mlir::Block;
using mlir::BlockArgument;
using mlir::DenseMap;
using mlir::DenseSet;
using mlir::FailureOr;
using mlir::IntegerAttr;
using mlir::IRMapping;
using mlir::Location;
using mlir::LogicalResult;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationState;
using mlir::Region;
using mlir::Type;
using mlir::TypedAttr;
using mlir::Value;
using mlir::ValueRange;
using mlir::WalkResult;

using loom::dataflow::CarryOp;
using loom::dataflow::GateOp;
using loom::dataflow::InvariantOp;
using loom::dataflow::StreamOp;

//===----------------------------------------------------------------------===//
// Shared types
//===----------------------------------------------------------------------===//

// Access kind for memory operations.
enum class AccessKind { Load, Store };

// SCF path entry for tracking nesting context of memory accesses.
struct PathEntry {
  Operation *op = nullptr;
  unsigned region = 0;
};
using ScfPath = llvm::SmallVector<PathEntry, 4>;

// Tracked memory access information.
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

// State for a region being converted (function body, loop body, if branch).
struct RegionState {
  Region *region = nullptr;
  RegionState *parent = nullptr;
  DenseMap<Value, Value> valueMap;
  Value invariantCond;
  bool pendingCond = false;
  llvm::SmallVector<InvariantOp, 4> pendingInvariants;
  Value controlToken;
};

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

inline std::optional<unsigned> getModuleUIntAttr(ModuleOp module,
                                                 llvm::StringRef name) {
  if (!module)
    return std::nullopt;
  if (auto attr = module->getAttrOfType<IntegerAttr>(name))
    return static_cast<unsigned>(attr.getInt());
  return std::nullopt;
}

inline Value buildCappedJoinTree(OpBuilder &builder, Location loc,
                                 llvm::ArrayRef<Value> inputs,
                                 unsigned maxFanin) {
  assert(maxFanin > 0 && "maxFanin must be positive");
  if (inputs.empty())
    return Value();
  if (inputs.size() == 1)
    return inputs.front();
  if (inputs.size() <= maxFanin)
    return circt::handshake::JoinOp::create(builder, loc,
                                            builder.getNoneType(), inputs)
        .getResult();

  llvm::SmallVector<Value, 8> current(inputs.begin(), inputs.end());
  llvm::SmallVector<Value, 8> next;
  while (current.size() > maxFanin) {
    next.clear();
    for (size_t base = 0; base < current.size(); base += maxFanin) {
      auto chunk = llvm::ArrayRef<Value>(current).slice(
          base, std::min<size_t>(maxFanin, current.size() - base));
      if (chunk.size() == 1)
        next.push_back(chunk.front());
      else
        next.push_back(circt::handshake::JoinOp::create(
                           builder, loc, builder.getNoneType(), chunk)
                           .getResult());
    }
    current.assign(next.begin(), next.end());
  }

  if (current.size() == 1)
    return current.front();
  return circt::handshake::JoinOp::create(builder, loc, builder.getNoneType(),
                                          current)
      .getResult();
}

inline LogicalResult capHandshakeJoinFanIn(ModuleOp module,
                                           unsigned maxFanin) {
  if (maxFanin == 0)
    return mlir::success();

  llvm::SmallVector<circt::handshake::JoinOp, 8> joins;
  module.walk([&](circt::handshake::JoinOp joinOp) {
    if (joinOp->getNumOperands() > maxFanin)
      joins.push_back(joinOp);
  });

  if (joins.empty())
    return mlir::success();

  if (maxFanin < 2) {
    if (auto func = joins.front()->getParentOfType<circt::handshake::FuncOp>())
      return func.emitError("ADG max join fan-in is too small to legalize a "
                            "multi-input handshake.join");
    return joins.front()->emitError(
        "ADG max join fan-in is too small to legalize a multi-input "
        "handshake.join");
  }

  for (circt::handshake::JoinOp joinOp : joins) {
    OpBuilder builder(joinOp);
    llvm::SmallVector<Value, 8> operands(joinOp->getOperands().begin(),
                                         joinOp->getOperands().end());
    Value replacement =
        buildCappedJoinTree(builder, joinOp.getLoc(), operands, maxFanin);
    joinOp.getResult().replaceAllUsesWith(replacement);
    joinOp.erase();
  }
  return mlir::success();
}

inline Value getMemrefRoot(Value v) {
  while (v) {
    if (auto cast = v.getDefiningOp<mlir::memref::CastOp>()) {
      v = cast.getSource();
      continue;
    }
    if (auto subview = v.getDefiningOp<mlir::memref::SubViewOp>()) {
      v = subview.getSource();
      continue;
    }
    if (auto reinterpret =
            v.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
      v = reinterpret.getSource();
      continue;
    }
    if (auto collapse = v.getDefiningOp<mlir::memref::CollapseShapeOp>()) {
      v = collapse.getSrc();
      continue;
    }
    if (auto expand = v.getDefiningOp<mlir::memref::ExpandShapeOp>()) {
      v = expand.getSrc();
      continue;
    }
    break;
  }
  return v;
}

inline ScfPath computeScfPath(Operation *op) {
  ScfPath path;
  Operation *current = op->getParentOp();
  while (current) {
    if (mlir::isa<mlir::scf::ForOp, mlir::scf::WhileOp, mlir::scf::IfOp,
                  mlir::scf::IndexSwitchOp>(current)) {
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

inline bool pathMatchesPrefix(const ScfPath &path, const ScfPath &prefix) {
  if (path.size() < prefix.size())
    return false;
  for (size_t i = 0; i < prefix.size(); ++i) {
    if (path[i].op != prefix[i].op || path[i].region != prefix[i].region)
      return false;
  }
  return true;
}

inline bool isLocalToRegion(Value value, Region *region) {
  if (!value || !region)
    return false;
  if (auto blockArg = mlir::dyn_cast<BlockArgument>(value))
    return blockArg.getOwner()->getParent() == region;
  if (Operation *def = value.getDefiningOp())
    return region->isAncestor(def->getParentRegion());
  return false;
}

inline bool isValueDefinedInsideRoot(Operation *root, Value value) {
  if (!value || !root)
    return false;
  if (Operation *def = value.getDefiningOp())
    return root->isAncestor(def);
  if (auto blockArg = mlir::dyn_cast<BlockArgument>(value)) {
    Operation *owner = blockArg.getOwner()->getParentOp();
    return owner && root->isAncestor(owner);
  }
  return false;
}

inline bool isSideEffectFreeOp(Operation *op) {
  if (!op || op->getNumRegions() != 0)
    return false;
  if (auto memEffect = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op))
    return memEffect.hasNoEffect();
  return false;
}

inline bool isLiveOutOfRoot(Operation *root, Value value) {
  if (!value || !root)
    return false;
  for (mlir::OpOperand &use : value.getUses()) {
    if (!root->isAncestor(use.getOwner()))
      return true;
  }
  return false;
}

inline void setAccessCtrl(MemAccess &access, Value ctrl) {
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

//===----------------------------------------------------------------------===//
// Input collection and candidate extraction
//===----------------------------------------------------------------------===//

inline void collectCandidateInputs(Value value, Operation *root,
                                   llvm::SmallVectorImpl<Value> &inputs,
                                   DenseSet<Value> &seenInputs,
                                   DenseSet<Operation *> &visitedPureDefs) {
  if (!value || isValueDefinedInsideRoot(root, value))
    return;

  if (auto blockArg = mlir::dyn_cast<BlockArgument>(value)) {
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

inline Operation *getExtractionRoot(Operation *selectedRoot) {
  Operation *root = selectedRoot;
  while (auto parentIf =
             mlir::dyn_cast_or_null<mlir::scf::IfOp>(root->getParentOp())) {
    if (!parentIf.getElseRegion().empty())
      break;
    if (!parentIf.getThenRegion().hasOneBlock())
      break;

    Block &thenBlock = parentIf.getThenRegion().front();
    Operation *soleOp = nullptr;
    for (Operation &op : thenBlock) {
      if (mlir::isa<mlir::scf::YieldOp>(op))
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

inline FailureOr<mlir::func::FuncOp>
extractCandidateFunc(mlir::func::FuncOp sourceFunc, Operation *selectedRoot) {
  ModuleOp module = sourceFunc->getParentOfType<ModuleOp>();
  if (!module)
    return mlir::failure();

  Operation *root = getExtractionRoot(selectedRoot);
  llvm::SmallVector<Value, 8> inputValues;
  DenseSet<Value> seenInputs;
  DenseSet<Operation *> visitedPureDefs;
  root->walk([&](Operation *op) {
    for (Value operand : op->getOperands())
      collectCandidateInputs(operand, root, inputValues, seenInputs,
                             visitedPureDefs);
  });

  llvm::SmallVector<Value, 4> liveOuts;
  for (Value result : root->getResults()) {
    if (isLiveOutOfRoot(root, result))
      liveOuts.push_back(result);
  }

  OpBuilder moduleBuilder(sourceFunc.getContext());
  moduleBuilder.setInsertionPoint(sourceFunc);

  llvm::SmallVector<Type, 8> inputTypes;
  for (Value input : inputValues)
    inputTypes.push_back(input.getType());
  llvm::SmallVector<Type, 4> resultTypes;
  for (Value liveOut : liveOuts)
    resultTypes.push_back(liveOut.getType());

  std::string tempName =
      "__loom_dfg_candidate_" + sourceFunc.getName().str();
  auto tempFunc = mlir::func::FuncOp::create(
      moduleBuilder, root->getLoc(), tempName,
      moduleBuilder.getFunctionType(inputTypes, resultTypes));
  if (auto visibility = sourceFunc.getSymVisibilityAttr())
    tempFunc->setAttr(mlir::SymbolTable::getVisibilityAttrName(), visibility);

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
      return mlir::success();
    if (mapping.lookupOrNull(value))
      return mlir::success();

    if (auto blockArg = mlir::dyn_cast<BlockArgument>(value)) {
      auto it = inputIndex.find(blockArg);
      if (it == inputIndex.end()) {
        sourceFunc.emitError("missing extracted candidate input");
        return mlir::failure();
      }
      mapping.map(value, entry->getArgument(it->second));
      return mlir::success();
    }

    Operation *def = value.getDefiningOp();
    if (!def || !isSideEffectFreeOp(def) || root->isAncestor(def)) {
      auto it = inputIndex.find(value);
      if (it == inputIndex.end()) {
        sourceFunc.emitError("missing extracted candidate input");
        return mlir::failure();
      }
      mapping.map(value, entry->getArgument(it->second));
      return mlir::success();
    }

    for (Value operand : def->getOperands()) {
      if (mlir::failed(materialize(operand)))
        return mlir::failure();
    }

    Operation *clone = bodyBuilder.clone(*def, mapping);
    for (auto [orig, repl] :
         llvm::zip(def->getResults(), clone->getResults()))
      mapping.map(orig, repl);
    return mlir::success();
  };

  bool failedMaterialize = false;
  root->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if (!isValueDefinedInsideRoot(root, operand) &&
          mlir::failed(materialize(operand))) {
        failedMaterialize = true;
        return;
      }
    }
  });
  if (failedMaterialize) {
    tempFunc.erase();
    return mlir::failure();
  }

  Operation *rootClone = bodyBuilder.clone(*root, mapping);
  for (auto [orig, repl] :
       llvm::zip(root->getResults(), rootClone->getResults()))
    mapping.map(orig, repl);

  llvm::SmallVector<Value, 4> returnOperands;
  for (Value liveOut : liveOuts) {
    Value mapped = mapping.lookupOrNull(liveOut);
    if (!mapped) {
      tempFunc.erase();
      sourceFunc.emitError("missing extracted candidate live-out");
      return mlir::failure();
    }
    returnOperands.push_back(mapped);
  }
  mlir::func::ReturnOp::create(bodyBuilder, root->getLoc(), returnOperands);
  return tempFunc;
}

//===----------------------------------------------------------------------===//
// DFGConverter class declaration
//===----------------------------------------------------------------------===//

class DFGConverter {
public:
  DFGConverter(mlir::func::FuncOp func)
      : func(func), builder(func.getContext()), returnLoc(func.getLoc()) {}

  LogicalResult run();
  circt::handshake::FuncOp getHandshakeFunc() const { return handshakeFunc; }

private:
  friend class MemoryCtrlBuilder;

  // Create handshake constants.
  Value makeConstant(Location loc, Attribute value, Type type, Value ctrl);
  Value makeDummyData(Location loc, Type type);

  // Value mapping across regions.
  Value mapValue(Value value, RegionState &state, Location loc);

  // Update pending invariant conditions.
  void updateInvariantCond(RegionState &state, Value cond);

  // Convert individual operations.
  LogicalResult convertOp(Operation *op, RegionState &state);
  LogicalResult convertReturn(mlir::func::ReturnOp op, RegionState &state);
  LogicalResult convertLoad(mlir::memref::LoadOp op, RegionState &state);
  LogicalResult convertStore(mlir::memref::StoreOp op, RegionState &state);
  LogicalResult convertFor(mlir::scf::ForOp op, RegionState &state);
  LogicalResult convertWhile(mlir::scf::WhileOp op, RegionState &state);
  LogicalResult convertIf(mlir::scf::IfOp op, RegionState &state);

  // Memory finalization: connect loads/stores to extmemory ops.
  LogicalResult finalizeMemory();

  // Memory control: build ctrl-done chains for ordering.
  LogicalResult buildMemoryControl();

  mlir::func::FuncOp func;
  OpBuilder builder;

  // The created handshake function.
  circt::handshake::FuncOp handshakeFunc;

  // Entry token from the handshake function's start argument.
  Value entryToken;
  Value entrySignal;

  // Memory tracking.
  llvm::SmallVector<MemAccess, 8> memAccesses;
  unsigned orderCounter = 0;
  unsigned memoryId = 0;

  // SCF condition maps.
  // For scf.for we keep both:
  // - raw N+1 control from dataflow.stream
  // - body-visible N control from dataflow.gate
  DenseMap<Operation *, Value> forRawConds;
  DenseMap<Operation *, Value> forBodyConds;
  DenseMap<Operation *, Value> whileConds;
  DenseMap<Operation *, Value> ifConds;

  // Return values to finalize.
  llvm::SmallVector<Value, 4> pendingReturnValues;
  Location returnLoc;
  bool sawReturn = false;

  // Done token from memory control.
  Value memoryDoneToken;
};

//===----------------------------------------------------------------------===//
// MemoryCtrlBuilder: recursive memory control chain builder
//===----------------------------------------------------------------------===//

class MemoryCtrlBuilder {
public:
  MemoryCtrlBuilder(OpBuilder &builder, llvm::ArrayRef<MemAccess *> accesses,
                    DenseMap<Operation *, Value> &forBodyConds,
                    DenseMap<Operation *, Value> &whileConds,
                    DenseMap<Operation *, Value> &ifConds, Value entryControl)
      : builder(builder), forBodyConds(forBodyConds),
        whileConds(whileConds),
        ifConds(ifConds), entryControl(entryControl) {
    sortedAccesses.append(accesses.begin(), accesses.end());
    llvm::sort(sortedAccesses,
               [](MemAccess *a, MemAccess *b) { return a->order < b->order; });
  }

  LogicalResult run() {
    if (sortedAccesses.empty())
      return mlir::success();
    cursor = 0;
    doneToken = processLevel(ScfPath{}, entryControl);
    if (failed && cursor != sortedAccesses.size())
      return mlir::failure();
    if (!failed && cursor != sortedAccesses.size()) {
      sortedAccesses[cursor]->origOp->emitError(
          "memory control did not consume all accesses");
      return mlir::failure();
    }
    return failed ? mlir::failure() : mlir::success();
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

      setAccessCtrl(*access, ctrl);
      ctrl = access->doneToken ? access->doneToken : ctrl;
      ++cursor;
    }
    return ctrl;
  }

  Value processChild(const ScfPath &parentPath, const PathEntry &child,
                     Value ctrl) {
    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(child.op))
      return processFor(forOp, parentPath, ctrl);
    if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(child.op))
      return processWhile(whileOp, parentPath, ctrl);
    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(child.op))
      return processIf(ifOp, parentPath, ctrl);
    child.op->emitError("unsupported SCF op in memory control");
    failed = true;
    return ctrl;
  }

  Value processFor(mlir::scf::ForOp op, const ScfPath &parentPath,
                   Value ctrl) {
    auto condIt = forBodyConds.find(op.getOperation());
    if (condIt == forBodyConds.end()) {
      op.emitError("missing after_cond for scf.for in memory control");
      failed = true;
      return ctrl;
    }
    Value wc = condIt->second;
    Location loc = op.getLoc();

    auto carry =
        CarryOp::create(builder, loc, ctrl.getType(), wc, ctrl, ctrl);
    Value loopCtrl = carry.getO();

    ScfPath bodyPath = parentPath;
    bodyPath.push_back(PathEntry{op, 0});
    Value bodyDone = processLevel(bodyPath, loopCtrl);

    auto doneBranch = circt::handshake::ConditionalBranchOp::create(
        builder, loc, wc, bodyDone);
    carry->setOperand(2, doneBranch.getTrueResult());
    return doneBranch.getFalseResult();
  }

  Value processIf(mlir::scf::IfOp op, const ScfPath &parentPath, Value ctrl) {
    auto condIt = ifConds.find(op.getOperation());
    if (condIt == ifConds.end()) {
      op.emitError("missing condition for scf.if in memory control");
      failed = true;
      return ctrl;
    }
    Value cond = condIt->second;
    Location loc = op.getLoc();

    auto branch = circt::handshake::ConditionalBranchOp::create(
        builder, loc, cond, ctrl);
    Value thenCtrl = branch.getTrueResult();
    Value elseCtrl = branch.getFalseResult();

    ScfPath thenPath = parentPath;
    thenPath.push_back(PathEntry{op, 0});
    Value thenDone = processLevel(thenPath, thenCtrl);

    ScfPath elsePath = parentPath;
    elsePath.push_back(PathEntry{op, 1});
    Value elseDone = processLevel(elsePath, elseCtrl);

    Value zero =
        makeConstant(loc, builder.getIndexAttr(0), builder.getIndexType(),
                     ctrl);
    Value one =
        makeConstant(loc, builder.getIndexAttr(1), builder.getIndexType(),
                     ctrl);
    Value sel =
        mlir::arith::SelectOp::create(builder, loc, cond, one, zero);
    auto mux = circt::handshake::MuxOp::create(
        builder, loc, sel, ValueRange{elseDone, thenDone});
    return mux.getResult();
  }

  Value processWhile(mlir::scf::WhileOp op, const ScfPath &parentPath,
                     Value ctrl) {
    auto condIt = whileConds.find(op.getOperation());
    if (condIt == whileConds.end()) {
      op.emitError("missing condition for scf.while in memory control");
      failed = true;
      return ctrl;
    }
    Value wc = condIt->second;
    Location loc = op.getLoc();

    auto carry =
        CarryOp::create(builder, loc, ctrl.getType(), wc, ctrl, ctrl);
    Value loopCtrl = carry.getO();

    ScfPath beforePath = parentPath;
    beforePath.push_back(PathEntry{op, 0});
    Value beforeDone = processLevel(beforePath, loopCtrl);

    auto doneBranch = circt::handshake::ConditionalBranchOp::create(
        builder, loc, wc, beforeDone);
    Value afterCtrl = doneBranch.getTrueResult();
    Value exitCtrl = doneBranch.getFalseResult();

    ScfPath afterPath = parentPath;
    afterPath.push_back(PathEntry{op, 1});
    Value afterDone = processLevel(afterPath, afterCtrl);
    carry->setOperand(2, afterDone);
    return exitCtrl;
  }

  Value makeConstant(Location loc, Attribute value, Type type, Value ctrl) {
    auto typedValue = mlir::dyn_cast<TypedAttr>(value);
    if (!typedValue)
      typedValue = IntegerAttr::get(
          type, mlir::cast<IntegerAttr>(value).getInt());
    return circt::handshake::ConstantOp::create(builder, loc, type, typedValue,
                                                ctrl)
        .getResult();
  }

  OpBuilder &builder;
  DenseMap<Operation *, Value> &forBodyConds;
  DenseMap<Operation *, Value> &whileConds;
  DenseMap<Operation *, Value> &ifConds;
  llvm::SmallVector<MemAccess *, 16> sortedAccesses;
  Value entryControl;
  Value doneToken;
  size_t cursor = 0;
  bool failed = false;
};

} // namespace detail
} // namespace loom

#endif // LOOM_CONVERSION_SCFTODFG_INTERNAL_H
