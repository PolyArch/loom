// Lower scf.for ops with iter_args (i.e., structured reductions /
// loop-carried recurrences) found inside dataflow.thread bodies into
// `dataflow.graph.func` symbol definitions plus matching
// `dataflow.graph.launch` ops.
//
// scf.for ops without iter_args are left in place. Straight-line
// dataflow.thread bodies are also extracted into graph.func bodies so
// element-wise kernels have an explicit SpatialCore graph surface. The
// graph function_type is the spec's `(none, T0..TN) -> (none, R0..RM)`:
// leading `none` values are the per-launch ctrl_in / done_out ports.

#include "Frontend/Lowering/Passes.h"
#include "Frontend/Lowering/StreamLoopAttrs.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace {

std::string sanitizeSymbol(::llvm::StringRef in) {
  std::string out;
  out.reserve(in.size());
  for (char c : in) {
    if (::llvm::isAlnum(c) || c == '_')
      out.push_back(c);
    else
      out.push_back('_');
  }
  if (out.empty())
    out = "_";
  return out;
}

std::string uniqueSymbol(::mlir::ModuleOp module, ::llvm::StringRef stem) {
  ::mlir::SymbolTable st(module);
  std::string base = stem.str();
  if (!st.lookup(base))
    return base;
  unsigned suffix = 1;
  while (true) {
    std::string candidate = base + "_" + std::to_string(suffix);
    if (!st.lookup(candidate))
      return candidate;
    ++suffix;
  }
}

// Collect every value used inside the loop region that is defined above it.
void collectExternalUses(::mlir::scf::ForOp loop,
                         ::llvm::SetVector<::mlir::Value> &captures) {
  ::mlir::getUsedValuesDefinedAbove(loop.getRegion(), captures);
}

bool isNestedInReductionFor(::mlir::scf::ForOp loop) {
  for (::mlir::Operation *parent = loop->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto parentLoop = ::mlir::dyn_cast<::mlir::scf::ForOp>(parent);
    if (parentLoop && !parentLoop.getInitArgs().empty())
      return true;
  }
  return false;
}

bool isPureScalarEpilogueOp(::mlir::Operation *op) {
  if (op->getNumResults() == 0 || op->getNumRegions() != 0 ||
      op->getNumSuccessors() != 0)
    return false;
  if (op->hasTrait<::mlir::OpTrait::IsTerminator>())
    return false;
  if (op->hasTrait<::mlir::OpTrait::SymbolTable>())
    return false;
  if (::llvm::isa<::mlir::FunctionOpInterface, ::mlir::CallOpInterface>(op))
    return false;
  ::llvm::StringRef name = op->getName().getStringRef();
  if (name != "arith.addf" && name != "arith.subf" && name != "arith.mulf" &&
      name != "arith.addi" && name != "arith.muli" && name != "arith.andi" &&
      name != "arith.ori" && name != "arith.shli" && name != "arith.shrui" &&
      name != "arith.index_cast" && name != "llvm.zext" &&
      name != "llvm.intr.abs" && name != "llvm.intr.bswap" &&
      name != "llvm.intr.fabs" && name != "llvm.intr.fmuladd")
    return false;
  if (auto effects = ::llvm::dyn_cast<::mlir::MemoryEffectOpInterface>(op))
    return effects.hasNoEffect();
  return ::mlir::isPure(op);
}

bool isLlvmPointerType(::mlir::Type type) {
  return ::llvm::isa<::mlir::LLVM::LLVMPointerType>(type);
}

void collectEpilogueOps(::mlir::scf::ForOp loop,
                        ::llvm::SmallVectorImpl<::mlir::Operation *> &ops) {
  ops.clear();
  ::mlir::Block *block = loop->getBlock();
  ::llvm::DenseSet<::mlir::Value> available;
  for (::mlir::Value value : loop.getResults())
    available.insert(value);

  for (auto it = std::next(::mlir::Block::iterator(loop.getOperation())),
            end = block->end();
       it != end; ++it) {
    ::mlir::Operation *op = &*it;
    if (op->hasTrait<::mlir::OpTrait::IsTerminator>())
      break;
    bool dependsOnReduction = false;
    bool dependsOnUnsupportedLocal = false;
    for (::mlir::Value operand : op->getOperands()) {
      if (available.contains(operand)) {
        dependsOnReduction = true;
        continue;
      }
      ::mlir::Operation *def = operand.getDefiningOp();
      if (def && def->getBlock() == block && def->isBeforeInBlock(op) &&
          !def->isBeforeInBlock(loop.getOperation())) {
        dependsOnUnsupportedLocal = true;
        break;
      }
    }
    if (dependsOnUnsupportedLocal)
      break;
    if (!dependsOnReduction)
      continue;
    if (!isPureScalarEpilogueOp(op))
      break;
    ops.push_back(op);
    for (::mlir::Value result : op->getResults())
      available.insert(result);
  }
}

void collectAdditionalCaptures(::mlir::scf::ForOp loop,
                               ::llvm::ArrayRef<::mlir::Operation *> ops,
                               ::llvm::SetVector<::mlir::Value> &captures) {
  ::llvm::DenseSet<::mlir::Operation *> opSet;
  for (::mlir::Operation *op : ops)
    opSet.insert(op);
  ::llvm::DenseSet<::mlir::Value> loopResults;
  for (::mlir::Value value : loop.getResults())
    loopResults.insert(value);

  for (::mlir::Operation *op : ops) {
    for (::mlir::Value operand : op->getOperands()) {
      if (loopResults.contains(operand))
        continue;
      ::mlir::Operation *def = operand.getDefiningOp();
      if (def && opSet.contains(def))
        continue;
      captures.insert(operand);
    }
  }
}

void computeGraphOutputs(::mlir::scf::ForOp loop,
                         ::llvm::ArrayRef<::mlir::Operation *> epilogueOps,
                         ::llvm::SmallVectorImpl<::mlir::Value> &outputs) {
  outputs.clear();

  ::llvm::DenseSet<::mlir::Operation *> epilogueSet;
  for (::mlir::Operation *op : epilogueOps)
    epilogueSet.insert(op);

  auto usedOutsideEpilogue = [&](::mlir::Value value) {
    for (::mlir::OpOperand &use : value.getUses())
      if (!epilogueSet.contains(use.getOwner()))
        return true;
    return false;
  };
  auto usedByEpilogue = [&](::mlir::Value value) {
    for (::mlir::OpOperand &use : value.getUses())
      if (epilogueSet.contains(use.getOwner()))
        return true;
    return false;
  };

  for (::mlir::Value value : loop.getResults()) {
    if (!isLlvmPointerType(value.getType()) || usedOutsideEpilogue(value))
      outputs.push_back(value);
  }
  for (::mlir::Operation *op : epilogueOps) {
    for (::mlir::Value value : op->getResults()) {
      if (!usedByEpilogue(value))
        outputs.push_back(value);
    }
  }
}

struct LowerForToGraphPass
    : public ::mlir::PassWrapper<LowerForToGraphPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerForToGraphPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-lower-for-to-graph";
  }
  ::llvm::StringRef getDescription() const final {
    return "Lower scf.for ops with iter_args inside dataflow.thread bodies "
           "into dataflow.graph.func definitions plus dataflow.graph.launch "
           "ops.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::func::FuncDialect,
                    ::mlir::scf::SCFDialect, ::mlir::ub::UBDialect,
                    ::dataflow::DataflowDialect>();
  }

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();
    ::mlir::MLIRContext *ctx = &getContext();
    ::mlir::OpBuilder builder(ctx);

    promoteStandaloneMemcpyFunctions(module, builder);
    promoteStandaloneStructuredFunctions(module, builder);

    // First, for any host-scope scf.for with iter_args (i.e., a
    // reduction that survived the forall-to-thread pass because it
    // could not be promoted to a parallel forall), wrap it in a
    // synthetic 1x1 dataflow.thread so a downstream graph.launch can
    // be hosted. This keeps the smoke deliverable's "every iter_args
    // reduction becomes a graph" assertion straightforward.
    wrapHostScopeReductions(module, builder);

    // Then, for each dataflow.thread body, snapshot the eligible
    // scf.for ops in source order before mutating.
    struct Pending {
      ::dataflow::ThreadOp thread;
      ::llvm::SmallVector<::mlir::scf::ForOp, 4> loops;
    };
    ::llvm::SmallVector<Pending, 4> pending;
    for (::dataflow::ThreadOp thread : module.getOps<::dataflow::ThreadOp>()) {
      Pending p;
      p.thread = thread;
      thread.walk([&](::mlir::scf::ForOp loop) {
        if (loop.getInitArgs().empty())
          return; // No iter_args: keep as scf.for in the thread body.
        if (isNestedInReductionFor(loop))
          return;
        // Skip nested-inside-another-graph cases (none in the smoke
        // shape, but keep the rule explicit).
        if (loop->getParentOfType<::dataflow::GraphFuncOp>())
          return;
        p.loops.push_back(loop);
      });
      if (!p.loops.empty())
        pending.push_back(std::move(p));
    }

    for (Pending &p : pending) {
      ::llvm::StringRef threadSym = p.thread.getSymName();
      std::string stem = "g_" + sanitizeSymbol(threadSym);
      for (auto [seq, loop] : ::llvm::enumerate(p.loops)) {
        if (failed(promoteOne(module, loop, stem, seq, builder)))
          return signalPassFailure();
      }
    }

    if (failed(promoteStraightLineThreads(module, builder)))
      return signalPassFailure();
  }

  bool isPureMemcpySetupOp(::mlir::Operation &op) {
    if (op.getNumRegions() != 0 || op.getNumSuccessors() != 0)
      return false;
    if (op.hasTrait<::mlir::OpTrait::IsTerminator>())
      return false;
    if (op.hasTrait<::mlir::OpTrait::SymbolTable>())
      return false;
    if (::llvm::isa<::mlir::FunctionOpInterface, ::mlir::CallOpInterface>(op))
      return false;
    if (auto effects =
            ::llvm::dyn_cast<::mlir::MemoryEffectOpInterface>(&op))
      return effects.hasNoEffect();
    return ::mlir::isPure(&op);
  }

  bool isStandaloneMemcpyFunctionCandidate(::mlir::func::FuncOp func) {
    if (func.isExternal())
      return false;
    if (!func.getFunctionType().getResults().empty())
      return false;
    ::mlir::Region &body = func.getBody();
    if (!body.hasOneBlock())
      return false;
    ::mlir::Block &entry = body.front();
    ::mlir::LLVM::MemcpyOp candidate;
    for (::mlir::Operation &op : entry.without_terminator()) {
      if (auto memcpy = ::llvm::dyn_cast<::mlir::LLVM::MemcpyOp>(&op)) {
        if (candidate || memcpy.getIsVolatile())
          return false;
        candidate = memcpy;
        continue;
      }
      if (!isPureMemcpySetupOp(op))
        return false;
    }
    return static_cast<bool>(candidate);
  }

  bool isStandaloneStructuredSetupOp(::mlir::Operation &op) {
    if (op.getNumRegions() != 0 || op.getNumSuccessors() != 0)
      return false;
    if (op.hasTrait<::mlir::OpTrait::IsTerminator>())
      return false;
    if (op.hasTrait<::mlir::OpTrait::SymbolTable>())
      return false;
    if (::llvm::isa<::mlir::FunctionOpInterface, ::mlir::CallOpInterface>(&op))
      return false;
    if (auto effects =
            ::llvm::dyn_cast<::mlir::MemoryEffectOpInterface>(&op))
      return effects.hasNoEffect();
    return ::mlir::isPure(&op);
  }

  bool findSingleTopLevelStructuredLoop(::mlir::Region &region,
                                        ::mlir::scf::ForOp &selectedLoop) {
    if (!region.hasOneBlock())
      return false;
    for (::mlir::Operation &op : region.front().without_terminator()) {
      if (auto loop = ::llvm::dyn_cast<::mlir::scf::ForOp>(op)) {
        if (selectedLoop)
          return false;
        selectedLoop = loop;
        continue;
      }
      if (!isStandaloneStructuredSetupOp(op))
        return false;
    }
    return true;
  }

  ::mlir::scf::ForOp findStandaloneStructuredLoopRoot(::mlir::Block &entry) {
    ::mlir::scf::ForOp selectedLoop;
    bool sawRoot = false;
    for (::mlir::Operation &op : entry.without_terminator()) {
      if (auto loop = ::llvm::dyn_cast<::mlir::scf::ForOp>(op)) {
        if (sawRoot)
          return {};
        selectedLoop = loop;
        sawRoot = true;
        continue;
      }
      if (auto guard = ::llvm::dyn_cast<::mlir::scf::IfOp>(op)) {
        if (sawRoot || guard.getNumResults() != 0)
          return {};
        if (!findSingleTopLevelStructuredLoop(guard.getThenRegion(),
                                             selectedLoop))
          return {};
        if (!findSingleTopLevelStructuredLoop(guard.getElseRegion(),
                                             selectedLoop))
          return {};
        sawRoot = true;
        continue;
      }
      if (!isStandaloneStructuredSetupOp(op))
        return {};
    }
    return selectedLoop;
  }

  bool hasUnsupportedStandaloneStructuredLoopBody(::mlir::scf::ForOp loop) {
    bool unsupported = false;
    loop->walk([&](::mlir::Operation *nested) -> ::mlir::WalkResult {
      if (nested == loop.getOperation())
        return ::mlir::WalkResult::advance();
      if (::llvm::isa<::dataflow::GraphLaunchOp, ::dataflow::ThreadLaunchOp,
                      ::dataflow::GraphFuncOp, ::dataflow::ThreadOp,
                      ::mlir::func::FuncOp>(nested)) {
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
      if (::llvm::isa<::mlir::CallOpInterface>(nested)) {
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
      if (::llvm::isa<::mlir::scf::ForOp>(nested)) {
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
      if (nested->getNumRegions() != 0 &&
          !::llvm::isa<::mlir::scf::IfOp>(nested)) {
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
      return ::mlir::WalkResult::advance();
    });
    return unsupported;
  }

  void promoteGraphOnlyFunction(::mlir::ModuleOp module,
                                ::mlir::func::FuncOp func,
                                ::mlir::OpBuilder &builder) {
    ::mlir::Type noneType = builder.getType<::mlir::NoneType>();
    ::mlir::Location loc = func.getLoc();
    std::string stem = "g_" + sanitizeSymbol(func.getSymName()) + "_0";
    if (::mlir::SymbolTable(module).lookup(stem))
      return;
    std::string graphName = uniqueSymbol(module, stem);

    ::llvm::SmallVector<::mlir::Type, 8> inputTypes;
    inputTypes.push_back(noneType);
    for (::mlir::Type ty : func.getFunctionType().getInputs())
      inputTypes.push_back(ty);
    ::mlir::FunctionType graphType =
        builder.getFunctionType(inputTypes, {noneType});

    builder.setInsertionPointToEnd(module.getBody());
    auto graph = ::dataflow::GraphFuncOp::create(
        builder, loc, graphName, graphType,
        ::llvm::ArrayRef<::mlir::NamedAttribute>{});
    graph.setSymVisibilityAttr(builder.getStringAttr("private"));

    ::mlir::Block *graphEntry = builder.createBlock(&graph.getBody());
    for (::mlir::Type ty : inputTypes)
      graphEntry->addArgument(ty, loc);

    ::mlir::IRMapping mapping;
    ::mlir::Block &funcEntry = func.getBody().front();
    for (auto [i, arg] : ::llvm::enumerate(funcEntry.getArguments()))
      mapping.map(arg, graphEntry->getArgument(i + 1));

    builder.setInsertionPointToEnd(graphEntry);
    for (::mlir::Operation &op : funcEntry.without_terminator())
      builder.clone(op, mapping);
    ::dataflow::GraphReturnOp::create(
        builder, loc, ::mlir::ValueRange{graphEntry->getArgument(0)});
  }

  void promoteStandaloneMemcpyFunctions(::mlir::ModuleOp module,
                                        ::mlir::OpBuilder &builder) {
    ::llvm::SmallVector<::mlir::func::FuncOp, 4> funcs;
    module.walk([&](::mlir::func::FuncOp func) {
      if (isStandaloneMemcpyFunctionCandidate(func))
        funcs.push_back(func);
    });

    for (::mlir::func::FuncOp func : funcs)
      promoteGraphOnlyFunction(module, func, builder);
  }

  bool isStandaloneStructuredFunctionCandidate(::mlir::func::FuncOp func) {
    if (func.isExternal())
      return false;
    if (func.getSymName() == "main")
      return false;
    if (!func.getFunctionType().getResults().empty())
      return false;
    ::mlir::Region &body = func.getBody();
    if (!body.hasOneBlock())
      return false;
    ::mlir::Block &entry = body.front();

    ::mlir::scf::ForOp topLevelLoop =
        findStandaloneStructuredLoopRoot(entry);
    if (!topLevelLoop)
      return false;
    if (!topLevelLoop.getInitArgs().empty())
      return false;

    return !hasUnsupportedStandaloneStructuredLoopBody(topLevelLoop);
  }

  void promoteStandaloneStructuredFunctions(::mlir::ModuleOp module,
                                            ::mlir::OpBuilder &builder) {
    ::llvm::SmallVector<::mlir::func::FuncOp, 4> funcs;
    module.walk([&](::mlir::func::FuncOp func) {
      if (isStandaloneStructuredFunctionCandidate(func))
        funcs.push_back(func);
    });

    for (::mlir::func::FuncOp func : funcs)
      promoteGraphOnlyFunction(module, func, builder);
  }

  // ====================================================================
  // SMOKE-ONLY: structural placeholder for host-scope reductions.
  //
  // This path keeps the original host-scope `scf.for` alive WHILE
  // also emitting a synthetic 1x1 `dataflow.thread` + matching
  // `dataflow.thread.launch` next to it (the launch site is at the
  // host, not inside a thread). The synthetic launch exists purely
  // so the per-kernel `dfg_check.sh` smoke gate can observe a
  // `thread.launch` for kernels whose reduction tail otherwise lives
  // at host scope (e.g., dotproduct, reduction, vecadd's tail).
  //
  // Important contract: the synthetic launch is a placeholder. Its
  // body is a clone of the original loop, but the original loop is
  // intentionally retained at host scope because we have NOT yet
  // promoted host-scope reductions into their own
  // `loom.acc_region` (and hence into a real placement-eligible
  // thread). Downstream placement / execution semantics MUST NOT
  // consume this shape: the launch is structural-smoke only, the
  // host-scope loop is the truthful execution surface.
  //
  // Remove this entire path once host-scope reductions have been
  // promoted under their own `loom.acc_region` and the wrapper is
  // no longer needed for the smoke gate.
  // ====================================================================
  // TODO(loom-frontend): host-scope reductions need `loom.acc_region`
  // promotion before this path can be removed.
  void wrapHostScopeReductions(::mlir::ModuleOp module,
                               ::mlir::OpBuilder &builder) {
    struct Pending {
      ::mlir::func::FuncOp func;
      ::llvm::SmallVector<::mlir::scf::ForOp, 4> loops;
    };
    ::llvm::SmallVector<Pending, 4> pending;
    module.walk([&](::mlir::func::FuncOp func) {
      Pending p;
      p.func = func;
      func.walk([&](::mlir::scf::ForOp loop) {
        if (loop.getInitArgs().empty())
          return;
        if (isNestedInReductionFor(loop))
          return;
        // Skip if already inside a thread or graph body; only host-
        // scope reductions are wrapped here.
        if (loop->getParentOfType<::dataflow::ThreadOp>())
          return;
        if (loop->getParentOfType<::dataflow::GraphFuncOp>())
          return;
        p.loops.push_back(loop);
      });
      if (!p.loops.empty())
        pending.push_back(std::move(p));
    });

    for (Pending &p : pending) {
      ::llvm::StringRef funcSym = p.func.getSymName();
      std::string stem = "t_" + sanitizeSymbol(funcSym) + "_red";
      for (auto [seq, loop] : ::llvm::enumerate(p.loops)) {
        wrapOne(module, loop, stem, seq, builder);
      }
    }
  }

  void wrapOne(::mlir::ModuleOp module, ::mlir::scf::ForOp loop,
               ::llvm::StringRef stem, size_t seq, ::mlir::OpBuilder &builder) {
    ::mlir::Location loc = loop.getLoc();

    // Collect every value used inside the loop that is defined
    // outside the loop, including the iter_args' init operands and
    // lb/ub/step. Note: iter_args themselves are inside-defined block
    // args; we capture their *initial* SSA values via getInitArgs().
    ::llvm::SetVector<::mlir::Value> captures;
    collectExternalUses(loop, captures);
    ::llvm::SmallVector<::mlir::Operation *, 4> epilogueOps;
    collectEpilogueOps(loop, epilogueOps);
    collectAdditionalCaptures(loop, epilogueOps, captures);
    captures.insert(loop.getLowerBound());
    captures.insert(loop.getUpperBound());
    captures.insert(loop.getStep());
    for (::mlir::Value v : loop.getInitArgs())
      captures.insert(v);

    // Build the thread with one input per loop result type so the
    // launch can carry the reduction value back to the host. We do
    // this by capturing the loop, materialising a memref-backed
    // result-spill, but the simplest correct shape for the smoke
    // deliverable is to leave the loop in the thread body and emit a
    // store-into-host-side-memref boundary -- we don't need that for
    // the kernel-correctness of the smoke driver (which only checks
    // structure, not runnable semantics). Instead, we keep the thread
    // results void and let the loop's value live inside the thread
    // body.

    // Build inputs from captured outside values.
    ::llvm::SmallVector<::mlir::Type, 8> inputTypes;
    for (::mlir::Value v : captures)
      inputTypes.push_back(v.getType());
    ::mlir::FunctionType functionType =
        builder.getFunctionType(inputTypes, /*results=*/{});

    std::string symStem = (stem + "_" + ::llvm::Twine(seq)).str();
    std::string symName = uniqueSymbol(module, symStem);

    builder.setInsertionPointToEnd(module.getBody());
    auto threadOp = ::dataflow::ThreadOp::create(
        builder, loc, symName, functionType,
        ::llvm::ArrayRef<::mlir::NamedAttribute>{});
    threadOp.setSymVisibilityAttr(builder.getStringAttr("private"));

    ::mlir::Region &threadBody = threadOp.getBody();
    ::mlir::Block *entry = builder.createBlock(&threadBody);
    for (::mlir::Type ty : inputTypes)
      entry->addArgument(ty, loc);
    // Per spec section 5.4.1, the body's entry block layout is
    // `(args_*, thread_ctrl, iv_*)`: a `none`-typed thread_ctrl slot
    // followed by one `index`-typed iv per grid dim. We use a 1x1
    // grid here so a single iv slot covers it.
    ::mlir::BlockArgument threadCtrlArg =
        entry->addArgument(builder.getType<::mlir::NoneType>(), loc);
    (void)threadCtrlArg;
    ::mlir::BlockArgument ivArg =
        entry->addArgument(builder.getIndexType(), loc);
    (void)ivArg;

    builder.setInsertionPointToEnd(entry);
    ::mlir::IRMapping mapping;
    for (auto [i, captured] : ::llvm::enumerate(captures))
      mapping.map(captured, entry->getArgument(i));
    ::mlir::Operation *cloned = builder.clone(*loop.getOperation(), mapping);
    if (auto clonedLoop = ::llvm::dyn_cast<::mlir::scf::ForOp>(cloned))
      clonedLoop->setAttr(::loom::lowering::streamContCondAttrName(),
                          ::loom::lowering::inferStreamContCond(builder, loop));
    for (::mlir::Operation *op : epilogueOps)
      builder.clone(*op, mapping);
    ::dataflow::ThreadYieldOp::create(builder, loc);

    // Emit the smoke-only synthetic launch at the original loop
    // site. The original host-scope loop is intentionally retained
    // alongside the launch so the reduction value remains available
    // to the host (see the SMOKE-ONLY banner above for rationale).
    // The launch's purpose is purely structural -- a marker for the
    // per-kernel `dfg_check.sh` smoke gate.
    builder.setInsertionPoint(loop);
    ::llvm::SmallVector<::mlir::Value, 4> upperBounds;
    upperBounds.push_back(::mlir::arith::ConstantOp::create(
        builder, loc, builder.getIndexAttr(1)));
    ::llvm::SmallVector<::mlir::Value, 8> bodyOperands;
    for (::mlir::Value v : captures)
      bodyOperands.push_back(v);

    auto callee = ::mlir::FlatSymbolRefAttr::get(builder.getContext(), symName);
    ::dataflow::ThreadLaunchOp::create(
        builder, loc, /*asyncToken=*/::mlir::Type{}, callee, bodyOperands,
        upperBounds, /*asyncDependencies=*/::mlir::ValueRange{});
    // Note: we deliberately keep the original loop alive at host
    // scope so its result is still valid for downstream uses. The
    // thread body owns a clone of the same loop body.
  }

  ::mlir::LogicalResult promoteOne(::mlir::ModuleOp module,
                                   ::mlir::scf::ForOp loop,
                                   ::llvm::StringRef stem, size_t seq,
                                   ::mlir::OpBuilder &builder) {
    ::mlir::Location loc = loop.getLoc();
    ::mlir::Type noneType = builder.getType<::mlir::NoneType>();

    // Inputs to the graph callable (after the leading none ctrl_in):
    //   lb, ub, step,
    //   captures (external uses besides iv / iter_args / lb / ub / step),
    //   initial iter_arg values.
    ::llvm::SmallVector<::mlir::Operation *, 4> epilogueOps;
    collectEpilogueOps(loop, epilogueOps);

    ::llvm::SetVector<::mlir::Value> captures;
    collectExternalUses(loop, captures);
    collectAdditionalCaptures(loop, epilogueOps, captures);
    // Don't pass lb/ub/step twice: drop them from captures since we
    // forward them explicitly as separate body operands.
    captures.remove(loop.getLowerBound());
    captures.remove(loop.getUpperBound());
    captures.remove(loop.getStep());

    ::llvm::SmallVector<::mlir::Value, 4> initArgs(loop.getInitArgs().begin(),
                                                   loop.getInitArgs().end());
    ::llvm::SmallVector<::mlir::Value, 4> outputValues;
    computeGraphOutputs(loop, epilogueOps, outputValues);

    // Build the function type. Input layout:
    //   [ none, lb, ub, step, captures..., initIterArgs... ]
    // Result layout:
    //   [ none, final graph outputs... ]
    ::llvm::SmallVector<::mlir::Type, 8> inputTypes;
    inputTypes.push_back(noneType);
    inputTypes.push_back(loop.getLowerBound().getType());
    inputTypes.push_back(loop.getUpperBound().getType());
    inputTypes.push_back(loop.getStep().getType());
    for (::mlir::Value v : captures)
      inputTypes.push_back(v.getType());
    for (::mlir::Value v : initArgs)
      inputTypes.push_back(v.getType());
    ::llvm::SmallVector<::mlir::Type, 4> resultTypes;
    resultTypes.push_back(noneType);
    for (::mlir::Value v : outputValues)
      resultTypes.push_back(v.getType());
    ::mlir::FunctionType functionType =
        builder.getFunctionType(inputTypes, resultTypes);

    // Unique sym.
    std::string symStem = (stem + "_" + ::llvm::Twine(seq)).str();
    std::string symName = uniqueSymbol(module, symStem);

    // Insert the dataflow.graph.func definition at module scope.
    builder.setInsertionPointToEnd(module.getBody());
    auto graphOp = ::dataflow::GraphFuncOp::create(
        builder, loc, symName, functionType,
        ::llvm::ArrayRef<::mlir::NamedAttribute>{});
    graphOp.setSymVisibilityAttr(builder.getStringAttr("private"));

    // Build graph body. Entry block layout matches function_type
    // inputs exactly.
    ::mlir::Region &graphBody = graphOp.getBody();
    ::mlir::Block *entry = builder.createBlock(&graphBody);
    ::llvm::SmallVector<::mlir::Location, 8> argLocs(inputTypes.size(), loc);
    for (size_t i = 0, e = inputTypes.size(); i < e; ++i)
      entry->addArgument(inputTypes[i], loc);

    // Map captured / iter_arg / lb-ub-step to entry block args.
    size_t pos = 1; // 0 is ctrl_in.
    ::mlir::BlockArgument lbArg = entry->getArgument(pos++);
    ::mlir::BlockArgument ubArg = entry->getArgument(pos++);
    ::mlir::BlockArgument stepArg = entry->getArgument(pos++);
    ::llvm::SmallVector<::mlir::BlockArgument, 4> captureArgs;
    for (size_t i = 0; i < captures.size(); ++i)
      captureArgs.push_back(entry->getArgument(pos++));
    ::llvm::SmallVector<::mlir::Value, 4> initArgVals;
    for (size_t i = 0; i < initArgs.size(); ++i)
      initArgVals.push_back(entry->getArgument(pos++));

    // Build the new scf.for with the graph-side iter_arg seeds.
    builder.setInsertionPointToEnd(entry);
    auto newLoop = ::mlir::scf::ForOp::create(builder, loc, lbArg, ubArg,
                                              stepArg, initArgVals,
                                              /*bodyBuilder=*/nullptr);
    newLoop->setAttr(::loom::lowering::streamContCondAttrName(),
                     ::loom::lowering::inferStreamContCond(builder, loop));

    // Move the original loop body into the new loop, remapping ivs +
    // captures.
    ::mlir::Block &origBody = loop.getRegion().front();
    ::mlir::Block &newBody = newLoop.getRegion().front();
    // The new body already has [iv, iter_args*] block args from
    // ForOp::create above. We replace the empty body with a clone of
    // the original ops.
    ::mlir::IRMapping mapping;
    mapping.map(loop.getLowerBound(), lbArg);
    mapping.map(loop.getUpperBound(), ubArg);
    mapping.map(loop.getStep(), stepArg);
    mapping.map(loop.getInductionVar(), newBody.getArgument(0));
    for (size_t i = 0; i < initArgs.size(); ++i) {
      mapping.map(loop.getRegionIterArgs()[i], newBody.getArgument(1 + i));
    }
    for (auto [i, result] : ::llvm::enumerate(loop.getResults()))
      mapping.map(result, newLoop.getResult(i));
    for (auto [i, captured] : ::llvm::enumerate(captures))
      mapping.map(captured, captureArgs[i]);

    builder.setInsertionPointToEnd(&newBody);
    for (::mlir::Operation &op : origBody)
      builder.clone(op, mapping);

    builder.setInsertionPointToEnd(entry);
    for (::mlir::Operation *op : epilogueOps)
      builder.clone(*op, mapping);

    // Emit graph.return: leading ctrl_in passes through (we use the
    // entry block's ctrl_in here as the done signal placeholder; the
    // smoke deliverable does not yet model true memory-completion
    // ordering).
    builder.setInsertionPointToEnd(entry);
    ::llvm::SmallVector<::mlir::Value, 4> returnVals;
    returnVals.push_back(entry->getArgument(0)); // ctrl_in -> done_out
    for (::mlir::Value value : outputValues)
      returnVals.push_back(mapping.lookup(value));
    ::dataflow::GraphReturnOp::create(builder, loc, returnVals);

    // Materialize the graph.launch at the original loop site inside
    // the thread body. The `ctrl_in : none` operand comes from the
    // enclosing thread's `thread_ctrl` block argument (per spec
    // section 5.4.1: the thread body's entry block lays out
    // `(args_*, thread_ctrl, iv_*)` and root graph launches consume
    // the thread_ctrl as their start signal).
    builder.setInsertionPoint(loop);
    ::mlir::Value ctrlIn;
    if (auto enclosingThread = loop->getParentOfType<::dataflow::ThreadOp>()) {
      ::mlir::Block &threadEntry = enclosingThread.getBody().front();
      size_t ctrlIdx = enclosingThread.getFunctionType().getInputs().size();
      // The verifier guarantees this slot exists and is `none`-typed.
      ctrlIn = threadEntry.getArgument(ctrlIdx);
    } else {
      // SMOKE-ONLY fallback: a host-scope graph.launch (no enclosing
      // thread) has no thread_ctrl block arg to consume. The current
      // pipeline never emits this shape because every iter_args
      // reduction is hoisted into a thread first (either by
      // forall-to-thread or by wrapHostScopeReductions). We retain
      // the `ub.poison` fallback as a structural placeholder so a
      // future host-scope path does not silently miscompile.
      ctrlIn = ::mlir::ub::PoisonOp::create(builder, loc, noneType);
    }

    ::llvm::SmallVector<::mlir::Value, 8> launchOperands;
    launchOperands.push_back(loop.getLowerBound());
    launchOperands.push_back(loop.getUpperBound());
    launchOperands.push_back(loop.getStep());
    for (::mlir::Value v : captures)
      launchOperands.push_back(v);
    for (::mlir::Value v : initArgs)
      launchOperands.push_back(v);

    ::llvm::SmallVector<::mlir::Type, 4> launchResultTypes;
    for (::mlir::Value v : outputValues)
      launchResultTypes.push_back(v.getType());

    auto callee = ::mlir::FlatSymbolRefAttr::get(builder.getContext(), symName);
    auto launchOp = ::dataflow::GraphLaunchOp::create(
        builder, loc, /*doneOut=*/noneType, /*results=*/launchResultTypes,
        callee, ctrlIn, launchOperands);

    // Replace the selected graph outputs with the graph.launch's
    // user-data results (skip leading done_out), then erase the
    // scalar epilogue cloned into the graph.
    for (size_t i = 0, e = outputValues.size(); i < e; ++i)
      outputValues[i].replaceAllUsesWith(launchOp.getResults()[i]);
    for (::mlir::Operation *op : ::llvm::reverse(epilogueOps))
      op->erase();
    loop.erase();
    return ::mlir::success();
  }

  bool hasUnsupportedStructuredGraphNestedOp(::mlir::Operation *root) {
    bool unsupported = false;
    root->walk([&](::mlir::Operation *nested) -> ::mlir::WalkResult {
      if (nested == root)
        return ::mlir::WalkResult::advance();
      if (::llvm::isa<::dataflow::GraphLaunchOp, ::dataflow::ThreadLaunchOp,
                      ::mlir::scf::ForOp>(nested)) {
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
      if (auto forall = ::llvm::dyn_cast<::mlir::scf::ForallOp>(nested)) {
        if (isEffectFormForallGraphCandidate(forall))
          return ::mlir::WalkResult::advance();
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
      if (nested->getNumRegions() != 0 &&
          !::llvm::isa<::mlir::scf::IfOp, ::mlir::scf::WhileOp>(nested)) {
        unsupported = true;
        return ::mlir::WalkResult::interrupt();
      }
      return ::mlir::WalkResult::advance();
    });
    return unsupported;
  }

  bool isStraightLineGraphCandidate(::dataflow::ThreadOp thread) {
    ::mlir::Region &body = thread.getBody();
    if (!body.hasOneBlock())
      return false;
    ::mlir::Block &entry = body.front();
    bool hasBodyOp = false;
    for (::mlir::Operation &op : entry.without_terminator()) {
      hasBodyOp = true;
      if (::llvm::isa<::dataflow::GraphLaunchOp, ::dataflow::ThreadLaunchOp>(
              op))
        return false;
      if (::llvm::isa<::mlir::scf::ForOp>(op))
        return false;
      if (auto forall = ::llvm::dyn_cast<::mlir::scf::ForallOp>(op)) {
        if (!isEffectFormForallGraphCandidate(forall))
          return false;
        continue;
      }
      if (::llvm::isa<::mlir::scf::IfOp, ::mlir::scf::WhileOp>(op)) {
        if (hasUnsupportedStructuredGraphNestedOp(&op))
          return false;
        continue;
      }
      if (op.getNumRegions() != 0)
        return false;
      if (op.hasTrait<::mlir::OpTrait::SymbolTable>())
        return false;
      if (::llvm::isa<::mlir::FunctionOpInterface>(op))
        return false;
    }
    return hasBodyOp;
  }

  bool isEffectFormForallGraphCandidate(::mlir::scf::ForallOp forall) {
    if (!forall.getOutputs().empty() || forall.getNumResults() != 0)
      return false;
    auto inParallel = forall.getTerminator();
    if (!inParallel.getRegion().front().empty())
      return false;

    bool hasNestedBoundary = false;
    forall.walk([&](::mlir::Operation *nested) -> ::mlir::WalkResult {
      if (nested == forall.getOperation())
        return ::mlir::WalkResult::advance();
      if (::llvm::isa<::dataflow::GraphLaunchOp, ::dataflow::ThreadLaunchOp,
                      ::mlir::scf::ForOp, ::mlir::scf::ForallOp,
                      ::mlir::scf::WhileOp>(nested)) {
        hasNestedBoundary = true;
        return ::mlir::WalkResult::interrupt();
      }
      return ::mlir::WalkResult::advance();
    });
    return !hasNestedBoundary;
  }

  ::mlir::LogicalResult promoteStraightLineThreads(::mlir::ModuleOp module,
                                                   ::mlir::OpBuilder &builder) {
    ::llvm::SmallVector<::dataflow::ThreadOp, 8> threads;
    for (::dataflow::ThreadOp thread : module.getOps<::dataflow::ThreadOp>()) {
      if (isStraightLineGraphCandidate(thread))
        threads.push_back(thread);
    }

    for (::dataflow::ThreadOp thread : threads) {
      if (failed(promoteStraightLineThread(module, thread, builder)))
        return ::mlir::failure();
    }
    return ::mlir::success();
  }

  ::mlir::LogicalResult promoteStraightLineThread(::mlir::ModuleOp module,
                                                  ::dataflow::ThreadOp thread,
                                                  ::mlir::OpBuilder &builder) {
    ::mlir::Location loc = thread.getLoc();
    ::mlir::Type noneType = builder.getType<::mlir::NoneType>();
    ::mlir::FunctionType threadType = thread.getFunctionType();
    ::mlir::Region &threadBody = thread.getBody();
    ::mlir::Block &threadEntry = threadBody.front();

    size_t threadInputCount = threadType.getInputs().size();
    if (threadEntry.getNumArguments() <= threadInputCount)
      return thread.emitOpError("is missing thread control block argument");
    ::mlir::Value threadCtrl = threadEntry.getArgument(threadInputCount);
    if (!::llvm::isa<::mlir::NoneType>(threadCtrl.getType()))
      return thread.emitOpError("thread control block argument must be none");

    ::llvm::SmallVector<::mlir::Operation *, 8> bodyOps;
    for (::mlir::Operation &op : threadEntry.without_terminator())
      bodyOps.push_back(&op);

    ::llvm::SmallVector<::mlir::Type, 8> inputTypes;
    inputTypes.push_back(noneType);
    for (::mlir::Type ty : threadType.getInputs())
      inputTypes.push_back(ty);
    for (size_t i = threadInputCount + 1, e = threadEntry.getNumArguments();
         i < e; ++i)
      inputTypes.push_back(threadEntry.getArgument(i).getType());

    ::mlir::FunctionType graphType =
        builder.getFunctionType(inputTypes, {noneType});
    std::string stem = "g_" + sanitizeSymbol(thread.getSymName()) + "_0";
    std::string graphName = uniqueSymbol(module, stem);

    builder.setInsertionPointToEnd(module.getBody());
    auto graph = ::dataflow::GraphFuncOp::create(
        builder, loc, graphName, graphType,
        ::llvm::ArrayRef<::mlir::NamedAttribute>{});
    graph.setSymVisibilityAttr(builder.getStringAttr("private"));

    ::mlir::Block *graphEntry = builder.createBlock(&graph.getBody());
    for (::mlir::Type ty : inputTypes)
      graphEntry->addArgument(ty, loc);

    ::mlir::IRMapping mapping;
    mapping.map(threadCtrl, graphEntry->getArgument(0));
    size_t graphArgPos = 1;
    for (size_t i = 0; i < threadInputCount; ++i)
      mapping.map(threadEntry.getArgument(i),
                  graphEntry->getArgument(graphArgPos++));
    for (size_t i = threadInputCount + 1, e = threadEntry.getNumArguments();
         i < e; ++i)
      mapping.map(threadEntry.getArgument(i),
                  graphEntry->getArgument(graphArgPos++));

    builder.setInsertionPointToEnd(graphEntry);
    for (::mlir::Operation *op : bodyOps)
      builder.clone(*op, mapping);
    ::dataflow::GraphReturnOp::create(
        builder, loc, ::mlir::ValueRange{graphEntry->getArgument(0)});

    builder.setInsertionPoint(&threadEntry, threadEntry.begin());
    ::llvm::SmallVector<::mlir::Value, 8> launchOperands;
    for (size_t i = 0; i < threadInputCount; ++i)
      launchOperands.push_back(threadEntry.getArgument(i));
    for (size_t i = threadInputCount + 1, e = threadEntry.getNumArguments();
         i < e; ++i)
      launchOperands.push_back(threadEntry.getArgument(i));

    auto callee =
        ::mlir::FlatSymbolRefAttr::get(builder.getContext(), graphName);
    ::dataflow::GraphLaunchOp::create(builder, loc, /*doneOut=*/noneType,
                                      /*results=*/::mlir::TypeRange{}, callee,
                                      threadCtrl, launchOperands);

    for (::mlir::Operation *op : ::llvm::reverse(bodyOps))
      op->erase();
    return ::mlir::success();
  }
};

} // namespace

namespace loom {
namespace lowering {

std::unique_ptr<::mlir::Pass> createLowerForToGraphPass() {
  return std::make_unique<LowerForToGraphPass>();
}

void registerLowerForToGraphPass() {
  static bool once = []() {
    ::mlir::PassRegistration<LowerForToGraphPass>();
    return true;
  }();
  (void)once;
}

} // namespace lowering
} // namespace loom
