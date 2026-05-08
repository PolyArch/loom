// Lift sequential scf.for loops into scf.forall when the body is
// trivially parallel. The match criterion is intentionally conservative
// because false positives change semantics:
//   * Zero iter_args (lifting would break a loop-carried recurrence).
//   * Loop-invariant lb/ub/step.
//   * No unsupported control flow: no nested scf.while/execute_region,
//     no func.call / llvm.call (unknown callee side effects).
//   * Every memory write inside the body (recursively) goes to an
//     address whose computation transitively depends on the loop iv.
//     We do not prove pairwise-distinct addresses; we refuse to lift
//     loops whose stores are iv-independent (almost certainly racy).
//
// A matched scf.for is rewritten into an scf.forall over the same
// range with no shared_outs and no mapping attribute. The original
// integer-typed iv is recovered with an arith.index_cast inserted at
// the top of the new body so the original body ops continue to consume
// an integer iv as before.

#include "Frontend/Raising/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace {

// True if `v` is defined outside `loop` (a constant in the enclosing
// region, a func/forall block argument, etc).
bool isDefinedOutside(::mlir::Value v, ::mlir::scf::ForOp loop) {
  if (auto blockArg = ::mlir::dyn_cast<::mlir::BlockArgument>(v))
    return !loop->isAncestor(blockArg.getOwner()->getParentOp());
  ::mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  return !loop->isAncestor(def);
}

// True if `v` transitively depends on `iv`. We walk back through
// defining ops; values that come from outside the loop or from other
// block arguments terminate the search without matching. We deliberately
// stop at op boundaries we cannot see through (e.g. results of nested
// scf ops with implicit captures); the caller's sufficiency check on
// stores does not need maximal precision -- a false negative just leaves
// an scf.for in place.
bool dependsOnIV(::mlir::Value v, ::mlir::Value iv,
                 ::llvm::DenseSet<::mlir::Value> &visited) {
  if (v == iv)
    return true;
  if (!visited.insert(v).second)
    return false;
  if (::mlir::isa<::mlir::BlockArgument>(v))
    return false;
  ::mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  for (::mlir::Value operand : def->getOperands()) {
    if (dependsOnIV(operand, iv, visited))
      return true;
  }
  return false;
}

bool dependsOnIV(::mlir::Value v, ::mlir::Value iv) {
  ::llvm::DenseSet<::mlir::Value> visited;
  return dependsOnIV(v, iv, visited);
}

// True if `op` is an unconditional bail-out: a call, a while loop,
// inline asm, or anything with effects we cannot reason about safely
// for a parallel-iteration rewrite.
bool isBailOutOp(::mlir::Operation *op) {
  if (::mlir::isa<::mlir::scf::WhileOp, ::mlir::scf::ExecuteRegionOp>(op))
    return true;
  if (::mlir::isa<::mlir::func::CallOp, ::mlir::func::CallIndirectOp>(op))
    return true;
  if (::mlir::isa<::mlir::LLVM::CallOp, ::mlir::LLVM::InvokeOp,
                  ::mlir::LLVM::InlineAsmOp>(op))
    return true;
  return false;
}

// True if `op` is a structural scf op (or an scf terminator we treat
// transparently). Such ops are themselves neither a memory write nor
// a bail-out; the walk descends into their bodies and any meaningful
// effects are picked up there.
bool isTransparentScfOp(::mlir::Operation *op) {
  return ::mlir::isa<::mlir::scf::ForOp, ::mlir::scf::IfOp,
                     ::mlir::scf::ForallOp, ::mlir::scf::YieldOp,
                     ::mlir::scf::InParallelOp,
                     ::mlir::scf::ConditionOp>(op);
}

// Returns the list of values that determine where a write-only op
// stores to. For llvm.store this is just the (already-computed)
// pointer operand; for memref.store and memref.atomic_rmw it is the
// indices (the memref base is loop-invariant by construction). An
// empty list signals "this is not a write op we recognise". Reads are
// not considered (parallel reads are safe).
//
// We deliberately return a SmallVector by value because the operand
// ranges differ in shape between the two dialects.
::llvm::SmallVector<::mlir::Value, 4> getStoreAddressOperands(
    ::mlir::Operation *op) {
  ::llvm::SmallVector<::mlir::Value, 4> result;
  if (auto store = ::mlir::dyn_cast<::mlir::LLVM::StoreOp>(op)) {
    result.push_back(store.getAddr());
    return result;
  }
  if (auto store = ::mlir::dyn_cast<::mlir::memref::StoreOp>(op)) {
    for (::mlir::Value idx : store.getIndices())
      result.push_back(idx);
    return result;
  }
  if (auto atomic = ::mlir::dyn_cast<::mlir::memref::AtomicRMWOp>(op)) {
    for (::mlir::Value idx : atomic.getIndices())
      result.push_back(idx);
    return result;
  }
  return result;
}

// Walk the body of `loop` (recursively into nested regions) and verify:
//   1) No bail-out op (call, while, execute_region, inline asm).
//   2) Every memory-write op writes to an address that depends on
//      `iv`, the induction variable of `loop`. Reads are ignored.
//   3) Other side-effecting ops we don't recognise as a clean store
//      (e.g. llvm.intr.lifetime.*, atomics, volatile loads) cause a
//      bail-out: we are conservative.
::mlir::LogicalResult checkBodyParallel(::mlir::scf::ForOp loop) {
  ::mlir::Value iv = loop.getInductionVar();
  auto walkResult = loop.getBody()->walk([&](::mlir::Operation *op) {
    if (op == loop.getBody()->getTerminator())
      return ::mlir::WalkResult::advance();
    if (isBailOutOp(op))
      return ::mlir::WalkResult::interrupt();

    // Structural scf ops are transparent: they themselves are neither a
    // memory write nor a bail-out. The walk descends into their bodies
    // and any meaningful effects (loads, stores, calls) are visited
    // independently.
    if (isTransparentScfOp(op))
      return ::mlir::WalkResult::advance();

    // Pure ops (including arith, math, llvm.getelementptr, llvm.trunc,
    // llvm.uitofp, llvm.intr.fmuladd, ...) and pure-read ops (loads
    // that only read) do not constrain parallelism here. We allow
    // them through.
    if (::mlir::isMemoryEffectFree(op))
      return ::mlir::WalkResult::advance();

    // Read-only ops are parallel-safe across iterations.
    if (auto load = ::mlir::dyn_cast<::mlir::LLVM::LoadOp>(op)) {
      if (load.getVolatile_())
        return ::mlir::WalkResult::interrupt();
      return ::mlir::WalkResult::advance();
    }
    if (::mlir::isa<::mlir::memref::LoadOp>(op))
      return ::mlir::WalkResult::advance();

    // Stores: at least one address operand must depend on the iv.
    // (For memref.store the operands are the indices; for llvm.store
    // it is the precomputed pointer.) If every address operand is
    // loop-invariant, the store would write the same address from
    // every iteration -- a race -- so we bail out.
    auto addrOperands = getStoreAddressOperands(op);
    if (!addrOperands.empty()) {
      if (auto store = ::mlir::dyn_cast<::mlir::LLVM::StoreOp>(op)) {
        if (store.getVolatile_())
          return ::mlir::WalkResult::interrupt();
      }
      bool sawIvDependence = false;
      for (::mlir::Value v : addrOperands) {
        if (dependsOnIV(v, iv)) {
          sawIvDependence = true;
          break;
        }
      }
      if (!sawIvDependence)
        return ::mlir::WalkResult::interrupt();
      return ::mlir::WalkResult::advance();
    }

    // Lifetime markers and similar -- we cannot prove they are
    // safe, so bail. Keep this branch conservative.
    if (auto intr = op->getName().getStringRef();
        intr.starts_with("llvm.intr.lifetime"))
      return ::mlir::WalkResult::interrupt();

    // Unknown side-effecting op: bail out.
    return ::mlir::WalkResult::interrupt();
  });
  if (walkResult.wasInterrupted())
    return ::mlir::failure();
  return ::mlir::success();
}

// Materialise an Index-typed value from `v`. If `v` is already index,
// return it unchanged; otherwise insert an arith.index_cast.
::mlir::Value toIndex(::mlir::OpBuilder &builder, ::mlir::Location loc,
                      ::mlir::Value v) {
  if (::mlir::isa<::mlir::IndexType>(v.getType()))
    return v;
  return ::mlir::arith::IndexCastOp::create(
      builder, loc, builder.getIndexType(), v);
}

// Rewrite a parallel scf.for into scf.forall.
struct ForToForall : public ::mlir::OpRewritePattern<::mlir::scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::scf::ForOp loop,
                  ::mlir::PatternRewriter &rewriter) const override {
    // No iter_args: pure parallel-shape only.
    if (!loop.getInitArgs().empty())
      return ::mlir::failure();

    // Bounds and step must be loop-invariant.
    ::mlir::Value lb = loop.getLowerBound();
    ::mlir::Value ub = loop.getUpperBound();
    ::mlir::Value step = loop.getStep();
    if (!isDefinedOutside(lb, loop) || !isDefinedOutside(ub, loop) ||
        !isDefinedOutside(step, loop))
      return ::mlir::failure();

    // The bound/step types must be representable as Index. arith
    // requires lhs/rhs share types; since for already enforces this,
    // checking lb is enough.
    ::mlir::Type ivType = lb.getType();
    if (!::mlir::isa<::mlir::IndexType, ::mlir::IntegerType>(ivType))
      return ::mlir::failure();

    // Body must be parallel-safe.
    if (failed(checkBodyParallel(loop)))
      return ::mlir::failure();

    ::mlir::Location loc = loop.getLoc();
    ::mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(loop);

    // Cast bounds/step to index for scf.forall.
    ::mlir::Value lbIdx = toIndex(rewriter, loc, lb);
    ::mlir::Value ubIdx = toIndex(rewriter, loc, ub);
    ::mlir::Value stepIdx = toIndex(rewriter, loc, step);

    ::llvm::SmallVector<::mlir::OpFoldResult, 1> lbs{lbIdx};
    ::llvm::SmallVector<::mlir::OpFoldResult, 1> ubs{ubIdx};
    ::llvm::SmallVector<::mlir::OpFoldResult, 1> steps{stepIdx};

    auto forallOp = ::mlir::scf::ForallOp::create(
        rewriter, loc, lbs, ubs, steps, /*outputs=*/::mlir::ValueRange{},
        /*mapping=*/std::nullopt);

    // The block builder used by ForallOp::create only sets up a body
    // block with the iv argument; we need to migrate the original
    // body into it. forallOp.getBody() has rank-many iv arguments
    // followed by the shared-out arguments (none here).
    ::mlir::Block *forallBody = forallOp.getBody();
    ::mlir::Value forallIv = forallOp.getInductionVar(0);

    ::mlir::OpBuilder::InsertionGuard innerGuard(rewriter);
    rewriter.setInsertionPointToStart(forallBody);
    ::mlir::Value ivAsOriginal = forallIv;
    if (forallIv.getType() != ivType) {
      ivAsOriginal = ::mlir::arith::IndexCastOp::create(
          rewriter, loc, ivType, forallIv);
    }

    // Map the original loop's iv to the recovered integer iv. Other
    // captures (values defined outside the loop) flow through unchanged
    // since the body is moved, not cloned.
    ::mlir::Block *origBody = loop.getBody();
    ::mlir::IRMapping mapping;
    mapping.map(loop.getInductionVar(), ivAsOriginal);

    // Clone every op except the trailing scf.yield (which had no
    // operands since iter_args is empty); the in_parallel terminator
    // already exists in the new body.
    for (::mlir::Operation &op : origBody->without_terminator()) {
      rewriter.clone(op, mapping);
    }

    // Remove the original scf.for; it had no results.
    rewriter.eraseOp(loop);
    return ::mlir::success();
  }
};

struct SCFForToForallPass
    : public ::mlir::PassWrapper<SCFForToForallPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SCFForToForallPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-scf-for-to-forall";
  }
  ::llvm::StringRef getDescription() const final {
    return "Lift trivially parallel scf.for loops (no iter_args, "
           "iv-dependent stores only) into scf.forall so downstream "
           "lowerings can see parallel intent natively.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::scf::SCFDialect>();
  }

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();
    ::mlir::MLIRContext *ctx = &getContext();

    ::mlir::RewritePatternSet patterns(ctx);
    patterns.add<ForToForall>(ctx);

    if (failed(::mlir::applyPatternsGreedily(module, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace loom {
namespace raising {

std::unique_ptr<::mlir::Pass> createSCFForToForallPass() {
  return std::make_unique<SCFForToForallPass>();
}

void registerSCFForToForallPass() {
  static bool once = []() {
    ::mlir::PassRegistration<SCFForToForallPass>();
    return true;
  }();
  (void)once;
}

} // namespace raising
} // namespace loom
