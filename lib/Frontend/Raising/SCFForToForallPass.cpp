// Lift sequential scf.for loops into scf.forall when the body matches a
// strict, syntactic affine-map shape that we can prove parallel without
// dataflow analysis. The match criterion is intentionally conservative
// because false positives change semantics:
//   * Zero iter_args (lifting would break a loop-carried recurrence).
//   * Loop-invariant lb/ub/step.
//   * No unsupported control flow inside the body: no execute_region,
//     no func.call / llvm.call to non-pure callees, no
//     freestanding cf.cond_br / cf.switch, no inline assembly.
//   * No volatile or atomic / monotonic memory ops.
//   * No memory-effect intrinsic other than `llvm.intr.lifetime.{start,end}`.
//   * Reads and writes inside the body must touch disjoint base pointers,
//     except for narrow in-place elementwise loops where every same-base
//     read/write uses the exact same iv-derived address. The base is
//     recovered by walking back through llvm.getelementptr chains and
//     memref.subview chains until a non-pointer-derived value is reached;
//     that value is the base we compare.
//   * Each store's address expression must depend on the iv (or on iv +
//     loop-invariant operands), with NO rem / mod / div, no nonlinear
//     operators, and no pointer-table indirection (no load-of-pointer fed
//     into the store base). This is a syntactic affine-style check, not
//     a polyhedral dependence proof; that is enough to recognise the
//     vecadd / gemm / conv1d input-init / output-init shapes.
//
// A matched scf.for is rewritten into an scf.forall over the same range
// with no shared_outs and no mapping attribute. The original integer-
// typed iv is recovered with an arith.index_cast inserted at the top of
// the new body so the original body ops continue to consume an integer
// iv as before.
//
// A loop that does not satisfy the matcher remains an scf.for so later
// lowerings can preserve its sequential semantics.

#include "Frontend/Raising/Passes.h"

#include "CallableRegions.h"
#include "ExactRewrite.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

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

// True if the index-computation tree rooted at `v` is "syntactic affine
// style": only iv, loop-invariant values, sub-loop induction variables
// (block args of nested scf.for / scf.forall / scf.if), integer
// index_cast / sext / zext / trunc, addi, subi, muli, shli (constant
// shift), and llvm.getelementptr nodes built from affine-style
// operands. Forbids signed/unsigned remainders, divisions, arbitrary
// loads (no pointer-table indirection feeding the store base), and
// arbitrary calls. A leaf is acceptable when it is the iv, a
// loop-invariant value, a constant, or a block arg of a nested SCF
// region; the dependsOnIV check the caller runs separately verifies
// that the iv actually appears in the expression.
bool isAffineStyle(::mlir::Value v, ::mlir::Value iv, ::mlir::scf::ForOp loop,
                   ::llvm::DenseSet<::mlir::Value> &visited) {
  if (v == iv)
    return true;
  if (isDefinedOutside(v, loop))
    return true;
  if (!visited.insert(v).second)
    return true; // already visited via another path
  if (auto blockArg = ::mlir::dyn_cast<::mlir::BlockArgument>(v)) {
    // Block argument inside the loop body that is neither the iv nor
    // loop-invariant. Accept it only when it is the induction variable
    // of a nested scf.for / scf.forall / scf.if (where it is bounded
    // and produces parallel-safe addresses). Reject when it is an
    // iter_arg of a nested scf.for (recurrence) or an scf.while arg
    // (irreducible control).
    ::mlir::Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (auto nestedFor = ::mlir::dyn_cast<::mlir::scf::ForOp>(parentOp)) {
      // Only accept the iv (argument 0); iter_args (args 1..) carry
      // recurrences and we cannot prove they are independent across
      // outer iterations without a proper dependence analysis.
      return blockArg == nestedFor.getInductionVar();
    }
    if (::mlir::isa<::mlir::scf::ForallOp>(parentOp))
      return true; // forall ivs and shared_outs are both affine-safe
    if (::mlir::isa<::mlir::scf::IfOp>(parentOp))
      return false; // scf.if has no block args inside its branches
    return false;
  }
  ::mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  // Allow casts that preserve numeric value semantics.
  if (::mlir::isa<::mlir::arith::IndexCastOp, ::mlir::arith::IndexCastUIOp,
                  ::mlir::arith::ExtSIOp, ::mlir::arith::ExtUIOp,
                  ::mlir::arith::TruncIOp, ::mlir::LLVM::SExtOp,
                  ::mlir::LLVM::ZExtOp, ::mlir::LLVM::TruncOp>(def)) {
    return isAffineStyle(def->getOperand(0), iv, loop, visited);
  }
  // Allow add / sub / mul / shl with constant shift amount.
  if (::mlir::isa<::mlir::arith::AddIOp, ::mlir::arith::SubIOp,
                  ::mlir::arith::MulIOp, ::mlir::arith::ShLIOp,
                  ::mlir::LLVM::AddOp, ::mlir::LLVM::SubOp, ::mlir::LLVM::MulOp,
                  ::mlir::LLVM::ShlOp>(def)) {
    for (::mlir::Value op : def->getOperands()) {
      if (!isAffineStyle(op, iv, loop, visited))
        return false;
    }
    return true;
  }
  // Allow llvm.getelementptr -- the resulting pointer is built from
  // affine-style integer operands. We recurse into every operand
  // including the base (the base must come from outside the loop).
  if (auto gep = ::mlir::dyn_cast<::mlir::LLVM::GEPOp>(def)) {
    for (::mlir::Value op : gep->getOperands()) {
      if (!isAffineStyle(op, iv, loop, visited))
        return false;
    }
    return true;
  }
  // Constants are always acceptable.
  if (::mlir::isa<::mlir::arith::ConstantOp, ::mlir::LLVM::ConstantOp>(def))
    return true;
  // Anything else (including div / rem / loads / calls / selects) is
  // rejected.
  return false;
}

bool isAffineStyle(::mlir::Value v, ::mlir::Value iv, ::mlir::scf::ForOp loop) {
  ::llvm::DenseSet<::mlir::Value> visited;
  return isAffineStyle(v, iv, loop, visited);
}

// Walk the LLVM GEP / memref subview chain rooted at `v` back to its
// base SSA value. The base is the first value reached that is not an
// llvm.getelementptr / memref.subview / memref.cast / memref.reshape on
// a pointer-typed input. For our purposes ANY value that is loop-
// invariant or comes from a non-pointer-derivation op is the base.
::mlir::Value getMemoryBase(::mlir::Value v) {
  while (::mlir::Operation *def = v.getDefiningOp()) {
    if (auto gep = ::mlir::dyn_cast<::mlir::LLVM::GEPOp>(def)) {
      v = gep.getBase();
      continue;
    }
    if (auto subview = ::mlir::dyn_cast<::mlir::memref::SubViewOp>(def)) {
      v = subview.getSource();
      continue;
    }
    if (auto cast = ::mlir::dyn_cast<::mlir::memref::CastOp>(def)) {
      v = cast.getSource();
      continue;
    }
    if (auto bitcast = ::mlir::dyn_cast<::mlir::LLVM::BitcastOp>(def)) {
      v = bitcast.getOperand();
      continue;
    }
    if (auto addrSpace = ::mlir::dyn_cast<::mlir::LLVM::AddrSpaceCastOp>(def)) {
      v = addrSpace.getOperand();
      continue;
    }
    break;
  }
  return v;
}

// Returns the "memory pointer / memref" value of `op` if it is a
// recognised load. Returns null if `op` is not a load-like op. We also
// flag the volatile bit via the `outVolatile` flag when relevant.
::mlir::Value getLoadPointer(::mlir::Operation *op, bool &outVolatile) {
  outVolatile = false;
  if (auto load = ::mlir::dyn_cast<::mlir::LLVM::LoadOp>(op)) {
    outVolatile = load.getVolatile_();
    return load.getAddr();
  }
  if (auto load = ::mlir::dyn_cast<::mlir::memref::LoadOp>(op)) {
    return load.getMemRef();
  }
  return {};
}

// Returns the "memory pointer / memref" value of `op` if it is a
// recognised store. Returns null otherwise. Flags volatile via the
// outVolatile parameter.
::mlir::Value getStorePointer(::mlir::Operation *op, bool &outVolatile) {
  outVolatile = false;
  if (auto store = ::mlir::dyn_cast<::mlir::LLVM::StoreOp>(op)) {
    outVolatile = store.getVolatile_();
    return store.getAddr();
  }
  if (auto store = ::mlir::dyn_cast<::mlir::memref::StoreOp>(op)) {
    return store.getMemRef();
  }
  return {};
}

// Returns the SSA values that determine the address a store-like op
// writes to. For llvm.store this is the (already-computed) pointer
// operand; for memref.store it is the indices (the memref base is
// loop-invariant by construction). The caller uses these to verify the
// address depends on the iv and is syntactic affine. Returns an empty
// list when `op` is not recognised.
::llvm::SmallVector<::mlir::Value, 4>
getStoreAddressOperands(::mlir::Operation *op) {
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
  return result;
}

struct LinearExpr {
  int64_t ivCoeff = 0;
  int64_t constant = 0;
};

int64_t abs64(int64_t value) { return value < 0 ? -value : value; }

int64_t positiveMod(int64_t value, int64_t modulus) {
  int64_t residue = value % modulus;
  return residue < 0 ? residue + modulus : residue;
}

std::optional<int64_t> getConstantInt(::mlir::Value value) {
  if (auto constant = value.getDefiningOp<::mlir::arith::ConstantOp>()) {
    if (auto intAttr =
            ::llvm::dyn_cast<::mlir::IntegerAttr>(constant.getValue()))
      return intAttr.getInt();
  }
  return std::nullopt;
}

std::optional<LinearExpr> addLinear(LinearExpr lhs, LinearExpr rhs) {
  return LinearExpr{lhs.ivCoeff + rhs.ivCoeff, lhs.constant + rhs.constant};
}

std::optional<LinearExpr> subLinear(LinearExpr lhs, LinearExpr rhs) {
  return LinearExpr{lhs.ivCoeff - rhs.ivCoeff, lhs.constant - rhs.constant};
}

std::optional<LinearExpr> scaleLinear(LinearExpr expr, int64_t scale) {
  return LinearExpr{expr.ivCoeff * scale, expr.constant * scale};
}

std::optional<LinearExpr> linearExpr(::mlir::Value value, ::mlir::Value iv,
                                     ::mlir::scf::ForOp loop,
                                     ::llvm::DenseSet<::mlir::Value> &visited) {
  if (value == iv)
    return LinearExpr{1, 0};
  if (auto constant = getConstantInt(value))
    return LinearExpr{0, *constant};
  if (isDefinedOutside(value, loop))
    return std::nullopt;
  if (!visited.insert(value).second)
    return std::nullopt;
  if (::mlir::isa<::mlir::BlockArgument>(value))
    return std::nullopt;

  ::mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;

  if (::mlir::isa<::mlir::arith::IndexCastOp, ::mlir::arith::IndexCastUIOp,
                  ::mlir::arith::ExtSIOp, ::mlir::arith::ExtUIOp,
                  ::mlir::arith::TruncIOp, ::mlir::LLVM::SExtOp,
                  ::mlir::LLVM::ZExtOp, ::mlir::LLVM::TruncOp>(def))
    return linearExpr(def->getOperand(0), iv, loop, visited);

  if (::mlir::isa<::mlir::arith::AddIOp, ::mlir::LLVM::AddOp>(def)) {
    auto lhs = linearExpr(def->getOperand(0), iv, loop, visited);
    auto rhs = linearExpr(def->getOperand(1), iv, loop, visited);
    if (!lhs || !rhs)
      return std::nullopt;
    return addLinear(*lhs, *rhs);
  }

  if (::mlir::isa<::mlir::arith::SubIOp, ::mlir::LLVM::SubOp>(def)) {
    auto lhs = linearExpr(def->getOperand(0), iv, loop, visited);
    auto rhs = linearExpr(def->getOperand(1), iv, loop, visited);
    if (!lhs || !rhs)
      return std::nullopt;
    return subLinear(*lhs, *rhs);
  }

  if (::mlir::isa<::mlir::arith::MulIOp, ::mlir::LLVM::MulOp>(def)) {
    auto lhsConst = getConstantInt(def->getOperand(0));
    auto rhsConst = getConstantInt(def->getOperand(1));
    if (lhsConst) {
      auto rhs = linearExpr(def->getOperand(1), iv, loop, visited);
      if (!rhs)
        return std::nullopt;
      return scaleLinear(*rhs, *lhsConst);
    }
    if (rhsConst) {
      auto lhs = linearExpr(def->getOperand(0), iv, loop, visited);
      if (!lhs)
        return std::nullopt;
      return scaleLinear(*lhs, *rhsConst);
    }
    return std::nullopt;
  }

  if (::mlir::isa<::mlir::arith::ShLIOp, ::mlir::LLVM::ShlOp>(def)) {
    auto shift = getConstantInt(def->getOperand(1));
    if (!shift || *shift < 0 || *shift >= 62)
      return std::nullopt;
    auto lhs = linearExpr(def->getOperand(0), iv, loop, visited);
    if (!lhs)
      return std::nullopt;
    return scaleLinear(*lhs, int64_t{1} << *shift);
  }

  return std::nullopt;
}

std::optional<LinearExpr> linearExpr(::mlir::Value value, ::mlir::Value iv,
                                     ::mlir::scf::ForOp loop) {
  ::llvm::DenseSet<::mlir::Value> visited;
  return linearExpr(value, iv, loop, visited);
}

std::optional<LinearExpr> linearGepOffset(::mlir::LLVM::GEPOp gep,
                                          ::mlir::Value iv,
                                          ::mlir::scf::ForOp loop) {
  LinearExpr total;
  bool sawIndex = false;
  for (::mlir::Value operand : gep->getOperands()) {
    if (operand == gep.getBase())
      continue;
    auto expr = linearExpr(operand, iv, loop);
    if (!expr)
      return std::nullopt;
    total = *addLinear(total, *expr);
    sawIndex = true;
  }
  if (!sawIndex)
    return std::nullopt;
  return total;
}

std::optional<LinearExpr> storeLinearAddress(::mlir::Operation *store,
                                             ::mlir::scf::ForOp loop) {
  ::mlir::Value iv = loop.getInductionVar();
  if (auto memrefStore = ::mlir::dyn_cast<::mlir::memref::StoreOp>(store)) {
    if (memrefStore.getIndices().size() != 1)
      return std::nullopt;
    return linearExpr(memrefStore.getIndices().front(), iv, loop);
  }
  if (auto llvmStore = ::mlir::dyn_cast<::mlir::LLVM::StoreOp>(store)) {
    ::mlir::Value addr = llvmStore.getAddr();
    if (auto gep = addr.getDefiningOp<::mlir::LLVM::GEPOp>())
      return linearGepOffset(gep, iv, loop);
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<LinearExpr> memoryLinearAddress(::mlir::Operation *op,
                                              ::mlir::scf::ForOp loop) {
  if (auto memrefLoad = ::mlir::dyn_cast<::mlir::memref::LoadOp>(op)) {
    if (memrefLoad.getIndices().size() != 1)
      return std::nullopt;
    return linearExpr(memrefLoad.getIndices().front(), loop.getInductionVar(),
                      loop);
  }
  if (auto memrefStore = ::mlir::dyn_cast<::mlir::memref::StoreOp>(op)) {
    if (memrefStore.getIndices().size() != 1)
      return std::nullopt;
    return linearExpr(memrefStore.getIndices().front(), loop.getInductionVar(),
                      loop);
  }
  if (auto llvmLoad = ::mlir::dyn_cast<::mlir::LLVM::LoadOp>(op)) {
    if (auto gep = llvmLoad.getAddr().getDefiningOp<::mlir::LLVM::GEPOp>())
      return linearGepOffset(gep, loop.getInductionVar(), loop);
    return std::nullopt;
  }
  if (auto llvmStore = ::mlir::dyn_cast<::mlir::LLVM::StoreOp>(op)) {
    if (auto gep = llvmStore.getAddr().getDefiningOp<::mlir::LLVM::GEPOp>())
      return linearGepOffset(gep, loop.getInductionVar(), loop);
    return std::nullopt;
  }
  return std::nullopt;
}

bool sameLinearExpr(LinearExpr lhs, LinearExpr rhs) {
  return lhs.ivCoeff == rhs.ivCoeff && lhs.constant == rhs.constant;
}

bool sameBaseReadWriteAccessesAreIterationLocal(
    ::llvm::ArrayRef<::mlir::Operation *> loads,
    ::llvm::ArrayRef<::mlir::Operation *> stores, ::mlir::scf::ForOp loop) {
  if (loads.empty() || stores.empty())
    return true;

  std::optional<LinearExpr> first;
  for (::mlir::Operation *op :
       ::llvm::concat<::mlir::Operation *const>(loads, stores)) {
    auto expr = memoryLinearAddress(op, loop);
    if (!expr || expr->ivCoeff == 0)
      return false;
    if (!first) {
      first = *expr;
      continue;
    }
    if (!sameLinearExpr(*first, *expr))
      return false;
  }
  return true;
}

bool sameBaseStoresAreLaneDisjoint(::llvm::ArrayRef<::mlir::Operation *> stores,
                                   ::mlir::scf::ForOp loop) {
  if (stores.size() <= 1)
    return true;
  auto stepConst = getConstantInt(loop.getStep());
  if (!stepConst || *stepConst == 0)
    return false;

  std::optional<int64_t> expectedCoeff;
  ::llvm::DenseSet<int64_t> residues;
  int64_t stride = 0;
  for (::mlir::Operation *store : stores) {
    auto expr = storeLinearAddress(store, loop);
    if (!expr || expr->ivCoeff == 0)
      return false;
    if (!expectedCoeff) {
      expectedCoeff = expr->ivCoeff;
      stride = abs64(expr->ivCoeff * *stepConst);
      if (stride <= 1)
        return false;
    } else if (*expectedCoeff != expr->ivCoeff) {
      return false;
    }
    int64_t residue = positiveMod(expr->constant, stride);
    if (!residues.insert(residue).second)
      return false;
  }
  return true;
}

// True if `op` is a call to a callee we cannot model. Pure callees (in
// the MLIR memory-effect sense -- MemoryEffects::None) are allowed
// because a parallel iteration that calls a pure function is still
// parallel.
bool isUnmodelledCall(::mlir::Operation *op) {
  if (!::mlir::isa<::mlir::func::CallOp, ::mlir::func::CallIndirectOp,
                   ::mlir::LLVM::CallOp, ::mlir::LLVM::InvokeOp,
                   ::mlir::LLVM::InlineAsmOp>(op))
    return false;
  // Pure call: the op declares no memory effects via the
  // MemoryEffectOpInterface. (LLVM intrinsics that are pure typically
  // model this; arbitrary calls do not.)
  if (auto memOp = ::mlir::dyn_cast<::mlir::MemoryEffectOpInterface>(op)) {
    if (memOp.hasNoEffect())
      return false;
  }
  return true;
}

// True if `op` is a structural body operation treated transparently. Such ops
// are themselves neither a memory write nor a bail-out; the ownership walk
// descends into their regions and checks the operations there. builtin.module
// is the one non-SCF entry because it contributes a symbol scope, not an
// execution effect of its own. scf.execute_region is rejected separately
// because it can hide arbitrary control flow.
bool isTransparentBodyOp(::mlir::Operation *op) {
  return ::mlir::isa<::mlir::ModuleOp, ::mlir::scf::ForOp, ::mlir::scf::IfOp,
                     ::mlir::scf::WhileOp, ::mlir::scf::ForallOp,
                     ::mlir::scf::YieldOp, ::mlir::scf::InParallelOp,
                     ::mlir::scf::ConditionOp>(op);
}

// True when the loop body contains any block with more than one
// successor (e.g. a free-standing cf.cond_br, cf.switch, or
// llvm.cond_br). A nested scf.for / scf.if is fine because the
// successor counts of its own internal blocks are not the outer body's
// concern. We only check blocks that belong to `loop.getBody()` and to regions
// of nested transparent body operations. A nested callable owns its own body
// and is pruned before either check can inspect it.
bool bodyHasMultipleSuccessorTerminator(::mlir::scf::ForOp loop) {
  ::mlir::WalkResult walked = loom::raising::forEachOwnedOperation(
      loop.getRegion(), [&](::mlir::Operation *op) {
        if (op == loop.getBody()->getTerminator())
          return ::mlir::WalkResult::advance();
        if (op->getNumSuccessors() <= 1)
          return ::mlir::WalkResult::advance();
        // Allow scf transparent ops -- they have no successors at the cf
        // level (they yield). The check for `op->getNumSuccessors()` already
        // protects against that since the SCF ops do not list terminator
        // successors.
        return ::mlir::WalkResult::interrupt();
      });
  return walked.wasInterrupted();
}

// Walk the body of `loop` (recursively into nested regions) and verify:
//   1) No bail-out op (call to non-pure callee, execute_region,
//      inline asm, llvm.invoke).
//   2) No volatile / atomic memory op.
//   3) Reads and writes use disjoint base pointers, or a narrow
//      same-element in-place form.
//   4) Each store's address expression is "syntactic affine style"
//      (depends on iv + loop-invariants only, no rem/mod/load).
//   5) No memory-effect intrinsic except llvm.intr.lifetime.{start,end}.
//   6) Body has no block with more than one successor.
::mlir::LogicalResult checkBodyParallel(::mlir::scf::ForOp loop) {
  ::mlir::Value iv = loop.getInductionVar();
  ::llvm::DenseSet<::mlir::Value> readBases;
  ::llvm::DenseSet<::mlir::Value> writeBases;
  ::llvm::DenseMap<::mlir::Value, ::llvm::SmallVector<::mlir::Operation *, 4>>
      loadsByBase;
  ::llvm::SmallVector<::mlir::Operation *, 8> stores;
  ::llvm::DenseMap<::mlir::Value, ::llvm::SmallVector<::mlir::Operation *, 4>>
      storesByBase;

  if (bodyHasMultipleSuccessorTerminator(loop))
    return ::mlir::failure();

  auto walkResult = loom::raising::forEachOwnedOperation(
      loop.getRegion(), [&](::mlir::Operation *op) {
        if (op == loop.getBody()->getTerminator())
          return ::mlir::WalkResult::advance();
        // Reject scf.execute_region inside the body.
        if (::mlir::isa<::mlir::scf::ExecuteRegionOp>(op))
          return ::mlir::WalkResult::interrupt();
        // Reject calls to non-pure callees and inline asm / invoke.
        if (isUnmodelledCall(op))
          return ::mlir::WalkResult::interrupt();
        // Atomic ops are conservative bail-outs.
        if (::mlir::isa<
                ::mlir::LLVM::AtomicRMWOp, ::mlir::LLVM::AtomicCmpXchgOp,
                ::mlir::memref::AtomicRMWOp, ::mlir::memref::AtomicYieldOp>(op))
          return ::mlir::WalkResult::interrupt();

        // builtin.module is a structural symbol scope. Its contents are still
        // visited here, while nested callable bodies are pruned by the common
        // ownership traversal.
        if (isTransparentBodyOp(op))
          return ::mlir::WalkResult::advance();

        // Pure ops do not constrain parallelism here.
        if (::mlir::isMemoryEffectFree(op))
          return ::mlir::WalkResult::advance();

        // Read-only ops: capture the base pointer for read/write disjoint
        // analysis below. Reject volatile loads.
        bool isVol = false;
        if (::mlir::Value loadPtr = getLoadPointer(op, isVol)) {
          if (isVol)
            return ::mlir::WalkResult::interrupt();
          ::mlir::Value base = getMemoryBase(loadPtr);
          readBases.insert(base);
          loadsByBase[base].push_back(op);
          return ::mlir::WalkResult::advance();
        }

        // Store ops: capture the base for disjoint analysis, address must
        // be syntactic affine. Reject volatile stores.
        if (::mlir::Value storePtr = getStorePointer(op, isVol)) {
          if (isVol)
            return ::mlir::WalkResult::interrupt();
          ::mlir::Value base = getMemoryBase(storePtr);
          writeBases.insert(base);
          stores.push_back(op);
          storesByBase[base].push_back(op);
          return ::mlir::WalkResult::advance();
        }

        // Lifetime markers are explicitly fine.
        if (auto name = op->getName().getStringRef();
            name == "llvm.intr.lifetime.start" ||
            name == "llvm.intr.lifetime.end")
          return ::mlir::WalkResult::advance();

        // Unknown side-effecting op (including other LLVM memory-effect
        // intrinsics like memcpy / memset): bail out.
        return ::mlir::WalkResult::interrupt();
      });
  if (walkResult.wasInterrupted())
    return ::mlir::failure();

  // A store-side address may share a base with reads only for the
  // same-element in-place form. Shifted read/write forms keep the loop
  // serial because they carry a cross-iteration dependence.
  for (::mlir::Value w : writeBases) {
    if (readBases.count(w) &&
        !sameBaseReadWriteAccessesAreIterationLocal(
            loadsByBase.lookup(w), storesByBase.lookup(w), loop))
      return ::mlir::failure();
  }
  // WAW: same-base stores are allowed only for fixed-width lane groups,
  // such as out[3*i + {0,1,2}], where the per-iteration address residue
  // classes are provably disjoint. Anything not proved by this narrow
  // linear check remains serial.
  for (auto &entry : storesByBase) {
    if (!sameBaseStoresAreLaneDisjoint(entry.second, loop))
      return ::mlir::failure();
  }

  // Each store's address expression must be syntactic affine in iv.
  // For llvm.store the "address" is the precomputed ptr operand; for
  // memref.store the "address" is the index list, all of which must be
  // affine-style and at least one of which must depend on the iv.
  for (::mlir::Operation *st : stores) {
    auto addrOps = getStoreAddressOperands(st);
    if (addrOps.empty())
      return ::mlir::failure();
    bool sawIvDep = false;
    for (::mlir::Value v : addrOps) {
      if (!isAffineStyle(v, iv, loop))
        return ::mlir::failure();
      if (dependsOnIV(v, iv))
        sawIvDep = true;
    }
    if (!sawIvDep)
      return ::mlir::failure();
  }

  return ::mlir::success();
}

// Materialise an Index-typed value from `v`. If `v` is already index,
// return it unchanged; otherwise insert an arith.index_cast.
::mlir::Value toIndex(::mlir::OpBuilder &builder, ::mlir::Location loc,
                      ::mlir::Value v) {
  if (::mlir::isa<::mlir::IndexType>(v.getType()))
    return v;
  return ::mlir::arith::IndexCastOp::create(builder, loc,
                                            builder.getIndexType(), v);
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
      ivAsOriginal =
          ::mlir::arith::IndexCastOp::create(rewriter, loc, ivType, forallIv);
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
           "syntactic affine-style stores, disjoint read / write bases) "
           "into scf.forall so downstream lowerings can see parallel "
           "intent natively.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::scf::SCFDialect>();
  }

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();
    ::mlir::MLIRContext *ctx = &getContext();

    ::mlir::RewritePatternSet patterns(ctx);
    patterns.add<ForToForall>(ctx);
    ::mlir::FrozenRewritePatternSet frozen(std::move(patterns));

    (void)loom::raising::forEachCallableRegion(
        module, [&](::mlir::Region &region) {
          loom::raising::applyExactPatternsOnce(region, frozen);
          return ::mlir::success();
        });
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
