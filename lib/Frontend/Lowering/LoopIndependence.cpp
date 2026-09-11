#include "Frontend/Lowering/LoopIndependence.h"
#include "Common/IndexWidth.h"
#include "Frontend/Analysis/CallableRegions.h"
#include "Frontend/Analysis/DenseParallelMemoryProjection.h"
#include "Frontend/Analysis/MemoryProvenance.h"
#include "Frontend/Lowering/GraphMemoryAddressing.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <limits>
#include <optional>

namespace {

// True if `v` is defined outside `loop` (a constant in the enclosing
// region, a func/forall block argument, etc).
bool isDefinedOutside(::mlir::Value v, ::mlir::Operation *loop) {
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

std::optional<unsigned> fixedIntegerWidth(::mlir::Type type,
                                          ::mlir::Operation *anchor) {
  if (auto integer = ::mlir::dyn_cast<::mlir::IntegerType>(type))
    return integer.getWidth();
  if (::mlir::isa<::mlir::IndexType>(type)) {
    auto width = ::loom::getIndexBitWidth(anchor);
    if (!width) {
      ::llvm::consumeError(width.takeError());
      return std::nullopt;
    }
    return *width;
  }
  return std::nullopt;
}

bool isInjectiveCast(::mlir::Operation *operation, ::mlir::Operation *loop) {
  if (!operation || operation->getNumOperands() != 1 ||
      operation->getNumResults() != 1)
    return false;
  const auto sourceWidth =
      fixedIntegerWidth(operation->getOperand(0).getType(), loop);
  const auto resultWidth =
      fixedIntegerWidth(operation->getResult(0).getType(), loop);
  if (!sourceWidth || !resultWidth)
    return false;
  if (::mlir::isa<::mlir::arith::TruncIOp, ::mlir::LLVM::TruncOp>(operation))
    return false;
  if (::mlir::isa<::mlir::arith::ExtSIOp, ::mlir::arith::ExtUIOp,
                  ::mlir::LLVM::SExtOp, ::mlir::LLVM::ZExtOp>(operation))
    return *resultWidth >= *sourceWidth;
  if (::mlir::isa<::mlir::arith::IndexCastOp, ::mlir::arith::IndexCastUIOp>(
          operation))
    return *resultWidth >= *sourceWidth;
  return false;
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
    if (!isInjectiveCast(def, loop))
      return false;
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
  if (auto read = ::mlir::dyn_cast<::mlir::vector::TransferReadOp>(op))
    return read.getBase();
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
  if (auto write = ::mlir::dyn_cast<::mlir::vector::TransferWriteOp>(op))
    return write.getBase();
  return {};
}

/// One memref access of the loop body: its index operands and the number of
/// contiguous elements it touches from that index. Vector transfers are
/// admitted only in the exact rank-one minor-identity form that the graph
/// lowering accepts; a mask can only shrink the touched interval.
struct MemrefAccessGeometry final {
  ::mlir::Operation *op = nullptr;
  ::llvm::SmallVector<::mlir::Value, 4> indices;
  std::int64_t lanes = 1;
};

std::optional<MemrefAccessGeometry> memrefAccessGeometry(::mlir::Operation *op) {
  MemrefAccessGeometry access;
  access.op = op;
  if (auto load = ::mlir::dyn_cast<::mlir::memref::LoadOp>(op)) {
    access.indices.assign(load.getIndices().begin(), load.getIndices().end());
    return access;
  }
  if (auto store = ::mlir::dyn_cast<::mlir::memref::StoreOp>(op)) {
    access.indices.assign(store.getIndices().begin(),
                          store.getIndices().end());
    return access;
  }
  ::mlir::VectorType vector;
  ::mlir::AffineMap permutation;
  ::mlir::ValueRange indices;
  if (auto read = ::mlir::dyn_cast<::mlir::vector::TransferReadOp>(op)) {
    vector = read.getVectorType();
    permutation = read.getPermutationMap();
    indices = read.getIndices();
  } else if (auto write =
                 ::mlir::dyn_cast<::mlir::vector::TransferWriteOp>(op)) {
    vector = write.getVectorType();
    permutation = write.getPermutationMap();
    indices = write.getIndices();
  } else {
    return std::nullopt;
  }
  if (vector.getRank() != 1 || vector.isScalable() || indices.size() != 1 ||
      !permutation.isMinorIdentity())
    return std::nullopt;
  access.indices.assign(indices.begin(), indices.end());
  access.lanes = vector.getDimSize(0);
  return access;
}

struct LinearExpr {
  int64_t ivCoeff = 0;
  int64_t constant = 0;

  friend bool operator==(LinearExpr lhs, LinearExpr rhs) {
    return lhs.ivCoeff == rhs.ivCoeff && lhs.constant == rhs.constant;
  }
};

std::optional<std::int64_t> abs64(std::int64_t value) {
  if (value == std::numeric_limits<std::int64_t>::min())
    return std::nullopt;
  return value < 0 ? -value : value;
}

std::optional<std::int64_t> positiveMod(std::int64_t value,
                                        std::int64_t modulus) {
  if (modulus <= 0)
    return std::nullopt;
  std::int64_t residue = value % modulus;
  return residue < 0 ? residue + modulus : residue;
}

std::optional<int64_t> getConstantInt(::mlir::Value value) {
  if (auto constant = value.getDefiningOp<::mlir::arith::ConstantOp>()) {
    if (auto intAttr =
            ::llvm::dyn_cast<::mlir::IntegerAttr>(constant.getValue())) {
      const ::llvm::APInt &bits = intAttr.getValue();
      if (!bits.isSignedIntN(64))
        return std::nullopt;
      return bits.getSExtValue();
    }
  }
  return std::nullopt;
}

std::optional<int64_t>
getConstantIntExpression(::mlir::Value value,
                         ::llvm::DenseSet<::mlir::Value> &visited) {
  if (auto constant = getConstantInt(value))
    return constant;
  if (!visited.insert(value).second)
    return std::nullopt;
  ::mlir::Operation *definition = value.getDefiningOp();
  if (auto cast =
          ::mlir::dyn_cast_or_null<::mlir::arith::IndexCastOp>(definition)) {
    auto sourceWidth = fixedIntegerWidth(cast.getIn().getType(), definition);
    auto resultWidth = fixedIntegerWidth(cast.getType(), definition);
    if (!sourceWidth || !resultWidth || *sourceWidth != *resultWidth)
      return std::nullopt;
    return getConstantIntExpression(cast.getIn(), visited);
  }
  if (!definition || definition->getNumOperands() != 2)
    return std::nullopt;
  auto lhs = getConstantIntExpression(definition->getOperand(0), visited);
  auto rhs = getConstantIntExpression(definition->getOperand(1), visited);
  if (!lhs || !rhs)
    return std::nullopt;
  std::int64_t result = 0;
  if (::mlir::isa<::mlir::arith::AddIOp>(definition)) {
    if (__builtin_add_overflow(*lhs, *rhs, &result))
      return std::nullopt;
  } else if (::mlir::isa<::mlir::arith::SubIOp>(definition)) {
    if (__builtin_sub_overflow(*lhs, *rhs, &result))
      return std::nullopt;
  } else if (::mlir::isa<::mlir::arith::MulIOp>(definition)) {
    if (__builtin_mul_overflow(*lhs, *rhs, &result))
      return std::nullopt;
  } else {
    return std::nullopt;
  }
  auto width = fixedIntegerWidth(value.getType(), definition);
  if (!width || *width == 0 || *width > 64 ||
      !::llvm::APInt(64, static_cast<std::uint64_t>(result))
           .isSignedIntN(*width))
    return std::nullopt;
  return result;
}

std::optional<int64_t> getConstantIntExpression(::mlir::Value value) {
  ::llvm::DenseSet<::mlir::Value> visited;
  return getConstantIntExpression(value, visited);
}

std::optional<LinearExpr> addLinear(LinearExpr lhs, LinearExpr rhs) {
  LinearExpr result;
  if (__builtin_add_overflow(lhs.ivCoeff, rhs.ivCoeff, &result.ivCoeff) ||
      __builtin_add_overflow(lhs.constant, rhs.constant, &result.constant))
    return std::nullopt;
  return result;
}

std::optional<LinearExpr> subLinear(LinearExpr lhs, LinearExpr rhs) {
  LinearExpr result;
  if (__builtin_sub_overflow(lhs.ivCoeff, rhs.ivCoeff, &result.ivCoeff) ||
      __builtin_sub_overflow(lhs.constant, rhs.constant, &result.constant))
    return std::nullopt;
  return result;
}

std::optional<LinearExpr> scaleLinear(LinearExpr expr, int64_t scale) {
  LinearExpr result;
  if (__builtin_mul_overflow(expr.ivCoeff, scale, &result.ivCoeff) ||
      __builtin_mul_overflow(expr.constant, scale, &result.constant))
    return std::nullopt;
  return result;
}

std::optional<LinearExpr> linearExpr(::mlir::Value value, ::mlir::Value iv,
                                     ::mlir::Operation *loop,
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
    if (!isInjectiveCast(def, loop))
      return std::nullopt;
  if (::mlir::isa<::mlir::arith::IndexCastOp, ::mlir::arith::IndexCastUIOp,
                  ::mlir::arith::ExtSIOp, ::mlir::arith::ExtUIOp,
                  ::mlir::LLVM::SExtOp, ::mlir::LLVM::ZExtOp>(def))
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
                                     ::mlir::Operation *loop) {
  ::llvm::DenseSet<::mlir::Value> visited;
  return linearExpr(value, iv, loop, visited);
}

bool sameBaseMemrefReadWriteAccessesAreIterationLocal(
    ::llvm::ArrayRef<MemrefAccessGeometry> loads,
    ::llvm::ArrayRef<MemrefAccessGeometry> stores, ::mlir::scf::ForOp loop) {
  if (loads.empty() || stores.empty())
    return true;

  std::optional<LinearExpr> first;
  std::optional<std::int64_t> lanes;
  const auto sameElements = [&](const MemrefAccessGeometry &access) {
    if (access.indices.size() != 1)
      return false;
    auto expr =
        linearExpr(access.indices.front(), loop.getInductionVar(), loop);
    if (!expr || expr->ivCoeff == 0)
      return false;
    if (!first) {
      first = *expr;
      lanes = access.lanes;
    }
    return *first == *expr && *lanes == access.lanes;
  };
  return ::llvm::all_of(loads, sameElements) &&
         ::llvm::all_of(stores, sameElements);
}

bool sameBaseMemrefStoresAreLaneDisjoint(
    ::llvm::ArrayRef<MemrefAccessGeometry> stores, ::mlir::scf::ForOp loop) {
  if (stores.empty())
    return true;
  auto stepConst = getConstantInt(loop.getStep());
  if (!stepConst || *stepConst == 0)
    return false;
  if (stores.size() == 1) {
    const MemrefAccessGeometry &store = stores.front();
    for (::mlir::Value index : store.indices) {
      auto expr = linearExpr(index, loop.getInductionVar(), loop);
      if (!expr || expr->ivCoeff == 0)
        continue;
      // Consecutive iterations touch [e(i), e(i) + lanes); they are disjoint
      // exactly when the index advances by at least the lane count.
      std::int64_t advance = 0;
      if (__builtin_mul_overflow(expr->ivCoeff, *stepConst, &advance))
        return false;
      auto magnitude = abs64(advance);
      if (magnitude && *magnitude >= store.lanes)
        return true;
    }
    return false;
  }
  // Several stores to one base are admitted only in the scalar residue-class
  // form; overlapping vector lane groups keep the loop serial.
  if (::llvm::any_of(stores, [](const MemrefAccessGeometry &store) {
        return store.lanes != 1;
      }))
    return false;

  std::optional<int64_t> expectedCoeff;
  ::llvm::DenseSet<int64_t> residues;
  int64_t stride = 0;
  for (const MemrefAccessGeometry &store : stores) {
    if (store.indices.size() != 1)
      return false;
    auto expr =
        linearExpr(store.indices.front(), loop.getInductionVar(), loop);
    if (!expr || expr->ivCoeff == 0)
      return false;
    if (!expectedCoeff) {
      expectedCoeff = expr->ivCoeff;
      std::int64_t scaled = 0;
      if (__builtin_mul_overflow(expr->ivCoeff, *stepConst, &scaled))
        return false;
      auto magnitude = abs64(scaled);
      if (!magnitude)
        return false;
      stride = *magnitude;
      if (stride == 0 || (stores.size() > 1 && stride <= 1))
        return false;
    } else if (*expectedCoeff != expr->ivCoeff) {
      return false;
    }
    auto residue = positiveMod(expr->constant, stride);
    if (!residue || !residues.insert(*residue).second)
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
  ::mlir::WalkResult walked = loom::frontend::analysis::forEachOwnedOperation(
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
  ::llvm::DenseMap<::mlir::Value, ::llvm::SmallVector<MemrefAccessGeometry, 4>>
      memrefLoadsByBase;
  ::llvm::DenseMap<::mlir::Value, ::llvm::SmallVector<MemrefAccessGeometry, 4>>
      memrefStoresByBase;
  ::llvm::SmallVector<::loom::lowering::ExactPointerPointAccess, 8>
      exactPointerAccesses;
  // Read-only roots need no cross-iteration address proof. An unsupported
  // read coordinate, such as one computed by an enclosing loop, is safe only
  // when the root is proven distinct from every write. LLVM writes always
  // require the shared byte-aware point proof.
  ::llvm::DenseSet<::mlir::Value> inexactPointerReadBases;

  if (bodyHasMultipleSuccessorTerminator(loop))
    return ::mlir::failure();

  auto walkResult = loom::frontend::analysis::forEachOwnedOperation(
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
          ::mlir::Value base =
              loom::frontend::analysis::projectMemoryRoot(loadPtr);
          if (::mlir::isa<::mlir::LLVM::LoadOp>(op)) {
            auto projected = ::loom::lowering::projectExactPointerPointAccess(
                op, loop.getOperation(), [&](::mlir::Value coordinate) {
                  return ::loom::lowering::isSameSignedMemoryCoordinate(
                      coordinate, iv, loop);
                });
            if (auto *access =
                    std::get_if<::loom::lowering::ExactPointerPointAccess>(
                        &projected))
              exactPointerAccesses.push_back(*access);
            else {
              if (std::get<::loom::lowering::ExactPointerPointAccessRefusal>(
                      projected) ==
                  ::loom::lowering::ExactPointerPointAccessRefusal::
                      UnsupportedEffect)
                return ::mlir::WalkResult::interrupt();
              inexactPointerReadBases.insert(base);
            }
          } else {
            auto geometry = memrefAccessGeometry(op);
            if (!geometry)
              return ::mlir::WalkResult::interrupt();
            memrefLoadsByBase[base].push_back(std::move(*geometry));
          }
          readBases.insert(base);
          return ::mlir::WalkResult::advance();
        }

        // Store ops: capture the base for disjoint analysis, address must
        // be syntactic affine. Reject volatile stores.
        if (::mlir::Value storePtr = getStorePointer(op, isVol)) {
          if (isVol)
            return ::mlir::WalkResult::interrupt();
          ::mlir::Value base =
              loom::frontend::analysis::projectMemoryRoot(storePtr);
          if (::mlir::isa<::mlir::LLVM::StoreOp>(op)) {
            auto projected = ::loom::lowering::projectExactPointerPointAccess(
                op, loop.getOperation(), [&](::mlir::Value coordinate) {
                  return ::loom::lowering::isSameSignedMemoryCoordinate(
                      coordinate, iv, loop);
                });
            auto *access =
                std::get_if<::loom::lowering::ExactPointerPointAccess>(
                    &projected);
            if (!access)
              return ::mlir::WalkResult::interrupt();
            exactPointerAccesses.push_back(*access);
          } else {
            auto geometry = memrefAccessGeometry(op);
            if (!geometry)
              return ::mlir::WalkResult::interrupt();
            memrefStoresByBase[base].push_back(std::move(*geometry));
          }
          writeBases.insert(base);
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

  for (std::size_t lhs = 0; lhs != exactPointerAccesses.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs != exactPointerAccesses.size(); ++rhs) {
      const auto pair = ::loom::lowering::classifyExactPointerPointAccessPair(
          exactPointerAccesses[lhs], exactPointerAccesses[rhs]);
      if (pair == ::loom::lowering::ExactPointerPointAccessPairKind::
                      ByteRelationNotEstablished ||
          pair == ::loom::lowering::ExactPointerPointAccessPairKind::
                      AliasNotEstablished)
        return ::mlir::failure();
    }
  }

  // A store-side address may share a base with reads only for the
  // same-element in-place form. Shifted read/write forms keep the loop
  // serial because they carry a cross-iteration dependence.
  for (::mlir::Value w : writeBases) {
    if (inexactPointerReadBases.contains(w) ||
        (readBases.count(w) &&
         !sameBaseMemrefReadWriteAccessesAreIterationLocal(
             memrefLoadsByBase.lookup(w), memrefStoresByBase.lookup(w), loop)))
      return ::mlir::failure();
    for (::mlir::Value read : readBases)
      if (read != w &&
          !loom::frontend::analysis::haveProvenDistinctMemoryRoots(w, read))
        return ::mlir::failure();
    for (::mlir::Value otherWrite : writeBases)
      if (otherWrite != w &&
          !loom::frontend::analysis::haveProvenDistinctMemoryRoots(w,
                                                                   otherWrite))
        return ::mlir::failure();
  }
  // WAW: same-base memref stores are allowed only for fixed-width lane groups,
  // such as out[3*i + {0,1,2}], where the per-iteration address residue
  // classes are provably disjoint. LLVM pointer stores already passed the
  // shared byte-aware point proof above. Anything else remains serial.
  for (auto &entry : memrefStoresByBase) {
    if (!sameBaseMemrefStoresAreLaneDisjoint(entry.second, loop))
      return ::mlir::failure();
  }

  // LLVM stores already carry the exact point-coordinate proof. Memref
  // stores still need affine-style indices with an induction dependency.
  for (const auto &entry : memrefStoresByBase) {
    for (const MemrefAccessGeometry &store : entry.second) {
      bool sawIvDep = false;
      for (::mlir::Value index : store.indices) {
        if (!isAffineStyle(index, iv, loop))
          return ::mlir::failure();
        sawIvDep |= dependsOnIV(index, iv);
      }
      if (!sawIvDep)
        return ::mlir::failure();
    }
  }

  return ::mlir::success();
}

bool hasZeroBasedUnitStep(::mlir::scf::ForOp loop) {
  return getConstantInt(loop.getLowerBound()) == std::optional<int64_t>{0} &&
         getConstantInt(loop.getStep()) == std::optional<int64_t>{1};
}

bool hasLosslessIndexDomain(::mlir::scf::ForOp loop) {
  auto indexWidth = ::loom::getIndexBitWidth(loop);
  if (!indexWidth) {
    ::llvm::consumeError(indexWidth.takeError());
    return false;
  }
  ::mlir::Type type = loop.getInductionVar().getType();
  if (::mlir::isa<::mlir::IndexType>(type))
    return true;
  auto integer = ::mlir::dyn_cast<::mlir::IntegerType>(type);
  if (!integer || integer.getWidth() < *indexWidth)
    return false;
  // An integer loop with unsigned comparison needs a range proof that the
  // conversion to index preserves unsigned ordering. The initial owner does
  // not have that proof, so retain it serially rather than changing bounds.
  if (loop.getUnsignedCmp())
    return false;
  if (integer.getWidth() > *indexWidth) {
    // A wider integer induction variable may be narrowed only when every
    // bound is a constant in the signed index range. Dynamic bounds have no
    // range proof in this owner and remain serial.
    const auto lower = getConstantInt(loop.getLowerBound());
    const auto upper = getConstantInt(loop.getUpperBound());
    const auto step = getConstantInt(loop.getStep());
    if (!lower || !upper || !step || *indexWidth == 0)
      return false;
    if (*indexWidth < 64) {
      const std::int64_t minimum = -(std::int64_t{1} << (*indexWidth - 1));
      const std::int64_t maximum = (std::int64_t{1} << (*indexWidth - 1)) - 1;
      if (*lower < minimum || *lower > maximum || *upper < minimum ||
          *upper > maximum || *step < minimum || *step > maximum)
        return false;
    }
  }
  return true;
}

struct ConstantLoopDomain final {
  std::int64_t step = 0;
  __int128 first = 0;
  __int128 last = 0;
};

std::optional<ConstantLoopDomain>
constantSignedLoopDomain(::mlir::Operation *loop, ::mlir::Value induction,
                         std::optional<int64_t> lower,
                         std::optional<int64_t> upper,
                         std::optional<int64_t> step) {
  if (!lower || !upper || !step || *step <= 0 || *lower >= *upper)
    return std::nullopt;
  const __int128 first = *lower;
  const __int128 distance = static_cast<__int128>(*upper) - first;
  const __int128 last = first + ((distance - 1) / *step) * *step;
  auto width = fixedIntegerWidth(induction.getType(), loop);
  if (!width || *width == 0 || *width > 64)
    return std::nullopt;
  const __int128 magnitude = static_cast<__int128>(1) << (*width - 1);
  const __int128 minimum = -magnitude;
  const __int128 maximum = magnitude - 1;
  if (first < minimum || last > maximum || last + *step > maximum)
    return std::nullopt;
  return ConstantLoopDomain{*step, first, last};
}

std::optional<ConstantLoopDomain>
constantSignedLoopDomain(::mlir::scf::ForOp loop) {
  if (!loop || loop.getUnsignedCmp())
    return std::nullopt;
  return constantSignedLoopDomain(
      loop, loop.getInductionVar(),
      getConstantIntExpression(loop.getLowerBound()),
      getConstantIntExpression(loop.getUpperBound()),
      getConstantIntExpression(loop.getStep()));
}

std::optional<int64_t> constantBound(::mlir::OpFoldResult bound) {
  if (auto value = ::mlir::dyn_cast<::mlir::Value>(bound))
    return getConstantIntExpression(value);
  auto integer = ::mlir::dyn_cast<::mlir::IntegerAttr>(
      ::mlir::cast<::mlir::Attribute>(bound));
  if (!integer || !integer.getValue().isSignedIntN(64))
    return std::nullopt;
  return integer.getValue().getSExtValue();
}

bool linearRangeFits(LinearExpr expression, const ConstantLoopDomain &domain,
                     unsigned width) {
  if (width == 0 || width > 64)
    return false;
  const __int128 first =
      static_cast<__int128>(expression.ivCoeff) * domain.first +
      expression.constant;
  const __int128 last =
      static_cast<__int128>(expression.ivCoeff) * domain.last +
      expression.constant;
  const __int128 lower = std::min(first, last);
  const __int128 upper = std::max(first, last);
  const __int128 magnitude = static_cast<__int128>(1) << (width - 1);
  return lower >= -magnitude && upper < magnitude;
}

bool isExactLinearExpression(::mlir::Value value, ::mlir::Value induction,
                             ::mlir::Operation *loop,
                             const ConstantLoopDomain &domain,
                             ::llvm::DenseSet<::mlir::Value> &visited) {
  if (value == induction)
    return true;
  if (getConstantInt(value))
    return true;
  if (isDefinedOutside(value, loop) || !visited.insert(value).second ||
      ::mlir::isa<::mlir::BlockArgument>(value))
    return false;
  ::mlir::Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  const bool cast =
      ::mlir::isa<::mlir::arith::IndexCastOp, ::mlir::arith::ExtSIOp,
                  ::mlir::LLVM::SExtOp>(definition);
  const bool arithmetic =
      ::mlir::isa<::mlir::arith::AddIOp, ::mlir::arith::SubIOp,
                  ::mlir::arith::MulIOp, ::mlir::arith::ShLIOp,
                  ::mlir::LLVM::AddOp, ::mlir::LLVM::SubOp, ::mlir::LLVM::MulOp,
                  ::mlir::LLVM::ShlOp>(definition);
  if ((!cast && !arithmetic) || (cast && !isInjectiveCast(definition, loop)))
    return false;
  for (::mlir::Value operand : definition->getOperands())
    if (!isExactLinearExpression(operand, induction, loop, domain, visited))
      return false;
  auto expression = linearExpr(value, induction, loop);
  auto width = fixedIntegerWidth(value.getType(), loop);
  return expression && width && linearRangeFits(*expression, domain, *width);
}

bool isExactLinearExpression(::mlir::Value value, ::mlir::Value induction,
                             ::mlir::Operation *loop,
                             const ConstantLoopDomain &domain) {
  ::llvm::DenseSet<::mlir::Value> visited;
  return isExactLinearExpression(value, induction, loop, domain, visited);
}

void collectGuaranteedLowerBounds(::mlir::Value value, ::mlir::Value induction,
                                  ::mlir::Operation *loop,
                                  const ConstantLoopDomain &domain,
                                  ::llvm::SmallVectorImpl<LinearExpr> &bounds) {
  auto expression = linearExpr(value, induction, loop);
  if (expression && isExactLinearExpression(value, induction, loop, domain)) {
    bounds.push_back(*expression);
    return;
  }
  if (auto maximum = value.getDefiningOp<::mlir::arith::MaxSIOp>()) {
    collectGuaranteedLowerBounds(maximum.getLhs(), induction, loop, domain,
                                 bounds);
    collectGuaranteedLowerBounds(maximum.getRhs(), induction, loop, domain,
                                 bounds);
  }
}

void collectGuaranteedUpperBounds(::mlir::Value value, ::mlir::Value induction,
                                  ::mlir::Operation *loop,
                                  const ConstantLoopDomain &domain,
                                  ::llvm::SmallVectorImpl<LinearExpr> &bounds) {
  auto expression = linearExpr(value, induction, loop);
  if (expression && isExactLinearExpression(value, induction, loop, domain)) {
    bounds.push_back(*expression);
    return;
  }
  if (auto minimum = value.getDefiningOp<::mlir::arith::MinSIOp>()) {
    collectGuaranteedUpperBounds(minimum.getLhs(), induction, loop, domain,
                                 bounds);
    collectGuaranteedUpperBounds(minimum.getRhs(), induction, loop, domain,
                                 bounds);
    return;
  }
  auto addition = value.getDefiningOp<::mlir::arith::AddIOp>();
  if (!addition)
    return;
  ::mlir::Value operand;
  std::optional<std::int64_t> offset = getConstantInt(addition.getLhs());
  if (offset)
    operand = addition.getRhs();
  else {
    offset = getConstantInt(addition.getRhs());
    operand = addition.getLhs();
  }
  if (!offset || *offset < 0)
    return;
  ::llvm::SmallVector<LinearExpr, 4> operandBounds;
  collectGuaranteedUpperBounds(operand, induction, loop, domain, operandBounds);
  auto width = fixedIntegerWidth(value.getType(), loop);
  if (!width)
    return;
  for (LinearExpr bound : operandBounds) {
    if (__builtin_add_overflow(bound.constant, *offset, &bound.constant) ||
        !linearRangeFits(bound, domain, *width))
      continue;
    bounds.push_back(bound);
  }
}

bool hasExactPartitionedPointMemoryGeometry(::mlir::Operation *outer,
                                            ::mlir::scf::ForOp pointLoop) {
  ::llvm::SmallVector<::loom::lowering::ExactPointerPointAccess, 8> accesses;
  bool rejected = false;
  // A rank-one memref access indexed by the point coordinate owns one
  // element-wide byte partition at that coordinate, exactly like a direct
  // inbounds GEP; a vector transfer owns the lane-group partition.
  const auto projectMemrefPointAccess =
      [&](::mlir::Operation *operation)
      -> std::optional<::loom::lowering::ExactPointerPointAccess> {
    auto geometry = memrefAccessGeometry(operation);
    if (!geometry || geometry->indices.size() != 1)
      return std::nullopt;
    bool isVolatile = false;
    ::mlir::Value memory = getStorePointer(operation, isVolatile);
    const bool writes = static_cast<bool>(memory);
    if (!memory)
      memory = getLoadPointer(operation, isVolatile);
    if (!memory || isVolatile)
      return std::nullopt;
    auto type = ::llvm::dyn_cast<::mlir::MemRefType>(memory.getType());
    if (!type || type.getRank() != 1 || !type.getElementType().isIntOrFloat() ||
        !::loom::lowering::isSameSignedMemoryCoordinate(
            geometry->indices.front(), pointLoop.getInductionVar(), outer))
      return std::nullopt;
    const std::uint64_t elementBytes =
        ::mlir::DataLayout::closest(operation).getTypeSize(
            type.getElementType());
    if (elementBytes == 0 || geometry->lanes <= 0)
      return std::nullopt;
    return ::loom::lowering::ExactPointerPointAccess{
        operation, loom::frontend::analysis::projectMemoryRoot(memory),
        ::mlir::LLVM::GEPOp{}, writes,
        elementBytes * static_cast<std::uint64_t>(geometry->lanes)};
  };
  auto walked = loom::frontend::analysis::forEachOwnedOperation(
      pointLoop.getRegion(), [&](::mlir::Operation *operation) {
        if (::mlir::isa<::mlir::memref::LoadOp, ::mlir::memref::StoreOp,
                        ::mlir::vector::TransferReadOp,
                        ::mlir::vector::TransferWriteOp>(operation)) {
          auto access = projectMemrefPointAccess(operation);
          if (!access) {
            rejected = true;
            return ::mlir::WalkResult::interrupt();
          }
          accesses.push_back(*access);
          return ::mlir::WalkResult::advance();
        }
        if (!::mlir::isa<::mlir::LLVM::LoadOp, ::mlir::LLVM::StoreOp>(
                operation))
          return ::mlir::WalkResult::advance();
        auto projected = ::loom::lowering::projectExactPointerPointAccess(
            operation, outer, [&](::mlir::Value coordinate) {
              return ::loom::lowering::isSameSignedMemoryCoordinate(
                  coordinate, pointLoop.getInductionVar(), outer);
            });
        auto *access =
            std::get_if<::loom::lowering::ExactPointerPointAccess>(&projected);
        if (!access) {
          rejected = true;
          return ::mlir::WalkResult::interrupt();
        }
        accesses.push_back(*access);
        return ::mlir::WalkResult::advance();
      });
  if (rejected || walked.wasInterrupted() || accesses.empty() ||
      ::llvm::none_of(accesses,
                      [](const auto &access) { return access.writes; }))
    return false;
  for (std::size_t lhs = 0; lhs != accesses.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs != accesses.size(); ++rhs) {
      const auto pair = ::loom::lowering::classifyExactPointerPointAccessPair(
          accesses[lhs], accesses[rhs]);
      if (pair == ::loom::lowering::ExactPointerPointAccessPairKind::
                      ByteRelationNotEstablished ||
          pair == ::loom::lowering::ExactPointerPointAccessPairKind::
                      AliasNotEstablished)
        return false;
    }
  }
  return true;
}

/// Proves the precise strip-mined shape emitted by the polyhedral
/// materializer without consulting its provider schedule. The outer tile owns
/// disjoint half-open point intervals; the ordinary inner-loop proof then
/// establishes every memory access within those intervals independently.
::mlir::LogicalResult
checkPartitionedNestedBodyParallel(::mlir::Operation *outer,
                                   ::mlir::Block &body, ::mlir::Value induction,
                                   const ConstantLoopDomain &domain) {
  ::mlir::scf::ForOp pointLoop;
  for (::mlir::Operation &operation : body.without_terminator()) {
    if (auto nested = ::mlir::dyn_cast<::mlir::scf::ForOp>(&operation)) {
      if (pointLoop)
        return ::mlir::failure();
      pointLoop = nested;
      continue;
    }
    if (operation.getNumRegions() != 0 ||
        !::mlir::isMemoryEffectFree(&operation))
      return ::mlir::failure();
  }
  if (!pointLoop || !pointLoop.getInitArgs().empty() ||
      pointLoop.getUnsignedCmp() ||
      getConstantInt(pointLoop.getStep()) != std::optional<std::int64_t>{1} ||
      !hasLosslessIndexDomain(pointLoop) ||
      !hasExactPartitionedPointMemoryGeometry(outer, pointLoop) ||
      ::mlir::failed(checkBodyParallel(pointLoop)))
    return ::mlir::failure();

  ::llvm::SmallVector<LinearExpr, 4> starts;
  ::llvm::SmallVector<LinearExpr, 4> ends;
  collectGuaranteedLowerBounds(pointLoop.getLowerBound(), induction, outer,
                               domain, starts);
  collectGuaranteedUpperBounds(pointLoop.getUpperBound(), induction, outer,
                               domain, ends);
  auto pointWidth =
      fixedIntegerWidth(pointLoop.getInductionVar().getType(), pointLoop);
  if (!pointWidth)
    return ::mlir::failure();
  for (const LinearExpr &start : starts) {
    if (start.ivCoeff <= 0)
      continue;
    const __int128 increment =
        static_cast<__int128>(start.ivCoeff) * domain.step;
    const __int128 nextConstant =
        static_cast<__int128>(start.constant) + increment;
    if (nextConstant < std::numeric_limits<std::int64_t>::min() ||
        nextConstant > std::numeric_limits<std::int64_t>::max())
      continue;
    const LinearExpr next{start.ivCoeff,
                          static_cast<std::int64_t>(nextConstant)};
    if (!linearRangeFits(start, domain, *pointWidth) ||
        !linearRangeFits(next, domain, *pointWidth))
      continue;
    if (::llvm::is_contained(ends, next))
      return ::mlir::success();
  }
  return ::mlir::failure();
}

::mlir::LogicalResult
checkPartitionedNestedBodyParallel(::mlir::scf::ForOp outer) {
  auto domain = constantSignedLoopDomain(outer);
  if (!domain || !hasLosslessIndexDomain(outer))
    return ::mlir::failure();
  return checkPartitionedNestedBodyParallel(outer, *outer.getBody(),
                                            outer.getInductionVar(), *domain);
}

bool isPerfectRectangularNest(::llvm::ArrayRef<::mlir::scf::ForOp> nest) {
  if (nest.size() < 2)
    return false;
  ::mlir::scf::ForOp root = nest.front();
  for (std::size_t dimension = 0; dimension < nest.size(); ++dimension) {
    ::mlir::scf::ForOp loop = nest[dimension];
    if (!loop || !loop.getInitArgs().empty() || !hasZeroBasedUnitStep(loop) ||
        !isDefinedOutside(loop.getLowerBound(), root) ||
        !isDefinedOutside(loop.getUpperBound(), root) ||
        !isDefinedOutside(loop.getStep(), root))
      return false;
    if (dimension + 1 == nest.size())
      continue;

    ::mlir::scf::ForOp child = nest[dimension + 1];
    if (child->getParentOp() != loop.getOperation())
      return false;
    for (::mlir::Operation &operation : loop.getBody()->without_terminator()) {
      if (&operation == child.getOperation())
        continue;
      if (operation.getNumRegions() != 0 ||
          !::mlir::isMemoryEffectFree(&operation))
        return false;
    }
  }
  return true;
}

bool hasIndependentDenseStores(::llvm::ArrayRef<::mlir::scf::ForOp> nest) {
  ::llvm::DenseSet<::mlir::Value> readRoots;
  ::llvm::DenseMap<::mlir::Value, ::mlir::Operation *> storesByRoot;
  ::llvm::SmallVector<::mlir::Operation *, 4> stores;
  ::mlir::scf::ForOp innermost = nest.back();
  auto walked = loom::frontend::analysis::forEachOwnedOperation(
      innermost.getRegion(), [&](::mlir::Operation *operation) {
        bool isVolatile = false;
        if (::mlir::Value pointer = getLoadPointer(operation, isVolatile)) {
          readRoots.insert(
              loom::frontend::analysis::projectMemoryRoot(pointer));
          return ::mlir::WalkResult::advance();
        }
        if (::mlir::Value pointer = getStorePointer(operation, isVolatile)) {
          ::mlir::Value root =
              loom::frontend::analysis::projectMemoryRoot(pointer);
          if (storesByRoot.count(root))
            return ::mlir::WalkResult::interrupt();
          storesByRoot[root] = operation;
          stores.push_back(operation);
        }
        return ::mlir::WalkResult::advance();
      });
  if (walked.wasInterrupted())
    return false;
  for (const auto &entry : storesByRoot)
    if (readRoots.count(entry.first))
      return false;
  auto indexWidth = ::loom::getIndexBitWidth(innermost);
  if (!indexWidth) {
    ::llvm::consumeError(indexWidth.takeError());
    return false;
  }
  ::llvm::SmallVector<::mlir::Value, 4> coordinates;
  ::llvm::SmallVector<::mlir::OpFoldResult, 4> upperBounds;
  coordinates.reserve(nest.size());
  upperBounds.reserve(nest.size());
  for (::mlir::scf::ForOp loop : nest) {
    coordinates.push_back(loop.getInductionVar());
    upperBounds.push_back(loop.getUpperBound());
  }
  for (::mlir::Operation *store : stores)
    if (!::loom::frontend::analysis::hasExactDenseCoordinateStoreProjection(
            store, coordinates, upperBounds, *indexWidth))
      return false;
  return true;
}

} // namespace

namespace loom::lowering {

ParallelDependenceResult proveIndependentIterations(::mlir::scf::ForOp loop) {
  if (loop.getInitArgs().empty() == false)
    return ParallelDependenceResult::ProvenDependent;
  if (!hasLosslessIndexDomain(loop))
    return ParallelDependenceResult::ProofNotEstablished;
  return (::mlir::succeeded(checkBodyParallel(loop)) ||
          ::mlir::succeeded(checkPartitionedNestedBodyParallel(loop)))
             ? ParallelDependenceResult::ProvenIndependent
             : ParallelDependenceResult::ProofNotEstablished;
}

ParallelDependenceResult
proveIndependentIterations(::llvm::ArrayRef<::mlir::scf::ForOp> nest) {
  if (!isPerfectRectangularNest(nest))
    return ParallelDependenceResult::ProofNotEstablished;
  for (::mlir::scf::ForOp loop : nest)
    if (loop.getInitArgs().empty() == false)
      return ParallelDependenceResult::ProvenDependent;
  if (::llvm::any_of(nest, [](::mlir::scf::ForOp loop) {
        return !hasLosslessIndexDomain(loop);
      }))
    return ParallelDependenceResult::ProofNotEstablished;
  ::mlir::scf::ForOp innermost = nest.back();
  return ::mlir::succeeded(checkBodyParallel(innermost)) &&
                 hasIndependentDenseStores(nest)
             ? ParallelDependenceResult::ProvenIndependent
             : ParallelDependenceResult::ProofNotEstablished;
}

ParallelDependenceResult
proveIndependentIterations(::mlir::scf::ForallOp loop) {
  if (loop.getRank() != 1 || !loop.getOutputs().empty() ||
      loop.getNumResults() != 0 ||
      !loop.getTerminator().getRegion().front().empty())
    return ParallelDependenceResult::ProofNotEstablished;
  auto domain =
      constantSignedLoopDomain(loop, loop.getInductionVar(0),
                               constantBound(loop.getMixedLowerBound().front()),
                               constantBound(loop.getMixedUpperBound().front()),
                               constantBound(loop.getMixedStep().front()));
  if (!domain)
    return ParallelDependenceResult::ProofNotEstablished;
  return ::mlir::succeeded(checkPartitionedNestedBodyParallel(
             loop, *loop.getBody(), loop.getInductionVar(0), *domain))
             ? ParallelDependenceResult::ProvenIndependent
             : ParallelDependenceResult::ProofNotEstablished;
}

} // namespace loom::lowering
