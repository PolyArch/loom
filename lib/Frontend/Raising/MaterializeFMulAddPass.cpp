// Materialize the execution shape of `llvm.intr.fmuladd`.
//
// `llvm.intr.fmuladd` is not a computation, it is an unmade choice: the target
// may contract it into one fused multiply-add with a single rounding, or
// evaluate an ordinary multiply followed by an ordinary add with two. The two
// results differ, so nothing downstream may pick one implicitly and no shape
// can be inferred from the intrinsic spelling. Mechanical raising therefore
// leaves the intrinsic alone, and this pass materializes exactly the one shape
// its caller selected:
//
//   Fused -> math.fma
//   Split -> arith.mulf then arith.addf
//
// The two shapes differ in what they permit, not only in what they spell.
// Fused carries the complete source fast-math contract onto the one fused
// operation. Split consumes the source's `contract` permission: the multiply
// and the add each round on their own, and neither may be contracted back
// into a single rounding by a later pass or by target code generation.
//
// The selected shape is the entire decision this pass makes, so it is a
// required typed option rather than a defaulted one, in the same shape as the
// typed Dataflow rewrite catalog.
//
// A materialization is legal only when the target operations restate the whole
// source computation: exact numeric types, the operation's fast-math contract,
// the default floating-point environment the intrinsic is evaluated in, and
// the enclosing callable's floating-point environment. `math.fma` and the
// `arith` floating operations state no environment of their own, so a callable
// stating one that they cannot restate cannot receive either shape.
//
// Representability is intrinsic-local. An intrinsic whose complete semantics
// the selected standard form cannot restate remains explicit; it does not
// prevent representable siblings from receiving the selected shape.

#include "Frontend/Raising/Passes.h"

#include "CallableRegions.h"
#include "ExactStandardSpelling.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

namespace {

using loom::raising::FMulAddExecutionShape;

// Replace one proved-representable intrinsic with the selected shape. The
// replacement keeps the source location and the exact operand and result
// types, and carries the source fast-math flags the selected shape still
// permits.
void materializeOne(::mlir::LLVM::FMulAddOp op, FMulAddExecutionShape shape,
                    ::mlir::IRRewriter &rewriter) {
  rewriter.setInsertionPoint(op);
  ::mlir::Location loc = op.getLoc();
  ::mlir::Type type = op.getRes().getType();
  ::mlir::arith::FastMathFlags fastmath =
      loom::raising::exactFastMathFlags(op.getFastmathFlags());
  // No materialized operation states a rounding mode. An arith or math
  // operation that states one is a constrained operation: standard lowering
  // turns it into `llvm.intr.experimental.constrained.*` under an explicit
  // rounding and exception mode, and drops the fast-math flags on the way.
  // llvm.intr.fmuladd is an ordinary non-constrained intrinsic in the default
  // environment, so both shapes leave the mode absent and lower back to
  // ordinary LLVM floating operations.
  if (shape == FMulAddExecutionShape::Fused) {
    // Fusing is what the shape decided, so the complete source contract,
    // `contract` included, carries onto the one fused operation.
    rewriter.replaceOpWithNewOp<::mlir::math::FmaOp>(
        op, type, op.getA(), op.getB(), op.getC(), fastmath);
    return;
  }

  // `contract` is the source's permission to fuse this multiply and add into
  // one rounding. Selecting Split is the decision that declines it, so the
  // permission is consumed here rather than restated on the result: a
  // multiply and an add that still carried it would let any later contraction
  // -- upstream's own arith-to-math.fma uplift, or a backend -- re-fuse them
  // and silently undo the shape. Every other source flag is a property of the
  // computation, not of fusion, and carries onto both operations unchanged.
  ::mlir::arith::FastMathFlags split = ::mlir::arith::bitEnumClear(
      fastmath, ::mlir::arith::FastMathFlags::contract);

  auto product =
      ::mlir::arith::MulFOp::create(rewriter, loc, type, op.getA(), op.getB());
  product.setFastmath(split);
  auto sum = ::mlir::arith::AddFOp::create(rewriter, loc, type,
                                           product.getResult(), op.getC());
  sum.setFastmath(split);
  rewriter.replaceOp(op, sum);
}

void appendRepresentable(
    ::mlir::Operation *operation,
    ::llvm::SmallVectorImpl<::mlir::LLVM::FMulAddOp> &selected) {
  auto fmuladd = ::mlir::dyn_cast<::mlir::LLVM::FMulAddOp>(operation);
  if (fmuladd && loom::raising::restatesExactly(fmuladd.getOperation(),
                                                /*floating=*/true))
    selected.push_back(fmuladd);
}

void materializeSelected(::mlir::MLIRContext &context,
                         ::llvm::ArrayRef<::mlir::LLVM::FMulAddOp> selected,
                         FMulAddExecutionShape shape) {
  ::mlir::IRRewriter rewriter(&context);
  for (::mlir::LLVM::FMulAddOp op : selected)
    materializeOne(op, shape, rewriter);
}

struct MaterializeFMulAddPass
    : public ::mlir::PassWrapper<MaterializeFMulAddPass,
                                 ::mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeFMulAddPass)

  MaterializeFMulAddPass() = default;
  explicit MaterializeFMulAddPass(FMulAddExecutionShape selected) {
    shape = selected;
  }
  MaterializeFMulAddPass(const MaterializeFMulAddPass &other)
      : ::mlir::PassWrapper<MaterializeFMulAddPass, ::mlir::OperationPass<>>(
            other) {
    // Assigning a pass option runs its callback and marks it as supplied, so
    // an unset shape must be left alone to survive pass cloning as unset.
    if (other.shape.hasValue())
      shape = other.shape.getValue();
  }

  ::llvm::StringRef getArgument() const final {
    return "loom-materialize-fmuladd";
  }
  ::llvm::StringRef getDescription() const final {
    return "Materialize one selected execution shape for each exactly "
           "representable llvm.intr.fmuladd in callable regions.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::LLVM::LLVMDialect,
                    ::mlir::math::MathDialect>();
  }

  // The shape is the decision, so there is no default: silently choosing one
  // would materialize a form the caller never selected.
  ::mlir::Pass::Option<FMulAddExecutionShape> shape{
      *this, "shape",
      ::llvm::cl::desc("execution shape to materialize for llvm.intr.fmuladd"),
      ::llvm::cl::values(
          clEnumValN(FMulAddExecutionShape::Fused, "fused",
                     "one math.fma with a single rounding"),
          clEnumValN(FMulAddExecutionShape::Split, "split",
                     "an arith.mulf followed by an arith.addf"))};

  void runOnOperation() final {
    if (!shape.hasValue()) {
      getOperation()->emitError(
          "loom-materialize-fmuladd requires an explicit 'shape' option");
      return signalPassFailure();
    }

    ::llvm::SmallVector<::mlir::LLVM::FMulAddOp> selected;
    (void)loom::raising::forEachCallableRegion(
        getOperation(), [&](::mlir::Region &region) {
          (void)loom::raising::forEachOwnedOperation(
              region, [&](::mlir::Operation *op) {
                appendRepresentable(op, selected);
                return ::mlir::WalkResult::advance();
              });
          return ::mlir::success();
        });

    if (selected.empty())
      return markAllAnalysesPreserved();

    materializeSelected(getContext(), selected, shape.getValue());
  }
};

} // namespace

namespace loom {
namespace raising {

bool canMaterializeFMulAdd(::mlir::Operation &operation) {
  auto fmuladd = ::mlir::dyn_cast<::mlir::LLVM::FMulAddOp>(&operation);
  return fmuladd && restatesExactly(fmuladd.getOperation(),
                                    /*floating=*/true);
}

::mlir::LogicalResult materializeFMulAdd(::mlir::Operation &operation,
                                         FMulAddExecutionShape shape) {
  auto fmuladd = ::mlir::dyn_cast<::mlir::LLVM::FMulAddOp>(&operation);
  if (!fmuladd || !canMaterializeFMulAdd(operation))
    return ::mlir::failure();
  ::mlir::IRRewriter rewriter(operation.getContext());
  materializeOne(fmuladd, shape, rewriter);
  return ::mlir::success();
}

void materializeFMulAddInOperation(::mlir::Operation &root,
                                   FMulAddExecutionShape shape) {
  ::llvm::SmallVector<::mlir::LLVM::FMulAddOp> selected;
  root.walk<::mlir::WalkOrder::PreOrder>(
      [&](::mlir::Operation *operation) -> ::mlir::WalkResult {
        if (operation != &root && isCallableOp(operation))
          return ::mlir::WalkResult::skip();
        appendRepresentable(operation, selected);
        return ::mlir::WalkResult::advance();
      });
  materializeSelected(*root.getContext(), selected, shape);
}

std::unique_ptr<::mlir::Pass>
createMaterializeFMulAddPass(FMulAddExecutionShape shape) {
  return std::make_unique<MaterializeFMulAddPass>(shape);
}

void registerMaterializeFMulAddPass() {
  static bool once = []() {
    ::mlir::PassRegistration<MaterializeFMulAddPass>();
    return true;
  }();
  (void)once;
}

} // namespace raising
} // namespace loom
