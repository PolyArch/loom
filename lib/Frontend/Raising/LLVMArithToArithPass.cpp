// Convert llvm.* arith / compare / constant ops with builtin numeric
// types into the matching arith dialect ops. Operations whose operand or
// result types are not builtin integer/float types are intentionally left
// alone so they can continue through the pipeline unchanged.
//
// Pointer arithmetic (llvm.getelementptr), pointer ops (llvm.alloca,
// llvm.load, llvm.store), and ext/trunc with non-builtin element types
// stay in llvm form by design -- the spec allows multi-dialect output.
//
// This pass runs as a function-level pass strictly under func.func. The
// pipeline (see Pipeline.cpp) raises lift-able llvm.func ops into
// func.func first, then nests this pass under each func.func. Aggregate-
// signature llvm.func ops left as llvm.func by func-to-func are
// untouched, so their bodies stay in pristine LLVM form.

#include "Frontend/Raising/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {

// True if the type is something arith ops can directly consume:
// builtin integer (signless, no width restriction here -- arith handles
// arbitrary widths just like llvm), index, or builtin float.
bool isArithCompatibleType(::mlir::Type t) {
  return ::mlir::isa<::mlir::IntegerType, ::mlir::IndexType,
                     ::mlir::FloatType>(t);
}

bool allArithCompatible(::mlir::ValueRange values) {
  for (::mlir::Value v : values) {
    if (!isArithCompatibleType(v.getType()))
      return false;
  }
  return true;
}

// Generic pattern: rewrite a binary llvm op into the matching binary
// arith op when both operands have arith-compatible types. The arith
// op receives the same operand types and result type by SameOperandsAndResult.
template <typename LLVMOp, typename ArithOp>
struct BinaryRewrite : public ::mlir::OpRewritePattern<LLVMOp> {
  using ::mlir::OpRewritePattern<LLVMOp>::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(LLVMOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (!allArithCompatible(op.getOperands()))
      return ::mlir::failure();
    if (!isArithCompatibleType(op.getResult().getType()))
      return ::mlir::failure();
    rewriter.replaceOpWithNewOp<ArithOp>(op, op.getResult().getType(),
                                         op.getLhs(), op.getRhs());
    return ::mlir::success();
  }
};

// llvm.icmp -> arith.cmpi. The two predicate enums share the same
// numeric values for the supported predicates (eq=0, ne=1, slt=2, sle=3,
// sgt=4, sge=5, ult=6, ule=7, ugt=8, uge=9), see LLVMOpsEnums.h.inc and
// ArithOpsEnums.h.inc.
struct ICmpRewrite : public ::mlir::OpRewritePattern<::mlir::LLVM::ICmpOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::LLVM::ICmpOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (!allArithCompatible(op.getOperands()))
      return ::mlir::failure();
    auto predicate =
        static_cast<::mlir::arith::CmpIPredicate>(op.getPredicate());
    rewriter.replaceOpWithNewOp<::mlir::arith::CmpIOp>(
        op, predicate, op.getLhs(), op.getRhs());
    return ::mlir::success();
  }
};

// llvm.fcmp -> arith.cmpf. The CmpFPredicate enums also share numeric
// values (_false=0, oeq=1, ogt=2, oge=3, ..., _true=15) -- see
// LLVMOpsEnums.h.inc / ArithOpsEnums.h.inc.
struct FCmpRewrite : public ::mlir::OpRewritePattern<::mlir::LLVM::FCmpOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::LLVM::FCmpOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (!allArithCompatible(op.getOperands()))
      return ::mlir::failure();
    auto predicate =
        static_cast<::mlir::arith::CmpFPredicate>(op.getPredicate());
    rewriter.replaceOpWithNewOp<::mlir::arith::CmpFOp>(
        op, predicate, op.getLhs(), op.getRhs());
    return ::mlir::success();
  }
};

struct SelectRewrite
    : public ::mlir::OpRewritePattern<::mlir::LLVM::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::LLVM::SelectOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (!isArithCompatibleType(op.getResult().getType()))
      return ::mlir::failure();
    if (!isArithCompatibleType(op.getTrueValue().getType()) ||
        !isArithCompatibleType(op.getFalseValue().getType()))
      return ::mlir::failure();
    rewriter.replaceOpWithNewOp<::mlir::arith::SelectOp>(
        op, op.getResult().getType(), op.getCondition(), op.getTrueValue(),
        op.getFalseValue());
    return ::mlir::success();
  }
};

// llvm.mlir.constant with builtin int/float type -> arith.constant.
// llvm.mlir.constant with pointer / struct / vector elt of !llvm.* stays
// in llvm form because arith.constant cannot represent it directly.
struct ConstantRewrite
    : public ::mlir::OpRewritePattern<::mlir::LLVM::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::LLVM::ConstantOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    ::mlir::Type ty = op.getResult().getType();
    if (!isArithCompatibleType(ty))
      return ::mlir::failure();
    auto valueAttr =
        ::mlir::dyn_cast<::mlir::TypedAttr>(op.getValueAttr());
    if (!valueAttr || valueAttr.getType() != ty)
      return ::mlir::failure();
    rewriter.replaceOpWithNewOp<::mlir::arith::ConstantOp>(op, ty, valueAttr);
    return ::mlir::success();
  }
};

struct LLVMArithToArithPass
    : public ::mlir::PassWrapper<
          LLVMArithToArithPass,
          ::mlir::OperationPass<::mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMArithToArithPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-llvm-arith-to-arith";
  }
  ::llvm::StringRef getDescription() const final {
    return "Rewrite llvm.* arithmetic / compare / constant ops with builtin "
           "numeric types into the matching arith dialect ops, scoped to "
           "func.func bodies. Skipped (aggregate-signature) llvm.func ops "
           "are left untouched.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() final {
    ::mlir::func::FuncOp funcOp = getOperation();
    if (funcOp.isExternal())
      return;
    ::mlir::MLIRContext *ctx = &getContext();

    ::mlir::RewritePatternSet patterns(ctx);
    patterns.add<
        // integer arithmetic
        BinaryRewrite<::mlir::LLVM::AddOp, ::mlir::arith::AddIOp>,
        BinaryRewrite<::mlir::LLVM::SubOp, ::mlir::arith::SubIOp>,
        BinaryRewrite<::mlir::LLVM::MulOp, ::mlir::arith::MulIOp>,
        BinaryRewrite<::mlir::LLVM::SDivOp, ::mlir::arith::DivSIOp>,
        BinaryRewrite<::mlir::LLVM::UDivOp, ::mlir::arith::DivUIOp>,
        BinaryRewrite<::mlir::LLVM::SRemOp, ::mlir::arith::RemSIOp>,
        BinaryRewrite<::mlir::LLVM::URemOp, ::mlir::arith::RemUIOp>,
        BinaryRewrite<::mlir::LLVM::ShlOp, ::mlir::arith::ShLIOp>,
        BinaryRewrite<::mlir::LLVM::LShrOp, ::mlir::arith::ShRUIOp>,
        BinaryRewrite<::mlir::LLVM::AShrOp, ::mlir::arith::ShRSIOp>,
        BinaryRewrite<::mlir::LLVM::AndOp, ::mlir::arith::AndIOp>,
        BinaryRewrite<::mlir::LLVM::OrOp, ::mlir::arith::OrIOp>,
        BinaryRewrite<::mlir::LLVM::XOrOp, ::mlir::arith::XOrIOp>,

        // float arithmetic
        BinaryRewrite<::mlir::LLVM::FAddOp, ::mlir::arith::AddFOp>,
        BinaryRewrite<::mlir::LLVM::FSubOp, ::mlir::arith::SubFOp>,
        BinaryRewrite<::mlir::LLVM::FMulOp, ::mlir::arith::MulFOp>,
        BinaryRewrite<::mlir::LLVM::FDivOp, ::mlir::arith::DivFOp>,
        BinaryRewrite<::mlir::LLVM::FRemOp, ::mlir::arith::RemFOp>,

        // compares + constants
        ICmpRewrite, FCmpRewrite, SelectRewrite, ConstantRewrite>(ctx);

    if (failed(::mlir::applyPatternsGreedily(funcOp.getBody(),
                                             std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace loom {
namespace raising {

std::unique_ptr<::mlir::Pass> createLLVMArithToArithPass() {
  return std::make_unique<LLVMArithToArithPass>();
}

void registerLLVMArithToArithPass() {
  static bool once = []() {
    ::mlir::PassRegistration<LLVMArithToArithPass>();
    return true;
  }();
  (void)once;
}

} // namespace raising
} // namespace loom
