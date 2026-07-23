// Convert llvm.* arith / compare / constant ops with builtin numeric
// types into the matching arith dialect ops. Operations whose operand or
// result types are not exactly representable in arith (see
// isExactNumericType below) are intentionally left alone so they can
// continue through the pipeline unchanged.
//
// The rewrite is exact in both directions: every semantic flag the source
// op carries is transferred to the target op, and an op carrying a flag
// arith cannot express stays in llvm form instead of being weakened.
//
// Calls are not rewritten here. A recognized libm symbol does not prove the
// pure-math contract an arith/math op states -- an absent LLVM memory-effects
// attribute is the default read/write effect set, and the spelling of a name
// establishes nothing about errno, the FP environment, termination, or
// whether the caller asked for the builtin at all. Math lowering needs a
// source form that owns that contract explicitly.
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
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <optional>
#include <utility>

namespace {

// Numeric exactness gate ----------------------------------------------
//
// A rewrite may replace an LLVM operation with its arith counterpart only
// when every source semantic fact has an exact target representation: which
// types have an exact builtin counterpart, and how LLVM's numeric flag enums
// map onto arith's. Facts the target cannot express have no entry here on
// purpose -- an `llvm.or` with `disjoint` and fast-math flags on
// `llvm.select` are the current cases. A rewrite that meets one fails and
// leaves the operation in llvm form rather than emitting a weaker target op.

// True when `type` has an exact arith counterpart: a signless integer of
// non-zero width, `index`, a float, or a fixed-shape vector of those.
// arith rejects zero-width and signed integers, and a scalable vector's
// length is a runtime fact no existing authority defines a raising for, so
// those types keep their operations in llvm form.
bool isExactNumericType(::mlir::Type type) {
  if (auto vectorType = ::mlir::dyn_cast<::mlir::VectorType>(type)) {
    if (vectorType.isScalable())
      return false;
    type = vectorType.getElementType();
  }
  if (auto integerType = ::mlir::dyn_cast<::mlir::IntegerType>(type))
    return integerType.isSignless() && integerType.getWidth() > 0;
  return ::mlir::isa<::mlir::IndexType, ::mlir::FloatType>(type);
}

bool allExactNumericTypes(::mlir::ValueRange values) {
  for (::mlir::Value value : values) {
    if (!isExactNumericType(value.getType()))
      return false;
  }
  return true;
}

// arith counterpart of LLVM's integer overflow flags. Both enums carry
// exactly nsw and nuw with the same meaning, so nothing is lost.
::mlir::arith::IntegerOverflowFlags
exactOverflowFlags(::mlir::LLVM::IntegerOverflowFlags flags) {
  ::mlir::arith::IntegerOverflowFlags result{};
  if (::mlir::LLVM::bitEnumContainsAll(flags,
                                       ::mlir::LLVM::IntegerOverflowFlags::nsw))
    result = result | ::mlir::arith::IntegerOverflowFlags::nsw;
  if (::mlir::LLVM::bitEnumContainsAll(flags,
                                       ::mlir::LLVM::IntegerOverflowFlags::nuw))
    result = result | ::mlir::arith::IntegerOverflowFlags::nuw;
  return result;
}

// arith counterpart of LLVM's fast-math flags. Both enums name the same
// seven facts but assign them different bit positions, so each flag is
// mapped by name instead of being reinterpreted.
::mlir::arith::FastMathFlags
exactFastMathFlags(::mlir::LLVM::FastmathFlags flags) {
  const std::pair<::mlir::LLVM::FastmathFlags, ::mlir::arith::FastMathFlags>
      equivalents[] = {
          {::mlir::LLVM::FastmathFlags::nnan,
           ::mlir::arith::FastMathFlags::nnan},
          {::mlir::LLVM::FastmathFlags::ninf,
           ::mlir::arith::FastMathFlags::ninf},
          {::mlir::LLVM::FastmathFlags::nsz, ::mlir::arith::FastMathFlags::nsz},
          {::mlir::LLVM::FastmathFlags::arcp,
           ::mlir::arith::FastMathFlags::arcp},
          {::mlir::LLVM::FastmathFlags::contract,
           ::mlir::arith::FastMathFlags::contract},
          {::mlir::LLVM::FastmathFlags::afn, ::mlir::arith::FastMathFlags::afn},
          {::mlir::LLVM::FastmathFlags::reassoc,
           ::mlir::arith::FastMathFlags::reassoc}};

  ::mlir::arith::FastMathFlags result{};
  for (auto [llvmFlag, arithFlag] : equivalents) {
    if (::mlir::LLVM::bitEnumContainsAll(flags, llvmFlag))
      result = result | arithFlag;
  }
  return result;
}

// Exact arith counterpart of an LLVM integer compare predicate, or nothing
// when the predicate has no explicitly stated counterpart. The two enums are
// spelled independently upstream, so the correspondence is stated case by
// case rather than reinterpreted from the ordinal, and a predicate added
// upstream falls through to the failure return and keeps its llvm.icmp.
::std::optional<::mlir::arith::CmpIPredicate>
exactCmpIPredicate(::mlir::LLVM::ICmpPredicate predicate) {
  using Source = ::mlir::LLVM::ICmpPredicate;
  using Target = ::mlir::arith::CmpIPredicate;
  switch (predicate) {
  case Source::eq:
    return Target::eq;
  case Source::ne:
    return Target::ne;
  case Source::slt:
    return Target::slt;
  case Source::sle:
    return Target::sle;
  case Source::sgt:
    return Target::sgt;
  case Source::sge:
    return Target::sge;
  case Source::ult:
    return Target::ult;
  case Source::ule:
    return Target::ule;
  case Source::ugt:
    return Target::ugt;
  case Source::uge:
    return Target::uge;
  }
  return ::std::nullopt;
}

// Exact arith counterpart of an LLVM float compare predicate, stated case by
// case and failing closed for the same reason.
::std::optional<::mlir::arith::CmpFPredicate>
exactCmpFPredicate(::mlir::LLVM::FCmpPredicate predicate) {
  using Source = ::mlir::LLVM::FCmpPredicate;
  using Target = ::mlir::arith::CmpFPredicate;
  switch (predicate) {
  case Source::_false:
    return Target::AlwaysFalse;
  case Source::oeq:
    return Target::OEQ;
  case Source::ogt:
    return Target::OGT;
  case Source::oge:
    return Target::OGE;
  case Source::olt:
    return Target::OLT;
  case Source::ole:
    return Target::OLE;
  case Source::one:
    return Target::ONE;
  case Source::ord:
    return Target::ORD;
  case Source::ueq:
    return Target::UEQ;
  case Source::ugt:
    return Target::UGT;
  case Source::uge:
    return Target::UGE;
  case Source::ult:
    return Target::ULT;
  case Source::ule:
    return Target::ULE;
  case Source::une:
    return Target::UNE;
  case Source::uno:
    return Target::UNO;
  case Source::_true:
    return Target::AlwaysTrue;
  }
  return ::std::nullopt;
}

// Generic pattern: rewrite a binary llvm op into the matching binary
// arith op when both operands have exactly representable types. The arith
// op receives the same operand types and result type by SameOperandsAndResult.
//
// Which semantic flags the source op can carry is read from the upstream
// LLVM flag interfaces, so every instantiation below is covered without a
// per-opcode table. Each paired arith op declares the matching attribute;
// a pairing that ever loses one fails to compile rather than dropping it.
template <typename LLVMOp, typename ArithOp>
struct BinaryRewrite : public ::mlir::OpRewritePattern<LLVMOp> {
  using ::mlir::OpRewritePattern<LLVMOp>::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(LLVMOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (!allExactNumericTypes(op.getOperands()))
      return ::mlir::failure();
    if (!isExactNumericType(op.getResult().getType()))
      return ::mlir::failure();

    // `disjoint` states that the operands share no set bit, which arith.ori
    // cannot record. Keep such an op in llvm form.
    if constexpr (LLVMOp::template hasTrait<
                      ::mlir::LLVM::DisjointFlagInterface::Trait>()) {
      if (op.getIsDisjoint())
        return ::mlir::failure();
    }

    ArithOp raised =
        ArithOp::create(rewriter, op.getLoc(), op.getResult().getType(),
                        op.getLhs(), op.getRhs());
    if constexpr (LLVMOp::template hasTrait<
                      ::mlir::LLVM::IntegerOverflowFlagsInterface::Trait>())
      raised.setOverflowFlags(exactOverflowFlags(op.getOverflowFlags()));
    if constexpr (LLVMOp::template hasTrait<
                      ::mlir::LLVM::ExactFlagInterface::Trait>())
      raised.setIsExact(op.getIsExact());
    if constexpr (LLVMOp::template hasTrait<
                      ::mlir::LLVM::FastmathFlagsInterface::Trait>())
      raised.setFastmath(exactFastMathFlags(op.getFastmathFlags()));

    rewriter.replaceOp(op, raised);
    return ::mlir::success();
  }
};

// llvm.icmp -> arith.cmpi.
struct ICmpRewrite : public ::mlir::OpRewritePattern<::mlir::LLVM::ICmpOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::LLVM::ICmpOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (!allExactNumericTypes(op.getOperands()))
      return ::mlir::failure();
    auto predicate = exactCmpIPredicate(op.getPredicate());
    if (!predicate)
      return ::mlir::failure();
    rewriter.replaceOpWithNewOp<::mlir::arith::CmpIOp>(
        op, *predicate, op.getLhs(), op.getRhs());
    return ::mlir::success();
  }
};

// llvm.fcmp -> arith.cmpf.
struct FCmpRewrite : public ::mlir::OpRewritePattern<::mlir::LLVM::FCmpOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::LLVM::FCmpOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (!allExactNumericTypes(op.getOperands()))
      return ::mlir::failure();
    auto predicate = exactCmpFPredicate(op.getPredicate());
    if (!predicate)
      return ::mlir::failure();
    rewriter.replaceOpWithNewOp<::mlir::arith::CmpFOp>(
        op, *predicate, op.getLhs(), op.getRhs(),
        exactFastMathFlags(op.getFastmathFlags()));
    return ::mlir::success();
  }
};

struct SelectRewrite
    : public ::mlir::OpRewritePattern<::mlir::LLVM::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::LLVM::SelectOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (!isExactNumericType(op.getResult().getType()))
      return ::mlir::failure();
    if (!isExactNumericType(op.getTrueValue().getType()) ||
        !isExactNumericType(op.getFalseValue().getType()))
      return ::mlir::failure();
    // arith.select has no fast-math carrier, so a flagged llvm.select keeps
    // its llvm form.
    if (op.getFastmathFlags() != ::mlir::LLVM::FastmathFlags::none)
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
    if (!isExactNumericType(ty))
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
    return "Rewrite llvm.* arithmetic / compare / constant ops with exactly "
           "representable numeric types into the matching arith dialect ops, "
           "scoped to func.func bodies. Skipped llvm.func ops are left "
           "untouched.";
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

        // compares and constants
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
