// Normalize each LLVM computation to the exact standard `arith` or `math`
// spelling, so Canonical Dataflow sees one operation-schema identity per basic
// computation instead of a family of dialect aliases.
//
// Every pattern below replaces one LLVM operation with the standard operation
// that the pinned upstream `arith`/`math`-to-LLVM lowering produces exactly
// that LLVM operation from. That inverse relation is what proves the two
// spellings are one computation; a familiar opcode name proves nothing on its
// own, and the signedness reading is part of the operation identity on both
// sides. A source fact the standard operation cannot carry -- `disjoint` on an
// or, a fast-math contract on a select, a floating-point environment the
// enclosing callable states, a scalable element count -- has no entry on
// purpose, and the operation stating it stays in llvm form rather than being
// weakened. Multi-dialect S0 output is the intended result, not a fallback.
//
// No replacement is given a rounding mode. A standard operation that states
// one is a constrained operation, which standard lowering turns into
// `llvm.intr.experimental.constrained.*` under an explicit rounding and
// exception mode, dropping the fast-math flags on the way. Every source here
// is an ordinary non-constrained LLVM operation in the default floating-point
// environment, which is exactly what an unrounded standard operation lowers
// back to.
//
// Calls are deliberately absent. A recognized libm symbol does not prove the
// pure-math contract an arith or math operation states: an absent LLVM
// memory-effects attribute is the default read/write effect set, and a name
// establishes nothing about errno, the floating-point environment,
// termination, or whether the caller asked for the builtin at all. Pointer
// arithmetic (llvm.getelementptr) and the pointer memory operations stay in
// llvm form for the same kind of reason: no standard operation states an LLVM
// pointer's address computation or its memory model.
//
// The rewrite is scoped to callable regions, so an imported callable's body is
// normalized in place while a constant region such as an llvm.mlir.global
// initializer, which must stay expressible as an LLVM constant, is left alone.
// Each declared pattern is offered once per operation and nothing else runs:
// no folding, no constant CSE, no dead-code or unreachable-block removal.

#include "Frontend/Raising/Passes.h"

#include "CallableRegions.h"
#include "ExactRewrite.h"
#include "ExactStandardSpelling.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

namespace {

using loom::raising::allExactNumericTypes;
using loom::raising::enclosingFloatingPolicyBlocksRewrite;
using loom::raising::exactFastMathFlags;
using loom::raising::exactOverflowFlags;
using loom::raising::isExactNumericType;
using loom::raising::restatesExactly;

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

// Carry every semantic flag the source operation can state onto the standard
// operation that replaces it. Which flags a source can state is read from the
// upstream LLVM flag interfaces, so a new instantiation is covered without a
// per-opcode table, and a pairing whose target lacks the matching setter fails
// to compile rather than dropping the flag.
template <typename LLVMOp, typename StandardOp>
void carryExactFlags(LLVMOp op, StandardOp raised) {
  if constexpr (LLVMOp::template hasTrait<
                    ::mlir::LLVM::IntegerOverflowFlagsInterface::Trait>())
    raised.setOverflowFlags(exactOverflowFlags(op.getOverflowFlags()));
  if constexpr (LLVMOp::template hasTrait<
                    ::mlir::LLVM::ExactFlagInterface::Trait>())
    raised.setIsExact(op.getIsExact());
  if constexpr (LLVMOp::template hasTrait<
                    ::mlir::LLVM::FastmathFlagsInterface::Trait>())
    raised.setFastmath(exactFastMathFlags(op.getFastmathFlags()));
}

// Binary computation with two operands and one result of the operation type:
// integer and floating arithmetic, and integer and floating minimum and
// maximum. `Floating` selects whether the enclosing callable's floating-point
// environment participates.
template <typename LLVMOp, typename StandardOp, bool Floating>
struct BinaryAlias : public ::mlir::OpRewritePattern<LLVMOp> {
  using ::mlir::OpRewritePattern<LLVMOp>::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(LLVMOp op, ::mlir::PatternRewriter &rewriter) const override {
    if (!restatesExactly(op, Floating))
      return ::mlir::failure();

    // `disjoint` states that the operands share no set bit, which arith.ori
    // cannot record. Keep such an op in llvm form.
    if constexpr (LLVMOp::template hasTrait<
                      ::mlir::LLVM::DisjointFlagInterface::Trait>()) {
      if (op.getIsDisjoint())
        return ::mlir::failure();
    }

    StandardOp raised =
        StandardOp::create(rewriter, op.getLoc(), op->getResult(0).getType(),
                           op->getOperand(0), op->getOperand(1));
    carryExactFlags(op, raised);
    rewriter.replaceOp(op, raised);
    return ::mlir::success();
  }
};

// Unary floating computation with one operand and one result of the operation
// type: negation and absolute value.
template <typename LLVMOp, typename StandardOp>
struct UnaryFloatAlias : public ::mlir::OpRewritePattern<LLVMOp> {
  using ::mlir::OpRewritePattern<LLVMOp>::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(LLVMOp op, ::mlir::PatternRewriter &rewriter) const override {
    if (!restatesExactly(op, /*floating=*/true))
      return ::mlir::failure();
    StandardOp raised = StandardOp::create(
        rewriter, op.getLoc(), op->getResult(0).getType(), op->getOperand(0));
    carryExactFlags(op, raised);
    rewriter.replaceOp(op, raised);
    return ::mlir::success();
  }
};

// llvm.intr.fma -> math.fma.
//
// llvm.intr.fma is the exact fused multiply-add: one operation, one rounding,
// in the default floating-point environment, and math.fma without a rounding
// mode states the same non-constrained computation. llvm.intr.fmuladd is
// deliberately absent: it states a choice between that fused form and a
// separate multiply and add, which no single standard operation restates, so
// it survives mechanical raising until one typed materialization decision
// names the form.
struct FMAAlias : public ::mlir::OpRewritePattern<::mlir::LLVM::FMAOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::LLVM::FMAOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (!restatesExactly(op, /*floating=*/true))
      return ::mlir::failure();
    rewriter.replaceOpWithNewOp<::mlir::math::FmaOp>(
        op, op.getRes().getType(), op.getA(), op.getB(), op.getC(),
        exactFastMathFlags(op.getFastmathFlags()));
    return ::mlir::success();
  }
};

// The math zero-count operations define zero to return the operand width.
// That is exactly the non-poisoning LLVM form. The poisoning form retains its
// LLVM spelling because math has no attribute that can carry that contract.
template <typename LLVMOp, typename MathOp>
struct CountZerosAlias : public ::mlir::OpRewritePattern<LLVMOp> {
  using ::mlir::OpRewritePattern<LLVMOp>::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(LLVMOp op, ::mlir::PatternRewriter &rewriter) const override {
    if (op.getIsZeroPoison() || !restatesExactly(op, /*floating=*/false))
      return ::mlir::failure();
    rewriter.replaceOpWithNewOp<MathOp>(op, op.getRes().getType(), op.getIn());
    return ::mlir::success();
  }
};

// llvm.intr.abs -> compare, subtract, select.
//
// Integer magnitude is deliberately expressed using ordinary S0 actors: the
// builtin Fabric ALU inventory does not advertise a distinct integer-abs
// capability. LLVM's optional INT_MIN poison contract belongs on the only
// overflowing operation in the expansion, the signed negation.
struct IntegerAbsExpansion
    : public ::mlir::OpRewritePattern<::mlir::LLVM::AbsOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::LLVM::AbsOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (!restatesExactly(op, /*floating=*/false))
      return ::mlir::failure();

    ::mlir::Type type = op.getRes().getType();
    auto zeroAttr = rewriter.getZeroAttr(type);
    if (!zeroAttr)
      return ::mlir::failure();
    auto zero = ::mlir::arith::ConstantOp::create(rewriter, op.getLoc(), type,
                                                  zeroAttr);
    auto isNegative = ::mlir::arith::CmpIOp::create(
        rewriter, op.getLoc(), ::mlir::arith::CmpIPredicate::slt, op.getIn(),
        zero);
    auto negated = ::mlir::arith::SubIOp::create(rewriter, op.getLoc(), type,
                                                 zero, op.getIn());
    if (op.getIsIntMinPoison())
      negated.setOverflowFlags(::mlir::arith::IntegerOverflowFlags::nsw);
    rewriter.replaceOpWithNewOp<::mlir::arith::SelectOp>(op, type, isNegative,
                                                         negated, op.getIn());
    return ::mlir::success();
  }
};

// llvm.icmp -> arith.cmpi.
struct ICmpAlias : public ::mlir::OpRewritePattern<::mlir::LLVM::ICmpOp> {
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
struct FCmpAlias : public ::mlir::OpRewritePattern<::mlir::LLVM::FCmpOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::LLVM::FCmpOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (!allExactNumericTypes(op.getOperands()))
      return ::mlir::failure();
    if (enclosingFloatingPolicyBlocksRewrite(op))
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

struct SelectAlias : public ::mlir::OpRewritePattern<::mlir::LLVM::SelectOp> {
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
struct ConstantAlias
    : public ::mlir::OpRewritePattern<::mlir::LLVM::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::LLVM::ConstantOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    ::mlir::Type ty = op.getResult().getType();
    if (!isExactNumericType(ty))
      return ::mlir::failure();
    auto valueAttr = ::mlir::dyn_cast<::mlir::TypedAttr>(op.getValueAttr());
    if (!valueAttr || valueAttr.getType() != ty)
      return ::mlir::failure();
    rewriter.replaceOpWithNewOp<::mlir::arith::ConstantOp>(op, ty, valueAttr);
    return ::mlir::success();
  }
};

// Cast that states nothing beyond the two types: sign extension, and the four
// integer/floating domain conversions. `Floating` selects whether the
// enclosing callable's floating-point environment participates.
template <typename LLVMOp, typename StandardOp, bool Floating>
struct PlainCastAlias : public ::mlir::OpRewritePattern<LLVMOp> {
  using ::mlir::OpRewritePattern<LLVMOp>::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(LLVMOp op, ::mlir::PatternRewriter &rewriter) const override {
    if (!restatesExactly(op, Floating))
      return ::mlir::failure();
    rewriter.replaceOpWithNewOp<StandardOp>(op, op.getRes().getType(),
                                            op.getArg());
    return ::mlir::success();
  }
};

// Cast whose source may state that the operand's most significant bit is
// clear: zero extension and unsigned integer-to-float. Both target operations
// carry the same `nneg` fact with the same poison rule when it is violated.
template <typename LLVMOp, typename StandardOp, bool Floating>
struct NonNegativeCastAlias : public ::mlir::OpRewritePattern<LLVMOp> {
  using ::mlir::OpRewritePattern<LLVMOp>::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(LLVMOp op, ::mlir::PatternRewriter &rewriter) const override {
    if (!restatesExactly(op, Floating))
      return ::mlir::failure();
    rewriter.replaceOpWithNewOp<StandardOp>(op, op.getRes().getType(),
                                            op.getArg(), op.getNonNeg());
    return ::mlir::success();
  }
};

// llvm.trunc -> arith.trunci. Both carry nsw and nuw with the same
// poison-on-violation rule.
struct IntegerTruncationAlias
    : public ::mlir::OpRewritePattern<::mlir::LLVM::TruncOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::LLVM::TruncOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (!restatesExactly(op, /*floating=*/false))
      return ::mlir::failure();
    rewriter.replaceOpWithNewOp<::mlir::arith::TruncIOp>(
        op, op.getRes().getType(), op.getArg(),
        exactOverflowFlags(op.getOverflowFlags()));
    return ::mlir::success();
  }
};

// llvm.fpext -> arith.extf and llvm.fptrunc -> arith.truncf. An unflagged
// source is the exact inverse of the pinned arith-to-llvm lowering, so it is
// restated directly and stays unflagged rather than acquiring an empty
// contract attribute. A flagged source is preserved in llvm form: the pinned
// convert-arith-to-llvm lowering of a fast-math arith.extf or arith.truncf
// does not carry that contract back onto the llvm operation, so the raised
// form is not roundtrippable -- it leaves a foreign arith fast-math attribute
// that mlir-translate rejects. Keeping the llvm spelling is exact where
// respelling it would weaken or break the round trip.
template <typename LLVMOp, typename StandardOp>
struct FloatResizeAlias : public ::mlir::OpRewritePattern<LLVMOp> {
  using ::mlir::OpRewritePattern<LLVMOp>::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(LLVMOp op, ::mlir::PatternRewriter &rewriter) const override {
    if (!restatesExactly(op, /*floating=*/true))
      return ::mlir::failure();
    if (op.getFastmathFlags() != ::mlir::LLVM::FastmathFlags::none)
      return ::mlir::failure();

    ::mlir::arith::FastMathFlagsAttr fastmath;
    if constexpr (::std::is_same_v<StandardOp, ::mlir::arith::TruncFOp>)
      rewriter.replaceOpWithNewOp<::mlir::arith::TruncFOp>(
          op, op.getRes().getType(), op.getArg(),
          ::mlir::arith::RoundingModeAttr{}, fastmath);
    else
      rewriter.replaceOpWithNewOp<StandardOp>(op, op.getRes().getType(),
                                              op.getArg(), fastmath);
    return ::mlir::success();
  }
};

template <typename LLVMOp, typename StandardOp>
using IntegerBinaryAlias = BinaryAlias<LLVMOp, StandardOp, /*Floating=*/false>;
template <typename LLVMOp, typename StandardOp>
using FloatBinaryAlias = BinaryAlias<LLVMOp, StandardOp, /*Floating=*/true>;

struct LLVMArithToArithPass
    : public ::mlir::PassWrapper<LLVMArithToArithPass,
                                 ::mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMArithToArithPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-llvm-arith-to-arith";
  }
  ::llvm::StringRef getDescription() const final {
    return "Rewrite each llvm computation whose complete semantics an arith "
           "or math operation restates exactly into that standard operation, "
           "scoped to callable regions.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::LLVM::LLVMDialect,
                    ::mlir::math::MathDialect>();
  }

  void runOnOperation() final {
    ::mlir::MLIRContext *ctx = &getContext();
    ::mlir::RewritePatternSet patterns(ctx);
    patterns.add<
        // integer arithmetic
        IntegerBinaryAlias<::mlir::LLVM::AddOp, ::mlir::arith::AddIOp>,
        IntegerBinaryAlias<::mlir::LLVM::SubOp, ::mlir::arith::SubIOp>,
        IntegerBinaryAlias<::mlir::LLVM::MulOp, ::mlir::arith::MulIOp>,
        IntegerBinaryAlias<::mlir::LLVM::SDivOp, ::mlir::arith::DivSIOp>,
        IntegerBinaryAlias<::mlir::LLVM::UDivOp, ::mlir::arith::DivUIOp>,
        IntegerBinaryAlias<::mlir::LLVM::SRemOp, ::mlir::arith::RemSIOp>,
        IntegerBinaryAlias<::mlir::LLVM::URemOp, ::mlir::arith::RemUIOp>,
        IntegerBinaryAlias<::mlir::LLVM::ShlOp, ::mlir::arith::ShLIOp>,
        IntegerBinaryAlias<::mlir::LLVM::LShrOp, ::mlir::arith::ShRUIOp>,
        IntegerBinaryAlias<::mlir::LLVM::AShrOp, ::mlir::arith::ShRSIOp>,
        IntegerBinaryAlias<::mlir::LLVM::AndOp, ::mlir::arith::AndIOp>,
        IntegerBinaryAlias<::mlir::LLVM::OrOp, ::mlir::arith::OrIOp>,
        IntegerBinaryAlias<::mlir::LLVM::XOrOp, ::mlir::arith::XOrIOp>,

        // integer minimum and maximum, in both signedness readings
        IntegerBinaryAlias<::mlir::LLVM::SMaxOp, ::mlir::arith::MaxSIOp>,
        IntegerBinaryAlias<::mlir::LLVM::SMinOp, ::mlir::arith::MinSIOp>,
        IntegerBinaryAlias<::mlir::LLVM::UMaxOp, ::mlir::arith::MaxUIOp>,
        IntegerBinaryAlias<::mlir::LLVM::UMinOp, ::mlir::arith::MinUIOp>,

        // float arithmetic
        FloatBinaryAlias<::mlir::LLVM::FAddOp, ::mlir::arith::AddFOp>,
        FloatBinaryAlias<::mlir::LLVM::FSubOp, ::mlir::arith::SubFOp>,
        FloatBinaryAlias<::mlir::LLVM::FMulOp, ::mlir::arith::MulFOp>,
        FloatBinaryAlias<::mlir::LLVM::FDivOp, ::mlir::arith::DivFOp>,
        FloatBinaryAlias<::mlir::LLVM::FRemOp, ::mlir::arith::RemFOp>,

        // floating minimum and maximum. The two families differ in what they
        // state about NaN and signed zero, so each keeps its own identity:
        // maxnum/minnum return the other operand for a NaN, maximum/minimum
        // propagate it and order -0.0 below +0.0.
        FloatBinaryAlias<::mlir::LLVM::MaxNumOp, ::mlir::arith::MaxNumFOp>,
        FloatBinaryAlias<::mlir::LLVM::MinNumOp, ::mlir::arith::MinNumFOp>,
        FloatBinaryAlias<::mlir::LLVM::MaximumOp, ::mlir::arith::MaximumFOp>,
        FloatBinaryAlias<::mlir::LLVM::MinimumOp, ::mlir::arith::MinimumFOp>,

        // floating negation, absolute value, and typed transcendental aliases
        UnaryFloatAlias<::mlir::LLVM::FNegOp, ::mlir::arith::NegFOp>,
        UnaryFloatAlias<::mlir::LLVM::FAbsOp, ::mlir::math::AbsFOp>,
        UnaryFloatAlias<::mlir::LLVM::CosOp, ::mlir::math::CosOp>,

        // exact fused multiply-add
        FMAAlias,

        // zero-count aliases whose zero behavior is fully defined
        CountZerosAlias<::mlir::LLVM::CountLeadingZerosOp,
                        ::mlir::math::CountLeadingZerosOp>,
        CountZerosAlias<::mlir::LLVM::CountTrailingZerosOp,
                        ::mlir::math::CountTrailingZerosOp>,

        // integer magnitude in terms of the ordinary ALU graph
        IntegerAbsExpansion,

        // compares, selection and constants
        ICmpAlias, FCmpAlias, SelectAlias, ConstantAlias,

        // integer width casts
        IntegerTruncationAlias,
        PlainCastAlias<::mlir::LLVM::SExtOp, ::mlir::arith::ExtSIOp,
                       /*Floating=*/false>,
        NonNegativeCastAlias<::mlir::LLVM::ZExtOp, ::mlir::arith::ExtUIOp,
                             /*Floating=*/false>,

        // integer to floating, in both signedness readings
        PlainCastAlias<::mlir::LLVM::SIToFPOp, ::mlir::arith::SIToFPOp,
                       /*Floating=*/true>,
        NonNegativeCastAlias<::mlir::LLVM::UIToFPOp, ::mlir::arith::UIToFPOp,
                             /*Floating=*/true>,

        // floating to integer, in both signedness readings
        PlainCastAlias<::mlir::LLVM::FPToSIOp, ::mlir::arith::FPToSIOp,
                       /*Floating=*/true>,
        PlainCastAlias<::mlir::LLVM::FPToUIOp, ::mlir::arith::FPToUIOp,
                       /*Floating=*/true>,

        // floating width casts
        FloatResizeAlias<::mlir::LLVM::FPExtOp, ::mlir::arith::ExtFOp>,
        FloatResizeAlias<::mlir::LLVM::FPTruncOp, ::mlir::arith::TruncFOp>>(
        ctx);
    ::mlir::FrozenRewritePatternSet frozen(std::move(patterns));

    (void)loom::raising::forEachCallableRegion(
        getOperation(), [&](::mlir::Region &region) {
          loom::raising::applyExactPatternsOnce(region, frozen);
          return ::mlir::success();
        });
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
