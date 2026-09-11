#include "Frontend/Analysis/ScalarGuardDomain.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <iterator>

namespace loom::frontend::analysis {
namespace {

/// Records `values` for `value`, intersecting with an already recorded domain
/// so that several guards over one value compose.
void mergeDomain(mlir::Value value, std::set<std::int64_t> values,
                 std::vector<GuardedValueDomain> &domains) {
  for (GuardedValueDomain &domain : domains)
    if (domain.value == value) {
      std::set<std::int64_t> intersection;
      std::set_intersection(values.begin(), values.end(),
                            domain.values.begin(), domain.values.end(),
                            std::inserter(intersection, intersection.end()));
      domain.values = std::move(intersection);
      return;
    }
  domains.push_back({value, std::move(values)});
}

/// Records one guarded domain at every level of the value's widening and
/// constant-offset chain. A zero extension that loses no value and a constant
/// addition inverted in its own bit-vector width carry the same guard, so both
/// the expression the guard compares and the narrower value it is derived from
/// are admitted.
void addGuardedValueDomain(mlir::Value value, std::set<std::int64_t> values,
                           const ScalarGuardContext &context,
                           std::vector<GuardedValueDomain> &domains) {
  value = context.canonicalValue(value);
  if (auto extend = value.getDefiningOp<mlir::arith::ExtUIOp>()) {
    unsigned width = extend.getIn().getType().getIntOrFloatBitWidth();
    bool fits = width < 63;
    for (std::int64_t item : values)
      fits &= item >= 0 && std::uint64_t(item) <
                               (std::uint64_t(1) << std::min(width, 63u));
    if (fits) {
      mergeDomain(value, values, domains);
      addGuardedValueDomain(extend.getIn(), std::move(values), context,
                            domains);
      return;
    }
  }
  // Invert constant addition in the actual bit-vector width. This handles
  // Clang's unsigned interval test (x - lower) < width without losing wrap.
  if (auto add = value.getDefiningOp<mlir::arith::AddIOp>()) {
    auto lhs = context.constantValue(add.getLhs()),
         rhs = context.constantValue(add.getRhs());
    mlir::Value input;
    std::optional<llvm::APInt> constant;
    if (lhs) {
      input = add.getRhs();
      constant = lhs;
    } else if (rhs) {
      input = add.getLhs();
      constant = rhs;
    }
    if (constant) {
      std::set<std::int64_t> inverted;
      for (std::int64_t item : values) {
        llvm::APInt candidate(constant->getBitWidth(), item);
        candidate -= *constant;
        if (!candidate.isSignedIntN(64))
          return;
        inverted.insert(candidate.getSExtValue());
      }
      mergeDomain(value, std::move(values), domains);
      addGuardedValueDomain(input, std::move(inverted), context, domains);
      return;
    }
  }
  mergeDomain(value, std::move(values), domains);
}

/// One unsigned upper bound a taken branch proves: `bounded` is at most
/// `bound`, exclusively unless `inclusive`.
struct UnsignedUpperBound final {
  mlir::Value bounded;
  mlir::Value bound;
  bool inclusive = false;
};

/// Reads the bound of one unsigned comparison under the truth its branch
/// proves. The failed edge of a comparison bounds the other operand, which is
/// the shape an early-exit range check leaves behind.
std::optional<UnsignedUpperBound> unsignedUpperBound(mlir::arith::CmpIOp compare,
                                                     bool truth) {
  using Predicate = mlir::arith::CmpIPredicate;
  mlir::Value lhs = compare.getLhs(), rhs = compare.getRhs();
  if (!llvm::isa<mlir::IntegerType>(lhs.getType()))
    return std::nullopt;
  switch (compare.getPredicate()) {
  case Predicate::ult:
    return truth ? UnsignedUpperBound{lhs, rhs, false}
                 : UnsignedUpperBound{rhs, lhs, true};
  case Predicate::ule:
    return truth ? UnsignedUpperBound{lhs, rhs, true}
                 : UnsignedUpperBound{rhs, lhs, false};
  case Predicate::ugt:
    return truth ? UnsignedUpperBound{rhs, lhs, false}
                 : UnsignedUpperBound{lhs, rhs, true};
  case Predicate::uge:
    return truth ? UnsignedUpperBound{rhs, lhs, true}
                 : UnsignedUpperBound{lhs, rhs, false};
  default:
    return std::nullopt;
  }
}

/// True when the guard bounds the value the caller is asking about. Two values
/// are the same subject when they reduce to one value through the widening and
/// constant-offset chain that carries a guard.
bool sameGuardSubject(mlir::Value bounded, mlir::Value interested,
                      const ScalarGuardContext &context) {
  if (!interested)
    return true;
  std::vector<GuardedValueDomain> source, target;
  addGuardedValueDomain(bounded, {0}, context, source);
  addGuardedValueDomain(interested, {0}, context, target);
  for (const GuardedValueDomain &left : source)
    for (const GuardedValueDomain &right : target)
      if (left.value == right.value)
        return true;
  return false;
}

void projectCondition(mlir::Value value, bool truth,
                      const ScalarGuardContext &context, mlir::Value interested,
                      llvm::DenseSet<std::pair<mlir::Value, unsigned>> &seen,
                      std::vector<GuardedValueDomain> &domains) {
  if (!value || !seen.insert({value, truth}).second)
    return;
  const auto project = [&](mlir::Value nested, bool nestedTruth) {
    projectCondition(nested, nestedTruth, context, interested, seen, domains);
  };
  if (auto conjunction = value.getDefiningOp<mlir::arith::AndIOp>();
      conjunction && truth) {
    project(conjunction.getLhs(), true);
    project(conjunction.getRhs(), true);
  }
  if (auto disjunction = value.getDefiningOp<mlir::arith::OrIOp>();
      disjunction && !truth) {
    project(disjunction.getLhs(), false);
    project(disjunction.getRhs(), false);
  }
  // A one-bit exclusive or against all ones is a logical negation, so the
  // taken edge proves the opposite truth of its input.
  if (auto negation = value.getDefiningOp<mlir::arith::XOrIOp>()) {
    auto lhs = context.constantValue(negation.getLhs()),
         rhs = context.constantValue(negation.getRhs());
    if (rhs && rhs->getBitWidth() == 1 && rhs->isAllOnes())
      project(negation.getLhs(), !truth);
    else if (lhs && lhs->getBitWidth() == 1 && lhs->isAllOnes())
      project(negation.getRhs(), !truth);
  }
  if (auto select = value.getDefiningOp<mlir::arith::SelectOp>()) {
    auto yes = context.constantValue(select.getTrueValue()),
         no = context.constantValue(select.getFalseValue());
    if (no && no->getBitWidth() == 1 && no->isZero() == truth) {
      project(select.getCondition(), true);
      project(select.getTrueValue(), truth);
    }
    if (yes && yes->getBitWidth() == 1 && yes->isZero() == truth) {
      project(select.getCondition(), false);
      project(select.getFalseValue(), truth);
    }
  }
  auto compare = value.getDefiningOp<mlir::arith::CmpIOp>();
  if (!compare)
    return;
  using Predicate = mlir::arith::CmpIPredicate;
  auto lhs = context.constantValue(compare.getLhs()),
       rhs = context.constantValue(compare.getRhs());
  if ((compare.getPredicate() == Predicate::eq && truth) ||
      (compare.getPredicate() == Predicate::ne && !truth)) {
    if (rhs && rhs->isSignedIntN(64))
      addGuardedValueDomain(compare.getLhs(), {rhs->getSExtValue()}, context,
                            domains);
    else if (lhs && lhs->isSignedIntN(64))
      addGuardedValueDomain(compare.getRhs(), {lhs->getSExtValue()}, context,
                            domains);
  }
  auto bounded = unsignedUpperBound(compare, truth);
  if (!bounded)
    return;
  std::optional<llvm::APInt> limit = context.constantValue(bounded->bound);
  if (!limit && context.boundValues &&
      sameGuardSubject(bounded->bounded, interested, context)) {
    auto candidates = context.boundValues(bounded->bound);
    if (candidates && !candidates->empty() && *candidates->begin() >= 0)
      limit = llvm::APInt(bounded->bound.getType().getIntOrFloatBitWidth(),
                          *candidates->rbegin());
  }
  if (!limit)
    return;
  // Every admitted value is below this exclusive limit, and unsigned ordering
  // also bounds it below by zero.
  llvm::APInt exclusive = *limit;
  if (bounded->inclusive) {
    if (exclusive.isMaxValue())
      return;
    ++exclusive;
  }
  if (!exclusive.ult(context.maximumValues + 1))
    return;
  std::set<std::int64_t> values;
  for (std::uint64_t item = 0; item < exclusive.getZExtValue(); ++item)
    values.insert(static_cast<std::int64_t>(item));
  addGuardedValueDomain(bounded->bounded, std::move(values), context, domains);
}

} // namespace

std::vector<GuardedValueDomain> projectGuardedValueDomains(
    llvm::ArrayRef<std::pair<mlir::Value, bool>> takenConditions,
    const ScalarGuardContext &context, mlir::Value interested) {
  std::vector<GuardedValueDomain> domains;
  llvm::DenseSet<std::pair<mlir::Value, unsigned>> seen;
  for (const auto &[condition, truth] : takenConditions)
    projectCondition(condition, truth, context, interested, seen, domains);
  return domains;
}

} // namespace loom::frontend::analysis
