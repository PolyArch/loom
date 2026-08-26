#ifndef LOOM_EVALUATION_NUMERICVALUE_H
#define LOOM_EVALUATION_NUMERICVALUE_H

#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::evaluation {

// Persistent Evaluation values never pair a bare floating-point number with a
// unit string. Discrete counts use IntegerValue, continuous physical
// quantities use normalized DecimalValue in the descriptor's canonical unit,
// and exact dimensionless ratios, probabilities, and reference-cycle
// coordinates use ExactRatio.

class IntegerValue {
public:
  explicit constexpr IntegerValue(std::int64_t value) : value_(value) {}

  constexpr std::int64_t value() const { return value_; }

  friend constexpr bool operator==(IntegerValue lhs, IntegerValue rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr bool operator!=(IntegerValue lhs, IntegerValue rhs) {
    return !(lhs == rhs);
  }

private:
  std::int64_t value_;
};

// A nonzero decimal removes trailing decimal zeros from its coefficient and
// adds them to its exponent; zero has coefficient zero and exponent zero.
// Overflow during normalization is invalid.
class DecimalValue {
public:
  static llvm::Expected<DecimalValue> get(std::int64_t coefficient,
                                          std::int64_t base10Exponent);

  std::int64_t coefficient() const { return coefficient_; }
  std::int64_t base10Exponent() const { return base10Exponent_; }

  friend bool operator==(DecimalValue lhs, DecimalValue rhs) {
    return lhs.coefficient_ == rhs.coefficient_ &&
           lhs.base10Exponent_ == rhs.base10Exponent_;
  }
  friend bool operator!=(DecimalValue lhs, DecimalValue rhs) {
    return !(lhs == rhs);
  }

private:
  DecimalValue(std::int64_t coefficient, std::int64_t base10Exponent)
      : coefficient_(coefficient), base10Exponent_(base10Exponent) {}

  std::int64_t coefficient_;
  std::int64_t base10Exponent_;
};

/// Exact numeric order of two canonical decimals: negative, zero, or positive.
int compareDecimalValue(DecimalValue lhs, DecimalValue rhs);

// Canonical exact rational used by typed Evaluation fields whose semantics are
// a dimensionless ratio, a probability, or a coordinate or phase in reference
// cycles. It is deliberately not a MetricValue form: absolute physical
// quantities stay DecimalValue, so Decimal and Ratio never compete to encode
// the same fact. The numerator and denominator are uint64, the denominator is
// positive, the pair is reduced by greatest common divisor, and zero has the
// sole encoding 0/1. All normalization arithmetic is checked.
class ExactRatio {
public:
  static llvm::Expected<ExactRatio> get(std::uint64_t numerator,
                                        std::uint64_t denominator);

  std::uint64_t numerator() const { return numerator_; }
  std::uint64_t denominator() const { return denominator_; }

  bool isZero() const { return numerator_ == 0; }

  /// Adds a nonnegative integer without renormalization. For canonical n/d,
  /// gcd(n + value*d, d) equals gcd(n, d), so the reduced denominator is
  /// invariant; only the checked numerator can overflow.
  llvm::Expected<ExactRatio> addInteger(std::uint64_t value) const;

  // Normalize this ratio modulo a positive modulus into the half-open range
  // [0, modulus). Fails when the modulus is zero or when the exact reduced
  // result does not fit uint64.
  llvm::Expected<ExactRatio> reducedModulo(ExactRatio modulus) const;

  friend bool operator==(ExactRatio lhs, ExactRatio rhs) {
    return lhs.numerator_ == rhs.numerator_ &&
           lhs.denominator_ == rhs.denominator_;
  }
  friend bool operator!=(ExactRatio lhs, ExactRatio rhs) {
    return !(lhs == rhs);
  }

private:
  ExactRatio(std::uint64_t numerator, std::uint64_t denominator)
      : numerator_(numerator), denominator_(denominator) {}

  std::uint64_t numerator_;
  std::uint64_t denominator_;
};

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_NUMERICVALUE_H
