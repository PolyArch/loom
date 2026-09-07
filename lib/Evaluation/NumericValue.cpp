#include "Evaluation/NumericValue.h"

#include "CanonicalSupport.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>

namespace loom::evaluation {
namespace {

std::uint64_t magnitude(std::int64_t value) {
  if (value >= 0)
    return static_cast<std::uint64_t>(value);
  return static_cast<std::uint64_t>(-(value + 1)) + 1;
}

unsigned __int128 gcdU128(unsigned __int128 lhs, unsigned __int128 rhs) {
  while (rhs != 0) {
    const unsigned __int128 remainder = lhs % rhs;
    lhs = rhs;
    rhs = remainder;
  }
  return lhs;
}

llvm::Expected<ExactRatio> reducedRatio(unsigned __int128 numerator,
                                        unsigned __int128 denominator) {
  const auto divisor = gcdU128(numerator, denominator);
  numerator /= divisor;
  denominator /= divisor;
  constexpr auto max = std::numeric_limits<std::uint64_t>::max();
  if (numerator > max || denominator > max)
    return detail::evaluationError("exact ratio overflow during normalization");
  return ExactRatio::get(static_cast<std::uint64_t>(numerator),
                         static_cast<std::uint64_t>(denominator));
}

} // namespace

llvm::Expected<DecimalValue> DecimalValue::get(std::int64_t coefficient,
                                               std::int64_t base10Exponent) {
  if (coefficient == 0)
    return DecimalValue(0, 0);
  while ((coefficient % 10) == 0) {
    if (base10Exponent == std::numeric_limits<std::int64_t>::max())
      return detail::evaluationError(
          "decimal exponent overflow during normalization");
    coefficient /= 10;
    ++base10Exponent;
  }
  return DecimalValue(coefficient, base10Exponent);
}

int compareDecimalValue(DecimalValue lhs, DecimalValue rhs) {
  const std::int64_t lhsCoefficient = lhs.coefficient();
  const std::int64_t rhsCoefficient = rhs.coefficient();
  if (lhsCoefficient == rhsCoefficient &&
      lhs.base10Exponent() == rhs.base10Exponent())
    return 0;
  if (lhsCoefficient == 0)
    return rhsCoefficient < 0 ? 1 : -1;
  if (rhsCoefficient == 0)
    return lhsCoefficient < 0 ? -1 : 1;
  if ((lhsCoefficient < 0) != (rhsCoefficient < 0))
    return lhsCoefficient < 0 ? -1 : 1;

  std::string lhsDigits = std::to_string(magnitude(lhsCoefficient));
  std::string rhsDigits = std::to_string(magnitude(rhsCoefficient));
  const __int128 lhsOrder = static_cast<__int128>(lhs.base10Exponent()) +
                            static_cast<__int128>(lhsDigits.size());
  const __int128 rhsOrder = static_cast<__int128>(rhs.base10Exponent()) +
                            static_cast<__int128>(rhsDigits.size());

  int magnitudeComparison = 0;
  if (lhsOrder != rhsOrder) {
    magnitudeComparison = lhsOrder < rhsOrder ? -1 : 1;
  } else {
    const std::size_t width = std::max(lhsDigits.size(), rhsDigits.size());
    lhsDigits.append(width - lhsDigits.size(), '0');
    rhsDigits.append(width - rhsDigits.size(), '0');
    if (lhsDigits != rhsDigits)
      magnitudeComparison = lhsDigits < rhsDigits ? -1 : 1;
  }

  return lhsCoefficient < 0 ? -magnitudeComparison : magnitudeComparison;
}

llvm::Expected<ExactRatio> ExactRatio::get(std::uint64_t numerator,
                                           std::uint64_t denominator) {
  if (denominator == 0)
    return detail::evaluationError("exact ratio denominator must be positive");
  if (numerator == 0)
    return ExactRatio(0, 1);
  const std::uint64_t divisor = std::gcd(numerator, denominator);
  return ExactRatio(numerator / divisor, denominator / divisor);
}

llvm::Expected<ExactRatio> ExactRatio::addInteger(std::uint64_t value) const {
  using u128 = unsigned __int128;
  const u128 numerator = static_cast<u128>(numerator_) +
                         static_cast<u128>(value) * denominator_;
  if (numerator > std::numeric_limits<std::uint64_t>::max())
    return detail::evaluationError("exact ratio overflow during integer add");
  return ExactRatio(static_cast<std::uint64_t>(numerator), denominator_);
}

llvm::Expected<ExactRatio> ExactRatio::add(ExactRatio value) const {
  if (value.denominator_ == 1)
    return addInteger(value.numerator_);
  using u128 = unsigned __int128;
  const std::uint64_t common = std::gcd(denominator_, value.denominator_);
  const u128 left =
      static_cast<u128>(numerator_) * (value.denominator_ / common);
  const u128 right =
      static_cast<u128>(value.numerator_) * (denominator_ / common);
  // Overflow here requires coprime denominators, for which no subsequent
  // reduction could make the numerator fit the canonical u64 domain.
  if (right > ~u128(0) - left)
    return detail::evaluationError("exact ratio overflow during addition");
  return reducedRatio(left + right, static_cast<u128>(denominator_) *
                                        (value.denominator_ / common));
}

llvm::Expected<ExactRatio> ExactRatio::subtract(ExactRatio value) const {
  using u128 = unsigned __int128;
  const u128 left = static_cast<u128>(numerator_) * value.denominator_;
  const u128 right = static_cast<u128>(value.numerator_) * denominator_;
  if (left < right)
    return detail::evaluationError("exact ratio subtraction is negative");
  return reducedRatio(left - right,
                      static_cast<u128>(denominator_) * value.denominator_);
}

llvm::Expected<ExactRatio>
ExactRatio::multiplyInteger(std::uint64_t value) const {
  return reducedRatio(static_cast<unsigned __int128>(numerator_) * value,
                      denominator_);
}

int compareExactRatio(ExactRatio lhs, ExactRatio rhs) {
  using u128 = unsigned __int128;
  const u128 left = static_cast<u128>(lhs.numerator()) * rhs.denominator();
  const u128 right = static_cast<u128>(rhs.numerator()) * lhs.denominator();
  return left < right ? -1 : left > right ? 1 : 0;
}

llvm::Expected<ExactRatio> ExactRatio::reducedModulo(ExactRatio modulus) const {
  if (modulus.numerator_ == 0)
    return detail::evaluationError("exact ratio modulus must be positive");

  // Bring both ratios onto the common denominator b*d and take the remainder of
  // the scaled numerators; the exact result is (a*d mod c*b) / (b*d) reduced.
  // All intermediates fit unsigned __int128, so no step overflows or invokes
  // undefined behavior; only the reduced result may exceed uint64.
  using u128 = unsigned __int128;
  const u128 scaledValue = static_cast<u128>(numerator_) * modulus.denominator_;
  const u128 scaledModulus =
      static_cast<u128>(modulus.numerator_) * denominator_;
  const u128 commonDenominator =
      static_cast<u128>(denominator_) * modulus.denominator_;
  const u128 remainder = scaledValue % scaledModulus;
  return reducedRatio(remainder, commonDenominator);
}

} // namespace loom::evaluation
