#include "Evaluation/Metric.h"

#include "llvm/Support/Error.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

using namespace loom::evaluation;

namespace {

void fail(const char *test, const std::string &message) {
  std::cerr << test << ": " << message << '\n';
  std::exit(1);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T>
T takeExpected(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectErrorContains(const char *test, llvm::Expected<T> value,
                         llvm::StringRef expected) {
  if (value)
    fail(test, "expected an error");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected),
          "unexpected error: " + message);
}

ExactRatio ratio(const char *test, std::uint64_t numerator,
                 std::uint64_t denominator) {
  return takeExpected(test, ExactRatio::get(numerator, denominator));
}

void exactRatioNormalizesAndChecksArithmetic() {
  const ExactRatio reduced = ratio(__func__, 6, 4);
  require(__func__, reduced.numerator() == 3 && reduced.denominator() == 2,
          "ratio was not reduced by greatest common divisor");
  const ExactRatio zero = ratio(__func__, 0, 7);
  require(__func__, zero.numerator() == 0 && zero.denominator() == 1,
          "zero must have the sole encoding 0/1");
  expectErrorContains(__func__, ExactRatio::get(5, 0),
                      "denominator must be positive");

  require(__func__,
          takeExpected(
              __func__,
              ratio(__func__, 5, 6).reducedModulo(ratio(__func__, 1, 2))) ==
              ratio(__func__, 1, 3),
          "5/6 mod 1/2 must normalize to 1/3");
  expectErrorContains(__func__,
                      ratio(__func__, 2, 4295967341ULL)
                          .reducedModulo(ratio(__func__, 1, 4294967311ULL)),
                      "overflow");
}

} // namespace

int main() {
  exactRatioNormalizesAndChecksArithmetic();
  return 0;
}
