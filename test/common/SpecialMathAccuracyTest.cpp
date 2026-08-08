#include "Common/SpecialMathAccuracy.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>

using namespace loom;

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
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

llvm::APFloat fromBits(const llvm::fltSemantics &semantics,
                       std::uint64_t bits) {
  return llvm::APFloat(
      semantics,
      llvm::APInt(llvm::APFloat::semanticsSizeInBits(semantics), bits));
}

void adjacentRepresentableValuesAreOneUlpApart() {
  const char *test = __func__;
  struct FormatCase {
    const llvm::fltSemantics *semantics;
    std::uint64_t one;
  };
  const FormatCase formats[] = {
      {&llvm::APFloat::IEEEhalf(), 0x3c00U},
      {&llvm::APFloat::BFloat(), 0x3f80U},
      {&llvm::APFloat::IEEEsingle(), 0x3f800000U},
      {&llvm::APFloat::IEEEdouble(), 0x3ff0000000000000ULL},
  };

  for (const FormatCase &format : formats) {
    const llvm::APFloat one = fromBits(*format.semantics, format.one);
    const llvm::APFloat next = fromBits(*format.semantics, format.one + 1);
    require(test, takeExpected(test, specialMathUlpDistance(one, next)) == 1,
            "adjacent destination values were not one ULP apart");
  }

  const llvm::APFloat negativeOne =
      fromBits(llvm::APFloat::IEEEsingle(), 0xbf800000U);
  const llvm::APFloat nextTowardNegativeInfinity =
      fromBits(llvm::APFloat::IEEEsingle(), 0xbf800001U);
  const llvm::APFloat nextTowardZero =
      fromBits(llvm::APFloat::IEEEsingle(), 0xbf7fffffU);
  require(test,
          takeExpected(test, specialMathUlpDistance(
                                 negativeOne, nextTowardNegativeInfinity)) == 1,
          "adjacent negative values toward negative infinity were not one ULP "
          "apart");
  require(test,
          takeExpected(
              test, specialMathUlpDistance(negativeOne, nextTowardZero)) == 1,
          "adjacent negative values toward zero were not one ULP apart");
}

void signedZerosShareOneUlpPosition() {
  const char *test = __func__;
  const llvm::fltSemantics &semantics = llvm::APFloat::IEEEsingle();
  const llvm::APFloat positiveZero = fromBits(semantics, 0x00000000U);
  const llvm::APFloat negativeZero = fromBits(semantics, 0x80000000U);
  require(test,
          takeExpected(test,
                       specialMathUlpDistance(positiveZero, negativeZero)) == 0,
          "signed zeros occupied different ULP positions");

  const llvm::APFloat negativeMinimum = fromBits(semantics, 0x80000001U);
  const llvm::APFloat positiveMinimum = fromBits(semantics, 0x00000001U);
  require(test,
          takeExpected(test, specialMathUlpDistance(negativeMinimum,
                                                    positiveMinimum)) == 2,
          "collapsed signed zero introduced an extra ULP across zero");
}

void conformanceUsesTheTierOwnedUlpLimit() {
  const char *test = __func__;
  const llvm::fltSemantics &semantics = llvm::APFloat::IEEEsingle();
  const llvm::APFloat reference = fromBits(semantics, 0x3f800000U);

  struct TierCase {
    SpecialMathAccuracyTier tier;
    std::uint64_t admittedDistance;
  };
  const TierCase tiers[] = {
      {SpecialMathAccuracyTier::CorrectlyRounded, 0},
      {SpecialMathAccuracyTier::Max1Ulp, 1},
      {SpecialMathAccuracyTier::Max2Ulp, 2},
      {SpecialMathAccuracyTier::Max4Ulp, 4},
  };

  for (const TierCase &tier : tiers) {
    const llvm::APFloat admitted =
        fromBits(semantics, 0x3f800000U + tier.admittedDistance);
    require(test,
            takeExpected(test, specialMathAccuracyConforms(tier.tier, reference,
                                                           admitted)),
            "tier rejected its owned ULP limit");
    const llvm::APFloat rejected =
        fromBits(semantics, 0x3f800001U + tier.admittedDistance);
    require(test,
            !takeExpected(test, specialMathAccuracyConforms(
                                    tier.tier, reference, rejected)),
            "tier admitted a result beyond its owned ULP limit");
  }
}

void exactInfinitiesConformWithoutDefiningOtherInfiniteDistances() {
  const char *test = __func__;
  const llvm::APFloat positiveInfinity =
      llvm::APFloat::getInf(llvm::APFloat::IEEEsingle());
  const llvm::APFloat negativeInfinity =
      llvm::APFloat::getInf(llvm::APFloat::IEEEsingle(), true);
  require(test,
          takeExpected(test, specialMathUlpDistance(positiveInfinity,
                                                    positiveInfinity)) == 0,
          "identical infinities were not exact");
  require(test,
          takeExpected(test, specialMathAccuracyConforms(
                                 SpecialMathAccuracyTier::CorrectlyRounded,
                                 positiveInfinity, positiveInfinity)),
          "identical infinities did not conform to correctly rounded");
  expectErrorContains(
      test, specialMathUlpDistance(positiveInfinity, negativeInfinity),
      "infinite");
}

void undefinedUlpComparisonsFailClosed() {
  const char *test = __func__;
  const llvm::APFloat single =
      fromBits(llvm::APFloat::IEEEsingle(), 0x3f800000U);
  const llvm::APFloat doubleValue =
      fromBits(llvm::APFloat::IEEEdouble(), 0x3ff0000000000000ULL);
  expectErrorContains(test, specialMathUlpDistance(single, doubleValue),
                      "semantics");

  const llvm::APFloat nan = llvm::APFloat::getNaN(llvm::APFloat::IEEEsingle());
  expectErrorContains(test, specialMathUlpDistance(nan, nan), "NaN");

  const llvm::APFloat quadZero =
      llvm::APFloat::getZero(llvm::APFloat::IEEEquad());
  expectErrorContains(test, specialMathUlpDistance(quadZero, quadZero),
                      "64 bits");

  expectErrorContains(
      test,
      specialMathAccuracyConforms(static_cast<SpecialMathAccuracyTier>(0xff),
                                  single, single),
      "unknown special-math accuracy tier");
}

} // namespace

int main() {
  adjacentRepresentableValuesAreOneUlpApart();
  signedZerosShareOneUlpPosition();
  conformanceUsesTheTierOwnedUlpLimit();
  exactInfinitiesConformWithoutDefiningOtherInfiniteDistances();
  undefinedUlpComparisonsFailClosed();
  llvm::outs() << "all special math accuracy tests passed\n";
  return EXIT_SUCCESS;
}
