#include "PnR/DeterministicSearchProtocol.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "deterministic search protocol test: " << message << '\n';
  std::exit(1);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void seedFramingAndXoshiroSequenceAreCanonical() {
  loom::pnr::DeterministicPnrRandomStream stream =
      loom::pnr::DeterministicPnrRandomStream::create(
          UINT64_C(0x0123456789abcdef), 7,
          loom::pnr::PnrRandomStreamPurpose::ActionProposal);
  constexpr std::array<std::uint64_t, 8> expected{
      UINT64_C(0x7183822365697657), UINT64_C(0x2f9429b9bcdc1b20),
      UINT64_C(0x5b499c1a64d4a010), UINT64_C(0x5c448abe844d0951),
      UINT64_C(0x67c9ef784e0e8d8e), UINT64_C(0xd132c06ef60ea8b7),
      UINT64_C(0x32ff3689fc9d338c), UINT64_C(0x8c5518163c46141e)};
  for (std::uint64_t value : expected)
    require(stream.nextU64() == value, "xoshiro sequence changed");
}

void boundedSelectionUsesCanonicalRejection() {
  loom::pnr::DeterministicPnrRandomStream stream =
      loom::pnr::DeterministicPnrRandomStream::create(
          UINT64_C(0x0123456789abcdef), 7,
          loom::pnr::PnrRandomStreamPurpose::ActionProposal);
  require(take(stream.nextBounded(UINT64_C(0x8000000000000001))) ==
              UINT64_C(0x5132c06ef60ea8b6),
          "bounded selection did not reject the canonical prefix");
  require(stream.nextU64() == UINT64_C(0x32ff3689fc9d338c),
          "bounded selection consumed the wrong number of words");
  llvm::Expected<std::uint64_t> zero = stream.nextBounded(0);
  require(!zero && llvm::toString(zero.takeError()).find("positive") !=
                       std::string::npos,
          "zero bounded domain was accepted");
}

void exponentialThresholdTableIsCanonical() {
  const llvm::ArrayRef<std::uint64_t> thresholds =
      loom::pnr::expNegativeQ64Thresholds();
  require(thresholds.size() == 11356,
          "acceptance table has the wrong number of entries");
  require(loom::pnr::expNegativeQ64Threshold(1) ==
                  UINT64_C(0xff007fd55ffdde38) &&
              loom::pnr::expNegativeQ64Threshold(256) ==
                  UINT64_C(0x5e2d58d8b3bcdf1a) &&
              loom::pnr::expNegativeQ64Threshold(11356) == 1 &&
              loom::pnr::expNegativeQ64Threshold(11357) == 0,
          "acceptance table boundary anchors changed");
  for (std::size_t index = 1; index < thresholds.size(); ++index)
    require(thresholds[index - 1] >= thresholds[index],
            "acceptance table is not monotonic");

  std::vector<std::uint8_t> bytes;
  bytes.reserve(thresholds.size() * sizeof(std::uint64_t));
  for (std::uint64_t threshold : thresholds)
    for (unsigned shift = 56;; shift -= 8) {
      bytes.push_back(static_cast<std::uint8_t>(threshold >> shift));
      if (shift == 0)
        break;
    }
  constexpr std::array<std::uint8_t, 32> expectedDigest{
      0x88, 0xa3, 0x5f, 0xea, 0x36, 0x8b, 0x5d, 0xf8, 0x90, 0xaa, 0x79,
      0x02, 0x39, 0xca, 0x68, 0x11, 0x54, 0xf6, 0x95, 0x41, 0xc3, 0xe7,
      0xda, 0xb0, 0x5c, 0xf6, 0x0d, 0xbc, 0x38, 0x90, 0xbf, 0xbf};
  require(llvm::SHA256::hash(bytes) == expectedDigest,
          "acceptance table digest changed");
}

void acceptanceConsumesOnlyItsSpecifiedWords() {
  using loom::dse::ObjectiveDifferenceSign;
  using loom::dse::ObjectiveSignedDifference;
  using loom::dse::ObjectiveWideValue;

  auto stream = [] {
    return loom::pnr::DeterministicPnrRandomStream::create(
        UINT64_C(0x0123456789abcdef), 7,
        loom::pnr::PnrRandomStreamPurpose::Acceptance);
  };
  auto first = stream();
  auto second = stream();
  require(take(loom::pnr::acceptAnnealingDelta(
              {ObjectiveDifferenceSign::Zero, {}}, 1, first)),
          "zero energy delta was rejected");
  require(first.nextU64() == second.nextU64(),
          "non-positive energy delta consumed a random word");

  auto likely = stream();
  require(take(loom::pnr::acceptAnnealingDelta(
              {ObjectiveDifferenceSign::Positive, ObjectiveWideValue{0, 1}},
              256, likely)),
          "ratio-index-one delta was unexpectedly rejected");

  auto unlikely = stream();
  require(!take(loom::pnr::acceptAnnealingDelta(
              {ObjectiveDifferenceSign::Positive, ObjectiveWideValue{0, 1}}, 1,
              unlikely)),
          "ratio-index-256 delta was unexpectedly accepted");

  auto zeroThreshold = stream();
  auto reference = stream();
  require(!take(loom::pnr::acceptAnnealingDelta(
              {ObjectiveDifferenceSign::Positive,
               ObjectiveWideValue{UINT64_MAX, UINT64_MAX}},
              1, zeroThreshold)),
          "zero-threshold delta was accepted");
  (void)reference.nextU64();
  require(zeroThreshold.nextU64() == reference.nextU64(),
          "zero-threshold delta did not consume exactly one word");

  auto invalid = stream();
  llvm::Expected<bool> result = loom::pnr::acceptAnnealingDelta(
      {ObjectiveDifferenceSign::Positive, ObjectiveWideValue{0, 1}}, 0,
      invalid);
  require(!result && llvm::toString(result.takeError()).find("temperature") !=
                         std::string::npos,
          "zero temperature was accepted");
}

} // namespace

int main() {
  seedFramingAndXoshiroSequenceAreCanonical();
  boundedSelectionUsesCanonicalRejection();
  exponentialThresholdTableIsCanonical();
  acceptanceConsumesOnlyItsSpecifiedWords();
  llvm::outs() << "deterministic search protocol tests passed\n";
  return 0;
}
