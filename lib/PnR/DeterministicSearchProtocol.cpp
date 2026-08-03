#include "PnR/DeterministicSearchProtocol.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <system_error>
#include <vector>

using namespace loom;
using namespace loom::pnr;

namespace {

constexpr char seedDomain[] =
    "loom.pnr.prng.sha256_seeded_xoshiro256starstar.1.0";
static_assert(sizeof(seedDomain) - 1 == 50,
              "PnR seed domain must match the protocol framing");

template <std::size_t Size>
void appendU32Be(std::array<std::uint8_t, Size> &bytes, std::size_t &offset,
                 std::uint32_t value) {
  bytes[offset++] = static_cast<std::uint8_t>(value >> 24);
  bytes[offset++] = static_cast<std::uint8_t>(value >> 16);
  bytes[offset++] = static_cast<std::uint8_t>(value >> 8);
  bytes[offset++] = static_cast<std::uint8_t>(value);
}

template <std::size_t Size>
void appendU64Be(std::array<std::uint8_t, Size> &bytes, std::size_t &offset,
                 std::uint64_t value) {
  for (unsigned shift = 56;; shift -= 8) {
    bytes[offset++] = static_cast<std::uint8_t>(value >> shift);
    if (shift == 0)
      break;
  }
}

std::uint64_t readU64Be(const std::uint8_t *bytes) {
  std::uint64_t result = 0;
  for (unsigned index = 0; index != 8; ++index)
    result = (result << 8) | bytes[index];
  return result;
}

std::uint64_t rotateLeft(std::uint64_t value, unsigned amount) {
  return (value << amount) | (value >> (64 - amount));
}

llvm::APInt wideMagnitude(dse::ObjectiveWideValue value) {
  llvm::APInt result(128, value.high);
  result <<= 64;
  result |= llvm::APInt(128, value.low);
  return result;
}

llvm::Expected<std::uint64_t>
positiveDeltaRatioIndex(dse::ObjectiveWideValue delta,
                        std::uint64_t temperature) {
  if (temperature == 0)
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "annealing temperature must be positive");
  if (delta.high == 0 && delta.low == 0)
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "positive annealing delta must have nonzero magnitude");

  llvm::APInt numerator = wideMagnitude(delta).zext(192);
  numerator *= llvm::APInt(192, 256);
  const llvm::APInt denominator(192, temperature);
  llvm::APInt quotient = numerator.udiv(denominator);
  if (!numerator.urem(denominator).isZero())
    ++quotient;
  if (quotient.ugt(expNegativeQ64Thresholds().size()))
    return static_cast<std::uint64_t>(expNegativeQ64Thresholds().size() + 1);
  return quotient.getZExtValue();
}

bool thresholdReachesTarget(std::uint64_t threshold,
                            ResolvedExactRatio target) {
  llvm::APInt left(128, threshold);
  left *= llvm::APInt(128, target.denominator);
  llvm::APInt right(128, target.numerator);
  right <<= 64;
  return left.uge(right);
}

std::optional<std::uint64_t>
maximumTargetRatioIndex(ResolvedExactRatio target) {
  const llvm::ArrayRef<std::uint64_t> thresholds = expNegativeQ64Thresholds();
  std::size_t firstMiss = 0;
  std::size_t end = thresholds.size();
  while (firstMiss < end) {
    const std::size_t middle = firstMiss + (end - firstMiss) / 2;
    if (thresholdReachesTarget(thresholds[middle], target))
      firstMiss = middle + 1;
    else
      end = middle;
  }
  if (firstMiss == 0)
    return std::nullopt;
  return static_cast<std::uint64_t>(firstMiss);
}

} // namespace

DeterministicPnrRandomStream
DeterministicPnrRandomStream::create(std::uint64_t masterSeed,
                                     std::uint64_t seedIndex,
                                     PnrRandomStreamPurpose purpose) {
  std::array<std::uint8_t, sizeof(seedDomain) - 1 + 8 + 8 + 4> preimage{};
  std::copy(seedDomain, seedDomain + sizeof(seedDomain) - 1, preimage.begin());
  std::size_t offset = sizeof(seedDomain) - 1;
  appendU64Be(preimage, offset, masterSeed);
  appendU64Be(preimage, offset, seedIndex);
  appendU32Be(preimage, offset, static_cast<std::uint32_t>(purpose));
  assert(offset == preimage.size());
  const auto digest = llvm::SHA256::hash(preimage);
  std::array<std::uint64_t, 4> state{
      readU64Be(digest.data()), readU64Be(digest.data() + 8),
      readU64Be(digest.data() + 16), readU64Be(digest.data() + 24)};
  if (state[0] == 0 && state[1] == 0 && state[2] == 0 && state[3] == 0)
    state[0] = UINT64_C(0x9e3779b97f4a7c15);
  return DeterministicPnrRandomStream(state);
}

std::uint64_t DeterministicPnrRandomStream::nextU64() {
  const std::uint64_t result = rotateLeft(state_[1] * 5, 7) * 9;
  const std::uint64_t shifted = state_[1] << 17;
  state_[2] ^= state_[0];
  state_[3] ^= state_[1];
  state_[1] ^= state_[2];
  state_[0] ^= state_[3];
  state_[2] ^= shifted;
  state_[3] = rotateLeft(state_[3], 45);
  return result;
}

llvm::Expected<std::uint64_t>
DeterministicPnrRandomStream::nextBounded(std::uint64_t upperBound) {
  if (upperBound == 0)
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "bounded random domain must be positive");
  const std::uint64_t threshold = (UINT64_C(0) - upperBound) % upperBound;
  std::uint64_t value = 0;
  do {
    value = nextU64();
  } while (value < threshold);
  return value % upperBound;
}

llvm::Expected<bool> loom::pnr::acceptAnnealingDelta(
    dse::ObjectiveSignedDifference delta, std::uint64_t temperature,
    DeterministicPnrRandomStream &acceptanceStream) {
  if (delta.sign != dse::ObjectiveDifferenceSign::Positive)
    return true;
  auto ratioIndex = positiveDeltaRatioIndex(delta.magnitude, temperature);
  if (!ratioIndex)
    return ratioIndex.takeError();
  const std::uint64_t threshold = expNegativeQ64Threshold(*ratioIndex);
  return acceptanceStream.nextU64() < threshold;
}

llvm::Expected<std::uint64_t> loom::pnr::calibrateAnnealingTemperature(
    const ResolvedPnrAnnealingPolicy &policy,
    llvm::ArrayRef<dse::ObjectiveWideValue> positiveDeltas) {
  if (llvm::Error error = validateResolvedPnrAnnealingPolicy(policy))
    return std::move(error);
  if (positiveDeltas.empty())
    return policy.minimumTemperature;

  std::vector<dse::ObjectiveWideValue> sorted(positiveDeltas.begin(),
                                              positiveDeltas.end());
  std::stable_sort(sorted.begin(), sorted.end());
  llvm::APInt quantileProduct(128, policy.positiveDeltaQuantile.numerator);
  quantileProduct *= llvm::APInt(128, sorted.size() - 1);
  const std::uint64_t quantileIndex =
      quantileProduct
          .udiv(llvm::APInt(128, policy.positiveDeltaQuantile.denominator))
          .getZExtValue();
  const dse::ObjectiveWideValue selected = sorted[quantileIndex];
  const std::optional<std::uint64_t> maximumRatio =
      maximumTargetRatioIndex(policy.targetInitialAcceptance);
  if (!maximumRatio || (selected.high == 0 && selected.low == 0))
    return policy.fallbackTemperature;

  llvm::APInt numerator = wideMagnitude(selected).zext(192);
  numerator *= llvm::APInt(192, 256);
  const llvm::APInt denominator(192, *maximumRatio);
  llvm::APInt temperature = numerator.udiv(denominator);
  if (!numerator.urem(denominator).isZero())
    ++temperature;
  if (temperature.ugt(std::numeric_limits<std::uint64_t>::max()))
    return policy.fallbackTemperature;
  return std::max(temperature.getZExtValue(), policy.minimumTemperature);
}

llvm::Expected<std::uint64_t>
loom::pnr::annealingProposalsPerLevel(const ResolvedPnrAnnealingPolicy &policy,
                                      std::uint64_t movableDecisionCount) {
  if (llvm::Error error = validateResolvedPnrAnnealingPolicy(policy))
    return std::move(error);
  llvm::APInt count(128, policy.proposalsPerMovableDecision);
  count *= llvm::APInt(128, movableDecisionCount);
  count += llvm::APInt(128, policy.proposalsPerLevelBase);
  if (count.ugt(std::numeric_limits<std::uint64_t>::max()))
    return llvm::createStringError(
        std::make_error_code(std::errc::value_too_large),
        "annealing proposal count overflow");
  return count.getZExtValue();
}

llvm::Expected<AnnealingTemperatureSchedule>
AnnealingTemperatureSchedule::create(const ResolvedPnrAnnealingPolicy &policy,
                                     std::uint64_t initialTemperature) {
  if (llvm::Error error = validateResolvedPnrAnnealingPolicy(policy))
    return std::move(error);
  if (initialTemperature == 0)
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "initial annealing temperature must be positive");
  return AnnealingTemperatureSchedule(
      policy.minimumTemperature, policy.coolingRatio,
      std::max(initialTemperature, policy.minimumTemperature));
}

bool AnnealingTemperatureSchedule::advanceAfterCompletedLevel() {
  if (isFinalLevel())
    return false;
  llvm::APInt next(128, temperature_);
  next *= llvm::APInt(128, coolingRatio_.numerator);
  next = next.udiv(llvm::APInt(128, coolingRatio_.denominator));
  temperature_ = std::max(next.getZExtValue(), minimumTemperature_);
  return true;
}
