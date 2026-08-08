#include "Common/SpecialMathAccuracy.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSwitch.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <system_error>
#include <vector>

namespace loom {

namespace {

constexpr char kDomain[] = "loom.special-math-accuracy-tier\0";
constexpr std::uint32_t kCodecMajor = 1;
constexpr std::uint32_t kCodecMinor = 0;

constexpr std::array<SpecialMathAccuracyTier, 4> kTiers = {
    SpecialMathAccuracyTier::CorrectlyRounded,
    SpecialMathAccuracyTier::Max1Ulp,
    SpecialMathAccuracyTier::Max2Ulp,
    SpecialMathAccuracyTier::Max4Ulp,
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "special_math_accuracy_invalid: " + message);
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

std::uint32_t readU32(llvm::ArrayRef<std::uint8_t> bytes) {
  return (static_cast<std::uint32_t>(bytes[0]) << 24) |
         (static_cast<std::uint32_t>(bytes[1]) << 16) |
         (static_cast<std::uint32_t>(bytes[2]) << 8) |
         static_cast<std::uint32_t>(bytes[3]);
}

llvm::Expected<unsigned> strengthRank(SpecialMathAccuracyTier tier) {
  auto found = std::find(kTiers.begin(), kTiers.end(), tier);
  if (found == kTiers.end())
    return invalid("unknown special-math accuracy tier");
  return static_cast<unsigned>(std::distance(kTiers.begin(), found));
}

} // namespace

llvm::ArrayRef<SpecialMathAccuracyTier> specialMathAccuracyTiers() {
  return kTiers;
}

llvm::StringRef stringifySpecialMathAccuracyTier(SpecialMathAccuracyTier tier) {
  switch (tier) {
  case SpecialMathAccuracyTier::CorrectlyRounded:
    return "CorrectlyRounded";
  case SpecialMathAccuracyTier::Max1Ulp:
    return "Max1Ulp";
  case SpecialMathAccuracyTier::Max2Ulp:
    return "Max2Ulp";
  case SpecialMathAccuracyTier::Max4Ulp:
    return "Max4Ulp";
  }
  return {};
}

std::optional<SpecialMathAccuracyTier>
symbolizeSpecialMathAccuracyTier(llvm::StringRef spelling) {
  return llvm::StringSwitch<std::optional<SpecialMathAccuracyTier>>(spelling)
      .Case("CorrectlyRounded", SpecialMathAccuracyTier::CorrectlyRounded)
      .Case("Max1Ulp", SpecialMathAccuracyTier::Max1Ulp)
      .Case("Max2Ulp", SpecialMathAccuracyTier::Max2Ulp)
      .Case("Max4Ulp", SpecialMathAccuracyTier::Max4Ulp)
      .Default(std::nullopt);
}

llvm::Error validateSpecialMathAccuracyContract(SpecialMathAccuracyTier tier,
                                                bool approximationPermitted) {
  auto rank = strengthRank(tier);
  if (!rank)
    return rank.takeError();
  if (tier != SpecialMathAccuracyTier::CorrectlyRounded &&
      !approximationPermitted)
    return invalid("non-correctly-rounded contract requires afn");
  return llvm::Error::success();
}

llvm::Expected<bool>
specialMathAccuracyRefines(SpecialMathAccuracyTier guarantee,
                           SpecialMathAccuracyTier acceptedMaximum) {
  auto guaranteeRank = strengthRank(guarantee);
  if (!guaranteeRank)
    return guaranteeRank.takeError();
  auto acceptedRank = strengthRank(acceptedMaximum);
  if (!acceptedRank)
    return acceptedRank.takeError();
  return *guaranteeRank <= *acceptedRank;
}

llvm::Expected<std::uint32_t>
specialMathAccuracyWireTag(SpecialMathAccuracyTier tier) {
  switch (tier) {
  case SpecialMathAccuracyTier::CorrectlyRounded:
    return 0x4c534101;
  case SpecialMathAccuracyTier::Max1Ulp:
    return 0x4c534102;
  case SpecialMathAccuracyTier::Max2Ulp:
    return 0x4c534103;
  case SpecialMathAccuracyTier::Max4Ulp:
    return 0x4c534104;
  }
  return invalid("unknown special-math accuracy tier");
}

llvm::Expected<SpecialMathAccuracyTier>
specialMathAccuracyTierFromWireTag(std::uint32_t tag) {
  switch (tag) {
  case 0x4c534101:
    return SpecialMathAccuracyTier::CorrectlyRounded;
  case 0x4c534102:
    return SpecialMathAccuracyTier::Max1Ulp;
  case 0x4c534103:
    return SpecialMathAccuracyTier::Max2Ulp;
  case 0x4c534104:
    return SpecialMathAccuracyTier::Max4Ulp;
  default:
    return invalid("unknown special-math accuracy tier wire tag");
  }
}

llvm::Expected<CanonicalSemanticBytes>
encodeSpecialMathAccuracyTier(SpecialMathAccuracyTier tier) {
  auto tag = specialMathAccuracyWireTag(tier);
  if (!tag)
    return tag.takeError();
  std::vector<std::uint8_t> bytes(
      reinterpret_cast<const std::uint8_t *>(kDomain),
      reinterpret_cast<const std::uint8_t *>(kDomain) + sizeof(kDomain) - 1);
  appendU32(bytes, kCodecMajor);
  appendU32(bytes, kCodecMinor);
  appendU32(bytes, *tag);
  return CanonicalSemanticBytes(std::move(bytes));
}

llvm::Expected<SpecialMathAccuracyTier>
decodeSpecialMathAccuracyTier(llvm::ArrayRef<std::uint8_t> bytes) {
  constexpr std::size_t domainSize = sizeof(kDomain) - 1;
  constexpr std::size_t encodedSize = domainSize + 12;
  if (bytes.size() < encodedSize)
    return invalid("truncated special-math accuracy tier");
  llvm::ArrayRef<std::uint8_t> expected(
      reinterpret_cast<const std::uint8_t *>(kDomain), domainSize);
  if (bytes.take_front(domainSize) != expected)
    return invalid("wrong semantic domain");
  if (readU32(bytes.slice(domainSize, 4)) != kCodecMajor ||
      readU32(bytes.slice(domainSize + 4, 4)) != kCodecMinor)
    return invalid("unsupported version");
  if (bytes.size() != encodedSize)
    return invalid("trailing bytes");
  return specialMathAccuracyTierFromWireTag(readU32(bytes.take_back(4)));
}

} // namespace loom
