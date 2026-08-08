#ifndef LOOM_COMMON_SPECIALMATHACCURACY_H
#define LOOM_COMMON_SPECIALMATHACCURACY_H

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace loom {

inline constexpr llvm::StringLiteral kSpecialMathAccuracyAttrName =
    "loom.special_math_accuracy";

enum class SpecialMathAccuracyTier : std::uint8_t {
  CorrectlyRounded,
  Max1Ulp,
  Max2Ulp,
  Max4Ulp,
};

llvm::ArrayRef<SpecialMathAccuracyTier> specialMathAccuracyTiers();
llvm::StringRef stringifySpecialMathAccuracyTier(SpecialMathAccuracyTier tier);
std::optional<SpecialMathAccuracyTier>
symbolizeSpecialMathAccuracyTier(llvm::StringRef spelling);

llvm::Error validateSpecialMathAccuracyContract(SpecialMathAccuracyTier tier,
                                                bool approximationPermitted);

/// Whether `guarantee` is at least as strong as `acceptedMaximum`.
llvm::Expected<bool>
specialMathAccuracyRefines(SpecialMathAccuracyTier guarantee,
                           SpecialMathAccuracyTier acceptedMaximum);

llvm::Expected<std::uint32_t>
specialMathAccuracyWireTag(SpecialMathAccuracyTier tier);
llvm::Expected<SpecialMathAccuracyTier>
specialMathAccuracyTierFromWireTag(std::uint32_t tag);
llvm::Expected<CanonicalSemanticBytes>
encodeSpecialMathAccuracyTier(SpecialMathAccuracyTier tier);
llvm::Expected<SpecialMathAccuracyTier>
decodeSpecialMathAccuracyTier(llvm::ArrayRef<std::uint8_t> bytes);

} // namespace loom

#endif // LOOM_COMMON_SPECIALMATHACCURACY_H
