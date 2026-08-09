#ifndef FABRIC_IR_CROSSPOINT_H
#define FABRIC_IR_CROSSPOINT_H

#include "llvm/Support/Error.h"

#include <cstdint>

namespace fabric {

inline constexpr std::uint64_t kPeCrosspointWarningThreshold = 16;
inline constexpr std::uint64_t kPeCrosspointLimit = 64;

/// Returns the exact product of two nonempty crosspoint dimensions. Overflow
/// is rejected before multiplication.
llvm::Expected<std::uint64_t> checkedCrosspointCount(std::uint64_t inputCount,
                                                     std::uint64_t outputCount);

/// Returns the exact PE boundary crosspoint count when it is within the
/// physical PE limit.
llvm::Expected<std::uint64_t>
validatedPeBoundaryCrosspointCount(std::uint64_t inputCount,
                                   std::uint64_t outputCount);

} // namespace fabric

#endif // FABRIC_IR_CROSSPOINT_H
