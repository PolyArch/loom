#ifndef FABRIC_IR_PHYSICALTAG_H
#define FABRIC_IR_PHYSICALTAG_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace fabric {

/// Returns the unique width-independent APInt representation of an unsigned
/// Physical Tag value. The Fabric owner, not APInt storage, supplies any port
/// width needed for encoding or hardware comparison.
inline llvm::APInt canonicalPhysicalTagValue(const llvm::APInt &value) {
  return value.zextOrTrunc(std::max(1u, value.getActiveBits()));
}

/// Compares two unsigned Physical Tag values irrespective of their APInt
/// storage widths; storage width is not semantic. Values whose storage fits
/// one machine word compare without widening.
inline int comparePhysicalTagValues(const llvm::APInt &lhs,
                                    const llvm::APInt &rhs) {
  const unsigned leftActiveBits = lhs.getActiveBits();
  const unsigned rightActiveBits = rhs.getActiveBits();
  if (leftActiveBits != rightActiveBits)
    return leftActiveBits < rightActiveBits ? -1 : 1;
  const unsigned wordCount = (leftActiveBits + 63) / 64;
  for (unsigned word = wordCount; word != 0; --word) {
    const std::uint64_t left = lhs.getRawData()[word - 1];
    const std::uint64_t right = rhs.getRawData()[word - 1];
    if (left != right)
      return left < right ? -1 : 1;
  }
  return 0;
}

/// Returns whether an unsigned Physical Tag value fits the exact owner width.
/// The APInt's storage width is not semantic; the Fabric owner supplies the
/// sole width authority.
bool isRepresentablePhysicalTagValue(std::uint32_t tagWidthBits,
                                     const llvm::APInt &value);

/// Encodes one Physical Tag as the exact owner-width, big-endian byte string.
/// Unused high bits in the first byte are zero. Width is intentionally absent
/// from the bytes because the exact ResourceUse owner supplies it.
llvm::Expected<std::vector<std::uint8_t>>
encodePhysicalTagValue(std::uint32_t tagWidthBits, const llvm::APInt &value);

/// Decodes and validates the unique encoding produced above. The returned
/// APInt has exactly `tagWidthBits` bits.
llvm::Expected<llvm::APInt>
decodePhysicalTagValue(std::uint32_t tagWidthBits,
                       llvm::ArrayRef<std::uint8_t> bytes);

} // namespace fabric

#endif // FABRIC_IR_PHYSICALTAG_H
