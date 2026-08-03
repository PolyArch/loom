#ifndef FABRIC_IR_PHYSICALTAG_H
#define FABRIC_IR_PHYSICALTAG_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace fabric {

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
