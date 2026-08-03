#include "Fabric/IR/PhysicalTag.h"

#include "llvm/Support/Errc.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <system_error>
#include <vector>

using namespace fabric;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 "invalid Physical Tag value: %s",
                                 message.str().c_str());
}

std::size_t encodedSize(std::uint32_t tagWidthBits) {
  return (static_cast<std::size_t>(tagWidthBits) + 7) / 8;
}

} // namespace

bool fabric::isRepresentablePhysicalTagValue(std::uint32_t tagWidthBits,
                                             const llvm::APInt &value) {
  return tagWidthBits != 0 && value.getActiveBits() <= tagWidthBits;
}

llvm::Expected<std::vector<std::uint8_t>>
fabric::encodePhysicalTagValue(std::uint32_t tagWidthBits,
                               const llvm::APInt &value) {
  if (tagWidthBits == 0)
    return invalid("owner width is zero");
  if (!isRepresentablePhysicalTagValue(tagWidthBits, value))
    return invalid("unsigned value is not representable by the owner width");

  const std::size_t byteCount = encodedSize(tagWidthBits);
  std::vector<std::uint8_t> result(byteCount, 0);
  const llvm::APInt normalized = value.zextOrTrunc(tagWidthBits);
  for (std::size_t littleByte = 0; littleByte < byteCount; ++littleByte) {
    const std::uint64_t bitOffset = littleByte * 8;
    const unsigned bitCount = static_cast<unsigned>(
        std::min<std::uint64_t>(8, tagWidthBits - bitOffset));
    result[byteCount - littleByte - 1] = static_cast<std::uint8_t>(
        normalized.extractBitsAsZExtValue(bitCount, bitOffset));
  }
  return result;
}

llvm::Expected<llvm::APInt>
fabric::decodePhysicalTagValue(std::uint32_t tagWidthBits,
                               llvm::ArrayRef<std::uint8_t> bytes) {
  if (tagWidthBits == 0)
    return invalid("owner width is zero");
  const std::size_t byteCount = encodedSize(tagWidthBits);
  if (bytes.size() != byteCount)
    return invalid("byte count does not match the owner width");

  const unsigned usedHighBits = ((tagWidthBits - 1) % 8) + 1;
  if (usedHighBits != 8) {
    const std::uint8_t paddingMask =
        static_cast<std::uint8_t>(0xffu << usedHighBits);
    if ((bytes.front() & paddingMask) != 0)
      return invalid("unused high bits are not zero");
  }

  llvm::APInt result(tagWidthBits, 0);
  const unsigned shift = std::min<std::uint32_t>(8, tagWidthBits);
  for (std::uint8_t byte : bytes) {
    result <<= shift;
    result |= llvm::APInt(tagWidthBits, byte);
  }
  return result;
}
