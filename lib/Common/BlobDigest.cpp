#include "Common/BlobDigest.h"

#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

namespace loom {

llvm::Expected<BlobDigest>
BlobDigest::fromBytes(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != byteSize)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "blob digest requires exactly 32 bytes");
  Storage storage;
  std::copy(bytes.begin(), bytes.end(), storage.begin());
  return BlobDigest(storage);
}

BlobDigest computeBlobDigest(llvm::ArrayRef<std::uint8_t> logicalBytes) {
  return BlobDigest(llvm::SHA256::hash(logicalBytes));
}

std::string formatBlobDigestHex(const BlobDigest &digest) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(BlobDigest::byteSize * 2);
  for (std::uint8_t byte : digest.bytes()) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 0x0f]);
  }
  return result;
}

llvm::Expected<BlobDigest> parseBlobDigestHex(llvm::StringRef spelling) {
  if (spelling.size() != BlobDigest::byteSize * 2)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "blob digest must use exactly 64 lowercase hexadecimal characters");

  auto parseNibble = [](char character) -> int {
    if (character >= '0' && character <= '9')
      return character - '0';
    if (character >= 'a' && character <= 'f')
      return character - 'a' + 10;
    return -1;
  };

  std::array<std::uint8_t, BlobDigest::byteSize> bytes;
  for (std::size_t index = 0; index < spelling.size(); index += 2) {
    const int high = parseNibble(spelling[index]);
    const int low = parseNibble(spelling[index + 1]);
    if (high < 0 || low < 0)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "blob digest must use lowercase hexadecimal");
    bytes[index / 2] = static_cast<std::uint8_t>((high << 4) | low);
  }
  return BlobDigest::fromBytes(bytes);
}

} // namespace loom
