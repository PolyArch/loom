#include "Common/ExternalFileFingerprint.h"

#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace loom {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "external_file_fingerprint_invalid: " + message);
}

} // namespace

llvm::Expected<ExternalFileFingerprint>
ExternalFileFingerprint::fromBytes(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != byteSize)
    return invalid("external file fingerprint requires exactly 32 bytes");
  Storage storage;
  std::copy(bytes.begin(), bytes.end(), storage.begin());
  return ExternalFileFingerprint(storage);
}

std::string
formatExternalFileFingerprint(const ExternalFileFingerprint &fingerprint) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(ExternalFileFingerprint::byteSize * 2);
  for (std::uint8_t byte : fingerprint.bytes()) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 0x0f]);
  }
  return result;
}

llvm::Expected<ExternalFileFingerprint>
parseExternalFileFingerprint(llvm::StringRef spelling) {
  if (spelling.size() != ExternalFileFingerprint::byteSize * 2)
    return invalid("external file fingerprint must use exactly 64 lowercase "
                   "hexadecimal characters");
  const auto nibble = [](char character) -> int {
    if (character >= '0' && character <= '9')
      return character - '0';
    if (character >= 'a' && character <= 'f')
      return character - 'a' + 10;
    return -1;
  };
  ExternalFileFingerprint::Storage bytes{};
  for (std::size_t index = 0; index < spelling.size(); index += 2) {
    const int high = nibble(spelling[index]);
    const int low = nibble(spelling[index + 1]);
    if (high < 0 || low < 0)
      return invalid("external file fingerprint must use lowercase "
                     "hexadecimal characters");
    bytes[index / 2] = static_cast<std::uint8_t>((high << 4) | low);
  }
  return ExternalFileFingerprint::fromBytes(bytes);
}

} // namespace loom
