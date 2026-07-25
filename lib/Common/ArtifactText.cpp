#include "Common/ArtifactText.h"

#include "llvm/ADT/Twine.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace loom {
namespace {

llvm::Error artifactTextError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

llvm::Expected<std::uint32_t>
parseSchemaVersionComponent(llvm::StringRef spelling) {
  if (spelling.empty() || (spelling.size() > 1 && spelling.front() == '0'))
    return artifactTextError(
        "schema version must use canonical X.Y uint32 spelling");

  std::uint32_t value = 0;
  for (char character : spelling) {
    if (character < '0' || character > '9')
      return artifactTextError(
          "schema version must use canonical X.Y uint32 spelling");
    const std::uint32_t digit = static_cast<std::uint32_t>(character - '0');
    if (value > (std::numeric_limits<std::uint32_t>::max() - digit) / 10)
      return artifactTextError(
          "schema version must use canonical X.Y uint32 spelling");
    value = value * 10 + digit;
  }
  return value;
}

} // namespace

std::string formatSchemaVersion(SchemaVersion version) {
  return std::to_string(version.major) + "." + std::to_string(version.minor);
}

llvm::Expected<SchemaVersion> parseSchemaVersion(llvm::StringRef spelling) {
  const std::size_t separator = spelling.find('.');
  if (separator == llvm::StringRef::npos || separator == 0 ||
      separator + 1 == spelling.size() ||
      spelling.find('.', separator + 1) != llvm::StringRef::npos)
    return artifactTextError(
        "schema version must use canonical X.Y uint32 spelling");

  auto major = parseSchemaVersionComponent(spelling.take_front(separator));
  if (!major)
    return major.takeError();
  auto minor = parseSchemaVersionComponent(spelling.drop_front(separator + 1));
  if (!minor)
    return minor.takeError();

  return SchemaVersion{*major, *minor};
}

std::string formatArtifactIdentityHex(const ArtifactIdentity &identity) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(ArtifactIdentity::byteSize * 2);
  for (std::uint8_t byte : identity.bytes()) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 0x0f]);
  }
  return result;
}

llvm::Expected<ArtifactIdentity>
parseArtifactIdentityHex(llvm::StringRef spelling) {
  if (spelling.size() != ArtifactIdentity::byteSize * 2)
    return artifactTextError("artifact identity must use exactly 64 lowercase "
                             "hexadecimal characters");

  auto parseNibble = [](char character) -> int {
    if (character >= '0' && character <= '9')
      return character - '0';
    if (character >= 'a' && character <= 'f')
      return character - 'a' + 10;
    return -1;
  };

  std::array<std::uint8_t, ArtifactIdentity::byteSize> bytes;
  for (std::size_t index = 0; index < spelling.size(); index += 2) {
    const int high = parseNibble(spelling[index]);
    const int low = parseNibble(spelling[index + 1]);
    if (high < 0 || low < 0)
      return artifactTextError(
          "artifact identity must use lowercase hexadecimal");
    bytes[index / 2] = static_cast<std::uint8_t>((high << 4) | low);
  }
  return ArtifactIdentity::fromBytes(bytes);
}

std::string
formatArtifactLocalPayloadHex(llvm::ArrayRef<std::uint8_t> payload) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(payload.size() * 2);
  for (std::uint8_t byte : payload) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 0x0f]);
  }
  return result;
}

llvm::Expected<std::vector<std::uint8_t>>
parseArtifactLocalPayloadHex(llvm::StringRef spelling) {
  auto parseNibble = [](char character) -> int {
    if (character >= '0' && character <= '9')
      return character - '0';
    if (character >= 'a' && character <= 'f')
      return character - 'a' + 10;
    return -1;
  };

  if ((spelling.size() % 2) != 0)
    return artifactTextError("an artifact local-reference payload must use "
                             "paired lowercase hexadecimal characters");
  std::vector<std::uint8_t> payload;
  payload.reserve(spelling.size() / 2);
  for (std::size_t index = 0; index < spelling.size(); index += 2) {
    const int high = parseNibble(spelling[index]);
    const int low = parseNibble(spelling[index + 1]);
    if (high < 0 || low < 0)
      return artifactTextError("an artifact local-reference payload must use "
                               "paired lowercase hexadecimal characters");
    payload.push_back(static_cast<std::uint8_t>((high << 4) | low));
  }
  return payload;
}

} // namespace loom
