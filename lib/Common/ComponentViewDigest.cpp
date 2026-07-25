#include "Common/ComponentViewDigest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace loom {
namespace {

constexpr char componentViewDigestDomain[] = "loom.component.view.digest.v1\0";
constexpr std::size_t componentViewDigestDomainSize =
    sizeof(componentViewDigestDomain) - 1;

/// Largest descriptor length representable in the framed u32 length field.
constexpr std::uint64_t maxFramedDescriptorLength =
    std::numeric_limits<std::uint32_t>::max();

/// Every canonical view length is representable in the framed u64 length field
/// because no supported host has a size_t wider than u64.
static_assert(std::numeric_limits<std::size_t>::max() <=
                  std::numeric_limits<std::uint64_t>::max(),
              "canonical view length must be exact in the framed u64 length");

void appendU32Be(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64Be(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

/// Frames a descriptor already checked against maxFramedDescriptorLength.
std::vector<std::uint8_t>
buildComponentViewDigestPreimage(llvm::ArrayRef<std::uint8_t> descriptorBytes,
                                 llvm::ArrayRef<std::uint8_t> viewBytes) {
  std::vector<std::uint8_t> preimage;
  preimage.reserve(componentViewDigestDomainSize + 4 + descriptorBytes.size() +
                   8 + viewBytes.size());
  preimage.insert(preimage.end(), componentViewDigestDomain,
                  componentViewDigestDomain + componentViewDigestDomainSize);
  appendU32Be(preimage, static_cast<std::uint32_t>(descriptorBytes.size()));
  preimage.insert(preimage.end(), descriptorBytes.begin(),
                  descriptorBytes.end());
  appendU64Be(preimage, static_cast<std::uint64_t>(viewBytes.size()));
  preimage.insert(preimage.end(), viewBytes.begin(), viewBytes.end());
  return preimage;
}

} // namespace

llvm::Expected<ComponentViewDigest>
ComponentViewDigest::fromBytes(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != byteSize)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "component view digest requires exactly 32 bytes");
  Storage storage;
  std::copy(bytes.begin(), bytes.end(), storage.begin());
  return ComponentViewDigest(storage);
}

llvm::Expected<ComponentViewDigest>
computeComponentViewDigest(llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
                           llvm::ArrayRef<std::uint8_t> canonicalViewBytes) {
  if (static_cast<std::uint64_t>(schemaDescriptorBytes.size()) >
      maxFramedDescriptorLength)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "component_view_digest_descriptor_too_large: schema descriptor "
        "length " +
            llvm::Twine(schemaDescriptorBytes.size()) +
            " is not representable in the framed u32 length");
  const std::vector<std::uint8_t> preimage = buildComponentViewDigestPreimage(
      schemaDescriptorBytes, canonicalViewBytes);
  return ComponentViewDigest(llvm::SHA256::hash(preimage));
}

llvm::Error
validateComponentViewDigest(llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
                            llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
                            const ComponentViewDigest &suppliedDigest) {
  llvm::Expected<ComponentViewDigest> recomputed =
      computeComponentViewDigest(schemaDescriptorBytes, canonicalViewBytes);
  if (!recomputed)
    return recomputed.takeError();
  if (*recomputed != suppliedDigest)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "component_view_digest_mismatch: supplied digest does not match the "
        "component view source bytes");
  return llvm::Error::success();
}

std::string formatComponentViewDigestHex(const ComponentViewDigest &digest) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(ComponentViewDigest::byteSize * 2);
  for (std::uint8_t byte : digest.bytes()) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 0x0f]);
  }
  return result;
}

llvm::Expected<ComponentViewDigest>
parseComponentViewDigestHex(llvm::StringRef spelling) {
  if (spelling.size() != ComponentViewDigest::byteSize * 2)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "component view digest must use exactly 64 lowercase hexadecimal "
        "characters");

  auto parseNibble = [](char character) -> int {
    if (character >= '0' && character <= '9')
      return character - '0';
    if (character >= 'a' && character <= 'f')
      return character - 'a' + 10;
    return -1;
  };

  std::array<std::uint8_t, ComponentViewDigest::byteSize> bytes;
  for (std::size_t index = 0; index < spelling.size(); index += 2) {
    const int high = parseNibble(spelling[index]);
    const int low = parseNibble(spelling[index + 1]);
    if (high < 0 || low < 0)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "component view digest must use lowercase hexadecimal");
    bytes[index / 2] = static_cast<std::uint8_t>((high << 4) | low);
  }
  return ComponentViewDigest::fromBytes(bytes);
}

} // namespace loom
