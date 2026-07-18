#include "Common/ArtifactFinalizer.h"

#include "ArtifactFinalizerInternal.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace loom {
namespace {

constexpr char identityDomain[] = "loom.artifact.identity.v1\0";
constexpr std::size_t identityDomainSize = sizeof(identityDomain) - 1;

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

llvm::Error invalidPreimage(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

llvm::Expected<std::uint32_t> readU32Be(llvm::ArrayRef<std::uint8_t> bytes,
                                        std::size_t &offset) {
  if (bytes.size() - offset < 4)
    return invalidPreimage("truncated artifact identity preimage");
  const std::uint32_t value =
      (static_cast<std::uint32_t>(bytes[offset]) << 24) |
      (static_cast<std::uint32_t>(bytes[offset + 1]) << 16) |
      (static_cast<std::uint32_t>(bytes[offset + 2]) << 8) |
      static_cast<std::uint32_t>(bytes[offset + 3]);
  offset += 4;
  return value;
}

llvm::Expected<std::uint64_t> readU64Be(llvm::ArrayRef<std::uint8_t> bytes,
                                        std::size_t &offset) {
  if (bytes.size() - offset < 8)
    return invalidPreimage("truncated artifact identity preimage");
  std::uint64_t value = 0;
  for (unsigned index = 0; index < 8; ++index)
    value = (value << 8) | bytes[offset + index];
  offset += 8;
  return value;
}

} // namespace

llvm::Expected<ArtifactIdentity>
ArtifactIdentity::fromBytes(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != byteSize)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "artifact identity requires exactly 32 bytes");
  Storage storage;
  std::copy(bytes.begin(), bytes.end(), storage.begin());
  return ArtifactIdentity(storage);
}

std::vector<std::uint8_t> detail::buildArtifactIdentityPreimage(
    const ArtifactSchemaDescriptor &schema,
    const CanonicalSemanticBytes &canonicalBytes) {
  assert(schema.identity.size() <= std::numeric_limits<std::uint32_t>::max());
  const std::size_t totalSize = identityDomainSize + 4 +
                                schema.identity.size() + 4 + 4 + 8 +
                                canonicalBytes.bytes().size();

  std::vector<std::uint8_t> preimage;
  preimage.reserve(totalSize);
  preimage.insert(preimage.end(), identityDomain,
                  identityDomain + identityDomainSize);
  appendU32Be(preimage, static_cast<std::uint32_t>(schema.identity.size()));
  preimage.insert(preimage.end(), schema.identity.bytes_begin(),
                  schema.identity.bytes_end());
  appendU32Be(preimage, schema.version.major);
  appendU32Be(preimage, schema.version.minor);
  appendU64Be(preimage, canonicalBytes.bytes().size());
  preimage.insert(preimage.end(), canonicalBytes.bytes().begin(),
                  canonicalBytes.bytes().end());
  return preimage;
}

llvm::Error detail::validateArtifactIdentityPreimage(
    llvm::ArrayRef<std::uint8_t> preimage) {
  if (preimage.size() < identityDomainSize ||
      !std::equal(identityDomain, identityDomain + identityDomainSize,
                  preimage.begin()))
    return invalidPreimage("invalid artifact identity domain");

  std::size_t offset = identityDomainSize;
  auto schemaLength = readU32Be(preimage, offset);
  if (!schemaLength)
    return schemaLength.takeError();
  if (*schemaLength > preimage.size() - offset)
    return invalidPreimage("truncated artifact schema identity");
  offset += *schemaLength;

  auto major = readU32Be(preimage, offset);
  if (!major)
    return major.takeError();
  auto minor = readU32Be(preimage, offset);
  if (!minor)
    return minor.takeError();
  auto semanticLength = readU64Be(preimage, offset);
  if (!semanticLength)
    return semanticLength.takeError();
  if (*semanticLength != preimage.size() - offset)
    return invalidPreimage("artifact semantic byte length mismatch");
  return llvm::Error::success();
}

ArtifactIdentity
finalizeArtifactIdentity(const ArtifactSchemaDescriptor &schema,
                         const CanonicalSemanticBytes &canonicalBytes) {
  const std::vector<std::uint8_t> preimage =
      detail::buildArtifactIdentityPreimage(schema, canonicalBytes);
  return ArtifactIdentity(llvm::SHA256::hash(preimage));
}

} // namespace loom
