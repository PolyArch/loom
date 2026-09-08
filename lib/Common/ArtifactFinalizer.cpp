#include "Common/ArtifactFinalizer.h"
#include "Common/BlobDigest.h"

#include "ArtifactFinalizerInternal.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
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

std::vector<std::uint8_t>
buildIdentityHeader(const ArtifactSchemaDescriptor &schema,
                    std::size_t semanticSize) {
  assert(schema.identity.size() <= std::numeric_limits<std::uint32_t>::max());
  std::vector<std::uint8_t> header;
  header.reserve(identityDomainSize + 4 + schema.identity.size() + 4 + 4 + 8);
  header.insert(header.end(), identityDomain,
                identityDomain + identityDomainSize);
  appendU32Be(header, static_cast<std::uint32_t>(schema.identity.size()));
  header.insert(header.end(), schema.identity.bytes_begin(),
                schema.identity.bytes_end());
  appendU32Be(header, schema.version.major);
  appendU32Be(header, schema.version.minor);
  appendU64Be(header, semanticSize);
  return header;
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

struct CanonicalSemanticBytes::Storage {
  explicit Storage(std::vector<std::uint8_t> bytes)
      : bytes(std::move(bytes)) {}

  struct IdentityMemo {
    std::string schemaIdentity;
    SchemaVersion schemaVersion;
    ArtifactIdentity identity;
  };

  const std::vector<std::uint8_t> bytes;
  std::mutex identityMutex;
  // Copies share immutable bytes and their last exact-schema identity. A
  // different schema replaces this bounded memo; it never changes the bytes.
  std::optional<IdentityMemo> identity;
};

CanonicalSemanticBytes::CanonicalSemanticBytes(std::vector<std::uint8_t> bytes)
    : storage_(std::make_shared<Storage>(std::move(bytes))) {}

llvm::ArrayRef<std::uint8_t> CanonicalSemanticBytes::bytes() const {
  return storage_->bytes;
}

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
  std::vector<std::uint8_t> preimage =
      buildIdentityHeader(schema, canonicalBytes.bytes().size());
  preimage.reserve(preimage.size() + canonicalBytes.bytes().size());
  preimage.insert(preimage.end(), canonicalBytes.bytes().begin(),
                  canonicalBytes.bytes().end());
  return preimage;
}

llvm::Expected<detail::ParsedArtifactIdentityPreimage>
detail::parseArtifactIdentityPreimage(llvm::ArrayRef<std::uint8_t> preimage) {
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
  const llvm::StringRef schemaIdentity(
      reinterpret_cast<const char *>(preimage.data() + offset), *schemaLength);
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
  return detail::ParsedArtifactIdentityPreimage{
      schemaIdentity, SchemaVersion{*major, *minor}, preimage.slice(offset)};
}

ArtifactIdentity detail::finalizeArtifactIdentityPreimage(
    llvm::ArrayRef<std::uint8_t> preimage) {
  return llvm::cantFail(
      ArtifactIdentity::fromBytes(computeBlobDigest(preimage).bytes()));
}

ArtifactIdentity
finalizeArtifactIdentity(const ArtifactSchemaDescriptor &schema,
                         const CanonicalSemanticBytes &canonicalBytes) {
  auto &storage = *canonicalBytes.storage_;
  std::lock_guard<std::mutex> lock(storage.identityMutex);
  if (storage.identity &&
      storage.identity->schemaIdentity == schema.identity &&
      storage.identity->schemaVersion == schema.version)
    return storage.identity->identity;

  const auto header = buildIdentityHeader(schema, storage.bytes.size());
  auto digest = llvm::cantFail(BlobDigestBuilder::create());
  llvm::cantFail(digest.update(header));
  llvm::cantFail(digest.update(storage.bytes));
  const ArtifactIdentity identity = llvm::cantFail(
      ArtifactIdentity::fromBytes(llvm::cantFail(digest.finish()).bytes()));
  storage.identity.emplace(CanonicalSemanticBytes::Storage::IdentityMemo{
      schema.identity.str(), schema.version, identity});
  return identity;
}

} // namespace loom
