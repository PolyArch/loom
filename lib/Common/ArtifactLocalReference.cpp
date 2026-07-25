#include "Common/ArtifactLocalReference.h"

#include "ArtifactLocalReferenceRegistry.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <mutex>
#include <vector>

namespace loom {
namespace {

llvm::Error framingError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

llvm::Error
ownerCodecUnavailable(const EncodedArtifactLocalReference &reference) {
  return llvm::make_error<ArtifactLocalReferenceError>(
      ArtifactLocalReferenceErrorKind::OwnerCodecUnavailable,
      ("no registered owner codec for local-reference kind " +
       llvm::Twine(reference.ownerLocalKind) + " of schema '" +
       reference.artifact.schemaIdentity + "'")
          .str());
}

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

struct LocalReferenceKindEntry {
  ArtifactSchemaDescriptor schema;
  std::uint32_t kind;
  ArtifactLocalReferenceCodec codec;
};

std::vector<LocalReferenceKindEntry> &localReferenceKinds() {
  static std::vector<LocalReferenceKindEntry> entries;
  return entries;
}

std::mutex &localReferenceKindMutex() {
  static std::mutex mutex;
  return mutex;
}

} // namespace

char ArtifactLocalReferenceError::ID = 0;

void ArtifactLocalReferenceError::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code ArtifactLocalReferenceError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

bool artifactRootReferenceLess(const ArtifactRootReference &lhs,
                               const ArtifactRootReference &rhs) {
  if (lhs.schemaIdentity != rhs.schemaIdentity)
    return lhs.schemaIdentity < rhs.schemaIdentity;
  if (lhs.schemaVersion.major != rhs.schemaVersion.major)
    return lhs.schemaVersion.major < rhs.schemaVersion.major;
  if (lhs.schemaVersion.minor != rhs.schemaVersion.minor)
    return lhs.schemaVersion.minor < rhs.schemaVersion.minor;
  return lhs.artifact.bytes() < rhs.artifact.bytes();
}

std::vector<std::uint8_t>
encodeArtifactRootReference(const ArtifactRootReference &reference) {
  std::vector<std::uint8_t> bytes;
  appendU32Be(bytes,
              static_cast<std::uint32_t>(reference.schemaIdentity.size()));
  bytes.insert(bytes.end(), reference.schemaIdentity.begin(),
               reference.schemaIdentity.end());
  appendU32Be(bytes, reference.schemaVersion.major);
  appendU32Be(bytes, reference.schemaVersion.minor);
  bytes.insert(bytes.end(), reference.artifact.bytes().begin(),
               reference.artifact.bytes().end());
  return bytes;
}

std::vector<std::uint8_t>
encodeArtifactLocalReference(const EncodedArtifactLocalReference &reference) {
  std::vector<std::uint8_t> bytes =
      encodeArtifactRootReference(reference.artifact);
  appendU32Be(bytes, reference.ownerLocalKind);
  appendU64Be(bytes, reference.payload.size());
  bytes.insert(bytes.end(), reference.payload.begin(), reference.payload.end());
  return bytes;
}

llvm::Error
registerArtifactLocalReferenceKind(const ArtifactSchemaDescriptor &ownerSchema,
                                   std::uint32_t ownerLocalKind,
                                   ArtifactLocalReferenceCodec codec) {
  if (!codec.strictDecode || !codec.validate)
    return framingError("an artifact local-reference kind requires both a "
                        "strict decoder and a validator");
  std::lock_guard<std::mutex> lock(localReferenceKindMutex());
  for (const LocalReferenceKindEntry &entry : localReferenceKinds()) {
    if (entry.schema == ownerSchema && entry.kind == ownerLocalKind) {
      if (entry.codec.strictDecode == codec.strictDecode &&
          entry.codec.validate == codec.validate)
        return llvm::Error::success();
      return framingError("conflicting registration for local-reference kind " +
                          llvm::Twine(ownerLocalKind) + " of schema '" +
                          ownerSchema.identity + "'");
    }
  }
  localReferenceKinds().push_back({ownerSchema, ownerLocalKind, codec});
  return llvm::Error::success();
}

std::optional<ArtifactLocalReferenceCodec>
findArtifactLocalReferenceKind(const ArtifactSchemaDescriptor &ownerSchema,
                               std::uint32_t ownerLocalKind) {
  std::lock_guard<std::mutex> lock(localReferenceKindMutex());
  for (const LocalReferenceKindEntry &entry : localReferenceKinds())
    if (entry.schema == ownerSchema && entry.kind == ownerLocalKind)
      return entry.codec;
  return std::nullopt;
}

std::optional<ArtifactSchemaDescriptor>
findArtifactLocalReferenceSchema(llvm::StringRef identity,
                                 SchemaVersion version) {
  std::lock_guard<std::mutex> lock(localReferenceKindMutex());
  for (const LocalReferenceKindEntry &entry : localReferenceKinds())
    if (entry.schema.identity == identity && entry.schema.version == version)
      return entry.schema;
  return std::nullopt;
}

llvm::Error
validateArtifactLocalReference(const EncodedArtifactLocalReference &reference) {
  std::optional<ArtifactSchemaDescriptor> ownerSchema =
      findArtifactLocalReferenceSchema(reference.artifact.schemaIdentity,
                                       reference.artifact.schemaVersion);
  if (!ownerSchema)
    return ownerCodecUnavailable(reference);
  std::optional<ArtifactLocalReferenceCodec> codec =
      findArtifactLocalReferenceKind(*ownerSchema, reference.ownerLocalKind);
  if (!codec)
    return ownerCodecUnavailable(reference);
  if (llvm::Error error = codec->strictDecode(reference.payload))
    return error;
  return codec->validate(reference);
}

} // namespace loom
