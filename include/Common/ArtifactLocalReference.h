#ifndef LOOM_COMMON_ARTIFACTLOCALREFERENCE_H
#define LOOM_COMMON_ARTIFACTLOCALREFERENCE_H

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

// The heterogeneous exact-reference carriers owned by the Common artifact
// contract (docs/spec-full-stack-traceability.md). An ArtifactIdentity digest
// alone does not select an importer or validator, so every heterogeneous
// reference carries the exact schema descriptor. Common owns only the outer
// framing and a lifetime-safe registry dispatch; each Artifact family owns
// its closed owner-local kind ordinals, canonical payload bytes, typed
// decoder, and target validation against the exact imported Artifact it
// resolves through its own typed view boundary.
//
// Registry lookups return value copies taken under the registry lock, so a
// later registration can never invalidate an earlier result. A consumer never
// reinterprets an owner payload, erases it to a consumer-local integer, or
// substitutes a tuple, path, symbol, or property bag.

namespace loom {

class ArtifactStore;

/// The complete local-reference type descriptor, derived exactly once as
/// (owner schema, owner-local kind). owner_local_kind is a stable closed
/// ordinal owned by that exact Artifact family and schema version; it is not
/// a global entity kind, consumer enum, textual type name, or native variant
/// index.
struct ArtifactLocalReferenceTypeDescriptor {
  std::string ownerSchemaIdentity;
  SchemaVersion ownerSchemaVersion;
  std::uint32_t ownerLocalKind = 0;

  ArtifactLocalReferenceTypeDescriptor(
      const ArtifactSchemaDescriptor &ownerSchema, std::uint32_t ownerLocalKind)
      : ArtifactLocalReferenceTypeDescriptor(
            ownerSchema.identity.str(), ownerSchema.version, ownerLocalKind) {}

  friend bool operator==(const ArtifactLocalReferenceTypeDescriptor &lhs,
                         const ArtifactLocalReferenceTypeDescriptor &rhs) {
    return lhs.ownerSchemaIdentity == rhs.ownerSchemaIdentity &&
           lhs.ownerSchemaVersion == rhs.ownerSchemaVersion &&
           lhs.ownerLocalKind == rhs.ownerLocalKind;
  }
  friend bool operator!=(const ArtifactLocalReferenceTypeDescriptor &lhs,
                         const ArtifactLocalReferenceTypeDescriptor &rhs) {
    return !(lhs == rhs);
  }

private:
  ArtifactLocalReferenceTypeDescriptor(std::string ownerSchemaIdentity,
                                       SchemaVersion ownerSchemaVersion,
                                       std::uint32_t ownerLocalKind)
      : ownerSchemaIdentity(std::move(ownerSchemaIdentity)),
        ownerSchemaVersion(ownerSchemaVersion), ownerLocalKind(ownerLocalKind) {
  }

  friend struct EncodedArtifactLocalReference;
};

/// The persistent or heterogeneous carrier used to recover a typed
/// ArtifactReference<T> through the owner codec. The payload is exactly the
/// owner-produced canonical bytes. An Artifact root is a separate
/// ArtifactRootReference variant; it is never represented by a reserved local
/// kind or sentinel payload.
struct EncodedArtifactLocalReference {
  ArtifactRootReference artifact;
  std::uint32_t ownerLocalKind = 0;
  std::vector<std::uint8_t> payload;

  ArtifactLocalReferenceTypeDescriptor type() const {
    return ArtifactLocalReferenceTypeDescriptor{
        artifact.schemaIdentity, artifact.schemaVersion, ownerLocalKind};
  }

  friend bool operator==(const EncodedArtifactLocalReference &lhs,
                         const EncodedArtifactLocalReference &rhs) {
    return lhs.artifact == rhs.artifact &&
           lhs.ownerLocalKind == rhs.ownerLocalKind &&
           lhs.payload == rhs.payload;
  }
  friend bool operator!=(const EncodedArtifactLocalReference &lhs,
                         const EncodedArtifactLocalReference &rhs) {
    return !(lhs == rhs);
  }
};

/// Total canonical order over exact root references: schema identity, schema
/// version, then identity bytes. Canonical collections order by this key;
/// authoring order has no meaning.
bool artifactRootReferenceLess(const ArtifactRootReference &lhs,
                               const ArtifactRootReference &rhs);

/// Canonical binary framing of one exact root reference:
/// u32be(length(schema identity)) || bytes(schema identity)
///   || u32be(schema major) || u32be(schema minor)
///   || 32-byte ArtifactIdentity
std::vector<std::uint8_t>
encodeArtifactRootReference(const ArtifactRootReference &reference);

/// One root reference decoded from the beginning of a larger canonical wire.
/// byteCount is the exact prefix length consumed by the Common-owned framing.
struct DecodedArtifactRootReferencePrefix {
  ArtifactRootReference reference;
  std::size_t byteCount;
};

/// Strictly decodes the Common-owned root-reference framing at the beginning
/// of bytes. Trailing bytes belong to the enclosing owner and are not consumed.
llvm::Expected<DecodedArtifactRootReferencePrefix>
decodeArtifactRootReferencePrefix(llvm::ArrayRef<std::uint8_t> bytes);

/// Canonical binary framing of one heterogeneous local reference:
/// the root-reference framing of the containing Artifact
///   || u32be(owner-local kind)
///   || u64be(payload length) || payload bytes
std::vector<std::uint8_t>
encodeArtifactLocalReference(const EncodedArtifactLocalReference &reference);

/// One registered owner-local kind. Canonical payload validation rejects
/// malformed or noncanonical payloads without resolving an Artifact. Target
/// validation receives the exact canonical bytes loaded from ArtifactStore and
/// checks the decoded typed target through the family-owned strict importer.
/// Any type erasure stays inside the owner codec: no untyped view crosses the
/// Common boundary.
struct ArtifactLocalReferenceCodec {
  llvm::Error (*validateCanonicalPayload)(llvm::ArrayRef<std::uint8_t> payload);
  llvm::Error (*validateTarget)(const CanonicalSemanticBytes &artifactBytes,
                                const EncodedArtifactLocalReference &reference);
};

enum class ArtifactLocalReferenceErrorKind : std::uint8_t {
  OwnerCodecUnavailable,
};

class ArtifactLocalReferenceError final
    : public llvm::ErrorInfo<ArtifactLocalReferenceError> {
public:
  static char ID;

  ArtifactLocalReferenceError(ArtifactLocalReferenceErrorKind kind,
                              std::string message)
      : kind_(kind), message_(std::move(message)) {}

  ArtifactLocalReferenceErrorKind kind() const { return kind_; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  ArtifactLocalReferenceErrorKind kind_;
  std::string message_;
};

/// The registered codec for one (owner schema, owner-local kind) pair, or
/// absent. The result is a value copy taken under the registry lock: later
/// registrations never invalidate it. An absent codec is an
/// implementation/capability error at the consumer, never a reason to fall
/// back to an untyped payload.
std::optional<ArtifactLocalReferenceCodec>
findArtifactLocalReferenceKind(const ArtifactSchemaDescriptor &ownerSchema,
                               std::uint32_t ownerLocalKind);

/// The owner-published schema descriptor under which at least one local kind
/// is registered, or absent. The result is a value copy taken under the
/// registry lock: later registrations never invalidate it. Heterogeneous
/// import resolves the exact schema descriptor through this lookup; a consumer
/// never reconstructs a descriptor from a string.
std::optional<ArtifactSchemaDescriptor>
findArtifactLocalReferenceSchema(llvm::StringRef identity,
                                 SchemaVersion version);

/// Resolves the registered owner codec and validates only the canonical owner
/// payload. This boundary performs no Artifact lookup and is suitable for
/// strict parsing before an exact ArtifactStore is available.
llvm::Error validateArtifactLocalReferencePayload(
    const EncodedArtifactLocalReference &reference);

/// Full import of one heterogeneous local reference: resolves the registered
/// owner codec, validates the canonical payload, loads the exact referenced
/// Artifact from the explicit store, and invokes the owner validator with its
/// canonical bytes. Unknown schemas or kinds remain capability failures;
/// missing objects propagate the store failure; malformed or noncanonical
/// payloads, Artifact bytes, and owner-rejected targets are invalid.
llvm::Error
validateArtifactLocalReference(const ArtifactStore &store,
                               const EncodedArtifactLocalReference &reference);

} // namespace loom

#endif // LOOM_COMMON_ARTIFACTLOCALREFERENCE_H
