#ifndef LOOM_COMMON_ARTIFACTLOCALREFERENCE_H
#define LOOM_COMMON_ARTIFACTLOCALREFERENCE_H

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
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

/// An exact reference to one finalized Artifact root. The schema is present
/// because the identity digest alone does not select an importer or
/// validator.
struct ArtifactRootReference {
  ArtifactSchemaDescriptor schema;
  ArtifactIdentity artifact;

  friend bool operator==(const ArtifactRootReference &lhs,
                         const ArtifactRootReference &rhs) {
    return lhs.schema == rhs.schema && lhs.artifact == rhs.artifact;
  }
  friend bool operator!=(const ArtifactRootReference &lhs,
                         const ArtifactRootReference &rhs) {
    return !(lhs == rhs);
  }
};

/// The complete local-reference type descriptor, derived exactly once as
/// (owner schema, owner-local kind). owner_local_kind is a stable closed
/// ordinal owned by that exact Artifact family and schema version; it is not
/// a global entity kind, consumer enum, textual type name, or native variant
/// index.
struct ArtifactLocalReferenceTypeDescriptor {
  ArtifactSchemaDescriptor ownerSchema;
  std::uint32_t ownerLocalKind = 0;

  friend bool operator==(const ArtifactLocalReferenceTypeDescriptor &lhs,
                         const ArtifactLocalReferenceTypeDescriptor &rhs) {
    return lhs.ownerSchema == rhs.ownerSchema &&
           lhs.ownerLocalKind == rhs.ownerLocalKind;
  }
  friend bool operator!=(const ArtifactLocalReferenceTypeDescriptor &lhs,
                         const ArtifactLocalReferenceTypeDescriptor &rhs) {
    return !(lhs == rhs);
  }
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
    return ArtifactLocalReferenceTypeDescriptor{artifact.schema, ownerLocalKind};
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

/// Canonical binary framing of one heterogeneous local reference:
/// the root-reference framing of the containing Artifact
///   || u32be(owner-local kind)
///   || u64be(payload length) || payload bytes
std::vector<std::uint8_t>
encodeArtifactLocalReference(const EncodedArtifactLocalReference &reference);

/// One registered owner-local kind. The strict decoder rejects malformed or
/// noncanonical payloads without any Artifact view; the validator then checks
/// the decoded typed target against the one exact imported Artifact, resolving
/// its family-owned importer view through the family's own typed resolver
/// boundary. Any type erasure stays inside the owner codec: no untyped view
/// ever crosses the Common boundary.
struct ArtifactLocalReferenceCodec {
  llvm::Error (*strictDecode)(llvm::ArrayRef<std::uint8_t> payload);
  llvm::Error (*validate)(const EncodedArtifactLocalReference &reference);
};

/// Statically registers one owner-local kind for one exact Artifact family.
/// Only the family that owns the schema may supply its kinds and codecs.
/// Registering the same kind with the same codec again is a no-op; a
/// conflicting registration is an error.
llvm::Error
registerArtifactLocalReferenceKind(const ArtifactSchemaDescriptor &ownerSchema,
                                   std::uint32_t ownerLocalKind,
                                   ArtifactLocalReferenceCodec codec);

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

/// Import of one heterogeneous local reference: resolves the registered
/// owner codec, strictly decodes the payload, and invokes the owner validator,
/// which resolves the exact imported Artifact's view through the family's own
/// typed boundary. Unknown schemas or kinds, malformed or noncanonical
/// payloads, and owner rejections are invalid.
llvm::Error
validateArtifactLocalReference(const EncodedArtifactLocalReference &reference);

} // namespace loom

#endif // LOOM_COMMON_ARTIFACTLOCALREFERENCE_H
