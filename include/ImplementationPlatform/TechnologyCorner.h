#ifndef LOOM_IMPLEMENTATIONPLATFORM_TECHNOLOGYCORNER_H
#define LOOM_IMPLEMENTATIONPLATFORM_TECHNOLOGYCORNER_H

#include "Common/Artifact.h"
#include "Common/ArtifactLocalReference.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>

// The typed technology-corner reference and canonical local payload specified
// by docs/spec-implementation-platform.md. The platform root importer resolves
// this owner-local ID against one exact finalized corner catalog.

namespace loom::platform {

/// The schema descriptor required by the specified platform Artifact family.
inline constexpr ArtifactSchemaDescriptor implementationPlatformSchema{
    "loom.implementation_platform", SchemaVersion{1, 0}};

/// A dense owner-local ordinal in [0, N) assigned at platform finalization by
/// the corner's complete semantic model-input key. It is not an author label,
/// payload ordinal, hash, or reusable across platform Artifacts.
class TechnologyCornerId {
public:
  explicit constexpr TechnologyCornerId(std::uint64_t value) : value_(value) {}

  constexpr std::uint64_t value() const { return value_; }

  friend constexpr bool operator==(TechnologyCornerId lhs,
                                   TechnologyCornerId rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr bool operator!=(TechnologyCornerId lhs,
                                   TechnologyCornerId rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint64_t value_;
};

/// The typed in-memory API for one exact technology-corner reference.
using TechnologyCornerRef = ArtifactReference<TechnologyCornerId>;

/// The family-owned closed ordinal space for heterogeneous local references.
enum class ImplementationPlatformLocalReferenceKind : std::uint32_t {
  TechnologyCorner = 0,
};

constexpr std::uint32_t
implementationPlatformLocalKind(ImplementationPlatformLocalReferenceKind kind) {
  return static_cast<std::uint32_t>(kind);
}

/// The family-owned existential-reference codec for local kind
/// TechnologyCorner: exactly u64be(corner_id), so the payload is exactly
/// eight bytes. Decoding any other length is invalid.
std::array<std::uint8_t, 8>
encodeTechnologyCornerPayload(TechnologyCornerId corner);
llvm::Expected<TechnologyCornerId>
decodeTechnologyCornerPayload(llvm::ArrayRef<std::uint8_t> payload);

/// The family-owned canonical bytes of one exact typed corner reference in
/// one exact platform Artifact.
EncodedArtifactLocalReference
encodeTechnologyCornerRef(const TechnologyCornerRef &reference);

/// The typed recovery of one heterogeneous corner reference: exact schema and
/// local kind, then strict payload decode.
llvm::Expected<TechnologyCornerRef>
decodeTechnologyCornerRef(const EncodedArtifactLocalReference &reference);

} // namespace loom::platform

#endif // LOOM_IMPLEMENTATIONPLATFORM_TECHNOLOGYCORNER_H
