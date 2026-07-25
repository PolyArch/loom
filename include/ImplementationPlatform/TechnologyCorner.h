#ifndef LOOM_IMPLEMENTATIONPLATFORM_TECHNOLOGYCORNER_H
#define LOOM_IMPLEMENTATIONPLATFORM_TECHNOLOGYCORNER_H

#include "Common/Artifact.h"
#include "Common/ArtifactLocalReference.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>

// The ImplementationPlatform family's owner projection for technology-corner
// local references (docs/spec-implementation-platform.md). The family owns
// the exact loom.implementation_platform 1.0 schema descriptor, the dense
// owner-local TechnologyCornerId ordinal space, the existential local kind
// ImplementationPlatformLocalReferenceKind::TechnologyCorner, its exactly
// eight-byte u64be canonical payload codec, and corner validation against one
// exact imported platform. Evaluation and EDA adapters invoke this codec and
// validator; they never reinterpret the ID or erase a reference to a bare
// integer.
//
// Validation resolves the family-owned importer view of the reference's exact
// Artifact only through this family's own typed resolver: the view enters
// exclusively as an ImplementationPlatformView and the identity-to-view
// lookup never leaves the family. No view crosses the Common boundary, typed
// or erased.

namespace loom::platform {

/// The family's own persistent schema descriptor, declared once by the C++
/// owner and reused by the root codec, Artifact store, typed references, and
/// every consumer. A caller may not reconstruct it from a string or maintain
/// a second version constant.
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

constexpr std::uint32_t implementationPlatformLocalKind(
    ImplementationPlatformLocalReferenceKind kind) {
  return static_cast<std::uint32_t>(kind);
}

/// The family-owned existential-reference codec for local kind
/// TechnologyCorner: exactly u64be(corner_id), so the payload is exactly
/// eight bytes. Decoding any other length is invalid.
std::array<std::uint8_t, 8> encodeTechnologyCornerPayload(TechnologyCornerId corner);
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

/// A read-only importer view over one exact validated ImplementationPlatform
/// root. The root was independently validated before this view exists; the
/// view resolves corner IDs against the one immutable corner catalog.
class ImplementationPlatformView {
public:
  virtual ~ImplementationPlatformView();

  virtual const ArtifactIdentity &identity() const = 0;

  /// The dense corner catalog size: valid IDs are exactly [0, count).
  virtual std::uint64_t technologyCornerCount() const = 0;
};

/// The owner validator: the reference must name this exact platform Artifact
/// and its ID must resolve to exactly one catalog entry.
llvm::Error
validateTechnologyCornerRef(const ImplementationPlatformView &view,
                            const TechnologyCornerRef &reference);

/// Publishes the family-owned read-only importer view of one exact imported
/// ImplementationPlatform Artifact into the family's typed resolver, keyed by
/// the view's own identity. The view must have static storage duration.
/// Publishing the same view again is a no-op; a different view for the same
/// identity is a conflict. Existential local-reference validation resolves
/// views only through this boundary.
llvm::Error
publishImplementationPlatformView(const ImplementationPlatformView &view);

/// Statically registers the family's local-reference kinds with the Common
/// framing. Idempotent.
llvm::Error registerImplementationPlatformLocalReferenceKinds();

} // namespace loom::platform

#endif // LOOM_IMPLEMENTATIONPLATFORM_TECHNOLOGYCORNER_H
