#ifndef LOOM_FABRIC_IDENTITY_FABRICLOCALREFERENCE_H
#define LOOM_FABRIC_IDENTITY_FABRICLOCALREFERENCE_H

#include "Common/Artifact.h"
#include "Common/ArtifactLocalReference.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/Support/Error.h"

#include <cstdint>

// The Fabric family's existential local-reference ownership
// (docs/spec-fabric-artifact.md, docs/spec-full-stack-traceability.md). The
// family owns the exact loom.fabric 1.0 schema descriptor, one local kind per
// typed entity reference, the canonical payload codec, and validation against
// one exact imported Fabric Artifact. The local kind ordinal of a typed
// entity reference is its persistent FabricEntityKind discriminant from the
// one reference catalog; the payload is exactly that reference's canonical
// bytes. Consumers use the Common heterogeneous framing and never reinterpret
// the payload.
//
// Validation resolves the family-owned importer view of the reference's exact
// Artifact only through this family's own typed resolver: the view enters
// exclusively as a FabricArtifactView and the identity-to-view lookup never
// leaves the family. No view crosses the Common boundary, typed or erased.

namespace loom::fabric {

/// The Fabric family's own persistent schema descriptor, declared once by the
/// C++ owner and reused by every consumer. A caller may not reconstruct it
/// from a string or maintain a second version constant.
inline constexpr ArtifactSchemaDescriptor fabricArtifactSchema{
    "loom.fabric", SchemaVersion{1, 0}};

/// The local kind ordinal of a typed Fabric entity reference is exactly its
/// persistent entity-kind discriminant.
template <FabricEntityKind Kind>
constexpr std::uint32_t fabricEntityLocalKind() {
  return static_cast<std::uint32_t>(Kind);
}

/// The family-owned canonical payload of one typed entity reference.
template <FabricEntityKind Kind>
EncodedArtifactLocalReference
encodeFabricEntityLocalReference(const ArtifactIdentity &artifact,
                                 const FabricTypedEntityRef<Kind> &ref) {
  return EncodedArtifactLocalReference{
      ArtifactRootReference{fabricArtifactSchema, artifact},
      fabricEntityLocalKind<Kind>(), canonicalFabricBytes(ref)};
}

/// Publishes the family-owned read-only importer view of one exact imported
/// Fabric Artifact into the family's typed resolver, keyed by the view's own
/// identity. The view must have static storage duration. Publishing the same
/// view again is a no-op; a different view for the same identity is a
/// conflict. Existential local-reference validation resolves views only
/// through this boundary.
llvm::Error publishFabricImporterView(const FabricArtifactView &view);

/// Statically registers every typed entity reference kind of the one
/// reference catalog with the Common framing. Idempotent.
llvm::Error registerFabricLocalReferenceKinds();

} // namespace loom::fabric

#endif // LOOM_FABRIC_IDENTITY_FABRICLOCALREFERENCE_H
