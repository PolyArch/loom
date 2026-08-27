#ifndef LOOM_FABRIC_ARTIFACT_FABRICARTIFACTLOCALREFERENCE_H
#define LOOM_FABRIC_ARTIFACT_FABRICARTIFACTLOCALREFERENCE_H

#include "Common/ArtifactLocalReference.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <type_traits>
#include <utility>

namespace loom::fabric {

class FabricArtifactView;

/// The closed loom.fabric 7.x owner-local kind space used by Common's
/// ArtifactLocalReference framing. Enumerators and typed traits are generated
/// from the one declaration in FabricRefs.def.
enum class FabricArtifactLocalReferenceKind : std::uint32_t {
#define LOOM_FABRIC_LOCAL_REFERENCE_KIND(Ordinal, Type) Type = Ordinal,
#include "Fabric/Identity/FabricRefs.def"
};

constexpr std::uint32_t fabricArtifactLocalReferenceKindCount() {
  std::uint32_t count = 0;
#define LOOM_FABRIC_LOCAL_REFERENCE_KIND(Ordinal, Type) ++count;
#include "Fabric/Identity/FabricRefs.def"
  return count;
}

constexpr std::uint32_t
fabricArtifactLocalReferenceKindOrdinal(FabricArtifactLocalReferenceKind kind) {
  return static_cast<std::uint32_t>(kind);
}

struct FabricArtifactLocalReferenceKindDescriptor {
  FabricArtifactLocalReferenceKind kind;
  llvm::StringLiteral typedTarget;
};

llvm::ArrayRef<FabricArtifactLocalReferenceKindDescriptor>
fabricArtifactLocalReferenceKindCatalog();

template <typename Ref> struct FabricArtifactLocalReferenceKindTraits;

#define LOOM_FABRIC_LOCAL_REFERENCE_KIND(Ordinal, Type)                        \
  template <> struct FabricArtifactLocalReferenceKindTraits<Type> {            \
    static constexpr FabricArtifactLocalReferenceKind kind =                   \
        FabricArtifactLocalReferenceKind::Type;                                \
    static constexpr llvm::StringLiteral typedTarget = #Type;                  \
  };
#include "Fabric/Identity/FabricRefs.def"

template <typename Ref, typename = void>
struct IsFabricArtifactLocalReference : std::false_type {};

template <typename Ref>
struct IsFabricArtifactLocalReference<
    Ref,
    std::void_t<decltype(FabricArtifactLocalReferenceKindTraits<Ref>::kind)>>
    : std::true_type {};

template <typename Ref>
inline constexpr bool isFabricArtifactLocalReference =
    IsFabricArtifactLocalReference<Ref>::value;

/// Family-owned production projection into Common's existential wire.
template <typename Ref>
EncodedArtifactLocalReference
encodeFabricArtifactLocalReference(const ArtifactReference<Ref> &reference) {
  static_assert(isFabricArtifactLocalReference<Ref>,
                "Ref is not a loom.fabric owner-local reference kind");
  return EncodedArtifactLocalReference{
      ArtifactRootReference{fabricArtifactSchema.identity.str(),
                            fabricArtifactSchema.version, reference.artifact},
      fabricArtifactLocalReferenceKindOrdinal(
          FabricArtifactLocalReferenceKindTraits<Ref>::kind),
      canonicalFabricBytes(reference.entity)};
}

/// Strict typed recovery from Common's existential wire. Exact target
/// resolution is a separate import step against the owning Fabric view.
template <typename Ref>
llvm::Expected<ArtifactReference<Ref>> decodeFabricArtifactLocalReference(
    const EncodedArtifactLocalReference &reference) {
  static_assert(isFabricArtifactLocalReference<Ref>,
                "Ref is not a loom.fabric owner-local reference kind");
  if (reference.artifact.schemaIdentity != fabricArtifactSchema.identity ||
      reference.artifact.schemaVersion != fabricArtifactSchema.version)
    return makeFabricRefError(
        FabricRefErrorKind::ForeignArtifact,
        "the local reference is not owned by the current loom.fabric schema");

  const std::uint32_t expected = fabricArtifactLocalReferenceKindOrdinal(
      FabricArtifactLocalReferenceKindTraits<Ref>::kind);
  if (reference.ownerLocalKind != expected)
    return makeFabricRefError(
        FabricRefErrorKind::MalformedSyntax,
        llvm::Twine("owner-local kind ") +
            llvm::Twine(reference.ownerLocalKind) + " does not encode " +
            FabricArtifactLocalReferenceKindTraits<Ref>::typedTarget);

  llvm::Expected<Ref> decoded = decodeFabricRef<Ref>(reference.payload);
  if (!decoded)
    return decoded.takeError();
  if (canonicalFabricBytes(*decoded) != reference.payload)
    return makeFabricRefError(FabricRefErrorKind::MalformedSyntax,
                              "noncanonical Fabric reference payload");
  return ArtifactReference<Ref>{reference.artifact.artifact,
                                std::move(*decoded)};
}

llvm::Error validateFabricArtifactLocalReference(
    const FabricArtifactView &view,
    const EncodedArtifactLocalReference &reference);

} // namespace loom::fabric

#endif // LOOM_FABRIC_ARTIFACT_FABRICARTIFACTLOCALREFERENCE_H
