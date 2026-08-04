#include "Fabric/Artifact/FabricArtifactLocalReference.h"

#include "Fabric/Identity/FabricRefImport.h"

#include <array>

namespace loom::fabric {
namespace {

constexpr std::array<FabricArtifactLocalReferenceKindDescriptor,
                     fabricArtifactLocalReferenceKindCount()>
    kindCatalog = {{
#define LOOM_FABRIC_LOCAL_REFERENCE_KIND(Ordinal, Type)                        \
  {FabricArtifactLocalReferenceKind::Type, llvm::StringLiteral(#Type)},
#include "Fabric/Identity/FabricRefs.def"
    }};

} // namespace

llvm::ArrayRef<FabricArtifactLocalReferenceKindDescriptor>
fabricArtifactLocalReferenceKindCatalog() {
  return kindCatalog;
}

llvm::Error validateFabricArtifactLocalReference(
    const FabricArtifactView &view,
    const EncodedArtifactLocalReference &reference) {
  if (reference.artifact.artifact != view.identity())
    return makeFabricRefError(FabricRefErrorKind::ForeignArtifact,
                              "the reference names a foreign Fabric artifact");
  switch (static_cast<FabricArtifactLocalReferenceKind>(
      reference.ownerLocalKind)) {
#define LOOM_FABRIC_LOCAL_REFERENCE_KIND(Ordinal, Type)                        \
  case FabricArtifactLocalReferenceKind::Type: {                              \
    auto decoded = decodeFabricArtifactLocalReference<Type>(reference);        \
    if (!decoded)                                                              \
      return decoded.takeError();                                              \
    return validateFabricRef(view, decoded->entity);                           \
  }
#include "Fabric/Identity/FabricRefs.def"
  }
  return makeFabricRefError(FabricRefErrorKind::MalformedSyntax,
                            "unknown Fabric owner-local reference kind");
}

} // namespace loom::fabric
