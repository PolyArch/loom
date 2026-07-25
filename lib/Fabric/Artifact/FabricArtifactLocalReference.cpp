#include "Fabric/Artifact/FabricArtifactLocalReference.h"

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

} // namespace loom::fabric
