#ifndef LOOM_LIB_COMMON_ARTIFACTFINALIZERINTERNAL_H
#define LOOM_LIB_COMMON_ARTIFACTFINALIZERINTERNAL_H

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::detail {

std::vector<std::uint8_t>
buildArtifactIdentityPreimage(const ArtifactSchemaDescriptor &schema,
                              const CanonicalSemanticBytes &canonicalBytes);

ArtifactIdentity
finalizeArtifactIdentityPreimage(llvm::ArrayRef<std::uint8_t> preimage);

llvm::Error
validateArtifactIdentityPreimage(llvm::ArrayRef<std::uint8_t> preimage);

} // namespace loom::detail

#endif // LOOM_LIB_COMMON_ARTIFACTFINALIZERINTERNAL_H
