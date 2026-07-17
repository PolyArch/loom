#ifndef LOOM_COMMON_ARTIFACTTEXT_H
#define LOOM_COMMON_ARTIFACTTEXT_H

#include "Common/Artifact.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace loom {

std::string formatSchemaVersion(SchemaVersion version);
llvm::Expected<SchemaVersion> parseSchemaVersion(llvm::StringRef spelling);

std::string formatArtifactIdentityHex(const ArtifactIdentity &identity);
llvm::Expected<ArtifactIdentity>
parseArtifactIdentityHex(llvm::StringRef spelling);

} // namespace loom

#endif // LOOM_COMMON_ARTIFACTTEXT_H
