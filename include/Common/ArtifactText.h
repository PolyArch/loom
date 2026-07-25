#ifndef LOOM_COMMON_ARTIFACTTEXT_H
#define LOOM_COMMON_ARTIFACTTEXT_H

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {

std::string formatSchemaVersion(SchemaVersion version);
llvm::Expected<SchemaVersion> parseSchemaVersion(llvm::StringRef spelling);

std::string formatArtifactIdentityHex(const ArtifactIdentity &identity);
llvm::Expected<ArtifactIdentity>
parseArtifactIdentityHex(llvm::StringRef spelling);

/// Lowercase hexadecimal text of a complete family-owned local-reference
/// payload of any length, exactly as canonical JSON frames it.
std::string formatArtifactLocalPayloadHex(llvm::ArrayRef<std::uint8_t> payload);
llvm::Expected<std::vector<std::uint8_t>>
parseArtifactLocalPayloadHex(llvm::StringRef spelling);

} // namespace loom

#endif // LOOM_COMMON_ARTIFACTTEXT_H
