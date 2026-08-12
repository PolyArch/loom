#ifndef LOOM_COMMON_ARTIFACTTEXT_H
#define LOOM_COMMON_ARTIFACTTEXT_H

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {

std::string formatSchemaVersion(SchemaVersion version);
llvm::Expected<SchemaVersion> parseSchemaVersion(llvm::StringRef spelling);

std::string formatArtifactIdentityHex(const ArtifactIdentity &identity);
llvm::Expected<ArtifactIdentity>
parseArtifactIdentityHex(llvm::StringRef spelling);

/// Common-owned canonical JSON fields for one exact root reference. Enclosing
/// schemas use the field form after validating their complete field set; the
/// strict object form accepts no fields beyond these three.
void writeArtifactRootReferenceJsonFields(
    llvm::json::OStream &json, const ArtifactRootReference &reference);
llvm::Expected<ArtifactRootReference>
parseArtifactRootReferenceJsonFields(const llvm::json::Object &object);
void writeArtifactRootReferenceJson(llvm::json::OStream &json,
                                    const ArtifactRootReference &reference);
llvm::Expected<ArtifactRootReference>
parseArtifactRootReferenceJson(const llvm::json::Object &object);
std::string
formatArtifactRootReferenceJson(const ArtifactRootReference &reference);
llvm::Error
writeArtifactRootReferenceJsonFile(llvm::StringRef path,
                                   const ArtifactRootReference &reference);
llvm::Expected<ArtifactRootReference>
loadArtifactRootReferenceJsonFile(llvm::StringRef path);

/// Canonical sorted unique non-empty root-reference set JSON. This is a text
/// authoring boundary only; each reference retains its family owner.
std::string formatArtifactRootReferenceSetJson(
    llvm::ArrayRef<ArtifactRootReference> references);
llvm::Error writeArtifactRootReferenceSetJsonFile(
    llvm::StringRef path, llvm::ArrayRef<ArtifactRootReference> references);
llvm::Expected<std::vector<ArtifactRootReference>>
loadArtifactRootReferenceSetJsonFile(llvm::StringRef path);

/// Lowercase hexadecimal text of a complete family-owned local-reference
/// payload of any length, exactly as canonical JSON frames it.
std::string formatArtifactLocalPayloadHex(llvm::ArrayRef<std::uint8_t> payload);
llvm::Expected<std::vector<std::uint8_t>>
parseArtifactLocalPayloadHex(llvm::StringRef spelling);

} // namespace loom

#endif // LOOM_COMMON_ARTIFACTTEXT_H
