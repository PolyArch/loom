#ifndef LOOM_COMMON_ARTIFACTSTORE_H
#define LOOM_COMMON_ARTIFACTSTORE_H

#include "Common/Artifact.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace loom {

class ArtifactStore {
public:
  /// Root must name an existing, durably provisioned non-symlink directory.
  explicit ArtifactStore(llvm::StringRef root) : root_(root.str()) {}

  llvm::Expected<ArtifactIdentity>
  put(const ArtifactSchemaDescriptor &schema,
      const CanonicalSemanticBytes &canonicalBytes) const;

  llvm::Expected<CanonicalSemanticBytes>
  get(const ArtifactSchemaDescriptor &expectedSchema,
      const ArtifactIdentity &identity) const;

  /// Resolves the schema framing carried by one exact root reference.
  llvm::Expected<CanonicalSemanticBytes>
  get(const ArtifactRootReference &reference) const;

private:
  llvm::Expected<CanonicalSemanticBytes>
  getExact(llvm::StringRef schemaIdentity, SchemaVersion schemaVersion,
           const ArtifactIdentity &identity) const;

  std::string root_;
};

} // namespace loom

#endif // LOOM_COMMON_ARTIFACTSTORE_H
