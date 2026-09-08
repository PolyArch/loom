#ifndef LOOM_COMMON_ARTIFACTSTORE_H
#define LOOM_COMMON_ARTIFACTSTORE_H

#include "Common/Artifact.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace loom {

class ArtifactStore {
public:
  enum class Durability { Durable, Transient };

  /// Root must name an existing non-symlink directory. Durable publication
  /// requires a durably provisioned root. Transient publication is restricted
  /// to invocation-owned workspaces with no crash-recovery contract.
  explicit ArtifactStore(llvm::StringRef root,
                         Durability durability = Durability::Durable);

  llvm::Expected<ArtifactIdentity>
  put(const ArtifactSchemaDescriptor &schema,
      const CanonicalSemanticBytes &canonicalBytes) const;

  llvm::Expected<CanonicalSemanticBytes>
  get(const ArtifactSchemaDescriptor &expectedSchema,
      const ArtifactIdentity &identity) const;

  /// Resolves the schema framing carried by one exact root reference.
  llvm::Expected<CanonicalSemanticBytes>
  get(const ArtifactRootReference &reference) const;

  /// Admits an exact reference into this store's immutable content-addressed
  /// domain. A bounded record of a validated read can satisfy later admission
  /// after its bytes are evicted. A newly opened store validates from disk.
  llvm::Error verifyReference(const ArtifactRootReference &reference) const;

  /// Returns the exact validated identity preimage stored under reference.
  /// This is the transport form used by content-addressed package projections;
  /// callers do not reconstruct schema framing around canonical semantic bytes.
  llvm::Expected<std::vector<std::uint8_t>>
  getStoredObject(const ArtifactRootReference &reference) const;

private:
  llvm::Expected<CanonicalSemanticBytes>
  getExact(llvm::StringRef schemaIdentity, SchemaVersion schemaVersion,
           const ArtifactIdentity &identity) const;

  struct VerifiedReadCache;

  std::string root_;
  Durability durability_;
  /// Copies of a store share one cache: the root names one content-addressed
  /// domain, so a validated read is valid for every copy.
  std::shared_ptr<VerifiedReadCache> verifiedReads_;
};

} // namespace loom

#endif // LOOM_COMMON_ARTIFACTSTORE_H
