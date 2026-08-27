#ifndef LOOM_COMMON_ARTIFACTSTORE_H
#define LOOM_COMMON_ARTIFACTSTORE_H

#include "Common/Artifact.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace loom {

class ArtifactStore {
public:
  /// Root must name an existing, durably provisioned non-symlink directory.
  explicit ArtifactStore(llvm::StringRef root)
      : root_(root.str()),
        verifiedReads_(std::make_shared<VerifiedReadCache>()) {}

  llvm::Expected<ArtifactIdentity>
  put(const ArtifactSchemaDescriptor &schema,
      const CanonicalSemanticBytes &canonicalBytes) const;

  llvm::Expected<CanonicalSemanticBytes>
  get(const ArtifactSchemaDescriptor &expectedSchema,
      const ArtifactIdentity &identity) const;

  /// Resolves the schema framing carried by one exact root reference.
  llvm::Expected<CanonicalSemanticBytes>
  get(const ArtifactRootReference &reference) const;

  /// Returns the exact validated identity preimage stored under reference.
  /// This is the transport form used by content-addressed package projections;
  /// callers do not reconstruct schema framing around canonical semantic bytes.
  llvm::Expected<std::vector<std::uint8_t>>
  getStoredObject(const ArtifactRootReference &reference) const;

private:
  llvm::Expected<CanonicalSemanticBytes>
  getExact(llvm::StringRef schemaIdentity, SchemaVersion schemaVersion,
           const ArtifactIdentity &identity) const;

  /// Bounded cache of reads whose stored identity preimage has already been
  /// validated. Stored objects are content addressed and immutable, so a
  /// validated read is a verified handle: a hit shares the immutable bytes
  /// without rereading or rehashing, and never trusts caller-supplied bytes.
  struct VerifiedRead final {
    std::string schemaIdentity;
    SchemaVersion schemaVersion;
    CanonicalSemanticBytes bytes;
  };
  struct IdentityLess final {
    bool operator()(const ArtifactIdentity &lhs,
                    const ArtifactIdentity &rhs) const {
      return lhs.bytes() < rhs.bytes();
    }
  };
  static constexpr std::size_t verifiedReadByteBudget = 256u << 20;
  struct VerifiedReadCache final {
    std::mutex mutex;
    std::map<ArtifactIdentity, VerifiedRead, IdentityLess> entries;
    std::list<ArtifactIdentity> order;
    std::size_t retainedBytes = 0;
  };

  std::string root_;
  /// Copies of a store share one cache: the root names one content-addressed
  /// domain, so a validated read is valid for every copy.
  std::shared_ptr<VerifiedReadCache> verifiedReads_;
};

} // namespace loom

#endif // LOOM_COMMON_ARTIFACTSTORE_H
