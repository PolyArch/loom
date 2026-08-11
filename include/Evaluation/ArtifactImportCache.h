#ifndef LOOM_EVALUATION_ARTIFACTIMPORTCACHE_H
#define LOOM_EVALUATION_ARTIFACTIMPORTCACHE_H

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <typeindex>
#include <utility>

namespace loom {
class ArtifactStore;
class BlobStore;
}

namespace loom::evaluation {

struct ArtifactImportCacheStatistics final {
  std::uint64_t hitCount = 0;
  std::uint64_t missCount = 0;
};

/// One invocation-local cache of strictly imported immutable typed views.
/// Exact references, store instances, and result type form every key. The
/// cache has no persistent identity and never survives its explicit scope.
class ArtifactImportCache final {
public:
  ArtifactImportCache();
  ~ArtifactImportCache();

  ArtifactImportCache(const ArtifactImportCache &) = delete;
  ArtifactImportCache &operator=(const ArtifactImportCache &) = delete;

  ArtifactImportCacheStatistics statistics() const;

  template <typename Value, typename Loader>
  llvm::Expected<std::shared_ptr<const Value>>
  import(const ArtifactStore &artifacts, const BlobStore *blobs,
         llvm::ArrayRef<ArtifactRootReference> references, Loader &&loader) {
    std::shared_ptr<const void> found =
        lookup(typeid(Value), artifacts, blobs, references);
    if (found)
      return std::static_pointer_cast<const Value>(std::move(found));
    auto imported = loader();
    if (!imported)
      return imported.takeError();
    auto sealed =
        std::make_shared<const Value>(std::move(*imported));
    return std::static_pointer_cast<const Value>(
        insert(typeid(Value), artifacts, blobs, references, sealed));
  }

private:
  class Impl;
  std::unique_ptr<Impl> impl_;

  std::shared_ptr<const void>
  lookup(std::type_index type, const ArtifactStore &artifacts,
         const BlobStore *blobs,
         llvm::ArrayRef<ArtifactRootReference> references);
  std::shared_ptr<const void>
  insert(std::type_index type, const ArtifactStore &artifacts,
         const BlobStore *blobs,
         llvm::ArrayRef<ArtifactRootReference> references,
         std::shared_ptr<const void> value);
};

/// Reuses an enclosing cache or installs a fresh cache for this synchronous
/// call tree. Nested scopes never replace the enclosing invocation cache.
class ArtifactImportCacheScope final {
public:
  ArtifactImportCacheScope();
  ~ArtifactImportCacheScope();

  ArtifactImportCacheScope(const ArtifactImportCacheScope &) = delete;
  ArtifactImportCacheScope &operator=(const ArtifactImportCacheScope &) =
      delete;

private:
  std::unique_ptr<ArtifactImportCache> owned_;
  ArtifactImportCache *previous_ = nullptr;
};

ArtifactImportCache *currentArtifactImportCache();

template <typename Value, typename Loader>
llvm::Expected<std::shared_ptr<const Value>> importCachedArtifact(
    const ArtifactStore &artifacts, const BlobStore *blobs,
    llvm::ArrayRef<ArtifactRootReference> references, Loader &&loader) {
  if (ArtifactImportCache *cache = currentArtifactImportCache())
    return cache->import<Value>(artifacts, blobs, references,
                                std::forward<Loader>(loader));
  auto imported = loader();
  if (!imported)
    return imported.takeError();
  return std::make_shared<const Value>(std::move(*imported));
}

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_ARTIFACTIMPORTCACHE_H
