#ifndef LOOM_EVALUATION_ARTIFACTIMPORTCACHE_H
#define LOOM_EVALUATION_ARTIFACTIMPORTCACHE_H

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <typeindex>
#include <utility>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::evaluation {

inline constexpr std::uint64_t artifactImportCacheAlgorithmVersion = 1;
inline constexpr std::size_t defaultArtifactImportCacheEntryLimit = 64;

enum class ArtifactImportCacheVerificationDomain : std::uint8_t {
  SourceInvocation,
  IndependentReplay,
};

struct ArtifactImportCacheStatistics final {
  std::uint64_t importRequests = 0;
  std::uint64_t cacheHits = 0;
  std::uint64_t cacheMisses = 0;
  std::uint64_t uniqueConstructions = 0;
  std::uint64_t uncachedConstructions = 0;
  std::uint64_t revalidationCount = 0;
  std::uint64_t revalidatedBytes = 0;
  std::uint64_t constructionNanoseconds = 0;
  std::uint64_t constructionNanosecondsSaved = 0;
  std::uint64_t deterministicWork = 0;
  std::uint64_t minimumRetainedBytes = 0;
  std::uint64_t minimumRetainedBytesReused = 0;
  std::uint64_t entryCount = 0;
};

void emitArtifactImportCacheStatistics(
    ArtifactImportCacheVerificationDomain domain,
    const ArtifactImportCacheStatistics &statistics);

/// One invocation-local cache of strictly imported immutable typed views.
/// Exact references, result type, and the import algorithm version form every
/// key. One cache is bound to one exact store domain, has an explicit entry
/// bound, and never survives its scope.
class ArtifactImportCache final {
public:
  ArtifactImportCache(const ArtifactStore &artifacts, const BlobStore *blobs,
                      std::size_t entryLimit);
  ~ArtifactImportCache();

  ArtifactImportCache(const ArtifactImportCache &) = delete;
  ArtifactImportCache &operator=(const ArtifactImportCache &) = delete;

  ArtifactImportCacheStatistics statistics() const;
  bool owns(const ArtifactStore &artifacts, const BlobStore *blobs) const;

  template <typename Value, typename Loader>
  llvm::Expected<std::shared_ptr<const Value>>
  import(const ArtifactStore &artifacts, const BlobStore *blobs,
         llvm::ArrayRef<ArtifactRootReference> references, Loader &&loader) {
    auto found = lookup(typeid(Value), references);
    if (found.value) {
      auto revalidated = revalidate(artifacts, references);
      if (!revalidated)
        return revalidated.takeError();
      recordHit(*revalidated, found.constructionNanoseconds,
                found.minimumRetainedBytes);
      return std::static_pointer_cast<const Value>(std::move(found.value));
    }
    const auto begin = std::chrono::steady_clock::now();
    auto imported = loader();
    const std::uint64_t constructionNanoseconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin)
            .count();
    if (!imported) {
      recordFailedConstruction(constructionNanoseconds);
      return imported.takeError();
    }
    auto sealed = std::make_shared<const Value>(std::move(*imported));
    const std::uint64_t minimumRetainedBytes =
        sizeof(Value) + references.size() * sizeof(ArtifactRootReference);
    return std::static_pointer_cast<const Value>(
        insert(typeid(Value), references, sealed, constructionNanoseconds,
               minimumRetainedBytes));
  }

private:
  class Impl;
  std::unique_ptr<Impl> impl_;

  struct LookupResult final {
    std::shared_ptr<const void> value;
    std::uint64_t constructionNanoseconds = 0;
    std::uint64_t minimumRetainedBytes = 0;
  };

  LookupResult lookup(std::type_index type,
                      llvm::ArrayRef<ArtifactRootReference> references);
  std::shared_ptr<const void>
  insert(std::type_index type, llvm::ArrayRef<ArtifactRootReference> references,
         std::shared_ptr<const void> value,
         std::uint64_t constructionNanoseconds,
         std::uint64_t minimumRetainedBytes);
  llvm::Expected<std::uint64_t>
  revalidate(const ArtifactStore &artifacts,
             llvm::ArrayRef<ArtifactRootReference> references);
  void recordHit(std::uint64_t revalidatedBytes,
                 std::uint64_t constructionNanoseconds,
                 std::uint64_t minimumRetainedBytes);
  void recordFailedConstruction(std::uint64_t constructionNanoseconds);
};

/// Reuses an enclosing cache or installs a fresh cache for this synchronous
/// call tree. Nested scopes never replace the enclosing invocation cache.
class ArtifactImportCacheScope final {
public:
  ArtifactImportCacheScope(
      const ArtifactStore &artifacts, const BlobStore *blobs,
      std::size_t entryLimit = defaultArtifactImportCacheEntryLimit);
  ~ArtifactImportCacheScope();

  ArtifactImportCacheScope(const ArtifactImportCacheScope &) = delete;
  ArtifactImportCacheScope &
  operator=(const ArtifactImportCacheScope &) = delete;

  ArtifactImportCacheStatistics statistics() const;

private:
  std::unique_ptr<ArtifactImportCache> owned_;
  ArtifactImportCache *active_ = nullptr;
  ArtifactImportCache *previous_ = nullptr;
};

ArtifactImportCache *currentArtifactImportCache();

template <typename Value, typename Loader>
llvm::Expected<std::shared_ptr<const Value>>
importCachedArtifact(const ArtifactStore &artifacts, const BlobStore *blobs,
                     llvm::ArrayRef<ArtifactRootReference> references,
                     Loader &&loader) {
  if (ArtifactImportCache *cache = currentArtifactImportCache())
    if (cache->owns(artifacts, blobs))
      return cache->import <Value>(artifacts, blobs, references,
                                   std::forward<Loader>(loader));
  auto imported = loader();
  if (!imported)
    return imported.takeError();
  return std::make_shared<const Value>(std::move(*imported));
}

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_ARTIFACTIMPORTCACHE_H
