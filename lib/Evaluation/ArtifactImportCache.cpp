#include "Evaluation/ArtifactImportCache.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <map>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::evaluation {
namespace {

thread_local ArtifactImportCache *currentCache = nullptr;

struct CacheKey final {
  std::type_index type;
  const ArtifactStore *artifacts = nullptr;
  const BlobStore *blobs = nullptr;
  std::vector<ArtifactRootReference> references;
};

bool operator<(const CacheKey &lhs, const CacheKey &rhs) {
  if (lhs.type != rhs.type)
    return lhs.type < rhs.type;
  const std::less<const void *> pointerLess;
  if (lhs.artifacts != rhs.artifacts)
    return pointerLess(lhs.artifacts, rhs.artifacts);
  if (lhs.blobs != rhs.blobs)
    return pointerLess(lhs.blobs, rhs.blobs);
  return std::lexicographical_compare(
      lhs.references.begin(), lhs.references.end(), rhs.references.begin(),
      rhs.references.end(), artifactRootReferenceLess);
}

CacheKey makeKey(std::type_index type, const ArtifactStore &artifacts,
                 const BlobStore *blobs,
                 llvm::ArrayRef<ArtifactRootReference> references) {
  return CacheKey{type, &artifacts, blobs,
                  std::vector<ArtifactRootReference>(references.begin(),
                                                     references.end())};
}

} // namespace

class ArtifactImportCache::Impl final {
public:
  std::mutex mutex;
  std::map<CacheKey, std::shared_ptr<const void>> entries;
  std::atomic<std::uint64_t> hitCount{0};
  std::atomic<std::uint64_t> missCount{0};
};

ArtifactImportCache::ArtifactImportCache() : impl_(std::make_unique<Impl>()) {}
ArtifactImportCache::~ArtifactImportCache() = default;

ArtifactImportCacheStatistics ArtifactImportCache::statistics() const {
  return {impl_->hitCount.load(std::memory_order_relaxed),
          impl_->missCount.load(std::memory_order_relaxed)};
}

std::shared_ptr<const void> ArtifactImportCache::lookup(
    std::type_index type, const ArtifactStore &artifacts,
    const BlobStore *blobs,
    llvm::ArrayRef<ArtifactRootReference> references) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto found = impl_->entries.find(makeKey(type, artifacts, blobs, references));
  if (found == impl_->entries.end()) {
    impl_->missCount.fetch_add(1, std::memory_order_relaxed);
    return {};
  }
  impl_->hitCount.fetch_add(1, std::memory_order_relaxed);
  return found->second;
}

std::shared_ptr<const void> ArtifactImportCache::insert(
    std::type_index type, const ArtifactStore &artifacts,
    const BlobStore *blobs,
    llvm::ArrayRef<ArtifactRootReference> references,
    std::shared_ptr<const void> value) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto [found, inserted] = impl_->entries.try_emplace(
      makeKey(type, artifacts, blobs, references), std::move(value));
  (void)inserted;
  return found->second;
}

ArtifactImportCacheScope::ArtifactImportCacheScope()
    : previous_(currentCache) {
  if (currentCache)
    return;
  owned_ = std::make_unique<ArtifactImportCache>();
  currentCache = owned_.get();
}

ArtifactImportCacheScope::~ArtifactImportCacheScope() {
  if (owned_)
    currentCache = previous_;
}

ArtifactImportCache *currentArtifactImportCache() { return currentCache; }

} // namespace loom::evaluation
