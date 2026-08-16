#include "Evaluation/ArtifactImportCache.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Common/InvocationDiagnosticLog.h"

#include <algorithm>
#include <limits>
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
  std::uint64_t algorithmVersion = artifactImportCacheAlgorithmVersion;
  std::vector<ArtifactRootReference> references;
};

bool operator<(const CacheKey &lhs, const CacheKey &rhs) {
  if (lhs.type != rhs.type)
    return lhs.type < rhs.type;
  if (lhs.algorithmVersion != rhs.algorithmVersion)
    return lhs.algorithmVersion < rhs.algorithmVersion;
  return std::lexicographical_compare(
      lhs.references.begin(), lhs.references.end(), rhs.references.begin(),
      rhs.references.end(), artifactRootReferenceLess);
}

CacheKey makeKey(std::type_index type,
                 llvm::ArrayRef<ArtifactRootReference> references) {
  return CacheKey{
      type, artifactImportCacheAlgorithmVersion,
      std::vector<ArtifactRootReference>(references.begin(), references.end())};
}

void add(std::uint64_t &destination, std::uint64_t value) {
  if (value > std::numeric_limits<std::uint64_t>::max() - destination)
    destination = std::numeric_limits<std::uint64_t>::max();
  else
    destination += value;
}

llvm::StringRef spelling(ArtifactImportCacheVerificationDomain domain) {
  switch (domain) {
  case ArtifactImportCacheVerificationDomain::SourceInvocation:
    return "source_invocation";
  case ArtifactImportCacheVerificationDomain::IndependentReplay:
    return "independent_replay";
  }
  llvm_unreachable("unknown artifact import verification domain");
}

} // namespace

class ArtifactImportCache::Impl final {
public:
  struct Entry final {
    std::shared_ptr<const void> value;
    std::uint64_t constructionNanoseconds = 0;
    std::uint64_t minimumRetainedBytes = 0;
  };

  const ArtifactStore *artifacts = nullptr;
  const BlobStore *blobs = nullptr;
  std::size_t entryLimit = 0;
  std::mutex mutex;
  std::map<CacheKey, Entry> entries;
  ArtifactImportCacheStatistics statistics;
};

ArtifactImportCache::ArtifactImportCache(const ArtifactStore &artifacts,
                                         const BlobStore *blobs,
                                         std::size_t entryLimit)
    : impl_(std::make_unique<Impl>()) {
  impl_->artifacts = &artifacts;
  impl_->blobs = blobs;
  impl_->entryLimit = entryLimit;
}
ArtifactImportCache::~ArtifactImportCache() = default;

ArtifactImportCacheStatistics ArtifactImportCache::statistics() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->statistics;
}

bool ArtifactImportCache::owns(const ArtifactStore &artifacts,
                               const BlobStore *blobs) const {
  return impl_->artifacts == &artifacts && (!blobs || impl_->blobs == blobs);
}

ArtifactImportCache::LookupResult
ArtifactImportCache::lookup(std::type_index type,
                            llvm::ArrayRef<ArtifactRootReference> references) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  add(impl_->statistics.importRequests, 1);
  add(impl_->statistics.deterministicWork, 1);
  auto found = impl_->entries.find(makeKey(type, references));
  if (found == impl_->entries.end()) {
    add(impl_->statistics.cacheMisses, 1);
    return {};
  }
  return {found->second.value, found->second.constructionNanoseconds,
          found->second.minimumRetainedBytes};
}

std::shared_ptr<const void> ArtifactImportCache::insert(
    std::type_index type, llvm::ArrayRef<ArtifactRootReference> references,
    std::shared_ptr<const void> value, std::uint64_t constructionNanoseconds,
    std::uint64_t minimumRetainedBytes) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  add(impl_->statistics.uniqueConstructions, 1);
  add(impl_->statistics.constructionNanoseconds, constructionNanoseconds);
  add(impl_->statistics.deterministicWork, 1);
  if (impl_->entries.size() >= impl_->entryLimit) {
    add(impl_->statistics.uncachedConstructions, 1);
    return value;
  }
  auto [found, inserted] = impl_->entries.try_emplace(
      makeKey(type, references),
      Impl::Entry{std::move(value), constructionNanoseconds,
                  minimumRetainedBytes});
  if (inserted) {
    add(impl_->statistics.minimumRetainedBytes, minimumRetainedBytes);
    impl_->statistics.entryCount = impl_->entries.size();
  }
  return found->second.value;
}

llvm::Expected<std::uint64_t> ArtifactImportCache::revalidate(
    const ArtifactStore &artifacts,
    llvm::ArrayRef<ArtifactRootReference> references) {
  std::uint64_t byteCount = 0;
  for (const ArtifactRootReference &reference : references) {
    auto bytes = artifacts.get(reference);
    if (!bytes)
      return bytes.takeError();
    add(byteCount, bytes->bytes().size());
  }
  return byteCount;
}

void ArtifactImportCache::recordHit(std::uint64_t revalidatedBytes,
                                    std::uint64_t constructionNanoseconds,
                                    std::uint64_t minimumRetainedBytes) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  add(impl_->statistics.cacheHits, 1);
  add(impl_->statistics.revalidationCount, 1);
  add(impl_->statistics.revalidatedBytes, revalidatedBytes);
  add(impl_->statistics.constructionNanosecondsSaved, constructionNanoseconds);
  add(impl_->statistics.minimumRetainedBytesReused, minimumRetainedBytes);
  add(impl_->statistics.deterministicWork, 1);
}

void ArtifactImportCache::recordFailedConstruction(
    std::uint64_t constructionNanoseconds) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  add(impl_->statistics.uniqueConstructions, 1);
  add(impl_->statistics.constructionNanoseconds, constructionNanoseconds);
  add(impl_->statistics.deterministicWork, 1);
}

ArtifactImportCacheScope::ArtifactImportCacheScope(
    const ArtifactStore &artifacts, const BlobStore *blobs,
    std::size_t entryLimit)
    : previous_(currentCache) {
  if (previous_ && previous_->owns(artifacts, blobs)) {
    active_ = previous_;
  } else {
    owned_ =
        std::make_unique<ArtifactImportCache>(artifacts, blobs, entryLimit);
    active_ = owned_.get();
  }
  currentCache = active_;
}

ArtifactImportCacheScope::~ArtifactImportCacheScope() {
  currentCache = previous_;
}

ArtifactImportCacheStatistics ArtifactImportCacheScope::statistics() const {
  return active_ ? active_->statistics() : ArtifactImportCacheStatistics{};
}

void emitArtifactImportCacheStatistics(
    ArtifactImportCacheVerificationDomain domain,
    const ArtifactImportCacheStatistics &statistics) {
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary, InvocationDiagnosticStage::Deployment,
      InvocationDiagnosticEvent::ArtifactImportSession, [&] {
        llvm::json::Object payload;
        payload["verification_domain"] = spelling(domain);
        payload["import_requests"] = statistics.importRequests;
        payload["cache_hits"] = statistics.cacheHits;
        payload["cache_misses"] = statistics.cacheMisses;
        payload["unique_constructions"] = statistics.uniqueConstructions;
        payload["uncached_constructions"] = statistics.uncachedConstructions;
        payload["revalidation_count"] = statistics.revalidationCount;
        payload["revalidated_bytes"] = statistics.revalidatedBytes;
        payload["construction_time_ns"] = statistics.constructionNanoseconds;
        payload["construction_time_saved_ns"] =
            statistics.constructionNanosecondsSaved;
        payload["deterministic_work"] = statistics.deterministicWork;
        payload["minimum_retained_bytes"] = statistics.minimumRetainedBytes;
        payload["minimum_retained_bytes_reused"] =
            statistics.minimumRetainedBytesReused;
        payload["entry_count"] = statistics.entryCount;
        return llvm::json::Value(std::move(payload));
      });
}

ArtifactImportCache *currentArtifactImportCache() { return currentCache; }

} // namespace loom::evaluation
