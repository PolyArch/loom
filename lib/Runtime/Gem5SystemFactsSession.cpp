#include "Runtime/Gem5SystemExecution.h"

#include "Gem5SystemExecutionInternal.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Evaluation/Request.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace loom::runtime {
namespace {

inline constexpr std::uint64_t gem5SystemFactsAlgorithmVersion = 1;

void addSaturated(std::uint64_t &destination, std::uint64_t value) {
  if (value > std::numeric_limits<std::uint64_t>::max() - destination)
    destination = std::numeric_limits<std::uint64_t>::max();
  else
    destination += value;
}

std::uint64_t retainedFactsBytes(
    const gem5_system::Gem5SystemFacts &facts) {
  std::uint64_t bytes = sizeof(facts);
  addSaturated(bytes, facts.artifactDependencies.size() *
                          sizeof(ArtifactRootReference));
  addSaturated(bytes,
               facts.blobDependencies.size() * sizeof(BlobDigest));
  for (const external_tool::MaterializedBundleFile &file :
       facts.semanticInputs) {
    addSaturated(bytes, file.relativePath.size());
    addSaturated(bytes, file.contents.size());
  }
  return bytes;
}

} // namespace

class Gem5SystemFactsSession::Impl final {
public:
  Impl(const ArtifactStore &artifacts, const BlobStore &blobs,
       std::size_t entryLimit)
      : artifacts_(&artifacts), blobs_(&blobs), entryLimit_(entryLimit) {}

  bool owns(const ArtifactStore &artifacts, const BlobStore &blobs) const {
    return artifacts_ == &artifacts && blobs_ == &blobs;
  }

  llvm::Expected<std::shared_ptr<const gem5_system::Gem5SystemFactsOrUnsupported>>
  get(const evaluation::EvaluationRequest &request,
      const evaluation::CaseArtifactResolution &resolution,
      const ArtifactStore &artifacts, const BlobStore &blobs) {
    addSaturated(statistics_.requests, 1);
    if (!owns(artifacts, blobs))
      return gem5_system::invalid(
          "Gem5SystemFacts session crosses its store verification domain");

    Key key{evaluation::evaluationRequestReference(request)};
    const auto found = llvm::find_if(entries_, [&](const Entry &entry) {
      return entry.key == key;
    });
    if (found != entries_.end()) {
      auto revalidated = revalidate(*found, artifacts, blobs);
      if (!revalidated)
        return revalidated.takeError();
      addSaturated(statistics_.cacheHits, 1);
      addSaturated(statistics_.revalidationCount, 1);
      addSaturated(statistics_.revalidatedArtifactBytes,
                   revalidated->first);
      addSaturated(statistics_.revalidatedBlobBytes, revalidated->second);
      addSaturated(statistics_.constructionNanosecondsSaved,
                   found->constructionNanoseconds);
      return found->facts;
    }

    addSaturated(statistics_.cacheMisses, 1);
    addSaturated(statistics_.constructionAttempts, 1);
    const auto begin = std::chrono::steady_clock::now();
    auto derived = gem5_system::deriveFactsUncached(request, resolution,
                                                    artifacts, blobs);
    const std::uint64_t constructionNanoseconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin)
            .count();
    addSaturated(statistics_.constructionNanoseconds,
                 constructionNanoseconds);
    if (!derived) {
      addSaturated(statistics_.failedConstructions, 1);
      return derived.takeError();
    }
    auto facts = std::make_shared<const gem5_system::Gem5SystemFactsOrUnsupported>(
        std::move(*derived));
    const auto *completed =
        std::get_if<gem5_system::Gem5SystemFacts>(facts.get());
    if (!completed) {
      addSaturated(statistics_.uncachedConstructions, 1);
      addSaturated(statistics_.unsupportedConstructions, 1);
      return facts;
    }

    addSaturated(statistics_.uniqueConstructions, 1);
    if (entries_.size() >= entryLimit_) {
      addSaturated(statistics_.uncachedConstructions, 1);
      return facts;
    }
    const std::uint64_t retainedBytes = retainedFactsBytes(*completed);
    entries_.push_back(
        {std::move(key), facts, constructionNanoseconds, retainedBytes});
    addSaturated(statistics_.minimumRetainedBytes, retainedBytes);
    statistics_.entryCount = entries_.size();
    return facts;
  }

  Gem5SystemFactsSessionStatistics statistics() const { return statistics_; }

private:
  struct Key final {
    ArtifactRootReference request;
    std::uint64_t algorithmVersion = gem5SystemFactsAlgorithmVersion;

    friend bool operator==(const Key &lhs, const Key &rhs) {
      return lhs.request == rhs.request &&
             lhs.algorithmVersion == rhs.algorithmVersion;
    }
  };

  struct Entry final {
    Key key;
    std::shared_ptr<const gem5_system::Gem5SystemFactsOrUnsupported> facts;
    std::uint64_t constructionNanoseconds = 0;
    std::uint64_t retainedBytes = 0;
  };

  static llvm::Expected<std::pair<std::uint64_t, std::uint64_t>>
  revalidate(const Entry &entry, const ArtifactStore &artifacts,
             const BlobStore &blobs) {
    const auto *facts =
        std::get_if<gem5_system::Gem5SystemFacts>(entry.facts.get());
    if (!facts)
      return gem5_system::invalid(
          "Gem5SystemFacts cache contains a non-completed entry");
    std::uint64_t artifactBytes = 0;
    for (const ArtifactRootReference &reference :
         facts->artifactDependencies) {
      auto bytes = artifacts.get(reference);
      if (!bytes)
        return bytes.takeError();
      addSaturated(artifactBytes, bytes->bytes().size());
    }
    std::uint64_t blobBytes = 0;
    for (const BlobDigest &digest : facts->blobDependencies) {
      auto bytes = blobs.verify(digest);
      if (!bytes)
        return bytes.takeError();
      addSaturated(blobBytes, *bytes);
    }
    return std::pair(artifactBytes, blobBytes);
  }

  const ArtifactStore *artifacts_ = nullptr;
  const BlobStore *blobs_ = nullptr;
  std::size_t entryLimit_ = 0;
  std::vector<Entry> entries_;
  Gem5SystemFactsSessionStatistics statistics_;
};

namespace {
thread_local std::shared_ptr<Gem5SystemFactsSession::Impl> currentFactsSession;
} // namespace

Gem5SystemFactsSession::Gem5SystemFactsSession(
    const ArtifactStore &artifacts, const BlobStore &blobs,
    Gem5SystemFactsSessionMode mode, std::size_t entryLimit)
    : previous_(currentFactsSession) {
  if (mode == Gem5SystemFactsSessionMode::ReuseEnclosing && previous_ &&
      previous_->owns(artifacts, blobs))
    active_ = previous_;
  else
    active_ = std::make_shared<Impl>(artifacts, blobs, entryLimit);
  currentFactsSession = active_;
}

Gem5SystemFactsSession::~Gem5SystemFactsSession() {
  currentFactsSession = previous_;
}

Gem5SystemFactsSessionStatistics Gem5SystemFactsSession::statistics() const {
  return active_ ? active_->statistics() : Gem5SystemFactsSessionStatistics{};
}

namespace gem5_system {

llvm::Expected<std::shared_ptr<const Gem5SystemFactsOrUnsupported>>
deriveFacts(const evaluation::EvaluationRequest &request,
            const evaluation::CaseArtifactResolution &resolution,
            const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (currentFactsSession && currentFactsSession->owns(artifacts, blobs))
    return currentFactsSession->get(request, resolution, artifacts, blobs);
  auto derived = deriveFactsUncached(request, resolution, artifacts, blobs);
  if (!derived)
    return derived.takeError();
  return std::make_shared<const Gem5SystemFactsOrUnsupported>(
      std::move(*derived));
}

} // namespace gem5_system
} // namespace loom::runtime
