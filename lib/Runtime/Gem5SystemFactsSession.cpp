#include "Runtime/Gem5SystemExecution.h"

#include "Gem5SystemExecutionInternal.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Common/InvocationDiagnosticLog.h"
#include "Evaluation/Request.h"
#include "ExternalTool/ExternalFile.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"

#include <sys/resource.h>
#include <time.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
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

std::optional<std::uint64_t> processCpuNanoseconds() {
  timespec current{};
  if (::clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &current) != 0 ||
      current.tv_sec < 0 || current.tv_nsec < 0 ||
      current.tv_nsec >= 1'000'000'000)
    return std::nullopt;
  constexpr std::uint64_t nanosecondsPerSecond = 1'000'000'000;
  const std::uint64_t seconds = current.tv_sec;
  if (seconds > (std::numeric_limits<std::uint64_t>::max() -
                 static_cast<std::uint64_t>(current.tv_nsec)) /
                    nanosecondsPerSecond)
    return std::nullopt;
  return seconds * nanosecondsPerSecond + current.tv_nsec;
}

std::optional<std::uint64_t> timevalNanoseconds(const timeval &value) {
  if (value.tv_sec < 0 || value.tv_usec < 0 || value.tv_usec >= 1'000'000)
    return std::nullopt;
  constexpr std::uint64_t nanosecondsPerSecond = 1'000'000'000;
  const std::uint64_t subsecond =
      static_cast<std::uint64_t>(value.tv_usec) * 1000;
  const std::uint64_t seconds = value.tv_sec;
  if (seconds > (std::numeric_limits<std::uint64_t>::max() - subsecond) /
                    nanosecondsPerSecond)
    return std::nullopt;
  return seconds * nanosecondsPerSecond + subsecond;
}

std::optional<std::uint64_t> childCpuNanoseconds() {
  rusage usage{};
  if (::getrusage(RUSAGE_CHILDREN, &usage) != 0)
    return std::nullopt;
  auto user = timevalNanoseconds(usage.ru_utime);
  auto system = timevalNanoseconds(usage.ru_stime);
  if (!user || !system ||
      *system > std::numeric_limits<std::uint64_t>::max() - *user)
    return std::nullopt;
  return *user + *system;
}

std::optional<std::uint64_t> difference(std::optional<std::uint64_t> end,
                                        std::optional<std::uint64_t> begin) {
  if (!end || !begin || *end < *begin)
    return std::nullopt;
  return *end - *begin;
}

std::uint64_t retainedFactsBytes(const gem5_system::Gem5SystemFacts &facts) {
  std::uint64_t bytes = sizeof(facts);
  addSaturated(bytes, facts.artifactDependencies.size() *
                          sizeof(ArtifactRootReference));
  addSaturated(bytes, facts.blobDependencies.size() * sizeof(BlobDigest));
  for (const external_tool::MaterializedBundleFile &file :
       facts.semanticInputs) {
    addSaturated(bytes, file.relativePath.size());
    addSaturated(bytes, file.contents.size());
  }
  return bytes;
}

} // namespace

namespace gem5_system {

class Gem5SystemFactsOperationTimer::Impl final {
public:
  explicit Impl(Gem5SystemFactsOperationStatistics &statistics)
      : statistics(&statistics), wallStarted(std::chrono::steady_clock::now()),
        selfCpuStarted(processCpuNanoseconds()),
        childCpuStarted(childCpuNanoseconds()) {}

  Gem5SystemFactsOperationStatistics *statistics;
  std::chrono::steady_clock::time_point wallStarted;
  std::optional<std::uint64_t> selfCpuStarted;
  std::optional<std::uint64_t> childCpuStarted;
};

Gem5SystemFactsOperationTimer::Gem5SystemFactsOperationTimer(
    Gem5SystemFactsOperationStatistics *statistics) {
  if (statistics)
    impl_ = std::make_unique<Impl>(*statistics);
}

Gem5SystemFactsOperationTimer::~Gem5SystemFactsOperationTimer() {
  if (!impl_)
    return;
  Gem5SystemFactsOperationStatistics &statistics = *impl_->statistics;
  addSaturated(statistics.invocations, 1);
  const auto wall = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - impl_->wallStarted);
  if (wall.count() > 0)
    addSaturated(statistics.wallNanoseconds,
                 static_cast<std::uint64_t>(wall.count()));
  if (auto elapsed =
          difference(processCpuNanoseconds(), impl_->selfCpuStarted)) {
    addSaturated(statistics.selfCpuNanoseconds, *elapsed);
    addSaturated(statistics.selfCpuObservationCount, 1);
  }
  if (auto elapsed =
          difference(childCpuNanoseconds(), impl_->childCpuStarted)) {
    addSaturated(statistics.childCpuNanoseconds, *elapsed);
    addSaturated(statistics.childCpuObservationCount, 1);
  }
}

} // namespace gem5_system

class Gem5SystemFactsSession::Impl final {
public:
  Impl(const ArtifactStore &artifacts, const BlobStore &blobs,
       std::size_t entryLimit)
      : artifacts_(&artifacts), blobs_(&blobs), entryLimit_(entryLimit) {}

  bool owns(const ArtifactStore &artifacts, const BlobStore &blobs) const {
    return artifacts_ == &artifacts && blobs_ == &blobs;
  }

  llvm::Expected<
      std::shared_ptr<const gem5_system::Gem5SystemFactsOrUnsupported>>
  get(const evaluation::EvaluationRequest &request,
      const evaluation::CaseArtifactResolution &resolution,
      const ArtifactStore &artifacts, const BlobStore &blobs) {
    std::lock_guard<std::mutex> lock(mutex_);
    addSaturated(statistics_.requests, 1);
    if (!owns(artifacts, blobs))
      return gem5_system::invalid(
          "Gem5SystemFacts session crosses its store verification domain");

    Key key{evaluation::evaluationRequestReference(request)};
    const auto found = llvm::find_if(
        entries_, [&](const Entry &entry) { return entry.key == key; });
    if (found != entries_.end()) {
      auto revalidated = revalidate(*found, artifacts, blobs);
      if (!revalidated)
        return revalidated.takeError();
      addSaturated(statistics_.cacheHits, 1);
      addSaturated(statistics_.revalidationCount, 1);
      addSaturated(statistics_.revalidatedArtifactBytes, revalidated->first);
      addSaturated(statistics_.revalidatedBlobBytes, revalidated->second);
      addSaturated(statistics_.constructionNanosecondsSaved,
                   found->constructionNanoseconds);
      return found->facts;
    }

    addSaturated(statistics_.cacheMisses, 1);
    addSaturated(statistics_.constructionAttempts, 1);
    const auto begin = std::chrono::steady_clock::now();
    auto derived = gem5_system::deriveFactsUncached(
        request, resolution, artifacts, blobs, &statistics_.construction);
    const std::uint64_t constructionNanoseconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin)
            .count();
    addSaturated(statistics_.constructionNanoseconds, constructionNanoseconds);
    if (!derived) {
      addSaturated(statistics_.failedConstructions, 1);
      return derived.takeError();
    }
    auto facts =
        std::make_shared<const gem5_system::Gem5SystemFactsOrUnsupported>(
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

  llvm::Expected<external_tool::ExternalFileFingerprint>
  externalFileFingerprint(llvm::StringRef path) {
    std::lock_guard<std::mutex> lock(mutex_);
    addSaturated(statistics_.externalFileFingerprintRequests, 1);
    auto identity = external_tool::observeExternalFileIdentity(path);
    if (!identity)
      return identity.takeError();
    const auto found = llvm::find_if(
        externalFiles_, [&](const ExternalFileEntry &entry) {
          return entry.path == path && entry.observation.identity == *identity;
        });
    if (found != externalFiles_.end()) {
      addSaturated(statistics_.externalFileFingerprintHits, 1);
      return found->observation.fingerprint;
    }
    addSaturated(statistics_.externalFileFingerprintMisses, 1);
    const auto begin = std::chrono::steady_clock::now();
    auto observation = external_tool::observeExternalFile(path);
    addSaturated(statistics_.externalFileFingerprintNanoseconds,
                 std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - begin)
                     .count());
    if (!observation)
      return observation.takeError();
    addSaturated(statistics_.externalFileFingerprintedBytes,
                 observation->identity.size);
    const auto stale = llvm::find_if(
        externalFiles_,
        [&](const ExternalFileEntry &entry) { return entry.path == path; });
    if (stale != externalFiles_.end())
      stale->observation = *observation;
    else
      externalFiles_.push_back({path.str(), *observation});
    statistics_.externalFileFingerprintEntryCount = externalFiles_.size();
    return observation->fingerprint;
  }

  Gem5SystemFactsSessionStatistics statistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return statistics_;
  }

private:
  struct ExternalFileEntry final {
    std::string path;
    external_tool::ExternalFileObservation observation;
  };

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
    for (const ArtifactRootReference &reference : facts->artifactDependencies) {
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
  mutable std::mutex mutex_;
  std::vector<Entry> entries_;
  std::vector<ExternalFileEntry> externalFiles_;
  Gem5SystemFactsSessionStatistics statistics_;
};

namespace {
thread_local std::shared_ptr<Gem5SystemFactsSession::Impl> currentFactsSession;
} // namespace

Gem5SystemFactsSession::Gem5SystemFactsSession(const ArtifactStore &artifacts,
                                               const BlobStore &blobs,
                                               Gem5SystemFactsSessionMode mode,
                                               std::size_t entryLimit)
    : previous_(currentFactsSession) {
  if (mode == Gem5SystemFactsSessionMode::ReuseEnclosing && previous_ &&
      previous_->owns(artifacts, blobs))
    active_ = previous_;
  else
    active_ = std::make_shared<Impl>(artifacts, blobs, entryLimit);
  currentFactsSession = active_;
}

Gem5SystemFactsSession::Gem5SystemFactsSession(const Attachment &attachment)
    : active_(attachment.state_), previous_(currentFactsSession) {
  currentFactsSession = active_;
}

Gem5SystemFactsSession::~Gem5SystemFactsSession() {
  currentFactsSession = previous_;
}

Gem5SystemFactsSessionStatistics Gem5SystemFactsSession::statistics() const {
  return active_ ? active_->statistics() : Gem5SystemFactsSessionStatistics{};
}

void emitGem5SystemFactsSessionStatistics(
    const Gem5SystemFactsSessionStatistics &s) {
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary, InvocationDiagnosticStage::Deployment,
      InvocationDiagnosticEvent::Gem5SystemFactsSession, [&] {
        llvm::json::Object payload;
        payload["requests"] = s.requests;
        payload["cache_hits"] = s.cacheHits;
        payload["cache_misses"] = s.cacheMisses;
        payload["construction_attempts"] = s.constructionAttempts;
        payload["unique_constructions"] = s.uniqueConstructions;
        payload["uncached_constructions"] = s.uncachedConstructions;
        payload["unsupported_constructions"] = s.unsupportedConstructions;
        payload["failed_constructions"] = s.failedConstructions;
        payload["revalidation_count"] = s.revalidationCount;
        payload["revalidated_artifact_bytes"] = s.revalidatedArtifactBytes;
        payload["revalidated_blob_bytes"] = s.revalidatedBlobBytes;
        payload["construction_time_ns"] = s.constructionNanoseconds;
        payload["construction_time_saved_ns"] =
            s.constructionNanosecondsSaved;
        payload["minimum_retained_bytes"] = s.minimumRetainedBytes;
        payload["entry_count"] = s.entryCount;
        payload["external_file_fingerprint_requests"] =
            s.externalFileFingerprintRequests;
        payload["external_file_fingerprint_hits"] =
            s.externalFileFingerprintHits;
        payload["external_file_fingerprint_misses"] =
            s.externalFileFingerprintMisses;
        payload["external_file_fingerprinted_bytes"] =
            s.externalFileFingerprintedBytes;
        payload["external_file_fingerprint_time_ns"] =
            s.externalFileFingerprintNanoseconds;
        payload["external_file_fingerprint_entry_count"] =
            s.externalFileFingerprintEntryCount;
        return llvm::json::Value(std::move(payload));
      });
}

namespace {

class Gem5SystemInvocationContextActivation final
    : public evaluation::EvaluationModelInvocationContext::Activation {
public:
  explicit Gem5SystemInvocationContextActivation(
      const Gem5SystemFactsSession::Attachment &attachment)
      : session_(attachment) {}

private:
  Gem5SystemFactsSession session_;
};

class Gem5SystemInvocationContext final
    : public evaluation::EvaluationModelInvocationContext {
public:
  Gem5SystemInvocationContext(const ArtifactStore &artifacts,
                              const BlobStore &blobs,
                              Gem5SystemFactsSession::Attachment attachment)
      : artifacts_(&artifacts), blobs_(&blobs),
        attachment_(std::move(attachment)) {}

  llvm::Expected<std::unique_ptr<Activation>>
  activate(const ArtifactStore &artifacts,
           const BlobStore &blobs) const override {
    if (&artifacts != artifacts_ || &blobs != blobs_)
      return gem5_system::invalid(
          "Gem5 System invocation context crosses its store verification "
          "domain");
    if (!attachment_)
      return gem5_system::invalid(
          "Gem5 System invocation context has no facts session");
    return std::unique_ptr<Activation>(
        std::make_unique<Gem5SystemInvocationContextActivation>(attachment_));
  }

private:
  const ArtifactStore *artifacts_ = nullptr;
  const BlobStore *blobs_ = nullptr;
  Gem5SystemFactsSession::Attachment attachment_;
};

} // namespace

llvm::Expected<
    std::shared_ptr<const evaluation::EvaluationModelInvocationContext>>
openGem5SystemInvocationContext(
    const evaluation::EvaluationRequest &request,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  (void)request;
  (void)resolution;
  Gem5SystemFactsSession session(artifacts, blobs);
  return std::shared_ptr<const evaluation::EvaluationModelInvocationContext>(
      std::make_shared<const Gem5SystemInvocationContext>(
          artifacts, blobs, session.attachment()));
}

namespace gem5_system {

llvm::Expected<external_tool::ExternalFileFingerprint>
sessionExternalFileFingerprint(llvm::StringRef path) {
  if (currentFactsSession)
    return currentFactsSession->externalFileFingerprint(path);
  return external_tool::fingerprintExternalFile(path);
}

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
