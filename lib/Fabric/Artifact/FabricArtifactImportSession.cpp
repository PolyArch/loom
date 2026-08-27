#include "FabricArtifactImportSessionInternal.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/InvocationDiagnosticLog.h"

#include "llvm/Support/Error.h"

#include <limits>

namespace loom::fabric {

namespace {

thread_local std::shared_ptr<detail::FabricArtifactImportSessionState>
    currentImportSession;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

} // namespace

detail::FabricArtifactImportSessionState::FabricArtifactImportSessionState(
    std::size_t entryLimit)
    : entryLimit_(entryLimit) {
  statistics_.entryLimit = entryLimit;
}

void detail::FabricArtifactImportSessionState::add(std::uint64_t &destination,
                                                   std::uint64_t value) {
  if (value > std::numeric_limits<std::uint64_t>::max() - destination)
    destination = std::numeric_limits<std::uint64_t>::max();
  else
    destination += value;
}

bool detail::FabricArtifactImportSessionState::KeyLess::operator()(
    const FabricArtifactImportSessionKey &lhs,
    const FabricArtifactImportSessionKey &rhs) const {
  if (lhs.reference != rhs.reference)
    return artifactRootReferenceLess(lhs.reference, rhs.reference);
  return lhs.algorithmVersion < rhs.algorithmVersion;
}

llvm::Expected<detail::FabricArtifactImportSessionState::Lookup>
detail::FabricArtifactImportSessionState::lookupOrReserve(
    const ArtifactRootReference &reference) {
  const FabricArtifactImportSessionKey key{reference};
  std::unique_lock<std::mutex> lock(mutex_);
  add(statistics_.importRequests, 1);
  add(statistics_.deterministicWork, 1);
  while (true) {
    const auto found = entries_.find(key);
    if (found != entries_.end()) {
      add(statistics_.cacheHits, 1);
      add(statistics_.constructionNanosecondsSaved,
          found->second->constructionNanoseconds);
      add(statistics_.retainedPayloadBytesReused,
          found->second->retainedPayloadBytes);
      return Lookup{found->second, false};
    }
    const auto constructing = constructing_.find(key);
    if (constructing == constructing_.end()) {
      constructing_.emplace(key, std::this_thread::get_id());
      add(statistics_.cacheMisses, 1);
      return Lookup{{}, true};
    }
    if (constructing->second == std::this_thread::get_id())
      return invalid("recursive Fabric artifact dependency import");
    add(statistics_.coalescedWaits, 1);
    condition_.wait(lock, [&] { return !constructing_.count(key); });
  }
}

void detail::FabricArtifactImportSessionState::recordRevalidation(
    std::uint64_t byteCount) {
  std::lock_guard<std::mutex> lock(mutex_);
  add(statistics_.revalidationCount, 1);
  add(statistics_.revalidatedBytes, byteCount);
  add(statistics_.bytesRead, byteCount);
  add(statistics_.bytesCopied, byteCount);
}

std::shared_ptr<const detail::FabricArtifactImportSessionEntry>
detail::FabricArtifactImportSessionState::complete(
    const ArtifactRootReference &reference,
    std::shared_ptr<const FabricStrictImportResult> imported,
    std::uint64_t retainedPayloadBytes, std::uint64_t constructionNanoseconds) {
  const FabricArtifactImportSessionKey key{reference};
  std::lock_guard<std::mutex> lock(mutex_);
  add(statistics_.uniqueConstructions, 1);
  add(statistics_.constructionNanoseconds, constructionNanoseconds);
  add(statistics_.deterministicWork, 1);
  auto entry = std::make_shared<const FabricArtifactImportSessionEntry>(
      FabricArtifactImportSessionEntry{
          std::move(imported), retainedPayloadBytes, constructionNanoseconds});
  if (entries_.size() >= entryLimit_) {
    add(statistics_.uncachedConstructions, 1);
  } else {
    entries_.emplace(key, entry);
    add(statistics_.retainedPayloadBytes, retainedPayloadBytes);
    statistics_.entryCount = entries_.size();
  }
  constructing_.erase(key);
  condition_.notify_all();
  return entry;
}

std::shared_ptr<const FabricHandshakeContext>
detail::FabricArtifactImportSessionState::lookupHandshakeContext(
    const ArtifactIdentity &fabric) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto found = handshakeContexts_.find(fabric);
  return found == handshakeContexts_.end() ? nullptr : found->second;
}

void detail::FabricArtifactImportSessionState::retainHandshakeContext(
    const ArtifactIdentity &fabric,
    std::shared_ptr<const FabricHandshakeContext> context) {
  std::lock_guard<std::mutex> lock(mutex_);
  handshakeContexts_.try_emplace(fabric, std::move(context));
}

void detail::FabricArtifactImportSessionState::abandon(
    const ArtifactRootReference &reference,
    std::uint64_t constructionNanoseconds) {
  const FabricArtifactImportSessionKey key{reference};
  std::lock_guard<std::mutex> lock(mutex_);
  add(statistics_.uniqueConstructions, 1);
  add(statistics_.constructionNanoseconds, constructionNanoseconds);
  add(statistics_.deterministicWork, 1);
  constructing_.erase(key);
  condition_.notify_all();
}

FabricArtifactImportSessionStatistics
detail::FabricArtifactImportSessionState::statistics() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return statistics_;
}

std::shared_ptr<detail::FabricArtifactImportSessionState>
detail::currentFabricArtifactImportSession() {
  return currentImportSession;
}

FabricArtifactImportSession::FabricArtifactImportSession(
    FabricArtifactImportSessionMode mode, std::size_t entryLimit)
    : previous_(currentImportSession) {
  if (mode == FabricArtifactImportSessionMode::ReuseEnclosing &&
      currentImportSession) {
    active_ = currentImportSession;
  } else {
    active_ =
        std::make_shared<detail::FabricArtifactImportSessionState>(entryLimit);
  }
  currentImportSession = active_;
}

FabricArtifactImportSession::FabricArtifactImportSession(
    const Attachment &attachment)
    : active_(attachment.state_), previous_(currentImportSession) {
  currentImportSession = active_;
}

FabricArtifactImportSession::~FabricArtifactImportSession() {
  currentImportSession = previous_;
}

FabricArtifactImportSession::Attachment
FabricArtifactImportSession::currentAttachment() {
  return Attachment(currentImportSession);
}

FabricArtifactImportSessionStatistics
FabricArtifactImportSession::statistics() const {
  return active_ ? active_->statistics()
                 : FabricArtifactImportSessionStatistics{};
}

void emitFabricArtifactImportSessionStatistics(
    FabricArtifactImportVerificationDomain domain,
    InvocationDiagnosticStage stage,
    const FabricArtifactImportSessionStatistics &statistics) {
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary, stage,
      InvocationDiagnosticEvent::ArtifactImportSession, [&] {
        llvm::json::Object payload;
        switch (domain) {
        case FabricArtifactImportVerificationDomain::SourceInvocation:
          payload["verification_domain"] = "source_invocation";
          break;
        case FabricArtifactImportVerificationDomain::IndependentReplay:
          payload["verification_domain"] = "independent_replay";
          break;
        }
        payload["artifact_domain"] = "fabric";
        payload["import_requests"] = statistics.importRequests;
        payload["cache_hits"] = statistics.cacheHits;
        payload["cache_misses"] = statistics.cacheMisses;
        payload["coalesced_waits"] = statistics.coalescedWaits;
        payload["unique_constructions"] = statistics.uniqueConstructions;
        payload["uncached_constructions"] = statistics.uncachedConstructions;
        payload["revalidation_count"] = statistics.revalidationCount;
        payload["revalidated_bytes"] = statistics.revalidatedBytes;
        payload["bytes_read"] = statistics.bytesRead;
        payload["bytes_copied"] = statistics.bytesCopied;
        payload["construction_time_ns"] = statistics.constructionNanoseconds;
        payload["construction_time_saved_ns"] =
            statistics.constructionNanosecondsSaved;
        payload["deterministic_work"] = statistics.deterministicWork;
        payload["retained_payload_bytes"] = statistics.retainedPayloadBytes;
        payload["retained_payload_bytes_reused"] =
            statistics.retainedPayloadBytesReused;
        payload["entry_count"] = statistics.entryCount;
        payload["entry_limit"] = statistics.entryLimit;
        return llvm::json::Value(std::move(payload));
      });
}

llvm::Expected<std::shared_ptr<const FabricHandshakeContext>>
acquireFabricHandshakeContext(const FabricArtifactView &view) {
  auto session = detail::currentFabricArtifactImportSession();
  if (session)
    if (auto retained = session->lookupHandshakeContext(view.identity())) {
      // Revalidation is the reuse oracle: a hit is never trusted on identity
      // alone.
      if (llvm::Error error =
              revalidateFabricHandshakeContext(*retained, view))
        return std::move(error);
      return retained;
    }
  auto built = buildFabricHandshakeContext(view);
  if (!built)
    return built.takeError();
  auto owned =
      std::make_shared<const FabricHandshakeContext>(std::move(*built));
  if (session)
    session->retainHandshakeContext(view.identity(), owned);
  return owned;
}

} // namespace loom::fabric
