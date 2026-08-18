#include "PnrDerivedContextSessionInternal.h"

#include <algorithm>
#include <limits>

namespace loom::pnr {
namespace {

thread_local std::shared_ptr<detail::PnrDerivedContextSessionState>
    currentSession;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "pnr_derived_context_invalid: " + message);
}

detail::PnrDerivedContextSessionKey makeKey(
    detail::PnrDerivedContextDomain domain,
    llvm::ArrayRef<std::uint8_t> digest) {
  detail::PnrDerivedContextSessionKey key;
  key.domain = domain;
  std::copy(digest.begin(), digest.end(), key.digest.begin());
  return key;
}

} // namespace

detail::PnrDerivedContextSessionState::PnrDerivedContextSessionState(
    std::size_t entryLimit)
    : entryLimit_(entryLimit) {
  statistics_.entryLimit = entryLimit;
}

void detail::PnrDerivedContextSessionState::add(std::uint64_t &destination,
                                                std::uint64_t value) {
  if (value > std::numeric_limits<std::uint64_t>::max() - destination)
    destination = std::numeric_limits<std::uint64_t>::max();
  else
    destination += value;
}

bool detail::PnrDerivedContextSessionState::KeyLess::operator()(
    const PnrDerivedContextSessionKey &lhs,
    const PnrDerivedContextSessionKey &rhs) const {
  if (lhs.domain != rhs.domain)
    return lhs.domain < rhs.domain;
  if (lhs.digest != rhs.digest)
    return lhs.digest < rhs.digest;
  return lhs.algorithmVersion < rhs.algorithmVersion;
}

llvm::Expected<detail::PnrDerivedContextSessionState::Lookup>
detail::PnrDerivedContextSessionState::lookupOrReserve(
    PnrDerivedContextDomain domain, llvm::ArrayRef<std::uint8_t> digest) {
  if (digest.size() != 32)
    return invalid("cache key digest is not SHA-256 sized");
  const PnrDerivedContextSessionKey key = makeKey(domain, digest);
  std::unique_lock<std::mutex> lock(mutex_);
  add(statistics_.requests, 1);
  add(statistics_.deterministicWork, 1);
  while (true) {
    const auto found = entries_.find(key);
    if (found != entries_.end()) {
      add(statistics_.cacheHits, 1);
      add(statistics_.constructionNanosecondsSaved,
          found->second->constructionNanoseconds);
      add(statistics_.retainedBytesReused, found->second->retainedBytes);
      return Lookup{found->second, false};
    }
    const auto constructing = constructing_.find(key);
    if (constructing == constructing_.end()) {
      constructing_.emplace(key, std::this_thread::get_id());
      add(statistics_.cacheMisses, 1);
      return Lookup{{}, true};
    }
    if (constructing->second == std::this_thread::get_id())
      return invalid("recursive derived-context construction");
    add(statistics_.coalescedWaits, 1);
    condition_.wait(lock, [&] { return !constructing_.count(key); });
  }
}

std::shared_ptr<const detail::PnrDerivedContextSessionEntry>
detail::PnrDerivedContextSessionState::complete(
    PnrDerivedContextDomain domain, llvm::ArrayRef<std::uint8_t> digest,
    std::shared_ptr<const void> context,
    std::uint64_t constructionNanoseconds, std::uint64_t retainedBytes,
    std::uint64_t deterministicWork) {
  const PnrDerivedContextSessionKey key = makeKey(domain, digest);
  std::lock_guard<std::mutex> lock(mutex_);
  add(statistics_.uniqueConstructions, 1);
  add(statistics_.constructionNanoseconds, constructionNanoseconds);
  add(statistics_.deterministicWork, deterministicWork);
  auto entry = std::make_shared<const PnrDerivedContextSessionEntry>(
      PnrDerivedContextSessionEntry{std::move(context),
                                    constructionNanoseconds, retainedBytes,
                                    deterministicWork});
  if (entries_.size() >= entryLimit_) {
    add(statistics_.uncachedConstructions, 1);
  } else {
    entries_.emplace(key, entry);
    add(statistics_.retainedBytes, retainedBytes);
    statistics_.entryCount = entries_.size();
  }
  constructing_.erase(key);
  condition_.notify_all();
  return entry;
}

void detail::PnrDerivedContextSessionState::abandon(
    PnrDerivedContextDomain domain, llvm::ArrayRef<std::uint8_t> digest,
    std::uint64_t constructionNanoseconds) {
  const PnrDerivedContextSessionKey key = makeKey(domain, digest);
  std::lock_guard<std::mutex> lock(mutex_);
  add(statistics_.uniqueConstructions, 1);
  add(statistics_.constructionNanoseconds, constructionNanoseconds);
  constructing_.erase(key);
  condition_.notify_all();
}

void detail::PnrDerivedContextSessionState::recordRevalidation() {
  std::lock_guard<std::mutex> lock(mutex_);
  add(statistics_.revalidationCount, 1);
  add(statistics_.deterministicWork, 1);
}

PnrDerivedContextSessionStatistics
detail::PnrDerivedContextSessionState::statistics() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return statistics_;
}

std::shared_ptr<detail::PnrDerivedContextSessionState>
detail::currentPnrDerivedContextSession() {
  return currentSession;
}

PnrDerivedContextSession::PnrDerivedContextSession(
    PnrDerivedContextSessionMode mode, std::size_t entryLimit)
    : previous_(currentSession) {
  if (mode == PnrDerivedContextSessionMode::ReuseEnclosing && currentSession)
    active_ = currentSession;
  else
    active_ =
        std::make_shared<detail::PnrDerivedContextSessionState>(entryLimit);
  currentSession = active_;
}

PnrDerivedContextSession::PnrDerivedContextSession(
    const Attachment &attachment)
    : active_(attachment.state_), previous_(currentSession) {
  currentSession = active_;
}

PnrDerivedContextSession::~PnrDerivedContextSession() {
  currentSession = previous_;
}

PnrDerivedContextSession::Attachment
PnrDerivedContextSession::currentAttachment() {
  return Attachment(currentSession);
}

PnrDerivedContextSessionStatistics
PnrDerivedContextSession::statistics() const {
  return active_ ? active_->statistics() : PnrDerivedContextSessionStatistics{};
}

} // namespace loom::pnr
