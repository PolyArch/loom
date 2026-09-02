#include "DSE/SiteScheduler.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "site_scheduler_invalid: " + message);
}

BlobDigest deriveResourceDigest(llvm::StringRef domain,
                                const BlobDigest &binding) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(domain.size() + 1 + binding.bytes().size());
  bytes.insert(bytes.end(), domain.bytes_begin(), domain.bytes_end());
  bytes.push_back(0);
  bytes.insert(bytes.end(), binding.bytes().begin(), binding.bytes().end());
  return computeBlobDigest(bytes);
}

bool resourceLess(const CountedSiteResource &lhs,
                  const CountedSiteResource &rhs) {
  return lhs.key < rhs.key;
}

llvm::Expected<std::vector<CountedSiteResource>>
validateResources(llvm::ArrayRef<CountedSiteResource> resources,
                  SiteResourceKind expectedKind, llvm::StringRef field) {
  std::vector<CountedSiteResource> result(resources.begin(), resources.end());
  if (!llvm::is_sorted(result, resourceLess))
    return invalid(field + " resources are not in canonical key order");
  for (std::size_t index = 0; index != result.size(); ++index) {
    if (result[index].key.kind() != expectedKind)
      return invalid(field + " resource has the wrong typed key");
    if (result[index].units == 0)
      return invalid(field + " resource has zero units");
    if (index != 0 && result[index - 1].key == result[index].key)
      return invalid(field + " resources contain a duplicate key");
  }
  return result;
}

const CountedSiteResource *
findResource(llvm::ArrayRef<CountedSiteResource> resources,
             const SiteResourceKey &key) {
  auto found = llvm::lower_bound(
      resources, key,
      [](const CountedSiteResource &entry, const SiteResourceKey &candidate) {
        return entry.key < candidate;
      });
  if (found == resources.end() || !(found->key == key))
    return nullptr;
  return &*found;
}

struct MutableUsage final {
  std::uint64_t cpuCores = 0;
  std::uint64_t memoryBytes = 0;
  std::uint64_t scratchBytes = 0;
  std::map<SiteResourceKey, std::uint64_t> externalTools;
  std::map<SiteResourceKey, std::uint64_t> licenses;
};

bool resourcesFit(llvm::ArrayRef<CountedSiteResource> claim,
                  llvm::ArrayRef<CountedSiteResource> capacity,
                  const std::map<SiteResourceKey, std::uint64_t> &allocated,
                  std::uint64_t undeclaredUnits) {
  for (const CountedSiteResource &entry : claim) {
    const CountedSiteResource *limit = findResource(capacity, entry.key);
    const std::uint64_t units = limit ? limit->units : undeclaredUnits;
    if (entry.units > units)
      return false;
    auto found = allocated.find(entry.key);
    const std::uint64_t used = found == allocated.end() ? 0 : found->second;
    if (used > units || entry.units > units - used)
      return false;
  }
  return true;
}

bool fits(const SiteResourceClaim &claim, const SiteCapacity &capacity,
          const MutableUsage &allocated) {
  if (allocated.cpuCores > capacity.cpuCores() ||
      claim.cpuCores() > capacity.cpuCores() - allocated.cpuCores)
    return false;
  if (allocated.memoryBytes > capacity.memoryBytes() ||
      claim.memoryBytes() > capacity.memoryBytes() - allocated.memoryBytes)
    return false;
  if (allocated.scratchBytes > capacity.scratchBytes() ||
      claim.scratchBytes() > capacity.scratchBytes() - allocated.scratchBytes)
    return false;
  return resourcesFit(claim.externalTools(), capacity.externalTools(),
                      allocated.externalTools,
                      capacity.undeclaredExternalToolUnits()) &&
         resourcesFit(claim.licenses(), capacity.licenses(), allocated.licenses,
                      0);
}

bool admitted(const SiteResourceClaim &claim, const SiteCapacity &capacity) {
  return fits(claim, capacity, MutableUsage{});
}

void addResources(std::map<SiteResourceKey, std::uint64_t> &destination,
                  llvm::ArrayRef<CountedSiteResource> resources) {
  for (const CountedSiteResource &resource : resources)
    destination[resource.key] += resource.units;
}

void subtractResources(std::map<SiteResourceKey, std::uint64_t> &destination,
                       llvm::ArrayRef<CountedSiteResource> resources) {
  for (const CountedSiteResource &resource : resources) {
    auto found = destination.find(resource.key);
    if (found == destination.end() || found->second < resource.units)
      llvm_unreachable("site scheduler resource accounting underflow");
    found->second -= resource.units;
    if (found->second == 0)
      destination.erase(found);
  }
}

void add(MutableUsage &destination, const SiteResourceClaim &claim) {
  destination.cpuCores += claim.cpuCores();
  destination.memoryBytes += claim.memoryBytes();
  destination.scratchBytes += claim.scratchBytes();
  addResources(destination.externalTools, claim.externalTools());
  addResources(destination.licenses, claim.licenses());
}

void subtract(MutableUsage &destination, const SiteResourceClaim &claim) {
  if (destination.cpuCores < claim.cpuCores() ||
      destination.memoryBytes < claim.memoryBytes() ||
      destination.scratchBytes < claim.scratchBytes())
    llvm_unreachable("site scheduler scalar accounting underflow");
  destination.cpuCores -= claim.cpuCores();
  destination.memoryBytes -= claim.memoryBytes();
  destination.scratchBytes -= claim.scratchBytes();
  subtractResources(destination.externalTools, claim.externalTools());
  subtractResources(destination.licenses, claim.licenses());
}

std::vector<CountedSiteResource>
copyResources(const std::map<SiteResourceKey, std::uint64_t> &resources) {
  std::vector<CountedSiteResource> result;
  result.reserve(resources.size());
  for (const auto &[key, units] : resources)
    result.push_back({key, units});
  return result;
}

bool containsKey(llvm::ArrayRef<ScheduledWorkUnit> records,
                 const WorkUnitKey &key) {
  return llvm::any_of(records, [&](const ScheduledWorkUnit &record) {
    return record.key == key;
  });
}

bool sameClaim(const SiteResourceClaim &lhs, const SiteResourceClaim &rhs) {
  return lhs.cpuCores() == rhs.cpuCores() &&
         lhs.memoryBytes() == rhs.memoryBytes() &&
         lhs.scratchBytes() == rhs.scratchBytes() &&
         lhs.externalTools() == rhs.externalTools() &&
         lhs.licenses() == rhs.licenses();
}

struct QueuedSiteResourceClaim final {
  ScheduledWorkUnit unit;
  std::uint64_t sequence = 0;
};

struct PendingCountedResourceBinding final {
  WorkUnitKey key;
  SiteResourceClaim target;
  SiteResourceClaim requested;
  std::uint64_t sequence = 0;
};

} // namespace

class SiteSchedulerState final {
public:
  explicit SiteSchedulerState(SiteCapacity capacity)
      : capacity(std::move(capacity)) {}

  SiteCapacity capacity;
  MutableUsage allocated;
  std::vector<ScheduledWorkUnit> running;
  std::vector<QueuedSiteResourceClaim> queued;
  std::vector<PendingCountedResourceBinding> pendingBindings;
  std::uint64_t nextWaitSequence = 0;
  std::mutex mutex;
  std::condition_variable changed;
};

namespace {

bool containsQueuedKey(llvm::ArrayRef<QueuedSiteResourceClaim> records,
                       const WorkUnitKey &key) {
  return llvm::any_of(records, [&](const QueuedSiteResourceClaim &record) {
    return record.unit.key == key;
  });
}

bool resourcesConflict(llvm::ArrayRef<CountedSiteResource> lhs,
                       llvm::ArrayRef<CountedSiteResource> rhs) {
  std::size_t lhsIndex = 0;
  std::size_t rhsIndex = 0;
  while (lhsIndex != lhs.size() && rhsIndex != rhs.size()) {
    if (lhs[lhsIndex].key < rhs[rhsIndex].key) {
      ++lhsIndex;
      continue;
    }
    if (rhs[rhsIndex].key < lhs[lhsIndex].key) {
      ++rhsIndex;
      continue;
    }
    return true;
  }
  return false;
}

bool countedClaimsConflict(const SiteResourceClaim &lhs,
                           const SiteResourceClaim &rhs) {
  return resourcesConflict(lhs.externalTools(), rhs.externalTools()) ||
         resourcesConflict(lhs.licenses(), rhs.licenses());
}

enum class ReadyWaiterKind : std::uint8_t {
  Acquisition,
  CountedResourceBinding,
};

struct ReadyWaiter final {
  ReadyWaiterKind kind;
  WorkUnitKey key;
  std::uint64_t sequence = 0;
};

bool isReady(const SiteSchedulerState &state, ReadyWaiterKind kind,
             const WorkUnitKey &key, const SiteResourceClaim &claim) {
  if (kind == ReadyWaiterKind::Acquisition) {
    auto candidate = llvm::find_if(
        state.queued, [&](const QueuedSiteResourceClaim &queued) {
          return queued.unit.key == key;
        });
    if (candidate == state.queued.end() ||
        !fits(candidate->unit.claim, state.capacity, state.allocated))
      return false;
    for (const QueuedSiteResourceClaim &queued : state.queued)
      if (queued.sequence < candidate->sequence &&
          fits(queued.unit.claim, state.capacity, state.allocated))
        return false;
    for (const PendingCountedResourceBinding &binding : state.pendingBindings)
      if (binding.sequence < candidate->sequence &&
          countedClaimsConflict(claim, binding.requested) &&
          fits(binding.requested, state.capacity, state.allocated))
        return false;
    return true;
  }

  auto candidate = llvm::find_if(
      state.pendingBindings, [&](const PendingCountedResourceBinding &binding) {
        return binding.key == key;
      });
  if (candidate == state.pendingBindings.end() ||
      !fits(candidate->requested, state.capacity, state.allocated))
    return false;
  for (const PendingCountedResourceBinding &binding : state.pendingBindings)
    if (binding.sequence < candidate->sequence &&
        countedClaimsConflict(candidate->requested, binding.requested) &&
        fits(binding.requested, state.capacity, state.allocated))
      return false;
  for (const QueuedSiteResourceClaim &queued : state.queued)
    if (queued.sequence < candidate->sequence &&
        countedClaimsConflict(candidate->requested, queued.unit.claim) &&
        fits(queued.unit.claim, state.capacity, state.allocated))
      return false;
  return true;
}

std::chrono::nanoseconds
boundedWaitDelay(std::optional<std::chrono::steady_clock::duration> remaining) {
  auto delay = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::milliseconds(10));
  if (remaining)
    delay = std::min(
        delay,
        std::chrono::duration_cast<std::chrono::nanoseconds>(*remaining));
  return delay;
}

struct ExecutionControlSample final {
  bool stopped = false;
  std::optional<std::chrono::steady_clock::duration> remaining;
};

ExecutionControlSample sampleExecutionControl(ExecutionControlView control) {
  const auto remaining = control.remainingTime();
  return {control.stopRequested() ||
              (remaining &&
               *remaining <= std::chrono::steady_clock::duration::zero()),
          remaining};
}

} // namespace

SiteResourceKey
SiteResourceKey::externalToolBinding(const BlobDigest &binding) {
  return SiteResourceKey(
      SiteResourceKind::ExternalTool,
      deriveResourceDigest("loom.dse.external_tool_resource.v1", binding));
}

SiteResourceKey SiteResourceKey::licenseBinding(const BlobDigest &binding) {
  return SiteResourceKey(
      SiteResourceKind::License,
      deriveResourceDigest("loom.dse.license_resource.v1", binding));
}

bool operator<(const SiteResourceKey &lhs, const SiteResourceKey &rhs) {
  if (lhs.kind_ != rhs.kind_)
    return static_cast<std::uint32_t>(lhs.kind_) <
           static_cast<std::uint32_t>(rhs.kind_);
  return std::lexicographical_compare(
      lhs.digest_.bytes().begin(), lhs.digest_.bytes().end(),
      rhs.digest_.bytes().begin(), rhs.digest_.bytes().end());
}

llvm::Expected<SiteResourceClaim>
SiteResourceClaim::get(std::uint64_t cpuCores, std::uint64_t memoryBytes,
                       std::uint64_t scratchBytes,
                       llvm::ArrayRef<CountedSiteResource> externalTools,
                       llvm::ArrayRef<CountedSiteResource> licenses) {
  auto checkedTools = validateResources(
      externalTools, SiteResourceKind::ExternalTool, "external-tool claim");
  if (!checkedTools)
    return checkedTools.takeError();
  auto checkedLicenses =
      validateResources(licenses, SiteResourceKind::License, "license claim");
  if (!checkedLicenses)
    return checkedLicenses.takeError();
  return SiteResourceClaim(cpuCores, memoryBytes, scratchBytes,
                           std::move(*checkedTools),
                           std::move(*checkedLicenses));
}

llvm::Expected<SiteCapacity>
SiteCapacity::get(std::uint64_t cpuCores, std::uint64_t memoryBytes,
                  std::uint64_t scratchBytes,
                  llvm::ArrayRef<CountedSiteResource> externalTools,
                  llvm::ArrayRef<CountedSiteResource> licenses,
                  std::uint64_t undeclaredExternalToolUnits) {
  if (cpuCores == 0)
    return invalid("site capacity requires at least one CPU core");
  auto checkedTools = validateResources(
      externalTools, SiteResourceKind::ExternalTool, "external-tool capacity");
  if (!checkedTools)
    return checkedTools.takeError();
  auto checkedLicenses = validateResources(licenses, SiteResourceKind::License,
                                           "license capacity");
  if (!checkedLicenses)
    return checkedLicenses.takeError();
  return SiteCapacity(cpuCores, memoryBytes, scratchBytes,
                      std::move(*checkedTools), std::move(*checkedLicenses),
                      undeclaredExternalToolUnits);
}

SiteResourceLease::SiteResourceLease(SiteResourceLease &&other) noexcept
    : state_(std::move(other.state_)), key_(std::move(other.key_)),
      claim_(std::move(other.claim_)) {}

SiteResourceLease &
SiteResourceLease::operator=(SiteResourceLease &&other) noexcept {
  if (this == &other)
    return *this;
  release();
  state_ = std::move(other.state_);
  key_ = std::move(other.key_);
  claim_ = std::move(other.claim_);
  return *this;
}

SiteResourceLease::~SiteResourceLease() { release(); }

void SiteResourceLease::release() {
  if (!state_)
    return;
  std::shared_ptr<SiteSchedulerState> state = std::move(state_);
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    auto found =
        llvm::find_if(state->running, [&](const ScheduledWorkUnit &unit) {
          return unit.key == key_;
        });
    if (found == state->running.end())
      llvm_unreachable("site scheduler lost an active lease");
    if (containsQueuedKey(state->queued, key_) ||
        llvm::any_of(state->pendingBindings,
                     [&](const PendingCountedResourceBinding &binding) {
                       return binding.key == key_;
                     }))
      llvm_unreachable("site scheduler released a transitioning lease");
    subtract(state->allocated, found->claim);
    state->running.erase(found);
  }
  state->changed.notify_all();
}

llvm::Expected<SiteScheduler> SiteScheduler::create(SiteCapacity capacity) {
  return SiteScheduler(
      std::make_shared<SiteSchedulerState>(std::move(capacity)));
}

llvm::Expected<std::optional<SiteResourceLease>>
SiteScheduler::tryAcquire(const WorkUnitKey &key,
                          const SiteResourceClaim &claim) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (!admitted(claim, state_->capacity))
    return invalid("claim exceeds declared site capacity");
  if (containsKey(state_->running, key) ||
      containsQueuedKey(state_->queued, key))
    return invalid("work unit already owns or awaits a resource claim");
  // A nonblocking claimant never jumps ahead of an already queued normal
  // acquisition.  Pending counted transitions are handled below by their
  // actual exclusive resource dependencies.
  if (!state_->queued.empty())
    return std::optional<SiteResourceLease>{};
  for (const PendingCountedResourceBinding &binding : state_->pendingBindings) {
    if (!fits(binding.requested, state_->capacity, state_->allocated) ||
        !countedClaimsConflict(claim, binding.requested))
      continue;
    MutableUsage projected = state_->allocated;
    add(projected, claim);
    if (!fits(binding.requested, state_->capacity, projected))
      return std::optional<SiteResourceLease>{};
  }
  if (!fits(claim, state_->capacity, state_->allocated))
    return std::optional<SiteResourceLease>{};
  add(state_->allocated, claim);
  state_->running.push_back({key, claim});
  return std::optional<SiteResourceLease>(
      SiteResourceLease(state_, key, claim));
}

llvm::Expected<SiteResourceLease>
SiteScheduler::acquire(const WorkUnitKey &key, const SiteResourceClaim &claim) {
  std::unique_lock<std::mutex> lock(state_->mutex);
  if (!admitted(claim, state_->capacity))
    return invalid("claim exceeds declared site capacity");
  if (containsKey(state_->running, key) ||
      containsQueuedKey(state_->queued, key))
    return invalid("work unit already owns or awaits a resource claim");
  state_->queued.push_back({{key, claim}, state_->nextWaitSequence++});
  state_->changed.wait(lock, [&] {
    return isReady(*state_, ReadyWaiterKind::Acquisition, key, claim);
  });
  auto found =
      llvm::find_if(state_->queued, [&](const QueuedSiteResourceClaim &queued) {
        return queued.unit.key == key;
      });
  if (found == state_->queued.end())
    llvm_unreachable("site scheduler lost a queued claim");
  add(state_->allocated, found->unit.claim);
  state_->running.push_back(std::move(found->unit));
  state_->queued.erase(found);
  lock.unlock();
  state_->changed.notify_all();
  return SiteResourceLease(state_, key, claim);
}

llvm::Expected<std::optional<SiteResourceLease>>
SiteScheduler::acquire(const WorkUnitKey &key, const SiteResourceClaim &claim,
                       ExecutionControlView executionControl) {
  std::unique_lock<std::mutex> lock(state_->mutex);
  if (!admitted(claim, state_->capacity))
    return invalid("claim exceeds declared site capacity");
  if (containsKey(state_->running, key) ||
      containsQueuedKey(state_->queued, key))
    return invalid("work unit already owns or awaits a resource claim");
  state_->queued.push_back({{key, claim}, state_->nextWaitSequence++});
  const auto removeQueued = [&] {
    auto found = llvm::find_if(state_->queued,
                               [&](const QueuedSiteResourceClaim &queued) {
                                 return queued.unit.key == key;
                               });
    if (found == state_->queued.end())
      llvm_unreachable("site scheduler lost a controlled queued claim");
    state_->queued.erase(found);
  };
  while (true) {
    lock.unlock();
    const ExecutionControlSample control =
        sampleExecutionControl(executionControl);
    lock.lock();
    if (control.stopped) {
      removeQueued();
      lock.unlock();
      state_->changed.notify_all();
      return std::optional<SiteResourceLease>{};
    }
    if (isReady(*state_, ReadyWaiterKind::Acquisition, key, claim)) {
      auto found = llvm::find_if(state_->queued,
                                 [&](const QueuedSiteResourceClaim &queued) {
                                   return queued.unit.key == key;
                                 });
      if (found == state_->queued.end())
        llvm_unreachable("site scheduler lost a controlled queued claim");
      add(state_->allocated, found->unit.claim);
      state_->running.push_back(std::move(found->unit));
      state_->queued.erase(found);
      lock.unlock();
      state_->changed.notify_all();
      return std::optional<SiteResourceLease>(
          SiteResourceLease(state_, key, claim));
    }
    state_->changed.wait_for(lock, boundedWaitDelay(control.remaining));
  }
}

llvm::Expected<bool>
SiteScheduler::bindCountedResources(SiteResourceLease &lease,
                                    const SiteResourceClaim &target,
                                    ExecutionControlView executionControl) {
  std::unique_lock<std::mutex> lock(state_->mutex);
  if (lease.state_.get() != state_.get())
    return invalid("resource binding lease belongs to another scheduler");
  auto running =
      llvm::find_if(state_->running, [&](const ScheduledWorkUnit &unit) {
        return unit.key == lease.key_;
      });
  if (running == state_->running.end() ||
      !sameClaim(running->claim, lease.claim_))
    return invalid("resource binding lease is not the active owner");
  if (containsQueuedKey(state_->queued, lease.key_) ||
      llvm::any_of(state_->pendingBindings,
                   [&](const PendingCountedResourceBinding &binding) {
                     return binding.key == lease.key_;
                   }))
    return invalid("work unit already awaits a resource binding");
  if (target.cpuCores() != lease.claim_.cpuCores() ||
      target.memoryBytes() != lease.claim_.memoryBytes() ||
      target.scratchBytes() != lease.claim_.scratchBytes())
    return invalid("counted-resource binding changed a scalar reservation");
  if (!admitted(target, state_->capacity))
    return invalid("bound claim exceeds declared site capacity");
  auto scalar = SiteResourceClaim::get(lease.claim_.cpuCores(),
                                       lease.claim_.memoryBytes(),
                                       lease.claim_.scratchBytes());
  if (!scalar)
    return scalar.takeError();
  auto requested = SiteResourceClaim::get(0, 0, 0, target.externalTools(),
                                          target.licenses());
  if (!requested)
    return requested.takeError();

  lock.unlock();
  ExecutionControlSample control = sampleExecutionControl(executionControl);
  lock.lock();
  running = llvm::find_if(state_->running, [&](const ScheduledWorkUnit &unit) {
    return unit.key == lease.key_;
  });
  if (running == state_->running.end() ||
      !sameClaim(running->claim, lease.claim_))
    return invalid("resource binding lease changed during control sampling");
  if (control.stopped)
    return false;
  if (sameClaim(lease.claim_, target))
    return true;

  subtract(state_->allocated, running->claim);
  add(state_->allocated, *scalar);
  running->claim = *scalar;
  lease.claim_ = *scalar;
  if (target.externalTools().empty() && target.licenses().empty()) {
    lock.unlock();
    state_->changed.notify_all();
    return true;
  }

  state_->pendingBindings.push_back(
      {lease.key_, target, std::move(*requested), state_->nextWaitSequence++});
  const auto removePending = [&] {
    auto found =
        llvm::find_if(state_->pendingBindings,
                      [&](const PendingCountedResourceBinding &binding) {
                        return binding.key == lease.key_;
                      });
    if (found == state_->pendingBindings.end())
      llvm_unreachable("site scheduler lost a pending resource binding");
    state_->pendingBindings.erase(found);
  };
  state_->changed.notify_all();
  while (true) {
    lock.unlock();
    control = sampleExecutionControl(executionControl);
    lock.lock();
    if (control.stopped) {
      removePending();
      lock.unlock();
      state_->changed.notify_all();
      return false;
    }
    if (isReady(*state_, ReadyWaiterKind::CountedResourceBinding, lease.key_,
                target)) {
      auto pending =
          llvm::find_if(state_->pendingBindings,
                        [&](const PendingCountedResourceBinding &candidate) {
                          return candidate.key == lease.key_;
                        });
      if (pending == state_->pendingBindings.end())
        llvm_unreachable("site scheduler lost a ready resource binding");
      running =
          llvm::find_if(state_->running, [&](const ScheduledWorkUnit &unit) {
            return unit.key == lease.key_;
          });
      if (running == state_->running.end() ||
          !sameClaim(running->claim, lease.claim_))
        llvm_unreachable("site scheduler lost a binding lease owner");
      add(state_->allocated, pending->requested);
      running->claim = pending->target;
      lease.claim_ = pending->target;
      state_->pendingBindings.erase(pending);
      lock.unlock();
      state_->changed.notify_all();
      return true;
    }
    state_->changed.wait_for(lock, boundedWaitDelay(control.remaining));
  }
}

llvm::Expected<SiteSchedulerSnapshot> SiteScheduler::snapshot() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto allocated = SiteResourceClaim::get(
      state_->allocated.cpuCores, state_->allocated.memoryBytes,
      state_->allocated.scratchBytes,
      copyResources(state_->allocated.externalTools),
      copyResources(state_->allocated.licenses));
  if (!allocated)
    return allocated.takeError();
  std::vector<QueuedSiteResourceClaim> waiters = state_->queued;
  waiters.reserve(waiters.size() + state_->pendingBindings.size());
  for (const PendingCountedResourceBinding &binding : state_->pendingBindings)
    waiters.push_back({{binding.key, binding.requested}, binding.sequence});
  llvm::sort(waiters, [](const QueuedSiteResourceClaim &lhs,
                         const QueuedSiteResourceClaim &rhs) {
    return lhs.sequence < rhs.sequence;
  });
  std::vector<ScheduledWorkUnit> queued;
  queued.reserve(waiters.size());
  for (QueuedSiteResourceClaim &waiter : waiters)
    queued.push_back(std::move(waiter.unit));
  return SiteSchedulerSnapshot{state_->capacity, std::move(*allocated),
                               state_->running, std::move(queued)};
}

} // namespace loom::dse
