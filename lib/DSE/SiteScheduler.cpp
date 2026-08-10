#include "DSE/SiteScheduler.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
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
                  const std::map<SiteResourceKey, std::uint64_t> &allocated) {
  for (const CountedSiteResource &entry : claim) {
    const CountedSiteResource *limit = findResource(capacity, entry.key);
    if (!limit || entry.units > limit->units)
      return false;
    auto found = allocated.find(entry.key);
    const std::uint64_t used = found == allocated.end() ? 0 : found->second;
    if (used > limit->units || entry.units > limit->units - used)
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
                      allocated.externalTools) &&
         resourcesFit(claim.licenses(), capacity.licenses(),
                      allocated.licenses);
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

} // namespace

class SiteSchedulerState final {
public:
  explicit SiteSchedulerState(SiteCapacity capacity)
      : capacity(std::move(capacity)) {}

  SiteCapacity capacity;
  MutableUsage allocated;
  std::vector<ScheduledWorkUnit> running;
  std::vector<ScheduledWorkUnit> queued;
  std::mutex mutex;
  std::condition_variable changed;
};

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
                  llvm::ArrayRef<CountedSiteResource> licenses) {
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
                      std::move(*checkedTools), std::move(*checkedLicenses));
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
  if (containsKey(state_->running, key) || containsKey(state_->queued, key))
    return invalid("work unit already owns or awaits a resource claim");
  if (!state_->queued.empty())
    return std::optional<SiteResourceLease>{};
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
  if (containsKey(state_->running, key) || containsKey(state_->queued, key))
    return invalid("work unit already owns or awaits a resource claim");
  state_->queued.push_back({key, claim});
  state_->changed.wait(lock, [&] {
    for (const ScheduledWorkUnit &queued : state_->queued) {
      if (!fits(queued.claim, state_->capacity, state_->allocated))
        continue;
      return queued.key == key;
    }
    return false;
  });
  auto found =
      llvm::find_if(state_->queued, [&](const ScheduledWorkUnit &unit) {
        return unit.key == key;
      });
  if (found == state_->queued.end())
    llvm_unreachable("site scheduler lost a queued claim");
  add(state_->allocated, found->claim);
  state_->running.push_back(std::move(*found));
  state_->queued.erase(found);
  return SiteResourceLease(state_, key, claim);
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
  return SiteSchedulerSnapshot{state_->capacity, std::move(*allocated),
                               state_->running, state_->queued};
}

} // namespace loom::dse
