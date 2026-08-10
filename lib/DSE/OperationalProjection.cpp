#include "DSE/CampaignRunner.h"

#include "Common/BlobDigest.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "dse_operational_projection_invalid: " +
                                     message);
}

llvm::Expected<std::uint64_t> unixNanosecondsNow() {
  const auto elapsed = std::chrono::system_clock::now().time_since_epoch();
  const auto nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
  if (nanoseconds <= 0)
    return invalid("system clock cannot represent a positive observation time");
  return static_cast<std::uint64_t>(nanoseconds);
}

bool terminal(JournalWorkUnitStatus status) {
  return status == JournalWorkUnitStatus::Completed ||
         status == JournalWorkUnitStatus::Failed ||
         status == JournalWorkUnitStatus::TimedOut ||
         status == JournalWorkUnitStatus::Unsupported;
}

llvm::Error increment(std::uint64_t &value, llvm::StringRef field) {
  if (value == std::numeric_limits<std::uint64_t>::max())
    return invalid(field + " count overflows uint64");
  ++value;
  return llvm::Error::success();
}

llvm::Error add(std::uint64_t &value, std::uint64_t amount,
                llvm::StringRef field) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - value)
    return invalid(field + " value overflows uint64");
  value += amount;
  return llvm::Error::success();
}

std::uint64_t percentile(llvm::ArrayRef<std::uint64_t> sorted,
                         std::uint64_t numerator,
                         std::uint64_t denominator) {
  const std::uint64_t count = sorted.size();
  const std::uint64_t rank = static_cast<std::uint64_t>(
      (static_cast<unsigned __int128>(count) * numerator + denominator - 1) /
      denominator);
  return sorted[static_cast<std::size_t>(rank - 1)];
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

struct ResourcePressure final {
  LimitingSiteResource value;
};

bool pressureLess(const ResourcePressure &lhs, const ResourcePressure &rhs) {
  const bool lhsUnbounded =
      lhs.value.capacity == 0 &&
      (lhs.value.allocated != 0 || lhs.value.queuedDemand != 0);
  const bool rhsUnbounded =
      rhs.value.capacity == 0 &&
      (rhs.value.allocated != 0 || rhs.value.queuedDemand != 0);
  if (lhsUnbounded != rhsUnbounded)
    return !lhsUnbounded;
  if (!lhsUnbounded) {
    const unsigned __int128 lhsDemand =
        static_cast<unsigned __int128>(lhs.value.allocated) +
        lhs.value.queuedDemand;
    const unsigned __int128 rhsDemand =
        static_cast<unsigned __int128>(rhs.value.allocated) +
        rhs.value.queuedDemand;
    const unsigned __int128 lhsScaled = lhsDemand * rhs.value.capacity;
    const unsigned __int128 rhsScaled = rhsDemand * lhs.value.capacity;
    if (lhsScaled != rhsScaled)
      return lhsScaled < rhsScaled;
  }
  if (lhs.value.kind != rhs.value.kind)
    return static_cast<std::uint32_t>(lhs.value.kind) <
           static_cast<std::uint32_t>(rhs.value.kind);
  if (lhs.value.key && rhs.value.key)
    return *lhs.value.key < *rhs.value.key;
  return !lhs.value.key && rhs.value.key;
}

llvm::Expected<std::optional<LimitingSiteResource>>
limitingResource(const SiteSchedulerSnapshot &snapshot) {
  std::uint64_t queuedCpu = 0;
  std::uint64_t queuedMemory = 0;
  std::uint64_t queuedScratch = 0;
  std::map<SiteResourceKey, std::uint64_t> queuedTools;
  std::map<SiteResourceKey, std::uint64_t> queuedLicenses;
  for (const ScheduledWorkUnit &unit : snapshot.queued) {
    if (llvm::Error error = add(queuedCpu, unit.claim.cpuCores(),
                                "queued CPU demand"))
      return std::move(error);
    if (llvm::Error error = add(queuedMemory, unit.claim.memoryBytes(),
                                "queued memory demand"))
      return std::move(error);
    if (llvm::Error error = add(queuedScratch, unit.claim.scratchBytes(),
                                "queued scratch demand"))
      return std::move(error);
    for (const CountedSiteResource &resource :
         unit.claim.externalTools())
      if (llvm::Error error =
              add(queuedTools[resource.key], resource.units,
                  "queued external-tool demand"))
        return std::move(error);
    for (const CountedSiteResource &resource : unit.claim.licenses())
      if (llvm::Error error =
              add(queuedLicenses[resource.key], resource.units,
                  "queued license demand"))
        return std::move(error);
  }

  std::vector<ResourcePressure> pressures;
  pressures.push_back({{SiteResourceKind::Cpu, std::nullopt,
                        snapshot.allocated.cpuCores(), queuedCpu,
                        snapshot.capacity.cpuCores()}});
  pressures.push_back({{SiteResourceKind::Memory, std::nullopt,
                        snapshot.allocated.memoryBytes(), queuedMemory,
                        snapshot.capacity.memoryBytes()}});
  pressures.push_back({{SiteResourceKind::Scratch, std::nullopt,
                        snapshot.allocated.scratchBytes(), queuedScratch,
                        snapshot.capacity.scratchBytes()}});

  for (const CountedSiteResource &capacity :
       snapshot.capacity.externalTools()) {
    const CountedSiteResource *allocated =
        findResource(snapshot.allocated.externalTools(), capacity.key);
    auto queued = queuedTools.find(capacity.key);
    pressures.push_back(
        {{SiteResourceKind::ExternalTool, capacity.key,
          allocated ? allocated->units : 0,
          queued == queuedTools.end() ? 0 : queued->second, capacity.units}});
  }
  for (const CountedSiteResource &capacity : snapshot.capacity.licenses()) {
    const CountedSiteResource *allocated =
        findResource(snapshot.allocated.licenses(), capacity.key);
    auto queued = queuedLicenses.find(capacity.key);
    pressures.push_back(
        {{SiteResourceKind::License, capacity.key,
          allocated ? allocated->units : 0,
          queued == queuedLicenses.end() ? 0 : queued->second,
          capacity.units}});
  }

  pressures.erase(
      std::remove_if(pressures.begin(), pressures.end(),
                     [](const ResourcePressure &pressure) {
                       return pressure.value.allocated == 0 &&
                              pressure.value.queuedDemand == 0;
                     }),
      pressures.end());
  if (pressures.empty())
    return std::optional<LimitingSiteResource>{};
  return std::optional<LimitingSiteResource>(
      llvm::max_element(pressures, pressureLess)->value);
}

llvm::StringRef resourceKindSpelling(SiteResourceKind kind) {
  switch (kind) {
  case SiteResourceKind::Cpu:
    return "cpu";
  case SiteResourceKind::Memory:
    return "memory";
  case SiteResourceKind::Scratch:
    return "scratch";
  case SiteResourceKind::ExternalTool:
    return "external_tool";
  case SiteResourceKind::License:
    return "license";
  }
  llvm_unreachable("unknown site resource kind");
}

llvm::Expected<std::int64_t> jsonInteger(std::uint64_t value,
                                         llvm::StringRef field) {
  if (value > static_cast<std::uint64_t>(
                  std::numeric_limits<std::int64_t>::max()))
    return invalid(field + " exceeds the JSON integer domain");
  return static_cast<std::int64_t>(value);
}

} // namespace

llvm::Expected<DseOperationalProjection> projectDseOperationalState(
    const ExecutionJournal &journal, const SiteScheduler &scheduler,
    std::uint64_t requestedWorkerCount,
    std::uint64_t recentWindowNanoseconds) {
  if (requestedWorkerCount == 0)
    return invalid("requested worker count must be positive");
  if (recentWindowNanoseconds == 0)
    return invalid("recent throughput window must be positive");

  auto observed = unixNanosecondsNow();
  if (!observed)
    return observed.takeError();
  auto records = journal.workUnits();
  if (!records)
    return records.takeError();
  auto schedulerSnapshot = scheduler.snapshot();
  if (!schedulerSnapshot)
    return schedulerSnapshot.takeError();

  DseOperationalProjection projection;
  projection.observedUnixTimeNanoseconds = *observed;
  std::map<WorkUnitDescriptorRef, std::vector<std::uint64_t>> durations;
  std::vector<std::uint64_t> allDurations;
  std::uint64_t recentTerminal = 0;
  const std::uint64_t recentBegin =
      *observed > recentWindowNanoseconds
          ? *observed - recentWindowNanoseconds
          : 0;

  for (const JournalWorkUnitRecord &record : *records) {
    switch (record.status) {
    case JournalWorkUnitStatus::Queued:
      if (llvm::Error error = increment(projection.status.queued, "queued"))
        return std::move(error);
      break;
    case JournalWorkUnitStatus::Running:
      if (llvm::Error error = increment(projection.status.running, "running"))
        return std::move(error);
      break;
    case JournalWorkUnitStatus::Prepared:
      if (llvm::Error error =
              increment(projection.status.prepared, "prepared"))
        return std::move(error);
      break;
    case JournalWorkUnitStatus::Completed:
      if (llvm::Error error =
              increment(projection.status.completed, "completed"))
        return std::move(error);
      break;
    case JournalWorkUnitStatus::Failed:
      if (llvm::Error error = increment(projection.status.failed, "failed"))
        return std::move(error);
      break;
    case JournalWorkUnitStatus::TimedOut:
      if (llvm::Error error =
              increment(projection.status.timedOut, "timed-out"))
        return std::move(error);
      break;
    case JournalWorkUnitStatus::Unsupported:
      if (llvm::Error error =
              increment(projection.status.unsupported, "unsupported"))
        return std::move(error);
      break;
    }
    std::uint64_t active = record.activeWallTimeNanoseconds();
    if (record.activeAttemptStartUnixTimeNanoseconds != 0) {
      if (record.activeAttemptStartUnixTimeNanoseconds > *observed)
        return invalid("Journal contains a future active interval start");
      const std::uint64_t ongoing =
          *observed - record.activeAttemptStartUnixTimeNanoseconds;
      if (ongoing > std::numeric_limits<std::uint64_t>::max() - active)
        return invalid("observed active duration overflows uint64");
      active += ongoing;
    }
    if (active != 0)
      allDurations.push_back(active);
    if (!terminal(record.status))
      continue;
    if (record.terminalUnixTimeNanoseconds > *observed)
      return invalid("Journal contains a future terminal timestamp");
    durations[record.key.descriptor()].push_back(active);
    if (record.terminalUnixTimeNanoseconds >= recentBegin)
      if (llvm::Error error =
              increment(recentTerminal, "recent terminal"))
        return std::move(error);
  }

  projection.recentThroughputPerSecond =
      static_cast<double>(recentTerminal) * 1000000000.0 /
      static_cast<double>(recentWindowNanoseconds);
  projection.durations.reserve(durations.size());
  for (auto &[descriptor, values] : durations) {
    llvm::sort(values);
    projection.durations.push_back(
        {descriptor, static_cast<std::uint64_t>(values.size()),
         percentile(values, 50, 100), percentile(values, 90, 100)});
  }

  std::uint64_t outstanding = projection.status.queued;
  if (llvm::Error error =
          add(outstanding, projection.status.running, "outstanding work"))
    return std::move(error);
  if (outstanding != 0 && projection.status.prepared == 0 &&
      !allDurations.empty()) {
    llvm::sort(allDurations);
    const std::uint64_t p90 = percentile(allDurations, 90, 100);
    const std::uint64_t lanes =
        std::max<std::uint64_t>(
            1, std::min(requestedWorkerCount,
                        schedulerSnapshot->capacity.cpuCores()));
    const std::uint64_t batches = outstanding / lanes +
                                  (outstanding % lanes == 0 ? 0 : 1);
    if (p90 != 0 &&
        batches > std::numeric_limits<std::uint64_t>::max() / p90)
      return invalid("estimated remaining duration overflows uint64");
    projection.estimatedRemainingNanoseconds = batches * p90;
  }

  auto limiting = limitingResource(*schedulerSnapshot);
  if (!limiting)
    return limiting.takeError();
  projection.limitingResource = std::move(*limiting);
  return projection;
}

llvm::Error writeDseOperationalProjectionJsonLine(
    const DseOperationalProjection &projection, llvm::raw_ostream &output) {
  auto observed =
      jsonInteger(projection.observedUnixTimeNanoseconds, "observation time");
  if (!observed)
    return observed.takeError();
  llvm::json::Object status;
  const auto addStatus = [&](llvm::StringRef name, std::uint64_t value,
                             llvm::StringRef field) -> llvm::Error {
    auto encoded = jsonInteger(value, field);
    if (!encoded)
      return encoded.takeError();
    status[name] = *encoded;
    return llvm::Error::success();
  };
  if (llvm::Error error =
          addStatus("completed", projection.status.completed, "completed count"))
    return error;
  if (llvm::Error error =
          addStatus("running", projection.status.running, "running count"))
    return error;
  if (llvm::Error error =
          addStatus("prepared", projection.status.prepared, "prepared count"))
    return error;
  if (llvm::Error error =
          addStatus("queued", projection.status.queued, "queued count"))
    return error;
  if (llvm::Error error =
          addStatus("failed", projection.status.failed, "failed count"))
    return error;
  if (llvm::Error error = addStatus("timed_out", projection.status.timedOut,
                                    "timed-out count"))
    return error;
  if (llvm::Error error = addStatus("unsupported", projection.status.unsupported,
                                    "unsupported count"))
    return error;

  llvm::json::Array durations;
  for (const WorkUnitDurationProjection &duration : projection.durations) {
    auto terminalCount =
        jsonInteger(duration.terminalCount, "terminal count");
    if (!terminalCount)
      return terminalCount.takeError();
    auto p50 = jsonInteger(duration.p50Nanoseconds, "p50 duration");
    if (!p50)
      return p50.takeError();
    auto p90 = jsonInteger(duration.p90Nanoseconds, "p90 duration");
    if (!p90)
      return p90.takeError();
    llvm::json::Object item;
    item["owner_registry"] =
        duration.descriptor.ownerRegistryIdentity().str();
    item["owner_major"] = static_cast<std::int64_t>(
        duration.descriptor.ownerRegistryVersion().major);
    item["owner_minor"] = static_cast<std::int64_t>(
        duration.descriptor.ownerRegistryVersion().minor);
    item["owner_local_kind"] =
        static_cast<std::int64_t>(duration.descriptor.ownerLocalKind());
    item["terminal_count"] = *terminalCount;
    item["p50_ns"] = *p50;
    item["p90_ns"] = *p90;
    durations.push_back(std::move(item));
  }

  llvm::json::Object root;
  root["observed_unix_time_ns"] = *observed;
  root["status"] = std::move(status);
  root["recent_throughput_per_second"] =
      projection.recentThroughputPerSecond;
  root["durations"] = std::move(durations);
  if (projection.estimatedRemainingNanoseconds) {
    auto eta = jsonInteger(*projection.estimatedRemainingNanoseconds,
                           "estimated remaining duration");
    if (!eta)
      return eta.takeError();
    root["estimated_remaining_ns"] = *eta;
  } else {
    root["estimated_remaining_ns"] = nullptr;
  }

  if (projection.limitingResource) {
    const LimitingSiteResource &limiting = *projection.limitingResource;
    llvm::json::Object resource;
    resource["kind"] = resourceKindSpelling(limiting.kind).str();
    if (limiting.key)
      resource["key"] = formatBlobDigestHex(limiting.key->digest());
    else
      resource["key"] = nullptr;
    auto allocated = jsonInteger(limiting.allocated, "allocated resource");
    if (!allocated)
      return allocated.takeError();
    auto queued = jsonInteger(limiting.queuedDemand, "queued resource");
    if (!queued)
      return queued.takeError();
    auto capacity = jsonInteger(limiting.capacity, "resource capacity");
    if (!capacity)
      return capacity.takeError();
    resource["allocated"] = *allocated;
    resource["queued_demand"] = *queued;
    resource["capacity"] = *capacity;
    root["limiting_resource"] = std::move(resource);
  } else {
    root["limiting_resource"] = nullptr;
  }

  output << llvm::formatv("{0}", llvm::json::Value(std::move(root))) << '\n';
  return llvm::Error::success();
}

} // namespace loom::dse
