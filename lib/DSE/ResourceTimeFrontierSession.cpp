#include "DSE/ResourceTimeFrontier.h"

#include "llvm/Support/ErrorHandling.h"

#include <chrono>
#include <condition_variable>
#include <limits>
#include <map>
#include <mutex>
#include <type_traits>

namespace loom::dse {
namespace {

std::uint64_t outcomeRetainedBytes(const ResourceTimeFrontierOutcome &outcome) {
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  const auto add = [](std::uint64_t lhs, std::uint64_t rhs) {
    return rhs > maximum - lhs ? maximum : lhs + rhs;
  };
  const auto product = [](std::size_t lhs, std::size_t rhs) {
    if (rhs != 0 && lhs > maximum / rhs)
      return maximum;
    return static_cast<std::uint64_t>(lhs * rhs);
  };
  const auto hintBytes = [&](const ResourceTimeScheduleHint &hint) {
    std::uint64_t bytes = sizeof(ResourceTimeScheduleHint);
    bytes = add(bytes, product(hint.actions.size(),
                               sizeof(ResourceTimeActionDelta)));
    bytes = add(bytes,
                product(hint.states.size(), sizeof(ResourceTimeHintState)));
    for (const ResourceTimeActionDelta &action : hint.actions) {
      bytes = add(bytes, product(action.completedRegions.size(),
                                 sizeof(::dataflow::RootThreadLaunchRef)));
      bytes = add(bytes, product(action.tokenReadyProducers.size(),
                                 sizeof(::dataflow::RootThreadLaunchRef)));
      bytes = add(bytes, product(action.newlyReadyRegions.size(),
                                 sizeof(::dataflow::RootThreadLaunchRef)));
    }
    for (const ResourceTimeHintState &state : hint.states) {
      bytes = add(bytes, product(state.active.size(),
                                 sizeof(ResourceTimeHintAllocation)));
      bytes = add(bytes, product(state.ready.size(),
                                 sizeof(::dataflow::RootThreadLaunchRef)));
      bytes = add(bytes, product(state.completed.size(),
                                 sizeof(::dataflow::RootThreadLaunchRef)));
      for (const ResourceTimeHintAllocation &allocation : state.active)
        bytes = add(bytes, product(allocation.resourceUnits.size(),
                                   sizeof(std::uint64_t)));
    }
    return bytes;
  };

  std::uint64_t bytes = sizeof(ResourceTimeFrontierOutcome);
  std::visit(
      [&](const auto &value) {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value,
                                     CompletedResourceTimeFrontier>) {
          for (const ResourceTimeScheduleHint &hint : value.finalists)
            bytes = add(bytes, hintBytes(hint));
        } else if constexpr (std::is_same_v<
                                 Value, IncompleteResourceTimeFrontier>) {
          for (const ResourceTimeScheduleHint &hint : value.retainedFinalists)
            bytes = add(bytes, hintBytes(hint));
        }
      },
      outcome);
  return bytes;
}

bool reusable(const ResourceTimeFrontierOutcome &outcome) {
  return std::holds_alternative<CompletedResourceTimeFrontier>(outcome) ||
         std::holds_alternative<ProvenInfeasibleResourceTimeFrontier>(outcome);
}

} // namespace

class ResourceTimeFrontierSession::Impl final {
public:
  struct Flight final {
    bool complete = false;
    bool retained = false;
    std::shared_ptr<const ResourceTimeFrontierOutcome> outcome;
  };

  Impl(std::uint64_t maximumEntries, std::uint64_t maximumRetainedBytes)
      : maximumEntries(maximumEntries),
        maximumRetainedBytes(maximumRetainedBytes) {}

  std::uint64_t maximumEntries = 0;
  std::uint64_t maximumRetainedBytes = 0;
  mutable std::mutex mutex;
  std::condition_variable changed;
  std::map<std::string, std::shared_ptr<const ResourceTimeFrontierOutcome>>
      results;
  std::map<std::string, std::shared_ptr<Flight>> flights;
  ResourceTimeFrontierSessionStatistics statistics;
};

ResourceTimeFrontierSession::ResourceTimeFrontierSession(
    std::uint64_t maximumEntries, std::uint64_t maximumRetainedBytes)
    : impl_(std::make_unique<Impl>(maximumEntries, maximumRetainedBytes)) {
  if (maximumEntries == 0 || maximumRetainedBytes == 0)
    llvm::report_fatal_error("resource-time frontier session limits must be "
                             "positive");
}

ResourceTimeFrontierSession::~ResourceTimeFrontierSession() = default;

ResourceTimeFrontierSessionStatistics
ResourceTimeFrontierSession::statistics() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  ResourceTimeFrontierSessionStatistics result = impl_->statistics;
  result.entryCount = impl_->results.size();
  return result;
}

llvm::Expected<ResourceTimeFrontierSession::LookupResult>
ResourceTimeFrontierSession::lookupOrCompute(
    std::string key, Compute compute, ExecutionControlView executionControl) {
  std::shared_ptr<Impl::Flight> ownedFlight;
  bool waited = false;
  {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    ++impl_->statistics.requests;
    while (true) {
      const auto cached = impl_->results.find(key);
      if (cached != impl_->results.end()) {
        ++impl_->statistics.cacheHits;
        return LookupResult{cached->second, true, false, waited, false, false,
                            false};
      }
      const auto active = impl_->flights.find(key);
      if (active == impl_->flights.end()) {
        ownedFlight = std::make_shared<Impl::Flight>();
        impl_->flights.emplace(key, ownedFlight);
        ++impl_->statistics.cacheMisses;
        break;
      }
      if (!waited) {
        waited = true;
        ++impl_->statistics.singleFlightWaits;
      }
      const std::shared_ptr<Impl::Flight> flight = active->second;
      if (executionControl.stopRequested()) {
        ++impl_->statistics.cancelledWaits;
        return LookupResult{nullptr, false, false, true, false, true, false};
      }
      impl_->changed.wait_for(lock, std::chrono::milliseconds(1),
                              [&] { return flight->complete; });
      if (!flight->complete)
        continue;
      if (flight->outcome) {
        if (flight->retained) {
          ++impl_->statistics.cacheHits;
          return LookupResult{flight->outcome, true, false, true, false,
                              false, false};
        }
        ++impl_->statistics.coalescedUncachedResults;
        return LookupResult{flight->outcome, false, false, true, true, false,
                            false};
      }
    }
  }

  auto computed = compute();
  if (!computed) {
    {
      std::lock_guard<std::mutex> lock(impl_->mutex);
      ownedFlight->complete = true;
      const auto active = impl_->flights.find(key);
      if (active != impl_->flights.end() && active->second == ownedFlight)
        impl_->flights.erase(active);
    }
    impl_->changed.notify_all();
    return computed.takeError();
  }

  auto outcome = std::make_shared<const ResourceTimeFrontierOutcome>(
      std::move(*computed));
  bool retained = false;
  bool capacityBypass = false;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (reusable(*outcome)) {
      const std::uint64_t payloadBytes = outcomeRetainedBytes(*outcome);
      const std::uint64_t fixedBytes =
          sizeof(std::shared_ptr<const ResourceTimeFrontierOutcome>) + 128;
      const std::uint64_t keyBytes =
          key.size() > std::numeric_limits<std::uint64_t>::max()
              ? std::numeric_limits<std::uint64_t>::max()
              : static_cast<std::uint64_t>(key.size());
      const std::uint64_t fixedAndKey =
          keyBytes > std::numeric_limits<std::uint64_t>::max() - fixedBytes
              ? std::numeric_limits<std::uint64_t>::max()
              : fixedBytes + keyBytes;
      const std::uint64_t entryBytes =
          payloadBytes >
                  std::numeric_limits<std::uint64_t>::max() - fixedAndKey
              ? std::numeric_limits<std::uint64_t>::max()
              : payloadBytes + fixedAndKey;
      const std::uint64_t available =
          impl_->maximumRetainedBytes >= impl_->statistics.retainedBytes
              ? impl_->maximumRetainedBytes -
                    impl_->statistics.retainedBytes
              : 0;
      if (impl_->results.size() < impl_->maximumEntries &&
          entryBytes <= available) {
        auto [position, inserted] = impl_->results.emplace(key, outcome);
        if (!inserted)
          outcome = position->second;
        else
          impl_->statistics.retainedBytes += entryBytes;
        retained = true;
      } else {
        capacityBypass = true;
        ++impl_->statistics.capacityBypasses;
      }
    }
    ownedFlight->outcome = outcome;
    ownedFlight->retained = retained;
    ownedFlight->complete = true;
    const auto active = impl_->flights.find(key);
    if (active != impl_->flights.end() && active->second == ownedFlight)
      impl_->flights.erase(active);
  }
  impl_->changed.notify_all();
  return LookupResult{std::move(outcome), false, true, false, false, false,
                      capacityBypass};
}

} // namespace loom::dse
