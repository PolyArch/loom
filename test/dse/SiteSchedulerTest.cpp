#include "DSE/SiteScheduler.h"

#include "Common/BlobDigest.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <utility>

namespace {

using namespace loom;
using namespace loom::dse;

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "site scheduler test failure: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireErrorContains(llvm::Error error, llvm::StringRef needle) {
  const std::string message = llvm::toString(std::move(error));
  if (message.find(needle.str()) == std::string::npos)
    fail("expected error containing '" + needle.str() + "', got: " + message);
}

WorkUnitKey makeKey(std::uint64_t ordinal) {
  return take(WorkUnitKey::get(
      0,
      take(WorkUnitDescriptorRef::get("loom.test.scheduler_registry",
                                      SchemaVersion{1, 0}, 3)),
      ordinal));
}

struct SchedulerDeadline final {
  std::chrono::steady_clock::time_point notAfter;
};

bool schedulerDeadlineReached(const void *opaque) {
  return std::chrono::steady_clock::now() >=
         static_cast<const SchedulerDeadline *>(opaque)->notAfter;
}

std::optional<std::chrono::steady_clock::duration>
schedulerDeadlineRemaining(const void *opaque) {
  const auto notAfter =
      static_cast<const SchedulerDeadline *>(opaque)->notAfter;
  const auto now = std::chrono::steady_clock::now();
  return now >= notAfter ? std::chrono::steady_clock::duration::zero()
                         : notAfter - now;
}

struct ReentrantStopContext final {
  SiteScheduler *scheduler = nullptr;
  bool observedSnapshot = false;
};

bool reentrantStopRequested(const void *opaque) {
  auto &context =
      *static_cast<ReentrantStopContext *>(const_cast<void *>(opaque));
  (void)take(context.scheduler->snapshot());
  context.observedSnapshot = true;
  return true;
}

void testExactClaimsAndRelease() {
  const BlobDigest binding = computeBlobDigest({0x11, 0x22});
  const SiteResourceKey tool = SiteResourceKey::externalToolBinding(binding);
  const SiteResourceKey license = SiteResourceKey::licenseBinding(binding);
  SiteCapacity capacity =
      take(SiteCapacity::get(2, 1024, 2048, {{tool, 1}}, {{license, 1}}));
  SiteScheduler scheduler = take(SiteScheduler::create(capacity));
  SiteResourceClaim exclusive =
      take(SiteResourceClaim::get(1, 256, 512, {{tool, 1}}, {{license, 1}}));

  std::optional<SiteResourceLease> first =
      take(scheduler.tryAcquire(makeKey(0), exclusive));
  if (!first)
    fail("scheduler rejected an available exact resource claim");
  std::optional<SiteResourceLease> blocked =
      take(scheduler.tryAcquire(makeKey(1), exclusive));
  if (blocked)
    fail("exclusive tool and license claim was admitted twice");
  SiteSchedulerSnapshot occupied = take(scheduler.snapshot());
  if (occupied.running.size() != 1 || !occupied.queued.empty() ||
      occupied.allocated.cpuCores() != 1 ||
      occupied.allocated.externalTools().size() != 1 ||
      occupied.allocated.licenses().size() != 1)
    fail("scheduler snapshot lost exact allocated resources");

  first->release();
  std::optional<SiteResourceLease> second =
      take(scheduler.tryAcquire(makeKey(1), exclusive));
  if (!second)
    fail("released exact resource claim did not become available");
  second->release();
  SiteSchedulerSnapshot empty = take(scheduler.snapshot());
  if (!empty.running.empty() || empty.allocated.cpuCores() != 0 ||
      !empty.allocated.externalTools().empty() ||
      !empty.allocated.licenses().empty())
    fail("lease destruction did not restore resource capacity");
}

void testStrictCapacityAdmission() {
  const BlobDigest binding = computeBlobDigest({0x31});
  const SiteResourceKey tool = SiteResourceKey::externalToolBinding(binding);
  auto zeroCpu = SiteCapacity::get(0, 0, 0);
  if (zeroCpu)
    fail("site capacity accepted zero CPU cores");
  requireErrorContains(zeroCpu.takeError(), "at least one CPU");

  auto duplicate = SiteCapacity::get(1, 0, 0, {{tool, 1}, {tool, 1}});
  if (duplicate)
    fail("site capacity accepted duplicate exact resource keys");
  requireErrorContains(duplicate.takeError(), "duplicate key");

  SiteScheduler scheduler =
      take(SiteScheduler::create(take(SiteCapacity::get(1, 8, 8))));
  SiteResourceClaim oversized = take(SiteResourceClaim::get(2, 0, 0));
  auto rejected = scheduler.tryAcquire(makeKey(4), oversized);
  if (rejected)
    fail("scheduler accepted a claim beyond declared capacity");
  requireErrorContains(rejected.takeError(), "exceeds declared site capacity");

  SiteResourceClaim unknownTool =
      take(SiteResourceClaim::get(1, 0, 0, {{tool, 1}}));
  rejected = scheduler.tryAcquire(makeKey(5), unknownTool);
  if (rejected)
    fail("scheduler accepted an undeclared external-tool binding");
  requireErrorContains(rejected.takeError(), "exceeds declared site capacity");
}

void testQueuedClaimsAreNotBypassed() {
  SiteScheduler scheduler =
      take(SiteScheduler::create(take(SiteCapacity::get(2, 0, 0))));
  const SiteResourceClaim oneCpu = take(SiteResourceClaim::get(1, 0, 0));
  const SiteResourceClaim twoCpus = take(SiteResourceClaim::get(2, 0, 0));
  std::optional<SiteResourceLease> running =
      take(scheduler.tryAcquire(makeKey(10), oneCpu));
  if (!running)
    fail("scheduler could not establish the fairness fixture");

  std::optional<SiteResourceLease> queued;
  std::thread waiter(
      [&] { queued.emplace(take(scheduler.acquire(makeKey(11), twoCpus))); });
  while (take(scheduler.snapshot()).queued.empty())
    std::this_thread::yield();
  auto bypass = take(scheduler.tryAcquire(makeKey(12), oneCpu));
  if (bypass)
    fail("tryAcquire bypassed an existing queued claim");
  running->release();
  waiter.join();
  if (!queued)
    fail("queued claim did not acquire after the prior lease released");
  queued->release();
}

void testControlledAcquireLeavesTheQueue() {
  SiteScheduler scheduler =
      take(SiteScheduler::create(take(SiteCapacity::get(1, 0, 0))));
  const SiteResourceClaim oneCpu = take(SiteResourceClaim::get(1, 0, 0));
  std::optional<SiteResourceLease> running =
      take(scheduler.tryAcquire(makeKey(20), oneCpu));
  if (!running)
    fail("scheduler could not establish the controlled-wait fixture");

  const SchedulerDeadline deadline{std::chrono::steady_clock::now() +
                                   std::chrono::milliseconds(100)};
  const ExecutionControlView executionControl{
      &deadline, schedulerDeadlineReached, schedulerDeadlineRemaining};
  bool cancelled = false;
  const auto begin = std::chrono::steady_clock::now();
  std::thread waiter([&] {
    std::optional<SiteResourceLease> lease =
        take(scheduler.acquire(makeKey(21), oneCpu, executionControl));
    cancelled = !lease;
  });
  waiter.join();
  if (!cancelled ||
      std::chrono::steady_clock::now() - begin >= std::chrono::seconds(1))
    fail("controlled acquire did not stop at its execution deadline");
  SiteSchedulerSnapshot snapshot = take(scheduler.snapshot());
  if (!snapshot.queued.empty() || snapshot.running.size() != 1 ||
      !(snapshot.running.front().key == makeKey(20)))
    fail("controlled acquire left stale scheduler ownership");
  running->release();
}

void testPrepareDiscoveredCountedResources() {
  const BlobDigest binding = computeBlobDigest({0x81, 0x82});
  const SiteResourceKey tool = SiteResourceKey::externalToolBinding(binding);
  const SiteResourceKey license = SiteResourceKey::licenseBinding(binding);
  SiteScheduler scheduler = take(SiteScheduler::create(
      take(SiteCapacity::get(2, 1024, 2048, {{tool, 1}}, {{license, 1}}))));
  SiteResourceClaim scalar = take(SiteResourceClaim::get(2, 512, 1024));
  SiteResourceClaim bound =
      take(SiteResourceClaim::get(2, 512, 1024, {{tool, 1}}, {{license, 1}}));
  std::optional<SiteResourceLease> lease =
      take(scheduler.tryAcquire(makeKey(30), scalar));
  if (!lease)
    fail("scheduler rejected a prepare-time scalar reservation");
  if (!take(scheduler.bindCountedResources(*lease, bound)))
    fail("scheduler stopped an available counted-resource binding");
  SiteSchedulerSnapshot snapshot = take(scheduler.snapshot());
  if (snapshot.running.size() != 1 || !snapshot.queued.empty() ||
      snapshot.allocated.cpuCores() != 2 ||
      snapshot.allocated.memoryBytes() != 512 ||
      snapshot.allocated.scratchBytes() != 1024 ||
      snapshot.allocated.externalTools().size() != 1 ||
      snapshot.allocated.licenses().size() != 1)
    fail("counted-resource binding changed or lost its scalar reservation");
  lease->release();
  snapshot = take(scheduler.snapshot());
  if (!snapshot.running.empty() || !snapshot.queued.empty() ||
      snapshot.allocated.cpuCores() != 0 ||
      !snapshot.allocated.externalTools().empty() ||
      !snapshot.allocated.licenses().empty())
    fail("bound lease did not release its complete resource claim");
}

void testCountedResourceBindingWaitsWithoutReleasingScalars() {
  const BlobDigest binding = computeBlobDigest({0x91, 0x92});
  const SiteResourceKey tool = SiteResourceKey::externalToolBinding(binding);
  SiteScheduler scheduler = take(
      SiteScheduler::create(take(SiteCapacity::get(3, 0, 0, {{tool, 1}}))));
  const SiteResourceClaim scalar = take(SiteResourceClaim::get(1, 0, 0));
  const SiteResourceClaim exact =
      take(SiteResourceClaim::get(1, 0, 0, {{tool, 1}}));
  std::optional<SiteResourceLease> holder =
      take(scheduler.tryAcquire(makeKey(40), exact));
  std::optional<SiteResourceLease> bindingLease =
      take(scheduler.tryAcquire(makeKey(41), scalar));
  if (!holder || !bindingLease)
    fail("scheduler could not establish the counted-resource wait fixture");

  const SchedulerDeadline deadline{std::chrono::steady_clock::now() +
                                   std::chrono::seconds(5)};
  const ExecutionControlView executionControl{
      &deadline, schedulerDeadlineReached, schedulerDeadlineRemaining};
  bool bound = false;
  std::thread waiter([&] {
    bound = take(
        scheduler.bindCountedResources(*bindingLease, exact, executionControl));
  });
  while (take(scheduler.snapshot()).queued.empty()) {
    if (schedulerDeadlineReached(&deadline))
      fail("counted-resource binding did not enter the scheduler queue");
    std::this_thread::yield();
  }
  SiteSchedulerSnapshot waiting = take(scheduler.snapshot());
  if (waiting.running.size() != 2 || waiting.queued.size() != 1 ||
      waiting.allocated.cpuCores() != 2 ||
      waiting.allocated.externalTools().size() != 1 ||
      waiting.queued.front().claim.cpuCores() != 0 ||
      waiting.queued.front().claim.externalTools().size() != 1)
    fail("pending resource binding lost scalar ownership or delta accounting");
  std::optional<SiteResourceLease> disjoint =
      take(scheduler.tryAcquire(makeKey(42), scalar));
  if (!disjoint)
    fail("pending counted resource blocked disjoint scalar work");
  disjoint->release();

  holder->release();
  waiter.join();
  if (!bound || bindingLease->claim().externalTools().size() != 1)
    fail("counted-resource binding did not acquire the released resource");
  SiteSchedulerSnapshot acquired = take(scheduler.snapshot());
  if (acquired.running.size() != 1 || !acquired.queued.empty() ||
      acquired.allocated.cpuCores() != 1 ||
      acquired.allocated.externalTools().size() != 1)
    fail("completed resource binding did not preserve exact ownership");
  bindingLease->release();
}

void testCountedResourceTransitionCannotHoldAndWait() {
  const SiteResourceKey firstTool =
      SiteResourceKey::externalToolBinding(computeBlobDigest({0xa1}));
  const SiteResourceKey secondTool =
      SiteResourceKey::externalToolBinding(computeBlobDigest({0xa2}));
  std::vector<CountedSiteResource> tools{{firstTool, 1}, {secondTool, 1}};
  llvm::sort(tools, [](const auto &lhs, const auto &rhs) {
    return lhs.key < rhs.key;
  });
  SiteScheduler scheduler =
      take(SiteScheduler::create(take(SiteCapacity::get(2, 0, 0, tools))));
  const SiteResourceClaim first =
      take(SiteResourceClaim::get(1, 0, 0, {{firstTool, 1}}));
  const SiteResourceClaim second =
      take(SiteResourceClaim::get(1, 0, 0, {{secondTool, 1}}));
  std::optional<SiteResourceLease> firstLease =
      take(scheduler.tryAcquire(makeKey(50), first));
  std::optional<SiteResourceLease> secondLease =
      take(scheduler.tryAcquire(makeKey(51), second));
  if (!firstLease || !secondLease)
    fail("scheduler could not establish the counted transition fixture");

  const SchedulerDeadline deadline{std::chrono::steady_clock::now() +
                                   std::chrono::seconds(5)};
  const ExecutionControlView executionControl{
      &deadline, schedulerDeadlineReached, schedulerDeadlineRemaining};
  std::atomic<bool> firstFinished = false;
  std::atomic<bool> secondFinished = false;
  bool firstBound = false;
  bool secondBound = false;
  std::thread firstWaiter([&] {
    firstBound = take(
        scheduler.bindCountedResources(*firstLease, second, executionControl));
    firstFinished.store(true, std::memory_order_release);
  });
  while (take(scheduler.snapshot()).queued.empty())
    std::this_thread::yield();
  std::thread secondWaiter([&] {
    secondBound = take(
        scheduler.bindCountedResources(*secondLease, first, executionControl));
    secondFinished.store(true, std::memory_order_release);
  });
  while (!firstFinished.load(std::memory_order_acquire) &&
         !secondFinished.load(std::memory_order_acquire)) {
    if (schedulerDeadlineReached(&deadline))
      fail("counted resource transitions retained a wait cycle");
    std::this_thread::yield();
  }
  if (firstFinished.load(std::memory_order_acquire)) {
    firstWaiter.join();
    if (!firstBound)
      fail("first counted resource transition stopped unexpectedly");
    firstLease->release();
    secondWaiter.join();
  } else {
    secondWaiter.join();
    if (!secondBound)
      fail("second counted resource transition stopped unexpectedly");
    secondLease->release();
    firstWaiter.join();
  }
  if (!firstBound || !secondBound)
    fail("counted resource transitions did not both make progress");
  if (*firstLease)
    firstLease->release();
  if (*secondLease)
    secondLease->release();
}

void testNoopBindingObservesReentrantCancellation() {
  SiteScheduler scheduler =
      take(SiteScheduler::create(take(SiteCapacity::get(1, 0, 0))));
  const SiteResourceClaim scalar = take(SiteResourceClaim::get(1, 0, 0));
  std::optional<SiteResourceLease> lease =
      take(scheduler.tryAcquire(makeKey(60), scalar));
  if (!lease)
    fail("scheduler could not establish the no-op binding fixture");
  ReentrantStopContext stop{&scheduler, false};
  const ExecutionControlView executionControl{&stop, reentrantStopRequested,
                                              nullptr};
  if (take(scheduler.bindCountedResources(*lease, scalar, executionControl)) ||
      !stop.observedSnapshot)
    fail("no-op binding ignored reentrant cancellation");
  lease->release();
}

} // namespace

int main() {
  testExactClaimsAndRelease();
  testStrictCapacityAdmission();
  testQueuedClaimsAreNotBypassed();
  testControlledAcquireLeavesTheQueue();
  testPrepareDiscoveredCountedResources();
  testCountedResourceBindingWaitsWithoutReleasingScalars();
  testCountedResourceTransitionCannotHoldAndWait();
  testNoopBindingObservesReentrantCancellation();
  return 0;
}
