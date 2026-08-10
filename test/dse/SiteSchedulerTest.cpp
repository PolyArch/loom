#include "DSE/SiteScheduler.h"

#include "Common/BlobDigest.h"

#include "llvm/Support/Error.h"

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

} // namespace

int main() {
  testExactClaimsAndRelease();
  testStrictCapacityAdmission();
  testQueuedClaimsAreNotBypassed();
  return 0;
}
