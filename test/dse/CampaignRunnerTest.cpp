#include "PlanExecutionTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "DSE/CampaignRunner.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

namespace {

using namespace loom;
using namespace loom::dse;
using namespace loom::dse::test_support;

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "campaign runner test failure: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

void requireErrorContains(llvm::Error error, llvm::StringRef needle) {
  const std::string message = llvm::toString(std::move(error));
  if (message.find(needle.str()) == std::string::npos)
    fail("expected error containing '" + needle.str() + "', got: " +
         message);
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-campaign-runner", path_))
      fail("cannot create temporary directory: " + error.message());
  }
  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path_); }

  std::string makeDirectory(llvm::StringRef name) const {
    llvm::SmallString<128> path(path_);
    llvm::sys::path::append(path, name);
    if (std::error_code error = llvm::sys::fs::create_directory(path))
      fail("cannot create test directory: " + error.message());
    return path.str().str();
  }

private:
  llvm::SmallString<128> path_;
};

std::uint64_t unixNanosecondsNow() {
  const auto count = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
  if (count <= 0)
    fail("system clock did not produce a positive test timestamp");
  return static_cast<std::uint64_t>(count);
}

WorkUnitKey makeProjectionKey(std::uint64_t ordinal) {
  return take(WorkUnitKey::get(
      0,
      take(WorkUnitDescriptorRef::get("loom.test.projection_registry",
                                      SchemaVersion{1, 0}, 5)),
      ordinal));
}

PlanExecutionPolicy makeExecutionPolicy(std::uint64_t workers) {
  return take(PlanExecutionPolicy::get(
      workers, take(SiteResourceClaim::get(1, 0, 0))));
}

void testOperationalProjection(const ArtifactStore &store,
                               llvm::StringRef runRoot) {
  PlanExecutionFixture fixture = take(makePlanExecutionFixture(
      store, 0, "loom.test.projection.v1"));
  ExecutionJournal journal =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
  const WorkUnitKey first = makeProjectionKey(0);
  const WorkUnitKey second = makeProjectionKey(1);
  const WorkUnitKey pending = makeProjectionKey(2);
  requireSuccess(journal.queue(first));
  requireSuccess(journal.markRunning(first));
  const std::uint64_t firstCompletion = unixNanosecondsNow();
  requireSuccess(journal.markTerminal(
      first, JournalWorkUnitStatus::Completed, 10, firstCompletion, {}));
  requireSuccess(journal.queue(second));
  requireSuccess(journal.markRunning(second));
  const std::uint64_t secondCompletion = unixNanosecondsNow();
  requireSuccess(journal.markTerminal(
      second, JournalWorkUnitStatus::Failed, 20, secondCompletion, {}));
  requireSuccess(journal.queue(pending));

  SiteScheduler scheduler =
      take(SiteScheduler::create(take(SiteCapacity::get(2, 0, 0))));
  auto lease = take(scheduler.tryAcquire(
      makeProjectionKey(9), take(SiteResourceClaim::get(1, 0, 0))));
  if (!lease)
    fail("projection fixture could not acquire its declared CPU claim");
  DseOperationalProjection projection =
      take(projectDseOperationalState(journal, scheduler, 2));
  if (projection.status.completed != 1 || projection.status.failed != 1 ||
      projection.status.queued != 1 || projection.durations.size() != 1 ||
      projection.durations.front().terminalCount != 2 ||
      projection.durations.front().p50Nanoseconds != 10 ||
      projection.durations.front().p90Nanoseconds != 20 ||
      projection.estimatedRemainingNanoseconds != 20 ||
      !projection.limitingResource ||
      projection.limitingResource->kind != SiteResourceKind::Cpu)
    fail("operational projection changed counts, percentiles, ETA, or limit");

  std::string jsonLine;
  llvm::raw_string_ostream output(jsonLine);
  requireSuccess(writeDseOperationalProjectionJsonLine(projection, output));
  output.flush();
  auto parsed = llvm::json::parse(jsonLine);
  if (!parsed || !parsed->getAsObject() || jsonLine.empty() ||
      jsonLine.back() != '\n')
    fail("operational projection did not emit one valid JSONL record");
}

void testCampaignPolicyAdmission() {
  auto zeroPilot = CampaignExecutionPolicy::get(0, 1);
  if (zeroPilot)
    fail("campaign policy accepted an empty pilot");
  requireErrorContains(zeroPilot.takeError(), "pilot dispatch count");

  auto excessiveSample = CampaignExecutionPolicy::get(
      1, 1,
      CampaignExecutionPolicy::maximumSampleActiveWallTimeNanoseconds + 1,
      CampaignExecutionPolicy::maximumCampaignActiveWallTimeNanoseconds);
  if (excessiveSample)
    fail("campaign policy accepted a sample limit above 600 seconds");
  requireErrorContains(excessiveSample.takeError(), "600-second bound");

  auto excessiveCampaign = CampaignExecutionPolicy::get(
      1, 1,
      CampaignExecutionPolicy::maximumSampleActiveWallTimeNanoseconds,
      CampaignExecutionPolicy::maximumCampaignActiveWallTimeNanoseconds + 1);
  if (excessiveCampaign)
    fail("campaign policy accepted a limit above 23 hours");
  requireErrorContains(excessiveCampaign.takeError(), "23-hour bound");
}

void testPilotContinuation(const ArtifactStore &store, const BlobStore &blobs,
                           llvm::StringRef runRoot) {
  PlanExecutionFixture fixture = take(makePlanExecutionFixture(
      store, 3, "loom.test.campaign.pilot.v1"));
  ExecutionJournal journal =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
  SiteScheduler scheduler =
      take(SiteScheduler::create(take(SiteCapacity::get(2, 0, 0))));
  resetPlanExecutionProviderObservations();

  CampaignExecutionResult result = take(runGroundTruthCampaign(
      fixture.view, fixture.closure,
      take(CampaignExecutionPolicy::get(1, 1)), makeExecutionPolicy(2),
      scheduler, journal, store, blobs));
  const auto *executed = std::get_if<CampaignExecution>(&result);
  if (!executed ||
      !std::holds_alternative<CompletedDsePlanExecution>(executed->outcome))
    fail("admitted pilot did not continue the same plan to completion");
  if (planExecutionProviderCalls() != 3)
    fail("campaign reran pilot work or omitted a missing work unit");
  const auto records = take(journal.workUnits());
  if (records.size() != 3)
    fail("campaign did not retain one stable key per finite work unit");
  for (std::size_t index = 0; index != records.size(); ++index)
    if (records[index].key.planNodeOrdinal() != index ||
        records[index].status != JournalWorkUnitStatus::Completed)
      fail("campaign changed prefix key order or completion state");
}

} // namespace

int main() {
  TemporaryDirectory directory;
  const std::string storeRoot = directory.makeDirectory("artifacts");
  const std::string blobRoot = directory.makeDirectory("blobs");
  const std::string projectionRun = directory.makeDirectory("projection-run");
  const std::string campaignRun = directory.makeDirectory("campaign-run");
  ArtifactStore store(storeRoot);
  BlobStore blobs(blobRoot);
  requireSuccess(registerPlanExecutionTestGenerator());
  testOperationalProjection(store, projectionRun);
  testCampaignPolicyAdmission();
  testPilotContinuation(store, blobs, campaignRun);
  return 0;
}
