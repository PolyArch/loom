#include "PlanExecutionTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "DSE/CampaignRunner.h"
#include "DSE/InvocationManifest.h"

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
#include <variant>

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

WorkUnitKey makePlanExecutionKey(const PlanExecutionFixture &fixture,
                                 std::uint64_t planNodeOrdinal) {
  if (planNodeOrdinal >= fixture.config.dse.planNodes.size())
    fail("plan execution key names an unknown plan node");
  const auto *definition = std::get_if<GeneratePlanNodeDefinition>(
      &fixture.config.dse.planNodes[planNodeOrdinal]);
  if (!definition)
    fail("plan execution key names a non-Generate plan node");
  auto owner = take(
      WorkUnitDescriptorRef::get(candidateGeneratorDescriptorSchema.identity,
                                 candidateGeneratorDescriptorSchema.version,
                                 definition->descriptor.kind().ordinal()));
  return take(WorkUnitKey::get(planNodeOrdinal, std::move(owner), 0));
}

PlanExecutionPolicy makeExecutionPolicy(std::uint64_t workers) {
  return take(PlanExecutionPolicy::get(
      workers, take(SiteResourceClaim::get(1, 0, 0))));
}

PlanExecutionPolicy makeBoundedExecutionPolicy(std::uint64_t workers,
                                               std::uint64_t dispatches) {
  return take(PlanExecutionPolicy::get(workers,
                                       take(SiteResourceClaim::get(1, 0, 0)),
                                       std::nullopt, {}, dispatches));
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
    fail("campaign policy accepted a sample limit above its configured bound");
  requireErrorContains(excessiveSample.takeError(), "configured bound");

  auto excessiveCampaign = CampaignExecutionPolicy::get(
      1, 1,
      CampaignExecutionPolicy::maximumSampleActiveWallTimeNanoseconds,
      CampaignExecutionPolicy::maximumCampaignActiveWallTimeNanoseconds + 1);
  if (excessiveCampaign)
    fail("campaign policy accepted a limit above its configured bound");
  requireErrorContains(excessiveCampaign.takeError(), "configured bound");
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

  InvocationManifestReference reference =
      take(finalizeDsePlanInvocation(fixture.closure, fixture.config,
                                     executed->outcome, journal, store, blobs));
  InvocationManifest manifest =
      take(importInvocationManifest(reference, store, blobs));
  const auto *selected =
      std::get_if<InvocationCompletedSelection>(&manifest.outcome());
  if (reference.occurrence().occurrenceOrdinal != 0 || !selected ||
      selected->selected.size() != 1 || !selected->satisfiedEvidence.empty() ||
      manifest.generateRecords().size() != 3)
    fail("campaign manifest lost its occurrence, terminal roots, or records");
  auto released = journal.currentInvocationOccurrence();
  if (released)
    fail("manifest finalization retained its completed occurrence lease");
  llvm::consumeError(released.takeError());
  requireErrorContains(journal.queue(makeProjectionKey(99)), "immutable");
}

void testRefusalManifest(const ArtifactStore &store, const BlobStore &blobs,
                         llvm::StringRef runRoot) {
  PlanExecutionFixture fixture = take(makePlanExecutionFixture(
      store, 3, "loom.test.campaign.refusal_manifest.v1"));
  ExecutionJournal journal =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
  SiteScheduler scheduler =
      take(SiteScheduler::create(take(SiteCapacity::get(1, 0, 0))));
  resetPlanExecutionProviderObservations();

  CampaignExecutionResult result = take(runGroundTruthCampaign(
      fixture.view, fixture.closure,
      take(CampaignExecutionPolicy::get(
          1, 1, CampaignExecutionPolicy::maximumSampleActiveWallTimeNanoseconds,
          1)),
      makeExecutionPolicy(1), scheduler, journal, store, blobs));
  const auto *refusal = std::get_if<CampaignAdmissionRefusal>(&result);
  if (!refusal)
    fail("bounded campaign did not retain an admission refusal");
  if (refusal->reason !=
      CampaignAdmissionFailureReason::CampaignActiveWallTimeLimit)
    fail("bounded campaign retained unexpected refusal reason " +
         std::to_string(static_cast<std::uint32_t>(refusal->reason)));

  InvocationManifestReference reference =
      take(finalizeDsePlanInvocation(fixture.closure, fixture.config,
                                     refusal->outcome, journal, store, blobs,
                                     refusal->reason));
  InvocationManifest manifest =
      take(importInvocationManifest(reference, store, blobs));
  const auto *incomplete =
      std::get_if<InvocationIncomplete>(&manifest.outcome());
  if (!incomplete || incomplete->planNodeOrdinal != 1 ||
      incomplete->retainedArtifacts.size() != 1 ||
      !incomplete->retainedEvidence.empty() ||
      manifest.campaignAdmissionFailure() != refusal->reason)
    fail("refused campaign manifest lost its typed incomplete outcome");
}

void testCompletedRefusalManifest(const ArtifactStore &store,
                                  const BlobStore &blobs,
                                  llvm::StringRef runRoot) {
  PlanExecutionFixture fixture = take(makePlanExecutionFixture(
      store, 1, "loom.test.campaign.completed_refusal_manifest.v1"));
  ExecutionJournal journal =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
  SiteScheduler scheduler =
      take(SiteScheduler::create(take(SiteCapacity::get(1, 0, 0))));
  resetPlanExecutionProviderObservations();

  CampaignExecutionResult result = take(runGroundTruthCampaign(
      fixture.view, fixture.closure,
      take(CampaignExecutionPolicy::get(
          1, 1, CampaignExecutionPolicy::maximumSampleActiveWallTimeNanoseconds,
          1)),
      makeExecutionPolicy(1), scheduler, journal, store, blobs));
  const auto *refusal = std::get_if<CampaignAdmissionRefusal>(&result);
  if (!refusal ||
      refusal->reason !=
          CampaignAdmissionFailureReason::CampaignActiveWallTimeLimit ||
      !std::holds_alternative<CompletedDsePlanExecution>(refusal->outcome))
    fail("completed pilot did not retain its typed campaign refusal");

  InvocationManifestReference reference = take(finalizeDsePlanInvocation(
      fixture.closure, fixture.config, refusal->outcome, journal, store, blobs,
      refusal->reason));
  InvocationManifest manifest =
      take(importInvocationManifest(reference, store, blobs));
  if (!std::holds_alternative<InvocationCompletedSelection>(
          manifest.outcome()) ||
      manifest.campaignAdmissionFailure() != refusal->reason)
    fail("completed refusal manifest lost its campaign disposition");
}

void testResumedPilotUsesRemainingBudget(const ArtifactStore &store,
                                         const BlobStore &blobs,
                                         llvm::StringRef runRoot) {
  PlanExecutionFixture fixture = take(makePlanExecutionFixture(
      store, 2, "loom.test.campaign.remaining_budget.v1"));
  ExecutionJournal journal =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
  SiteScheduler scheduler =
      take(SiteScheduler::create(take(SiteCapacity::get(1, 0, 0))));
  resetPlanExecutionProviderObservations();
  DsePlanExecutionResult prior =
      take(executeDsePlan(fixture.view, fixture.closure, journal, scheduler,
                          makeBoundedExecutionPolicy(1, 1), store, blobs));
  if (!std::holds_alternative<IncompleteDsePlanExecution>(prior) ||
      planExecutionProviderCalls() != 1)
    fail("active-time fixture did not consume one canonical dispatch");
  const auto priorRecords = take(journal.workUnits());
  if (priorRecords.empty() ||
      !(priorRecords.front().key == makePlanExecutionKey(fixture, 0)))
    fail("active-time fixture did not retain the canonical recovery key");
  resetPlanExecutionProviderObservations();

  CampaignExecutionResult result = take(runGroundTruthCampaign(
      fixture.view, fixture.closure,
      take(CampaignExecutionPolicy::get(
          1, 1, CampaignExecutionPolicy::maximumSampleActiveWallTimeNanoseconds,
          1)),
      makeExecutionPolicy(1), scheduler, journal, store, blobs));
  const auto *refusal = std::get_if<CampaignAdmissionRefusal>(&result);
  if (!refusal ||
      refusal->reason !=
          CampaignAdmissionFailureReason::CampaignActiveWallTimeLimit)
    fail("resumed campaign did not retain its consumed active-time refusal");
  if (planExecutionProviderCalls() != 0)
    fail("resumed pilot dispatched provider work after exhausting its budget");
}

void testResumedPilotConsumesCompletedPrefix(const ArtifactStore &store,
                                             const BlobStore &blobs,
                                             llvm::StringRef runRoot) {
  PlanExecutionFixture fixture = take(makePlanExecutionFixture(
      store, 4, "loom.test.campaign.remaining_pilot_prefix.v1"));
  ExecutionJournal journal =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
  SiteScheduler scheduler =
      take(SiteScheduler::create(take(SiteCapacity::get(2, 0, 0))));
  resetPlanExecutionProviderObservations();
  DsePlanExecutionResult prior =
      take(executeDsePlan(fixture.view, fixture.closure, journal, scheduler,
                          makeBoundedExecutionPolicy(1, 1), store, blobs));
  if (!std::holds_alternative<IncompleteDsePlanExecution>(prior) ||
      planExecutionProviderCalls() != 1)
    fail("resumed pilot fixture did not consume one canonical dispatch");
  resetPlanExecutionProviderObservations();

  CampaignExecutionResult result = take(runGroundTruthCampaign(
      fixture.view, fixture.closure, take(CampaignExecutionPolicy::get(2, 1)),
      makeExecutionPolicy(2), scheduler, journal, store, blobs));
  if (!std::get_if<CampaignExecution>(&result) ||
      planExecutionProviderCalls() != 3)
    fail("resumed pilot repeated its completed prefix instead of charging "
         "only the remaining pilot dispatch");
  if (maximumConcurrentPlanExecutionProviders() != 2)
    fail("resumed pilot consumed work that belonged to the admitted pass");
  const auto records = take(journal.workUnits());
  if (records.size() != 4)
    fail("resumed pilot did not close the remaining finite work units");
}

} // namespace

int main() {
  TemporaryDirectory directory;
  const std::string storeRoot = directory.makeDirectory("artifacts");
  const std::string blobRoot = directory.makeDirectory("blobs");
  const std::string projectionRun = directory.makeDirectory("projection-run");
  const std::string campaignRun = directory.makeDirectory("campaign-run");
  const std::string resumedRun = directory.makeDirectory("resumed-run");
  const std::string resumedPrefixRun =
      directory.makeDirectory("resumed-prefix-run");
  const std::string refusalRun = directory.makeDirectory("refusal-run");
  const std::string completedRefusalRun =
      directory.makeDirectory("completed-refusal-run");
  ArtifactStore store(storeRoot);
  BlobStore blobs(blobRoot);
  requireSuccess(registerPlanExecutionTestGenerator());
  testOperationalProjection(store, projectionRun);
  testCampaignPolicyAdmission();
  testPilotContinuation(store, blobs, campaignRun);
  testResumedPilotUsesRemainingBudget(store, blobs, resumedRun);
  testResumedPilotConsumesCompletedPrefix(store, blobs, resumedPrefixRun);
  testRefusalManifest(store, blobs, refusalRun);
  testCompletedRefusalManifest(store, blobs, completedRefusalRun);
  return 0;
}
