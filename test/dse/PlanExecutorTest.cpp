#include "PlanExecutionTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "DSE/CandidateGeneratorRecovery.h"
#include "DSE/PlanExecutor.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace loom;
using namespace loom::dse;
using namespace loom::dse::test_support;

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "plan executor test failure: " << message << '\n';
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
    if (std::error_code error =
            llvm::sys::fs::createUniqueDirectory("loom-plan-executor", path_))
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

PlanExecutionPolicy makePolicy(std::uint64_t workers,
                               std::optional<std::uint64_t> dispatches = {},
                               std::uint64_t memoryBytes = 0) {
  return take(PlanExecutionPolicy::get(
      workers, take(SiteResourceClaim::get(1, memoryBytes, 0)), std::nullopt,
      {}, dispatches));
}

SiteScheduler makeScheduler(std::uint64_t memoryBytes = 0) {
  return take(
      SiteScheduler::create(take(SiteCapacity::get(2, memoryBytes, 0))));
}

void testParallelExecutionAndTerminalReplay(const ArtifactStore &store,
                                            const BlobStore &blobs,
                                            llvm::StringRef runRoot) {
  PlanExecutionFixture fixture = take(makePlanExecutionFixture(
      store, 2, "loom.test.plan_executor.parallel.v1"));
  ExecutionJournal journal =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
  SiteScheduler scheduler = makeScheduler();
  resetPlanExecutionProviderObservations();
  requireConcurrentPlanExecutionProviders(2);

  DsePlanExecutionResult first =
      take(executeDsePlan(fixture.view, fixture.closure, journal, scheduler,
                          makePolicy(2), store, blobs));
  requireConcurrentPlanExecutionProviders(1);
  const auto *completed = std::get_if<CompletedDsePlanExecution>(&first);
  if (!completed || completed->generateInvocations().size() != 2)
    fail("independent Generate plan did not complete both nodes");
  if (planExecutionProviderCalls() != 2)
    fail("Generate provider did not run exactly once per stable work unit");
  if (maximumConcurrentPlanExecutionProviders() != 2)
    fail("independent Generate nodes did not use both admitted worker lanes");
  const auto records = take(journal.workUnits());
  if (records.size() != 2 ||
      records[0].status != JournalWorkUnitStatus::Completed ||
      records[1].status != JournalWorkUnitStatus::Completed ||
      !records[0].finalizedWorkRecord || !records[1].finalizedWorkRecord ||
      records[0].key.planNodeOrdinal() != 0 ||
      records[1].key.planNodeOrdinal() != 1)
    fail("journal did not retain canonical recoverable Generate work");

  DsePlanExecutionResult replay = take(resumeDsePlan(
      fixture.view, fixture.closure, journal, scheduler, makePolicy(2), store,
      blobs, InvocationManifestRetention::Release));
  if (!std::holds_alternative<CompletedDsePlanExecution>(replay))
    fail("terminal plan did not replay to the same completed outcome");
  if (planExecutionProviderCalls() != 2)
    fail("resume invoked a provider for already finalized Generate work");
  const auto replayed = take(journal.workUnits());
  if (replayed.size() != records.size())
    fail("resume renumbered or duplicated stable work keys");
  for (std::size_t index = 0; index != records.size(); ++index)
    if (!(replayed[index].key == records[index].key) ||
        replayed[index].finalizedOutputs != records[index].finalizedOutputs)
      fail("resume changed a finalized work key or root set");

  ExecutionJournal reopened =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
  DsePlanExecutionResult recovered = take(resumeDsePlan(
      fixture.view, fixture.closure, reopened, scheduler, makePolicy(2), store,
      blobs, InvocationManifestRetention::Release));
  if (!std::holds_alternative<CompletedDsePlanExecution>(recovered))
    fail("reopened plan did not reconstruct its completed outcome");
  if (planExecutionProviderCalls() != 2)
    fail("reopen invoked providers for finalized recoverable Generate work");
  const auto recoveredRecords = take(reopened.workUnits());
  if (recoveredRecords.size() != records.size())
    fail("reopen renumbered or duplicated stable work keys");
  for (std::size_t index = 0; index != records.size(); ++index)
    if (!(recoveredRecords[index].key == records[index].key) ||
        recoveredRecords[index].finalizedOutputs !=
            records[index].finalizedOutputs)
      fail("reopen changed a finalized work key or root set");
}

void testStopAndResumeMissingKeys(const ArtifactStore &store,
                                  const BlobStore &blobs,
                                  llvm::StringRef runRoot) {
  PlanExecutionFixture fixture = take(
      makePlanExecutionFixture(store, 2, "loom.test.plan_executor.stop.v1"));
  ExecutionJournal journal =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
  SiteScheduler scheduler = makeScheduler();
  resetPlanExecutionProviderObservations();

  requireSuccess(
      stopDseExecution(journal, GracefulStopPolicy::FinishAtomicOwnerBoundary));
  DsePlanExecutionResult stopped =
      take(executeDsePlan(fixture.view, fixture.closure, journal, scheduler,
                          makePolicy(2), store, blobs));
  if (!std::holds_alternative<IncompleteDsePlanExecution>(stopped) ||
      planExecutionProviderCalls() != 0)
    fail("graceful stop dispatched new Generate work");

  DsePlanExecutionResult resumed = take(resumeDsePlan(
      fixture.view, fixture.closure, journal, scheduler, makePolicy(2), store,
      blobs, InvocationManifestRetention::Release));
  if (!std::holds_alternative<CompletedDsePlanExecution>(resumed) ||
      planExecutionProviderCalls() != 2)
    fail("resume did not execute exactly the missing stable work keys");
}

void testRunningProviderObservesStop(const ArtifactStore &store,
                                     const BlobStore &blobs,
                                     llvm::StringRef runRoot) {
  PlanExecutionFixture fixture = take(makePlanExecutionFixture(
      store, 1, "loom.test.plan_executor.inflight_stop.v1"));
  ExecutionJournal journal =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
  SiteScheduler scheduler = makeScheduler();
  resetPlanExecutionProviderObservations();
  requirePlanExecutionProviderStopObservation();

  auto pending = std::async(std::launch::async, [&] {
    return executeDsePlan(fixture.view, fixture.closure, journal, scheduler,
                          makePolicy(1), store, blobs);
  });
  if (!waitForActivePlanExecutionProvider())
    fail("interruptible Generate provider did not enter execution");
  requireSuccess(
      stopDseExecution(journal, GracefulStopPolicy::FinishAtomicOwnerBoundary));
  DsePlanExecutionResult stopped = take(pending.get());
  if (!std::holds_alternative<IncompleteDsePlanExecution>(stopped) ||
      !planExecutionProviderObservedStop())
    fail("running Generate provider did not return typed interruption");
  const auto records = take(journal.workUnits());
  if (records.size() != 1 ||
      records.front().status != JournalWorkUnitStatus::TimedOut ||
      !records.front().finalizedOutputs.empty())
    fail("interrupted Generate work did not retain a typed empty terminal "
         "record");
}

void testDeterministicDispatchPrefix(const ArtifactStore &store,
                                     const BlobStore &blobs,
                                     llvm::StringRef runRoot) {
  PlanExecutionFixture fixture = take(
      makePlanExecutionFixture(store, 3, "loom.test.plan_executor.prefix.v1"));
  ExecutionJournal journal =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
  SiteScheduler scheduler = makeScheduler();
  resetPlanExecutionProviderObservations();

  DsePlanExecutionResult pilot =
      take(executeDsePlan(fixture.view, fixture.closure, journal, scheduler,
                          makePolicy(3, 1), store, blobs));
  if (!std::holds_alternative<IncompleteDsePlanExecution>(pilot) ||
      planExecutionProviderCalls() != 1)
    fail("bounded dispatch did not execute exactly one pilot work unit");
  const auto pilotRecords = take(journal.workUnits());
  if (pilotRecords.size() < 2 ||
      pilotRecords.front().key.planNodeOrdinal() != 0 ||
      pilotRecords.front().status != JournalWorkUnitStatus::Completed)
    fail("pilot dispatch was not the canonical resolved-plan prefix");

  DsePlanExecutionResult full = take(resumeDsePlan(
      fixture.view, fixture.closure, journal, scheduler, makePolicy(3), store,
      blobs, InvocationManifestRetention::Release));
  if (!std::holds_alternative<CompletedDsePlanExecution>(full) ||
      planExecutionProviderCalls() != 3)
    fail("resumed prefix did not finish only its missing work units");
}

void testProviderReceivesAdmittedResourceBudget(const ArtifactStore &store,
                                                const BlobStore &blobs,
                                                llvm::StringRef runRoot) {
  constexpr std::uint64_t memoryBudgetBytes = UINT64_C(64) * 1024 * 1024;
  PlanExecutionFixture fixture = take(makePlanExecutionFixture(
      store, 1, "loom.test.plan_executor.memory_budget.v1"));
  ExecutionJournal journal =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
  SiteScheduler scheduler = makeScheduler(memoryBudgetBytes);
  resetPlanExecutionProviderObservations();

  DsePlanExecutionResult result = take(executeDsePlan(
      fixture.view, fixture.closure, journal, scheduler,
      makePolicy(1, std::nullopt, memoryBudgetBytes), store, blobs));
  if (!std::holds_alternative<CompletedDsePlanExecution>(result) ||
      planExecutionProviderCpuBudgetCores() != 1 ||
      planExecutionProviderMemoryBudgetBytes() != memoryBudgetBytes)
    fail("Generate provider lost its admitted execution resource budget");
}

void testProvenInfeasibleRecoveryAndManifest(const ArtifactStore &store,
                                             const BlobStore &blobs,
                                             llvm::StringRef runRoot) {
  PlanExecutionFixture fixture = take(makePlanExecutionFixture(
      store, 1, "loom.test.plan_executor.proven_infeasible.v1", true));
  ExecutionJournal journal =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
  SiteScheduler scheduler = makeScheduler();
  resetPlanExecutionProviderObservations();
  setPlanExecutionProviderOutcome(
      PlanExecutionProviderOutcomeKind::ProvenInfeasible);

  DsePlanExecutionResult first = take(executeDsePlan(
      fixture.view, fixture.closure, journal, scheduler, makePolicy(1), store,
      blobs));
  const auto *completed = std::get_if<CompletedDsePlanExecution>(&first);
  if (!completed || completed->generateInvocations().size() != 1 ||
      !completed->generateInvocations().front().infeasibilityProof ||
      !completed->resolve(PlanOutputRef{0, 0}).empty())
    fail("ProvenInfeasible Generate result lost its terminal proof or gained "
         "an output");

  const auto records = take(journal.workUnits());
  if (records.size() != 1 ||
      records.front().status != JournalWorkUnitStatus::Completed ||
      !records.front().finalizedOutputs.empty() ||
      !records.front().finalizedWorkRecord ||
      records.front().finalizedWorkRecord->schemaVersion !=
          candidateGeneratorFinalizedWorkRecordSchemaVersion)
    fail("journal did not retain the proof-bearing finalized work owner");

  DsePlanGenerateInvocationRecords generated =
      projectDsePlanGenerateInvocationRecords(first);
  const DsePlanGenerateInvocationSummary summary = take(
      validateAndSummarizeDsePlanGenerateInvocations(generated, store, blobs));
  if (summary.completedInvocations != 1 ||
      summary.provenInfeasibleInvocations != 1 ||
      summary.incompleteInvocations != 0)
    fail("Generate summary lost the terminal ProvenInfeasible subtype");
  auto controllerOutcome = take(projectDsePlanInvocationOutcome(
      fixture.view, static_cast<const DsePlanExecutionOutcome &>(first)));
  if (!std::holds_alternative<InvocationCompletedNoFeasibleCandidate>(
          controllerOutcome))
    fail("proof-bearing empty plan did not project a completed empty selection");
  const GenerateInvocationRecord &invocation =
      completed->generateInvocations().front();
  CandidateGeneratorInfeasibilityProof forgedProof =
      *invocation.infeasibilityProof;
  forgedProof.kind = CandidateGeneratorInfeasibilityProofKindRef(1);
  requireErrorContains(
      validateCanonicalCandidateGeneratorInvocation(
          invocation.inputBindings, invocation.generatorBinding,
          invocation.outputBindings, invocation.lineageEdges, true,
          forgedProof, store, blobs),
      "infeasibility proof is not canonical");
  InvocationManifest manifest = take(InvocationManifest::get(
      fixture.closure, 0, std::nullopt, fixture.config, generated,
      std::move(controllerOutcome), store, blobs));
  InvocationManifest imported =
      take(adoptInvocationManifest(manifest.canonicalBytes(), fixture.config,
                                   store, blobs));
  if (imported.generateRecords().size() != 1 ||
      !imported.generateRecords().front().invocation.infeasibilityProof ||
      imported.generateRecords().front().invocation.incompleteReason)
    fail("InvocationManifest did not round-trip the terminal proof type");
  std::vector<std::uint8_t> malformed(manifest.canonicalBytes().begin(),
                                      manifest.canonicalBytes().end());
  malformed.back() = 0x02;
  auto rejected =
      adoptInvocationManifest(malformed, fixture.config, store, blobs);
  if (rejected)
    fail("InvocationManifest accepted a noncanonical proof witness");
  requireErrorContains(rejected.takeError(),
                       "proof is not established by its exact input");

  setPlanExecutionProviderOutcome(PlanExecutionProviderOutcomeKind::Candidate);
  DsePlanExecutionResult replay = take(resumeDsePlan(
      fixture.view, fixture.closure, journal, scheduler, makePolicy(1), store,
      blobs, InvocationManifestRetention::Release));
  const auto *replayed = std::get_if<CompletedDsePlanExecution>(&replay);
  if (!replayed || replayed->generateInvocations().size() != 1 ||
      !replayed->generateInvocations().front().infeasibilityProof ||
      replayed->generateInvocationWasDispatched(0) ||
      planExecutionProviderCalls() != 1)
    fail("terminal replay lost the proof or redispatched provider work");
}

void testCompletedEmptyRemainsUnproven(const ArtifactStore &store,
                                       const BlobStore &blobs,
                                       llvm::StringRef runRoot) {
  PlanExecutionFixture fixture = take(makePlanExecutionFixture(
      store, 1, "loom.test.plan_executor.completed_empty.v1"));
  ExecutionJournal journal =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
  SiteScheduler scheduler = makeScheduler();
  resetPlanExecutionProviderObservations();
  setPlanExecutionProviderOutcome(
      PlanExecutionProviderOutcomeKind::CompletedEmpty);

  DsePlanExecutionResult execution = take(executeDsePlan(
      fixture.view, fixture.closure, journal, scheduler, makePolicy(1), store,
      blobs));
  const auto *completed = std::get_if<CompletedDsePlanExecution>(&execution);
  if (!completed || completed->generateInvocations().size() != 1 ||
      completed->generateInvocations().front().infeasibilityProof ||
      !completed->resolve(PlanOutputRef{0, 0}).empty())
    fail("completed-empty fixture did not preserve its ordinary outcome");
  InvocationControllerOutcome projected =
      take(projectDsePlanInvocationOutcome(fixture.view, execution));
  if (!std::holds_alternative<InvocationCompletedNoFeasibleCandidate>(
          projected))
    fail("completed-empty plan lost its completed empty selection");
  DsePlanGenerateInvocationRecords records =
      projectDsePlanGenerateInvocationRecords(execution);
  const DsePlanGenerateInvocationSummary summary = take(
      validateAndSummarizeDsePlanGenerateInvocations(records, store, blobs));
  if (summary.completedInvocations != 1 ||
      summary.provenInfeasibleInvocations != 0 ||
      summary.incompleteInvocations != 0)
    fail("Generate summary changed Completed(empty) into ProvenInfeasible");
  InvocationManifest manifest = take(InvocationManifest::get(
      fixture.closure, 0, std::nullopt, fixture.config, records,
      std::move(projected), store, blobs));
  InvocationManifest imported = take(adoptInvocationManifest(
      manifest.canonicalBytes(), fixture.config, store, blobs));
  if (imported.generateRecords().size() != 1 ||
      imported.generateRecords().front().invocation.infeasibilityProof)
    fail("Manifest changed completed-empty Generate into ProvenInfeasible");
}

void testInvalidRetentionDoesNotLease(const ArtifactStore &store,
                                      const BlobStore &blobs,
                                      llvm::StringRef runRoot) {
  PlanExecutionFixture fixture = take(makePlanExecutionFixture(
      store, 1, "loom.test.plan_executor.invalid_retention.v1"));
  ExecutionJournal journal =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
  SiteScheduler scheduler = makeScheduler();
  auto rejected = resumeDsePlan(
      fixture.view, fixture.closure, journal, scheduler, makePolicy(1), store,
      blobs, static_cast<InvocationManifestRetention>(0xff));
  if (rejected)
    fail("resume accepted an unknown manifest retention policy");
  const std::string message = llvm::toString(rejected.takeError());
  if (message.find("unknown invocation manifest retention policy") ==
      std::string::npos)
    fail("invalid retention rejection lost its lifecycle reason");
  auto occurrence = journal.currentInvocationOccurrence();
  if (occurrence)
    fail("invalid retention policy opened an invocation occurrence");
  llvm::consumeError(occurrence.takeError());
}

} // namespace

int main() {
  TemporaryDirectory directory;
  const std::string storeRoot = directory.makeDirectory("artifacts");
  const std::string blobRoot = directory.makeDirectory("blobs");
  const std::string parallelRun = directory.makeDirectory("parallel-run");
  const std::string stoppedRun = directory.makeDirectory("stopped-run");
  const std::string inflightStoppedRun =
      directory.makeDirectory("inflight-stopped-run");
  const std::string prefixRun = directory.makeDirectory("prefix-run");
  const std::string resourceBudgetRun =
      directory.makeDirectory("resource-budget-run");
  const std::string provenInfeasibleRun =
      directory.makeDirectory("proven-infeasible-run");
  const std::string completedEmptyRun =
      directory.makeDirectory("completed-empty-run");
  const std::string invalidRetentionRun =
      directory.makeDirectory("invalid-retention-run");
  ArtifactStore store(storeRoot);
  BlobStore blobs(blobRoot);
  requireSuccess(registerPlanExecutionTestGenerator());
  testParallelExecutionAndTerminalReplay(store, blobs, parallelRun);
  testStopAndResumeMissingKeys(store, blobs, stoppedRun);
  testRunningProviderObservesStop(store, blobs, inflightStoppedRun);
  testDeterministicDispatchPrefix(store, blobs, prefixRun);
  testProviderReceivesAdmittedResourceBudget(store, blobs, resourceBudgetRun);
  testProvenInfeasibleRecoveryAndManifest(store, blobs, provenInfeasibleRun);
  testCompletedEmptyRemainsUnproven(store, blobs, completedEmptyRun);
  testInvalidRetentionDoesNotLease(store, blobs, invalidRetentionRun);
  return 0;
}
