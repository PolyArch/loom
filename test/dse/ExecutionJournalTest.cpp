#include "DSE/ExecutionJournal.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobDigest.h"
#include "Config/ResolvedConfig.h"
#include "DSE/ResolvedConfigView.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using namespace loom;
using namespace loom::dse;

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "execution journal test failure: " << message << '\n';
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
    fail("expected error containing '" + needle.str() + "', got: " + message);
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-execution-journal", path_))
      fail("cannot create temporary directory: " + error.message());
  }
  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path_); }

  llvm::StringRef path() const { return path_; }
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

constexpr ArtifactSchemaDescriptor sourceSchema{
    "loom.test.execution_journal_source", SchemaVersion{1, 0}};

ArtifactRootReference publish(const ArtifactStore &store, std::uint8_t byte) {
  const ArtifactIdentity identity = take(store.put(
      sourceSchema, CanonicalSemanticBytes(std::vector<std::uint8_t>{byte})));
  return {sourceSchema.identity.str(), sourceSchema.version, identity};
}

struct Fixture final {
  ResolvedConfig config;
  ResolvedDseConfigView view;
  DseRunClosure closure;
  ArtifactRootReference source;
};

Fixture makeFixture(const ArtifactStore &store, std::uint8_t sourceByte,
                    llvm::StringRef producer) {
  ResolvedConfig config = defaultResolvedConfig();
  const ArtifactIdentity configIdentity = take(store.put(
      ResolvedConfig::artifactSchema, canonicalResolvedConfigBytes(config)));
  if (configIdentity != resolvedConfigIdentity(config))
    fail("resolved config publication changed its identity");
  ResolvedDseConfigView view = take(projectResolvedDseConfigView(config));
  ArtifactRootReference source = publish(store, sourceByte);
  DseRunClosure closure = take(
      DseRunClosure::get(take(DseProducerSemanticBuildIdentity::get(producer)),
                         {source}, config, {}, store));
  return {std::move(config), std::move(view), std::move(closure),
          std::move(source)};
}

WorkUnitKey makeKey(std::uint64_t node, std::uint64_t ordinal) {
  return take(WorkUnitKey::get(
      node,
      take(WorkUnitDescriptorRef::get("loom.test.execution_registry",
                                      SchemaVersion{1, 0}, 9)),
      ordinal));
}

external_tool::PreparedExternalToolInvocation
makePrepared(llvm::StringRef runRoot, llvm::StringRef name,
             std::uint8_t digestByte) {
  return {(runRoot + "/" + name).str(), computeBlobDigest({digestByte})};
}

void prepare(ExecutionJournal &journal, const WorkUnitKey &key,
             const external_tool::PreparedExternalToolInvocation &prepared) {
  requireSuccess(journal.queue(key));
  requireSuccess(journal.markRunning(key));
  requireSuccess(journal.recordPrepared(key, prepared));
  requireSuccess(journal.beginPreparedExecution(key));
}

std::uint64_t unixNanosecondsNow() {
  const auto count = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
  if (count <= 0)
    fail("system clock did not produce a positive test timestamp");
  return static_cast<std::uint64_t>(count);
}

void testRecoveryAndTerminalAdmission(const ArtifactStore &store,
                                      llvm::StringRef runRoot) {
  Fixture fixture = makeFixture(store, 0x11, "loom.test.execution.build.v1");
  const WorkUnitKey runningKey = makeKey(2, 4);
  {
    ExecutionJournal journal =
        take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
    requireSuccess(journal.queue(runningKey));
    requireSuccess(journal.markRunning(runningKey));
  }

  ExecutionJournal recovered =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
  auto running = take(recovered.find(runningKey));
  if (!running || running->status != JournalWorkUnitStatus::Queued)
    fail("reopen did not return interrupted work to its stable queued key");
  const std::uint64_t recoveredActive = running->activeWallTimeNanoseconds();

  requireSuccess(recovered.markRunning(runningKey));
  const std::uint64_t terminalTime = unixNanosecondsNow();
  requireSuccess(recovered.markTerminal(runningKey,
                                        JournalWorkUnitStatus::Completed, 17,
                                        terminalTime, {fixture.source}));
  auto completed = take(recovered.find(runningKey));
  if (!completed || completed->status != JournalWorkUnitStatus::Completed ||
      completed->activeWallTimeNanoseconds() < recoveredActive + 17 ||
      completed->terminalUnixTimeNanoseconds != terminalTime ||
      completed->finalizedOutputs !=
          std::vector<ArtifactRootReference>{fixture.source})
    fail("terminal work record lost its exact observations or roots");

  llvm::Error overwrite =
      recovered.markTerminal(runningKey, JournalWorkUnitStatus::Failed, 17,
                             terminalTime, {fixture.source});
  requireErrorContains(std::move(overwrite), "cannot be overwritten");

  const WorkUnitKey preparedKey = makeKey(3, 1);
  requireSuccess(recovered.queue(preparedKey));
  requireSuccess(recovered.markRunning(preparedKey));
  const external_tool::PreparedExternalToolInvocation prepared{
      (runRoot + "/prepared").str(), computeBlobDigest({0x21, 0x22})};
  requireSuccess(recovered.recordPrepared(preparedKey, prepared));
  ExecutionJournal reopened =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
  auto retained = take(reopened.find(preparedKey));
  if (!retained || retained->status != JournalWorkUnitStatus::Prepared ||
      !retained->preparedInvocation ||
      retained->preparedInvocation->bundleRoot != prepared.bundleRoot ||
      retained->preparedInvocation->manifestDigest != prepared.manifestDigest)
    fail("reopen changed a prepared external attempt");

  requireSuccess(reopened.requestGracefulStop());
  if (!reopened.gracefulStopRequested())
    fail("graceful stop was not durably visible");
  ExecutionJournal stopped =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
  if (!stopped.gracefulStopRequested())
    fail("graceful stop did not survive reopen");
  requireSuccess(stopped.beginResume());
  if (stopped.gracefulStopRequested())
    fail("resume did not clear the durable graceful-stop flag");

  Fixture foreign = makeFixture(store, 0x12, "loom.test.execution.build.v1");
  auto rejected = openExecutionJournal(runRoot, foreign.closure, foreign.view);
  if (rejected)
    fail("journal accepted a different semantic run closure");
  requireErrorContains(rejected.takeError(), "another semantic run");
}

void testStrictAdmission(const ArtifactStore &store, llvm::StringRef runRoot) {
  Fixture fixture = makeFixture(store, 0x31, "loom.test.execution.build.v2");
  auto badDescriptor =
      WorkUnitDescriptorRef::get("bad owner", SchemaVersion{1, 0}, 0);
  if (badDescriptor)
    fail("work descriptor accepted noncanonical owner spelling");
  requireErrorContains(badDescriptor.takeError(), "canonical ASCII");

  {
    ExecutionJournal journal =
        take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
    auto unopened = journal.currentInvocationOccurrence();
    if (unopened)
      fail("journal exposed an occurrence before execution began");
    requireErrorContains(unopened.takeError(), "has not opened");

    const std::filesystem::path snapshot =
        std::filesystem::path(runRoot.str()) / "execution-journal.snapshot";
    const std::filesystem::path saved =
        std::filesystem::path(runRoot.str()) / "execution-journal.saved";
    std::error_code filesystemError;
    std::filesystem::rename(snapshot, saved, filesystemError);
    if (filesystemError ||
        !std::filesystem::create_directory(snapshot, filesystemError) ||
        filesystemError)
      fail("cannot prepare the failed occurrence publication fixture");
    requireErrorContains(journal.beginResume(), "publish snapshot atomically");
    auto unpublished = journal.currentInvocationOccurrence();
    if (unpublished)
      fail("failed snapshot publication exposed an invocation occurrence");
    requireErrorContains(unpublished.takeError(), "has not opened");
    if (!std::filesystem::remove(snapshot, filesystemError) || filesystemError)
      fail("cannot remove the failed occurrence publication fixture");
    std::filesystem::rename(saved, snapshot, filesystemError);
    if (filesystemError)
      fail("cannot restore the occurrence snapshot fixture");

    const std::filesystem::path lock =
        std::filesystem::path(runRoot.str()) / "execution-journal.lock";
    const std::filesystem::path savedLock =
        std::filesystem::path(runRoot.str()) / "execution-journal.saved-lock";
    std::filesystem::rename(lock, savedLock, filesystemError);
    std::ofstream(lock).close();
    requireErrorContains(journal.flush(), "lock inode changed");
    if (!std::filesystem::remove(lock, filesystemError) || filesystemError)
      fail("cannot remove the replacement journal lock");
    std::filesystem::rename(savedLock, lock, filesystemError);
    if (filesystemError)
      fail("cannot restore the owned journal lock");

    ExecutionJournal concurrent =
        take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
    requireSuccess(journal.beginResume());
    auto initialOccurrence = take(journal.currentInvocationOccurrence());
    if (initialOccurrence.first.occurrenceOrdinal != 0 ||
        initialOccurrence.second)
      fail("initial journal occurrence has incorrect resume provenance");
    requireErrorContains(concurrent.beginResume(), "already opened");
    if (take(concurrent.currentInvocationOccurrence()) != initialOccurrence)
      fail("journal aliases exposed different active occurrences");

    const pid_t child = ::fork();
    if (child < 0)
      fail("cannot fork the inherited-journal fixture");
    if (child == 0) {
      llvm::Error inherited = concurrent.beginResume();
      const std::string detail = llvm::toString(std::move(inherited));
      ::_exit(detail.find("inherited across a process fork") !=
                      std::string::npos
                  ? 0
                  : 1);
    }
    int childStatus = 0;
    if (::waitpid(child, &childStatus, 0) != child || !WIFEXITED(childStatus) ||
        WEXITSTATUS(childStatus) != 0)
      fail("forked process used an inherited journal handle");

    const WorkUnitKey key = makeKey(0, 0);
    llvm::Error unqueued = journal.markRunning(key);
    requireErrorContains(std::move(unqueued), "unqueued");
    requireSuccess(journal.queue(key));
    requireSuccess(journal.markRunning(key));
    llvm::Error nonterminal =
        journal.markTerminal(key, JournalWorkUnitStatus::Running, 1, 2, {});
    requireErrorContains(std::move(nonterminal), "terminal status");
    llvm::Error unordered =
        journal.markTerminal(key, JournalWorkUnitStatus::Completed, 1, 2,
                             {publish(store, 0x42), publish(store, 0x41)});
    requireErrorContains(std::move(unordered), "canonical and unique");

    const BlobDigest firstManifest = computeBlobDigest({0x32});
    std::filesystem::rename(snapshot, saved, filesystemError);
    if (filesystemError ||
        !std::filesystem::create_directory(snapshot, filesystemError) ||
        filesystemError)
      fail("cannot prepare the failed manifest commit fixture");
    requireErrorContains(journal.commitInvocationManifest(
                             initialOccurrence.first, firstManifest),
                         "publish snapshot atomically");
    if (!std::filesystem::remove(snapshot, filesystemError) || filesystemError)
      fail("cannot remove the failed manifest commit fixture");
    std::filesystem::rename(saved, snapshot, filesystemError);
    if (filesystemError)
      fail("cannot restore the manifest commit snapshot");
    requireSuccess(journal.commitInvocationManifest(initialOccurrence.first,
                                                    firstManifest));
    requireSuccess(concurrent.commitInvocationManifest(initialOccurrence.first,
                                                       firstManifest));
    requireErrorContains(journal.queue(makeKey(1, 0)), "immutable");
    requireErrorContains(
        journal.commitInvocationManifest(initialOccurrence.first,
                                         computeBlobDigest({0x33})),
        "already owns");
  }

  ExecutionJournal resumed =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
  requireSuccess(resumed.beginResume());
  auto resumedOccurrence = take(resumed.currentInvocationOccurrence());
  if (resumedOccurrence.first.occurrenceOrdinal != 1 ||
      !resumedOccurrence.second ||
      resumedOccurrence.second->occurrenceOrdinal != 0 ||
      resumedOccurrence.second->runKey != resumedOccurrence.first.runKey)
    fail("resumed journal occurrence lost its durable predecessor");
}

void testExternalToolWorkLedger(const ArtifactStore &store,
                                llvm::StringRef runRoot) {
  Fixture fixture =
      makeFixture(store, 0x51, "loom.test.execution.external_tool.v1");
  ExecutionJournal journal =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));

  const WorkUnitKey missKey = makeKey(4, 0);
  prepare(journal, missKey, makePrepared(runRoot, "miss", 0x61));
  requireSuccess(journal.recordPreparedExecutionInterval(
      missKey, 0, unixNanosecondsNow(),
      external_tool::ExternalToolInvocationExecutionObservation{
          0, external_tool::ExternalToolResultReusePolicy::AllowExactReuse,
          external_tool::ExternalToolResultCacheAvailability::Available,
          external_tool::ExternalToolResultCacheLookup::Miss,
          external_tool::ExternalToolResultCacheDiscard::Discarded,
          external_tool::ExternalToolResultCachePublication::Published, false,
          true}));

  const WorkUnitKey hitKey = makeKey(4, 1);
  prepare(journal, hitKey, makePrepared(runRoot, "hit", 0x62));
  requireSuccess(journal.recordPreparedExecutionInterval(
      hitKey, 0, unixNanosecondsNow(),
      external_tool::ExternalToolInvocationExecutionObservation{
          0, external_tool::ExternalToolResultReusePolicy::AllowExactReuse,
          external_tool::ExternalToolResultCacheAvailability::Available,
          external_tool::ExternalToolResultCacheLookup::Hit,
          external_tool::ExternalToolResultCacheDiscard::NotAttempted,
          external_tool::ExternalToolResultCachePublication::NotAttempted, true,
          false}));

  ExecutionJournal reopened =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
  const auto miss = take(reopened.find(missKey));
  const auto hit = take(reopened.find(hitKey));
  if (!miss ||
      miss->externalToolWork !=
          ExternalToolWorkLedger{1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0})
    fail("reopen changed external-tool cache-miss work accounting");
  if (!hit ||
      hit->externalToolWork !=
          ExternalToolWorkLedger{1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0})
    fail("reopen changed external-tool cache-hit work accounting");
  const InvocationExternalToolWorkLedger summary =
      take(reopened.externalToolWorkLedger());
  if (summary.planNodes().size() != 1 ||
      summary.planNodes().front().planNodeOrdinal != 4 ||
      summary.total() !=
          ExternalToolWorkLedger{2, 2, 1, 1, 0, 2, 0, 1, 1, 1, 1, 0, 1, 0})
    fail("Journal external-tool work did not aggregate by plan node");
}

void rewriteSnapshotAsHistorical(llvm::StringRef runRoot,
                                 std::uint8_t targetMinor) {
  const std::filesystem::path path =
      std::filesystem::path(runRoot.str()) / "execution-journal.snapshot";
  std::ifstream input(path, std::ios::binary);
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
  if (input.bad() || bytes.size() < sizeof(std::uint64_t))
    fail("cannot read the generated journal snapshot");
  std::uint64_t identityBytes = 0;
  for (std::size_t index = 0; index != sizeof(std::uint64_t); ++index)
    identityBytes = (identityBytes << 8) | bytes[index];
  const std::size_t minorOffset = sizeof(std::uint64_t) +
                                  static_cast<std::size_t>(identityBytes) +
                                  sizeof(std::uint32_t);
  if (minorOffset + sizeof(std::uint32_t) > bytes.size() ||
      bytes[minorOffset + 3] != 4)
    fail("generated journal snapshot has an unexpected schema version");
  if (targetMinor < 1 || targetMinor > 3)
    fail("historical journal fixture requested an unsupported version");
  bytes[minorOffset + 3] = targetMinor;
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  if (!output)
    fail("cannot write the legacy journal snapshot fixture");
}

void testHistoricalSnapshotRejection(const ArtifactStore &store,
                                     llvm::StringRef runRoot,
                                     std::uint8_t targetMinor) {
  Fixture fixture =
      makeFixture(store, 0x71, "loom.test.execution.legacy_external_tool.v1");
  const WorkUnitKey key = makeKey(5, 0);
  {
    ExecutionJournal journal =
        take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
    requireSuccess(journal.queue(key));
    requireSuccess(journal.markRunning(key));
    requireSuccess(
        journal.recordPrepared(key, makePrepared(runRoot, "legacy", 0x72)));
  }
  rewriteSnapshotAsHistorical(runRoot, targetMinor);
  auto adopted = openExecutionJournal(runRoot, fixture.closure, fixture.view);
  if (adopted)
    fail("historical journal was assigned a fabricated occurrence ordinal");
  requireErrorContains(adopted.takeError(),
                       "predates committed invocation manifest ownership");
}

} // namespace

int main() {
  TemporaryDirectory directory;
  const std::string storeRoot = directory.makeDirectory("artifacts");
  const std::string recoveryRun = directory.makeDirectory("recovery-run");
  const std::string admissionRun = directory.makeDirectory("admission-run");
  const std::string externalToolRun =
      directory.makeDirectory("external-tool-run");
  const std::string legacyRun = directory.makeDirectory("legacy-run");
  const std::string externalLedgerRun =
      directory.makeDirectory("external-ledger-run");
  const std::string occurrenceRun = directory.makeDirectory("occurrence-run");
  ArtifactStore store(storeRoot);
  testRecoveryAndTerminalAdmission(store, recoveryRun);
  testStrictAdmission(store, admissionRun);
  testExternalToolWorkLedger(store, externalToolRun);
  testHistoricalSnapshotRejection(store, legacyRun, 1);
  testHistoricalSnapshotRejection(store, externalLedgerRun, 2);
  testHistoricalSnapshotRejection(store, occurrenceRun, 3);
  return 0;
}
