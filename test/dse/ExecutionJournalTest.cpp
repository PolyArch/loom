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
#include <iostream>
#include <string>
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
    fail("expected error containing '" + needle.str() + "', got: " +
         message);
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
  DseRunClosure closure = take(DseRunClosure::get(
      take(DseProducerSemanticBuildIdentity::get(producer)), {source}, config,
      {}, store));
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
  const std::uint64_t recoveredActive =
      running->activeWallTimeNanoseconds();

  requireSuccess(recovered.markRunning(runningKey));
  const std::uint64_t terminalTime = unixNanosecondsNow();
  requireSuccess(recovered.markTerminal(
      runningKey, JournalWorkUnitStatus::Completed, 17, terminalTime,
      {fixture.source}));
  auto completed = take(recovered.find(runningKey));
  if (!completed || completed->status != JournalWorkUnitStatus::Completed ||
      completed->activeWallTimeNanoseconds() < recoveredActive + 17 ||
      completed->terminalUnixTimeNanoseconds != terminalTime ||
      completed->finalizedOutputs !=
          std::vector<ArtifactRootReference>{fixture.source})
    fail("terminal work record lost its exact observations or roots");

  llvm::Error overwrite = recovered.markTerminal(
      runningKey, JournalWorkUnitStatus::Failed, 17, terminalTime,
      {fixture.source});
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

  Fixture foreign =
      makeFixture(store, 0x12, "loom.test.execution.build.v1");
  auto rejected = openExecutionJournal(runRoot, foreign.closure, foreign.view);
  if (rejected)
    fail("journal accepted a different semantic run closure");
  requireErrorContains(rejected.takeError(), "another semantic run");
}

void testStrictAdmission(const ArtifactStore &store,
                         llvm::StringRef runRoot) {
  Fixture fixture = makeFixture(store, 0x31, "loom.test.execution.build.v2");
  auto badDescriptor =
      WorkUnitDescriptorRef::get("bad owner", SchemaVersion{1, 0}, 0);
  if (badDescriptor)
    fail("work descriptor accepted noncanonical owner spelling");
  requireErrorContains(badDescriptor.takeError(), "canonical ASCII");

  ExecutionJournal journal =
      take(openExecutionJournal(runRoot, fixture.closure, fixture.view));
  const WorkUnitKey key = makeKey(0, 0);
  llvm::Error unqueued = journal.markRunning(key);
  requireErrorContains(std::move(unqueued), "unqueued");
  requireSuccess(journal.queue(key));
  requireSuccess(journal.markRunning(key));
  llvm::Error nonterminal = journal.markTerminal(
      key, JournalWorkUnitStatus::Running, 1, 2, {});
  requireErrorContains(std::move(nonterminal), "terminal status");
  llvm::Error unordered = journal.markTerminal(
      key, JournalWorkUnitStatus::Completed, 1, 2,
      {publish(store, 0x42), publish(store, 0x41)});
  requireErrorContains(std::move(unordered), "canonical and unique");
}

} // namespace

int main() {
  TemporaryDirectory directory;
  const std::string storeRoot = directory.makeDirectory("artifacts");
  const std::string recoveryRun = directory.makeDirectory("recovery-run");
  const std::string admissionRun = directory.makeDirectory("admission-run");
  ArtifactStore store(storeRoot);
  testRecoveryAndTerminalAdmission(store, recoveryRun);
  testStrictAdmission(store, admissionRun);
  return 0;
}
