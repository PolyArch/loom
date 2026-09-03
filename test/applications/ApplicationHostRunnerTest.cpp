#include "Application/HostRunner.h"
#include "Application/Manifest.h"
#include "Common/BlobDigest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#include <sys/types.h>
#include <unistd.h>

#ifndef LOOM_TEST_REPOSITORY_ROOT
#define LOOM_TEST_REPOSITORY_ROOT ""
#endif

namespace {

using namespace loom::application;

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "application host runner test failure: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

class TemporaryTree final {
public:
  explicit TemporaryTree(llvm::StringRef path)
      : root_(std::filesystem::absolute(path.str())) {
    std::error_code error;
    std::filesystem::remove_all(root_, error);
    error.clear();
    std::filesystem::create_directories(root_, error);
    if (error)
      fail("cannot create temporary tree: " + error.message());
  }

  ~TemporaryTree() {
    std::error_code ignored;
    std::filesystem::remove_all(root_, ignored);
  }

  std::filesystem::path path(llvm::StringRef relative = {}) const {
    return relative.empty() ? root_ : root_ / relative.str();
  }

private:
  std::filesystem::path root_;
};

void writeFile(const std::filesystem::path &path, llvm::StringRef contents) {
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  if (error)
    fail("cannot create fixture directory: " + error.message());
  llvm::raw_fd_ostream output(path.string(), error, llvm::sys::fs::OF_None);
  if (error)
    fail("cannot open fixture file: " + error.message());
  output << contents;
  output.close();
  if (output.has_error())
    fail("cannot write fixture file");
}

std::string readFile(const std::filesystem::path &path) {
  auto buffer = llvm::MemoryBuffer::getFile(path.string(), false, false);
  if (!buffer)
    fail("cannot read fixture file: " + buffer.getError().message());
  return (*buffer)->getBuffer().str();
}

std::string digest(llvm::StringRef bytes) {
  return loom::formatBlobDigestHex(
      loom::computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
          reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size())));
}

std::string fixtureManifest(llvm::StringRef oracleKind,
                            std::uint64_t deadlineMilliseconds,
                            std::uint64_t warmupSamples,
                            std::uint64_t measuredSamples) {
  return R"json({
  "schema": "loom.application_portfolio",
  "version": "4.0",
  "applications": [{
    "identity": "host-fixture",
    "source": {"kind": "repository", "root": "source"},
    "build": {
      "entry": "main.c",
      "language": "c",
      "sources": ["main.c"],
      "compiler_options": ["-std=c11", "-O0"],
      "link_options": [],
      "operator_protocol_symbols": [],
      "product_execution": null
    },
    "cached_inputs": [],
    "inputs": [{
      "name": "fixture",
      "workload": "host-fixture-workload",
      "runtime_input": "host-fixture-input",
      "cached_inputs": [],
      "compiler_options": [],
      "oracle": {"kind": ")json" +
         oracleKind.str() + R"json(", "entry": "expected.txt", "sha256": ")json" +
         digest("expected\n") + R"json(", "encoding": "utf8"},
      "profile": {
        "warmup_samples": )json" +
         std::to_string(warmupSamples) + R"json(,
        "measured_samples": )json" +
         std::to_string(measuredSamples) + R"json(,
        "oracle_coverage": "all_measured_samples",
        "deadline_milliseconds": )json" +
         std::to_string(deadlineMilliseconds) + R"json(
      }
    }],
    "selection_inputs": {"smoke": ["fixture"]}
  }]
})json";
}

std::string cachedFixtureManifest(llvm::StringRef digestHex,
                                  std::uint64_t warmupSamples,
                                  std::uint64_t measuredSamples) {
  return R"json({
  "schema": "loom.application_portfolio",
  "version": "4.0",
  "applications": [{
    "identity": "host-fixture",
    "source": {"kind": "repository", "root": "source"},
    "build": {
      "entry": "main.c",
      "language": "c",
      "sources": ["main.c"],
      "compiler_options": ["-std=c11", "-O0"],
      "link_options": [],
      "operator_protocol_symbols": [],
      "product_execution": null
    },
    "cached_inputs": [{
      "logical_name": "payload",
      "path": "payload-link.bin",
      "sha256": ")json" +
         digestHex.str() + R"json("
    }],
    "inputs": [{
      "name": "fixture",
      "workload": "host-fixture-workload",
      "runtime_input": "host-fixture-input",
      "cached_inputs": ["payload"],
      "compiler_options": [],
      "oracle": {"kind": "exact", "entry": "expected.txt", "sha256": ")json" +
         digest("expected\n") + R"json(", "encoding": "utf8"},
      "profile": {
        "warmup_samples": )json" +
         std::to_string(warmupSamples) + R"json(,
        "measured_samples": )json" +
         std::to_string(measuredSamples) + R"json(,
        "oracle_coverage": "all_measured_samples",
        "deadline_milliseconds": 500
      }
    }],
    "selection_inputs": {"smoke": ["fixture"]}
  }]
})json";
}

ApplicationHostRunReport
runFixture(const TemporaryTree &tree, llvm::StringRef compiler,
           llvm::StringRef source, llvm::StringRef oracleKind = "exact",
           std::uint64_t deadlineMilliseconds = 500,
           std::uint64_t warmupSamples = 0, std::uint64_t measuredSamples = 1) {
  writeFile(tree.path("source/main.c"), source);
  writeFile(tree.path("expected.txt"), "expected\n");
  ApplicationManifest manifest = take(parseApplicationManifest(fixtureManifest(
      oracleKind, deadlineMilliseconds, warmupSamples, measuredSamples)));
  return take(runApplicationInputOnHost(
      manifest,
      ApplicationHostRunRequest{"host-fixture", "fixture", tree.path().string(),
                                std::nullopt, compiler.str()}));
}

std::string serialize(const ApplicationHostRunReport &report) {
  std::string text;
  llvm::raw_string_ostream output(text);
  writeApplicationHostRunReportJson(output, report);
  output.flush();
  return text;
}

std::string serialize(const ApplicationHostSelectionRunReport &report) {
  std::string text;
  llvm::raw_string_ostream output(text);
  writeApplicationHostSelectionRunReportJson(output, report);
  output.flush();
  return text;
}

void requireProjectedStatuses(const ApplicationHostRunReport &report,
                              llvm::StringRef outcome,
                              llvm::StringRef compileStatus,
                              llvm::StringRef executionStatus,
                              llvm::StringRef oracleStatus) {
  auto parsed = take(llvm::json::parse(serialize(report)));
  const llvm::json::Object *root = parsed.getAsObject();
  const llvm::json::Object *compile =
      root ? root->getObject("compile") : nullptr;
  const llvm::json::Object *execution =
      root ? root->getObject("execution") : nullptr;
  const llvm::json::Object *oracle =
      root ? root->getObject("oracle_result") : nullptr;
  require(root && root->getString("outcome") == outcome && compile &&
              compile->getString("status") == compileStatus && execution &&
              execution->getString("status") == executionStatus && oracle &&
              oracle->getString("status") == oracleStatus,
          "typed outcome and projected statuses diverged");
}

void exerciseTypedOutcomes(const TemporaryTree &tree,
                           llvm::StringRef compiler) {
  ApplicationHostRunReport success = runFixture(
      tree, compiler,
      "#warning host compiler diagnostic\n"
      "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
      "int main(void) { const char *locale = getenv(\"LC_ALL\"); "
      "if (!locale || strcmp(locale, \"C\") != 0 || getchar() != EOF) "
      "return 2; fputs(\"host runtime diagnostic\\n\", stderr); "
      "puts(\"expected\"); return 0; }\n");
  require(success.outcome == ApplicationHostRunOutcome::Succeeded &&
              success.oracleStatus == ApplicationHostOracleStatus::Matched &&
              success.compileExitStatus == 0 &&
              success.executionExitStatus == 0 &&
              success.hostWallTimeNanoseconds.has_value() &&
              llvm::StringRef(success.diagnostic)
                  .contains("host compiler diagnostic") &&
              llvm::StringRef(success.diagnostic)
                  .contains("host runtime diagnostic"),
          "explicit compiler host success lost a typed status");
  requireProjectedStatuses(success, "succeeded", "succeeded", "succeeded",
                           "matched");

  ApplicationHostRunReport compileFailure =
      runFixture(tree, compiler, "int main(void) { this is not C; }\n");
  require(compileFailure.outcome == ApplicationHostRunOutcome::CompileFailure &&
              compileFailure.compileExitStatus.has_value() &&
              *compileFailure.compileExitStatus != 0 &&
              !compileFailure.executionExitStatus &&
              compileFailure.oracleStatus ==
                  ApplicationHostOracleStatus::NotChecked,
          "compile failure was not preserved as a typed outcome");
  requireProjectedStatuses(compileFailure, "compile_failure", "failed",
                           "not_run", "not_checked");

  ApplicationHostRunReport executionFailure =
      runFixture(tree, compiler, "int main(void) { return 7; }\n");
  require(executionFailure.outcome ==
                  ApplicationHostRunOutcome::ExecutionFailure &&
              executionFailure.compileExitStatus == 0 &&
              executionFailure.executionExitStatus == 7 &&
              executionFailure.oracleStatus ==
                  ApplicationHostOracleStatus::NotChecked,
          "nonzero host exit was not preserved as execution failure");
  requireProjectedStatuses(executionFailure, "execution_failure", "succeeded",
                           "failed", "not_checked");

  ApplicationHostRunReport mismatch =
      runFixture(tree, compiler,
                 "#include <stdio.h>\nint main(void) { "
                 "fputs(\"mismatch runtime diagnostic\\n\", stderr); "
                 "puts(\"actual\"); return 0; }\n");
  require(mismatch.outcome == ApplicationHostRunOutcome::OracleMismatch &&
              mismatch.executionExitStatus == 0 &&
              mismatch.oracleStatus ==
                  ApplicationHostOracleStatus::Mismatched &&
              llvm::StringRef(mismatch.diagnostic)
                  .contains("mismatch runtime diagnostic") &&
              llvm::StringRef(mismatch.diagnostic).contains("stdout differs"),
          "exact stdout mismatch was not preserved as a typed outcome");
  requireProjectedStatuses(mismatch, "oracle_mismatch", "succeeded",
                           "succeeded", "mismatched");

  ApplicationHostRunReport timeout =
      runFixture(tree, compiler,
                 "int main(void) { volatile unsigned long value = 0; "
                 "for (;;) { ++value; } }\n",
                 "exact", 10);
  require(timeout.outcome == ApplicationHostRunOutcome::Timeout &&
              timeout.compileExitStatus == 0 && !timeout.executionExitStatus &&
              timeout.hostWallTimeNanoseconds.has_value() &&
              timeout.oracleStatus == ApplicationHostOracleStatus::NotChecked,
          "profile deadline was not preserved as a typed timeout");
  requireProjectedStatuses(timeout, "timeout", "succeeded", "timed_out",
                           "not_checked");

  ApplicationHostRunReport unsupported = runFixture(
      tree, compiler, "int main(void) { return 0; }\n", "typed_invariant");
  require(unsupported.outcome == ApplicationHostRunOutcome::UnsupportedOracle &&
              unsupported.oracleStatus ==
                  ApplicationHostOracleStatus::Unsupported &&
              !unsupported.compileExitStatus &&
              !unsupported.executionExitStatus,
          "typed invariant was not preserved as unsupported host oracle");
  requireProjectedStatuses(unsupported, "unsupported_oracle", "not_run",
                           "not_run", "unsupported");

  ApplicationHostRunReport unsupportedProfile = runFixture(
      tree, compiler, "int main(void) { return 0; }\n", "exact", 500, 1, 2);
  require(unsupportedProfile.outcome ==
                  ApplicationHostRunOutcome::UnsupportedProfile &&
              unsupportedProfile.oracleStatus ==
                  ApplicationHostOracleStatus::NotChecked &&
              !unsupportedProfile.compileExitStatus &&
              !unsupportedProfile.executionExitStatus,
          "no-cache sample counts were not rejected as unsupported profile");
  requireProjectedStatuses(unsupportedProfile, "unsupported_profile", "not_run",
                           "not_run", "not_checked");

  ApplicationHostRunReport signaled = runFixture(
      tree, compiler,
      "#include <signal.h>\nint main(void) { raise(SIGTERM); return 0; }\n");
  require(signaled.outcome == ApplicationHostRunOutcome::ExecutionFailure &&
              signaled.compileExitStatus == 0 && !signaled.executionExitStatus,
          "signal termination was projected as a host exit status");

  const std::filesystem::path crashingCompiler = tree.path("compiler-crash");
  writeFile(crashingCompiler, "#!/bin/sh\nkill -TERM $$\n");
  std::error_code permissionError;
  std::filesystem::permissions(
      crashingCompiler,
      std::filesystem::perms::owner_read | std::filesystem::perms::owner_write |
          std::filesystem::perms::owner_exec,
      std::filesystem::perm_options::replace, permissionError);
  if (permissionError)
    fail("cannot make compiler fixture executable: " +
         permissionError.message());
  ApplicationHostRunReport compilerSignaled = runFixture(
      tree, crashingCompiler.string(), "int main(void) { return 0; }\n");
  require(compilerSignaled.outcome ==
                  ApplicationHostRunOutcome::CompileFailure &&
              !compilerSignaled.compileExitStatus &&
              !compilerSignaled.executionExitStatus,
          "compiler signal termination was projected as an exit status");
}

void exerciseSelectionRun(const TemporaryTree &tree, llvm::StringRef compiler) {
  writeFile(tree.path("source/main.c"),
            "#include <stdio.h>\nint main(void) { puts(\"expected\"); "
            "return 0; }\n");
  writeFile(tree.path("expected.txt"), "expected\n");
  ApplicationManifest manifest =
      take(parseApplicationManifest(fixtureManifest("exact", 500, 0, 1)));
  ApplicationHostSelectionRunReport report = take(runApplicationSelectionOnHost(
      manifest, ApplicationHostSelectionRunRequest{
                    ExecutionSelection::Smoke, tree.path().string(),
                    std::nullopt, compiler.str()}));
  require(applicationHostSelectionRunSucceeded(report) &&
              report.reports.size() == 1 &&
              report.reports.front().selection.applicationIdentity ==
                  "host-fixture",
          "host selection runner did not execute its exact manifest row");

  auto parsed = take(llvm::json::parse(serialize(report)));
  const llvm::json::Object *root = parsed.getAsObject();
  const llvm::json::Array *reports = root ? root->getArray("reports") : nullptr;
  require(root &&
              root->getString("schema") ==
                  ApplicationHostSelectionRunReport::schemaIdentity &&
              root->getString("execution_selection") == "smoke" && reports &&
              reports->size() == 1,
          "host selection report lost its tier or member report");

  ApplicationHostSelectionRunReport empty = take(runApplicationSelectionOnHost(
      manifest, ApplicationHostSelectionRunRequest{
                    ExecutionSelection::ScaleEda, tree.path().string(),
                    std::nullopt, compiler.str()}));
  require(empty.reports.empty() && !applicationHostSelectionRunSucceeded(empty),
          "an empty manifest tier was reported as a successful run");
}

void exerciseSourceUnavailable(const TemporaryTree &tree) {
  writeFile(tree.path("source/main.c"), "int main(void) { return 0; }\n");
  writeFile(tree.path("expected.txt"), "expected\n");
  ApplicationManifest manifest = take(parseApplicationManifest(
      cachedFixtureManifest(std::string(64, '0'), 1, 2)));
  ApplicationHostRunReport report = take(runApplicationInputOnHost(
      manifest,
      ApplicationHostRunRequest{"host-fixture", "fixture", tree.path().string(),
                                std::nullopt, std::nullopt}));
  require(report.outcome == ApplicationHostRunOutcome::SourceUnavailable &&
              report.unavailableSource &&
              report.unavailableSource->reason ==
                  SourceUnavailableReason::CacheRoot &&
              !report.compilerExecutable && !report.compileExitStatus &&
              !report.executionExitStatus,
          "missing cache root was not preserved as source unavailable");
  requireProjectedStatuses(report, "source_unavailable", "not_run", "not_run",
                           "not_checked");

  auto parsed = take(llvm::json::parse(serialize(report)));
  const llvm::json::Object *root = parsed.getAsObject();
  const llvm::json::Object *admission =
      root ? root->getObject("source_admission") : nullptr;
  require(admission && admission->getString("status") == "unavailable" &&
              admission->getString("reason") == "cache_root" &&
              admission->getString("path") == "",
          "source unavailable JSON projection changed");
}

void exerciseAdmittedCacheAbi(const TemporaryTree &tree,
                              llvm::StringRef compiler) {
  const llvm::StringRef payload = "admitted payload\n";
  const std::filesystem::path cacheRoot = tree.path("cache");
  const std::filesystem::path payloadPath = cacheRoot / "payload.bin";
  const std::filesystem::path payloadLink = cacheRoot / "payload-link.bin";
  writeFile(payloadPath, payload);
  std::error_code linkError;
  std::filesystem::remove(payloadLink, linkError);
  linkError.clear();
  std::filesystem::create_symlink("payload.bin", payloadLink, linkError);
  if (linkError)
    fail("cannot create cached-input symlink: " + linkError.message());

  const std::string admittedPath =
      std::filesystem::canonical(payloadPath, linkError).string();
  if (linkError)
    fail("cannot canonicalize cached-input fixture: " + linkError.message());
  const std::string scratchPrefix = tree.path("temp").string() + "/";
  const std::string source =
      "#include <stdio.h>\n#include <string.h>\n"
      "int main(int argc, char **argv) { if (argc != 4) return 3; "
      "if (strncmp(argv[0], \"" +
      scratchPrefix + "\", " + std::to_string(scratchPrefix.size()) +
      ") != 0) return 4; printf(\"%s\\n%s\\n%s\\n\", argv[1], argv[2], "
      "argv[3]); return 0; }\n";
  writeFile(tree.path("source/main.c"), source);
  writeFile(tree.path("expected.txt"), admittedPath + "\n2\n3\n");
  ApplicationManifest manifest = take(
      parseApplicationManifest(cachedFixtureManifest(digest(payload), 2, 3)));
  ApplicationHostRunReport report = take(runApplicationInputOnHost(
      manifest,
      ApplicationHostRunRequest{"host-fixture", "fixture", tree.path().string(),
                                cacheRoot.string(), compiler.str()}));
  require(report.outcome == ApplicationHostRunOutcome::Succeeded &&
              report.oracleStatus == ApplicationHostOracleStatus::Matched &&
              report.executionExitStatus == 0,
          "admitted cache path and sample-count ABI changed");
}

void exerciseDescendantContainment(const TemporaryTree &tree,
                                   llvm::StringRef compiler) {
  const std::filesystem::path pidPath = tree.path("descendant.pid");
  const std::string source =
      "#include <stdio.h>\n#include <sys/types.h>\n#include <unistd.h>\n"
      "int main(void) { pid_t child = fork(); if (child < 0) return 2; "
      "if (child == 0) { for (;;) pause(); } FILE *file = fopen(\"" +
      pidPath.string() +
      "\", \"w\"); if (!file) return 3; fprintf(file, \"%ld\\n\", "
      "(long)child); if (fclose(file) != 0) return 4; return 0; }\n";
  ApplicationHostRunReport report = runFixture(tree, compiler, source);
  require(report.outcome == ApplicationHostRunOutcome::ExecutionFailure &&
              report.executionExitStatus == 0 &&
              llvm::StringRef(report.diagnostic).contains("left descendants"),
          "leader exit did not reject and terminate its process group");

  const long child = std::stol(readFile(pidPath));
  bool childExists = true;
  for (unsigned attempt = 0; attempt != 100; ++attempt) {
    errno = 0;
    if (::kill(static_cast<pid_t>(child), 0) != 0 && errno == ESRCH) {
      childExists = false;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  require(!childExists, "host descendant survived process-group cleanup");
}

void exerciseTinyMl(llvm::StringRef manifestPath,
                    llvm::StringRef repositoryRoot,
                    llvm::StringRef sharedRepositoryRoot) {
  ApplicationManifest manifest = take(loadApplicationManifest(manifestPath));
  ApplicationHostRunReport report = take(runApplicationInputOnHost(
      manifest,
      ApplicationHostRunRequest{"mlperf-tiny-anomaly-detection", "smoke",
                                repositoryRoot.str(),
                                sharedRepositoryRoot.str(), std::nullopt}));
  require(applicationHostRunSucceeded(report) &&
              report.oracleStatus == ApplicationHostOracleStatus::Matched &&
              report.compileExitStatus == 0 &&
              report.executionExitStatus == 0 &&
              report.hostWallTimeNanoseconds.has_value(),
          "real TinyML host selection did not pass its exact oracle");
  require(report.selection.cachedInputs.size() == 2 &&
              report.selection.cachedInputs[0].logicalName == "model" &&
              report.selection.cachedInputs[1].logicalName == "smoke-dataset" &&
              report.selection.input.profile.warmupSamples == 1 &&
              report.selection.input.profile.measuredSamples == 4 &&
              report.selection.input.profile.deadlineMilliseconds == 10000,
          "real TinyML host ABI selection changed");

  const std::string first = serialize(report);
  require(first == serialize(report),
          "one host report does not serialize deterministically");
  auto parsed = take(llvm::json::parse(first));
  const llvm::json::Object *root = parsed.getAsObject();
  require(root &&
              root->getString("schema") ==
                  ApplicationHostRunReport::schemaIdentity &&
              root->getString("version") ==
                  ApplicationHostRunReport::schemaVersion &&
              root->getString("outcome") == "succeeded",
          "host report schema or outcome changed");
  const llvm::json::Object *selection = root->getObject("selection");
  const llvm::json::Object *profile = root->getObject("profile");
  const llvm::json::Object *execution = root->getObject("execution");
  const llvm::json::Object *oracle = root->getObject("oracle_result");
  require(selection &&
              selection->getString("application_identity") ==
                  "mlperf-tiny-anomaly-detection" &&
              selection->getString("input_name") == "smoke" && profile &&
              profile->getInteger("warmup_samples") == 1 &&
              profile->getInteger("measured_samples") == 4 &&
              profile->getInteger("deadline_milliseconds") == 10000 &&
              execution && execution->getString("status") == "succeeded" &&
              execution->getInteger("exit_status") == 0 &&
              execution->getInteger("host_wall_time_nanoseconds") && oracle &&
              oracle->getString("status") == "matched",
          "host report lost selection, profile, exit, wall, or oracle status");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4)
    fail("expected <temporary-directory> <manifest> <repository-root>");
  llvm::ErrorOr<std::string> compiler = llvm::sys::findProgramByName("clang");
  if (!compiler)
    fail("clang is required by the application host runner anchor");
  TemporaryTree tree(argv[1]);
  exerciseTypedOutcomes(tree, *compiler);
  exerciseSelectionRun(tree, *compiler);
  exerciseSourceUnavailable(tree);
  exerciseAdmittedCacheAbi(tree, *compiler);
  exerciseDescendantContainment(tree, *compiler);
  const llvm::StringRef configuredSharedRoot = LOOM_TEST_REPOSITORY_ROOT;
  exerciseTinyMl(argv[2], argv[3],
                 configuredSharedRoot.empty() ? llvm::StringRef(argv[3])
                                              : configuredSharedRoot);
  return EXIT_SUCCESS;
}
