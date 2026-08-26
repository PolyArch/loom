#include "ExternalTool/InvocationBundle.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <variant>

using namespace loom::external_tool;

namespace {

constexpr int kFixtureToolExitCode = 93;

[[noreturn]] void fail(const char *test, const std::string &message) {
  std::cerr << test << ": " << message << '\n';
  std::exit(1);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

void requireSuccess(const char *test, llvm::Error error) {
  if (error)
    fail(test, llvm::toString(std::move(error)));
}

template <typename T> T take(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void requireFailure(const char *test, llvm::Expected<T> value,
                    const std::string &message) {
  if (value)
    fail(test, message);
  llvm::consumeError(value.takeError());
}

std::string readText(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    fail(__func__, "could not open " + path.string());
  std::ostringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

void writeText(const std::filesystem::path &path, llvm::StringRef contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream)
    fail(__func__, "could not open " + path.string());
  stream << contents.str();
}

void writeExecutable(const std::filesystem::path &path,
                     llvm::StringRef contents) {
  writeText(path, contents);
  std::filesystem::permissions(path,
                               std::filesystem::perms::owner_read |
                                   std::filesystem::perms::owner_write |
                                   std::filesystem::perms::owner_exec |
                                   std::filesystem::perms::group_read |
                                   std::filesystem::perms::group_exec,
                               std::filesystem::perm_options::replace);
}

loom::BlobDigest blobDigest(llvm::StringRef contents) {
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(contents.data());
  return loom::computeBlobDigest(
      llvm::ArrayRef<std::uint8_t>(bytes, contents.size()));
}

ExternalToolInvocationBundleSpec
baseSpec(const std::filesystem::path &tool, llvm::StringRef output,
         llvm::StringRef value = "receipt-output") {
  ExternalToolInvocationBundleSpec spec;
  spec.semanticContract.providerIdentity = "receipt_fixture@1";
  spec.semanticContract.semanticClosure =
      SemanticInvocationClosure(CandidateGeneratorInvocationClosure{
          {0x01}, {0x02}, blobDigest("receipt-binding").bytes()});
  spec.semanticContract.resultImporterIdentity = std::string(64, 'a');
  spec.tool = ResolvedToolBinding{"receipt_fixture",
                                  ToolBindingSource::Explicit,
                                  tool.string(),
                                  "Receipt Fixture 1.0",
                                  {},
                                  {},
                                  std::nullopt,
                                  std::nullopt};
  spec.toolVersionProbe = ToolVersionProbe{{"--version"}, "Receipt Fixture"};
  spec.runtime.kind = InvocationRuntimeKind::Host;
  spec.commands = {{tool.string(), "run", value.str(), output.str()}};
  spec.declaredOutputs = {output.str()};
  return spec;
}

ExternalToolInvocationImportExpectation
importExpectation(const ExternalToolInvocationBundleSpec &spec) {
  return ExternalToolInvocationImportExpectation{
      spec.semanticContract, {}, {}, {}, spec.declaredOutputs};
}

PreparedExternalToolInvocation
prepare(const char *test, const std::filesystem::path &root,
        llvm::StringRef name, const ExternalToolInvocationBundleSpec &spec) {
  return take(test, finalizeExternalToolInvocationBundle(
                        (root / name.str()).string(), spec));
}

void interleavedGenerationRejectsOldCompletionAndReceipt(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  const std::string output = "outputs/interleaved.txt";
  const ExternalToolInvocationBundleSpec spec = baseSpec(tool, output);
  const PreparedExternalToolInvocation prepared =
      prepare(__func__, root, "interleaved-generation", spec);
  const ExternalToolInvocationExecutionObservation oldExecution = take(
      __func__, executeExternalToolInvocationBundleObserved(
                    prepared, {}, ExternalToolResultReusePolicy::RequireFresh));
  const std::filesystem::path completion =
      root / "interleaved-generation" / "outputs" / "completion.json";
  const std::string oldCompletion = readText(completion);

  const loom::BlobDigest newToken =
      take(__func__, beginExternalToolInvocationAttempt(prepared));
  require(__func__, newToken != oldExecution.attemptToken,
          "a new execution generation reused the prior attempt token");
  writeText(completion, oldCompletion);

  requireFailure(
      __func__,
      importExternalToolInvocationAttempt(prepared, importExpectation(spec)),
      "raw import accepted a completion from an old generation");
  ExternalToolInvocationExecutionObservation reboundExecution = oldExecution;
  reboundExecution.attemptToken = newToken;
  requireSuccess(__func__, validateExternalToolInvocationExecutionObservation(
                               prepared, reboundExecution));
  requireFailure(
      __func__,
      importExternalToolInvocationAttempt(prepared, importExpectation(spec),
                                          reboundExecution),
      "receipt-aware import accepted a rebound receipt from an old generation");
}

void outputMutationCannotEscapeReceiptAwareImport(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  const std::string output = "outputs/output-mutation.txt";
  const ExternalToolInvocationBundleSpec spec =
      baseSpec(tool, output, "original-output");
  const PreparedExternalToolInvocation prepared =
      prepare(__func__, root, "output-mutation", spec);
  const ExternalToolInvocationExecutionObservation execution = take(
      __func__, executeExternalToolInvocationBundleObserved(
                    prepared, {}, ExternalToolResultReusePolicy::RequireFresh));
  writeText(root / "output-mutation" / output, "mutated-output");

  requireFailure(
      __func__,
      importExternalToolInvocationBundle(prepared, importExpectation(spec),
                                         execution),
      "receipt-aware import accepted output bytes changed after execution");
}

void completionReplacementInvalidatesReceipt(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  const std::string output = "outputs/completion-replacement.txt";
  const ExternalToolInvocationBundleSpec spec = baseSpec(tool, output);
  const PreparedExternalToolInvocation prepared =
      prepare(__func__, root, "completion-replacement", spec);
  const ExternalToolInvocationExecutionObservation execution = take(
      __func__, executeExternalToolInvocationBundleObserved(
                    prepared, {}, ExternalToolResultReusePolicy::RequireFresh));
  const std::filesystem::path completion =
      root / "completion-replacement" / "outputs" / "completion.json";
  std::string replacement = readText(completion);
  const std::string success = "\"status\":\"success\",\"exit_code\":0";
  const std::string failure = "\"status\":\"tool_exit\",\"exit_code\":" +
                              std::to_string(kFixtureToolExitCode);
  const std::size_t statusOffset = replacement.find(success);
  require(__func__, statusOffset != std::string::npos,
          "could not locate the canonical completion status");
  replacement.replace(statusOffset, success.size(), failure);
  constexpr llvm::StringLiteral outputMember = "\"output_sha256\":[";
  const std::size_t outputBegin = replacement.find(outputMember.str());
  require(__func__, outputBegin != std::string::npos,
          "could not locate the canonical completion outputs");
  const std::size_t digestBegin = outputBegin + outputMember.size();
  const std::size_t digestEnd = replacement.find(']', digestBegin);
  require(__func__, digestEnd != std::string::npos,
          "could not locate the canonical completion output terminator");
  replacement.erase(digestBegin, digestEnd - digestBegin);
  writeText(completion, replacement);

  const InvocationCompletion parsed =
      take(__func__, loadExternalToolInvocationCompletion(prepared));
  require(__func__,
          parsed.status == InvocationCompletionStatus::ToolExit &&
              parsed.exitCode == kFixtureToolExitCode,
          "the replacement completion was not a valid failed record");
  requireFailure(__func__,
                 importExternalToolInvocationAttempt(
                     prepared, importExpectation(spec), execution),
                 "receipt-aware import accepted a replaced completion record");
}

void cacheHitCarriesAnImportableReceipt(const std::filesystem::path &root,
                                        const std::filesystem::path &tool) {
  const std::filesystem::path cache = root / "result-cache";
  require(__func__,
          ::setenv("LOOM_EXTERNAL_TOOL_CACHE_ROOT", cache.c_str(), 1) == 0,
          "could not enable the result cache");
  const std::filesystem::path counter = root / "cache-tool-entry-count";
  const std::string output = "outputs/cache-hit.txt";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, output, "cached");
  spec.commands = {
      {tool.string(), "counted-run", counter.string(), "cached", output}};

  const PreparedExternalToolInvocation population =
      prepare(__func__, root, "cache-population", spec);
  const ExternalToolInvocationExecutionObservation populationExecution =
      take(__func__, executeExternalToolInvocationBundleObserved(population));
  require(__func__,
          populationExecution.cacheLookup ==
              ExternalToolResultCacheLookup::Miss,
          "the cache population was not a miss");

  const PreparedExternalToolInvocation hit =
      prepare(__func__, root, "cache-hit", spec);
  const ExternalToolInvocationExecutionObservation hitExecution =
      take(__func__, executeExternalToolInvocationBundleObserved(hit));
  require(__func__,
          hitExecution.cacheLookup == ExternalToolResultCacheLookup::Hit &&
              !hitExecution.invokedExternalTool && readText(counter) == "1",
          "the exact cache hit re-entered the external tool");
  const ImportedExternalToolInvocationBundle imported =
      take(__func__, importExternalToolInvocationBundle(
                         hit, importExpectation(spec), hitExecution));
  require(__func__,
          take(__func__, readExternalToolInvocationDeclaredOutput(
                             imported, output)) == "cached",
          "the cache-hit receipt did not import its exact output");
  require(__func__, ::unsetenv("LOOM_EXTERNAL_TOOL_CACHE_ROOT") == 0,
          "could not disable the result cache");
}

void failedExecutionCarriesAnImportableReceipt(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  ExternalToolInvocationBundleSpec spec =
      baseSpec(tool, "outputs/unused-failure.txt");
  spec.commands = {{tool.string(), "fail"}};
  spec.declaredOutputs.clear();
  const PreparedExternalToolInvocation prepared =
      prepare(__func__, root, "failed-execution", spec);
  const ExternalToolInvocationExecutionObservation execution = take(
      __func__, executeExternalToolInvocationBundleObserved(
                    prepared, {}, ExternalToolResultReusePolicy::RequireFresh));
  const ExternalToolInvocationAttemptOutcome imported =
      take(__func__, importExternalToolInvocationAttempt(
                         prepared, importExpectation(spec), execution));
  require(__func__,
          std::holds_alternative<FailedExternalToolInvocationAttempt>(imported),
          "a failed receipt did not import as a failed attempt");
  const auto &failure = std::get<FailedExternalToolInvocationAttempt>(imported);
  require(__func__,
          failure.status == InvocationCompletionStatus::ToolExit &&
              failure.exitCode == kFixtureToolExitCode,
          "a failed receipt lost its exact completion disposition");
}

struct StopWhenEntered final {
  std::filesystem::path marker;
};

bool stopWhenEntered(const void *opaque) {
  return std::filesystem::exists(
      static_cast<const StopWhenEntered *>(opaque)->marker);
}

void stoppedExecutionCarriesAnImportableReceipt(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  const std::string output = "outputs/stopped.txt";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, output);
  spec.commands = {{tool.string(), "controlled-block", "outputs/entered",
                    "outputs/late", output}};
  const PreparedExternalToolInvocation prepared =
      prepare(__func__, root, "stopped-execution", spec);
  const StopWhenEntered stop{root / "stopped-execution" / "outputs" /
                             "entered"};
  const loom::ExecutionControlView control{&stop, stopWhenEntered};
  const ExternalToolInvocationExecutionObservation execution =
      take(__func__,
           executeExternalToolInvocationBundleObserved(
               prepared, control, ExternalToolResultReusePolicy::RequireFresh));
  require(__func__,
          execution.exitCode == externalToolExecutionStoppedExitCode &&
              execution.invokedExternalTool &&
              !std::filesystem::exists(root / "stopped-execution" / "outputs" /
                                       "completion.json"),
          "controlled execution did not preserve its incomplete disposition");
  const ExternalToolInvocationAttemptOutcome imported =
      take(__func__, importExternalToolInvocationAttempt(
                         prepared, importExpectation(spec), execution));
  require(
      __func__,
      std::holds_alternative<IncompleteExternalToolInvocationAttempt>(imported),
      "a stopped receipt did not import as an incomplete attempt");
  std::this_thread::sleep_for(std::chrono::milliseconds(700));
  require(
      __func__,
      !std::filesystem::exists(root / "stopped-execution" / "outputs" / "late"),
      "a descendant survived the stopped external-tool process group");
}

void publicObservationCannotSubstituteForAReceipt(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  const std::string output = "outputs/unsealed.txt";
  const ExternalToolInvocationBundleSpec spec = baseSpec(tool, output);
  const PreparedExternalToolInvocation prepared =
      prepare(__func__, root, "unsealed-observation", spec);
  ExternalToolInvocationExecutionObservation execution = take(
      __func__, executeExternalToolInvocationBundleObserved(
                    prepared, {}, ExternalToolResultReusePolicy::RequireFresh));
  execution.receipt = {};
  requireFailure(
      __func__,
      importExternalToolInvocationBundle(prepared, importExpectation(spec),
                                         execution),
      "public execution fields substituted for a sealed executor receipt");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one test-directory argument");
  const std::filesystem::path root =
      std::filesystem::absolute(argv[1]).lexically_normal();
  std::filesystem::create_directories(root);
  const std::filesystem::path tool = root / "tool bin" / "receipt fixture";
  writeExecutable(tool, "#!/usr/bin/env bash\n"
                        "set -u\n"
                        "case \"${1-}\" in\n"
                        "  --version) printf '%s\\n' 'Receipt Fixture 1.0' ;;\n"
                        "  run) printf '%s' \"$2\" >\"$3\" ;;\n"
                        "  counted-run)\n"
                        "    value=0\n"
                        "    if [[ -f \"$2\" ]]; then IFS= read -r value "
                        "<\"$2\"; fi\n"
                        "    printf '%s' \"$((value + 1))\" >\"$2\"\n"
                        "    printf '%s' \"$3\" >\"$4\"\n"
                        "    ;;\n"
                        "  fail) exit 93 ;;\n"
                        "  controlled-block)\n"
                        "    printf '%s' entered >\"$2\"\n"
                        "    (sleep 0.6; printf '%s' late >\"$3\") &\n"
                        "    while :; do sleep 0.01; done\n"
                        "    ;;\n"
                        "  *) exit 64 ;;\n"
                        "esac\n");
  require("main", ::unsetenv("LOOM_EXTERNAL_TOOL_CACHE_ROOT") == 0,
          "could not isolate the result-cache fixture");

  interleavedGenerationRejectsOldCompletionAndReceipt(root, tool);
  outputMutationCannotEscapeReceiptAwareImport(root, tool);
  completionReplacementInvalidatesReceipt(root, tool);
  cacheHitCarriesAnImportableReceipt(root, tool);
  failedExecutionCarriesAnImportableReceipt(root, tool);
  stoppedExecutionCarriesAnImportableReceipt(root, tool);
  publicObservationCannotSubstituteForAReceipt(root, tool);
  return 0;
}
