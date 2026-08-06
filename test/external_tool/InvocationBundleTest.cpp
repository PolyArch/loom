#include "ExternalTool/InvocationBundle.h"
#include "ExternalTool/ExternalFile.h"

#include "Common/ArtifactText.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SHA256.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

using namespace loom::external_tool;

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  std::cerr << test << ": " << message << '\n';
  std::exit(1);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
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

template <typename T>
void requireFailureContains(const char *test, llvm::Expected<T> value,
                            const std::string &reason) {
  if (value)
    fail(test, "expected a failure containing: " + reason);
  const std::string message = llvm::toString(value.takeError());
  require(test, message.find(reason) != std::string::npos,
          "failure reason mismatch: " + message);
}

void requireFailureContains(const char *test, llvm::Error error,
                            const std::string &reason) {
  if (!error)
    fail(test, "expected a failure containing: " + reason);
  const std::string message = llvm::toString(std::move(error));
  require(test, message.find(reason) != std::string::npos,
          "failure reason mismatch: " + message);
}

void take(const char *test, llvm::Error error) {
  if (error)
    fail(test, llvm::toString(std::move(error)));
}

int waitForChild(const char *test, pid_t child) {
  int status = 0;
  pid_t waited = -1;
  do {
    waited = ::waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  require(test, waited == child, "could not wait for child process");
  return status;
}

std::string readFile(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    fail(__func__, "could not open " + path.string());
  std::ostringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

void writeExecutable(const std::filesystem::path &path,
                     llvm::StringRef contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary);
  if (!stream)
    fail(__func__, "could not open " + path.string());
  stream << contents.str();
  stream.close();
  std::filesystem::permissions(path,
                               std::filesystem::perms::owner_read |
                                   std::filesystem::perms::owner_write |
                                   std::filesystem::perms::owner_exec |
                                   std::filesystem::perms::group_read |
                                   std::filesystem::perms::group_exec,
                               std::filesystem::perm_options::replace);
}

void writeText(const std::filesystem::path &path, llvm::StringRef contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary);
  if (!stream)
    fail(__func__, "could not open " + path.string());
  stream << contents.str();
}

ExternalFileFingerprint fingerprint(llvm::StringRef contents) {
  const auto *bytes =
      reinterpret_cast<const std::uint8_t *>(contents.data());
  return take(__func__, ExternalFileFingerprint::fromBytes(llvm::SHA256::hash(
                            llvm::ArrayRef<std::uint8_t>(bytes,
                                                         contents.size()))));
}

loom::BlobDigest blobDigest(llvm::StringRef contents) {
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(contents.data());
  return loom::computeBlobDigest(
      llvm::ArrayRef<std::uint8_t>(bytes, contents.size()));
}

ResolvedToolBinding toolBinding(const std::filesystem::path &executable,
                                ToolBindingSource source) {
  return ResolvedToolBinding{"fake_eda",     source,      executable.string(),
                             "Fake EDA 1.2", {},          {},
                             std::nullopt,   std::nullopt};
}

ExternalToolInvocationBundleSpec baseSpec(const std::filesystem::path &tool,
                                          const std::filesystem::path &output) {
  ExternalToolInvocationBundleSpec spec;
  spec.providerIdentity = "fake_eda@1";
  spec.semanticClosure = SemanticInvocationClosure(
      CandidateGeneratorInvocationClosure{{0x01, 0x02},
                                          {0x03, 0x04},
                                          blobDigest("fake-model-binding")
                                              .bytes()});
  spec.resultImporterIdentity = "fake_importer@1";
  spec.tool = toolBinding(tool, ToolBindingSource::Explicit);
  spec.toolVersionProbe = ToolVersionProbe{{"--version"}, "Fake EDA"};
  spec.runtime.kind = InvocationRuntimeKind::Host;
  spec.commands = {{tool.string(), "run", "literal; $(touch never)",
                    output.generic_string()}};
  spec.inheritEnvironment = {"LOOM_BUNDLE_TEST_LICENSE"};
  spec.declaredOutputs = {output.generic_string()};
  spec.files = {{"drivers/driver with ' quote.tcl", "puts {driver}\n",
                 std::nullopt, false},
                {"inputs/input.bin", "input-bytes",
                 loom::ArtifactRootReference{
                     "loom.test_input",
                     {1, 0},
                     take(__func__, loom::parseArtifactIdentityHex(
                                        std::string(64, '1')))},
                 false}};
  return spec;
}

ExternalToolInvocationImportExpectation
importExpectation(const ExternalToolInvocationBundleSpec &spec) {
  ExternalToolInvocationImportExpectation expectation;
  expectation.providerIdentity = spec.providerIdentity;
  expectation.semanticClosure = spec.semanticClosure;
  expectation.resultImporterIdentity = spec.resultImporterIdentity;
  for (const MaterializedBundleFile &file : spec.files)
    if (file.sourceArtifact)
      expectation.semanticInputs.push_back(
          {file.relativePath, *file.sourceArtifact, blobDigest(file.contents)});
  for (const ResolvedExternalFile &file : spec.externalFiles)
    expectation.externalInputs.push_back(
        {file.providerInputSlot, file.fingerprint});
  expectation.declaredOutputs = spec.declaredOutputs;
  return expectation;
}

void deterministicHostBundleExecutes(const std::filesystem::path &root,
                                     const std::filesystem::path &tool) {
  const std::filesystem::path first = root / "bundle-a";
  const std::filesystem::path second = root / "bundle-b";
  const std::filesystem::path output = "outputs/nested/result with ' quote.txt";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, output);
  spec.tool.source = ToolBindingSource::Module;
  spec.tool.requestedModules = {"fake_eda"};
  spec.tool.loadedModules = {"dependency/1.0", "fake; touch never"};
  spec.tool.moduleInit = (root / "module init.sh").string();
  writeExecutable(*spec.tool.moduleInit,
                  "module() { [[ \"${1-}\" == load ]]; }\n");

  take(__func__, finalizeExternalToolInvocationBundle(first.string(), spec));
  take(__func__, finalizeExternalToolInvocationBundle(second.string(), spec));
  const std::string firstManifest = readFile(first / "tool-invocation.json");
  const std::string firstScript = readFile(first / "run.sh");
  require(__func__, firstManifest == readFile(second / "tool-invocation.json"),
          "identical inputs produced different manifests");
  require(__func__,
          firstManifest.find("\"content_sha256\"") != std::string::npos,
          "materialized file contents are not bound by the manifest");
  require(__func__, firstScript == readFile(second / "run.sh"),
          "identical inputs produced different scripts");
  require(__func__,
          firstManifest.find("license-secret-value") == std::string::npos &&
              firstScript.find("license-secret-value") == std::string::npos,
          "an inherited environment value leaked into the bundle");

  const int status =
      take(__func__, executeExternalToolInvocationBundle(first.string()));
  require(__func__, status == 0, "successful bundle returned a failure status");
  require(__func__, readFile(first / output) == "literal; $(touch never)",
          "command arguments were not preserved as data");
  require(__func__, !std::filesystem::exists(first / "never"),
          "command data was evaluated by the shell");
  InvocationCompletion completion =
      take(__func__, loadExternalToolInvocationCompletion(first.string()));
  require(__func__,
          completion.status == InvocationCompletionStatus::Success &&
              completion.exitCode == 0 &&
              completion.manifestDigest == blobDigest(firstManifest) &&
              completion.outputDigests ==
                  std::vector<loom::BlobDigest>{
                      blobDigest("literal; $(touch never)")},
          "successful completion was not imported");

  const std::filesystem::path completionPath =
      first / "outputs" / "completion.json";
  writeText(completionPath, " " + readFile(completionPath));
  llvm::Expected<InvocationCompletion> noncanonical =
      loadExternalToolInvocationCompletion(first.string());
  require(__func__, !noncanonical,
          "noncanonical completion record was accepted");
  llvm::consumeError(noncanonical.takeError());

  writeExecutable(second / "drivers" / "driver with ' quote.tcl", "tampered\n");
  const int tamperedStatus =
      take(__func__, executeExternalToolInvocationBundle(second.string()));
  require(__func__, tamperedStatus != 0,
          "bundle with tampered materialized content returned success");
  InvocationCompletion tampered =
      take(__func__, loadExternalToolInvocationCompletion(second.string()));
  require(__func__,
          tampered.status == InvocationCompletionStatus::BundleContentMismatch,
          "tampered bundle content was not distinguished in completion");
}

void containerBundleExecutes(const std::filesystem::path &root,
                             const std::filesystem::path &tool,
                             const std::filesystem::path &container) {
  const std::filesystem::path bundle = root / "container-bundle";
  const std::filesystem::path output = "outputs/container-result.txt";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, output);
  ResolvedToolBinding runtimeBinding{"polyarch_container",
                                     ToolBindingSource::Explicit,
                                     container.string(),
                                     "PolyArch container v0.1.0",
                                     {},
                                     {},
                                     std::nullopt,
                                     std::nullopt};
  spec.runtime.kind = InvocationRuntimeKind::PolyArchContainer;
  spec.runtime.polyArchContainer = std::move(runtimeBinding);
  spec.runtime.os = "almalinux9";
  spec.containerVersionProbe =
      ToolVersionProbe{{"--version"}, "PolyArch container"};

  take(__func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
  const int status =
      take(__func__, executeExternalToolInvocationBundle(bundle.string()));
  require(__func__, status == 0, "container bundle returned a failure status");
  require(__func__, readFile(bundle / output) == "literal; $(touch never)",
          "container command did not preserve arguments");
}

void externalFileIsRevalidated(const std::filesystem::path &root,
                               const std::filesystem::path &tool) {
  constexpr llvm::StringLiteral contents = "vendor-library-original\n";
  const std::filesystem::path external =
      root / "external files" / "vendor's typical.lib";
  writeText(external, contents);

  LocalToolConfig config;
  config.externalFiles.emplace("asic_typical_lib", external.string());
  const ExternalFileRequirement requirement{"asic.liberty", fingerprint(contents)};
  std::vector<ResolvedExternalFile> resolved =
      take(__func__, resolveExternalFiles({requirement}, config));

  const std::filesystem::path bundle = root / "external-file-bundle";
  const std::filesystem::path output = "outputs/external-file-result.txt";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, output);
  spec.externalFiles = resolved;
  take(__func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));

  const std::string manifest = readFile(bundle / "tool-invocation.json");
  require(__func__,
          manifest.find("\"provider_input_slot\": \"asic.liberty\"") !=
                  std::string::npos &&
              manifest.find("\"local_file_key\": \"asic_typical_lib\"") !=
                  std::string::npos &&
              manifest.find(external.string()) != std::string::npos &&
              manifest.find(formatExternalFileFingerprint(
                                resolved.front().fingerprint)) !=
                  std::string::npos,
          "resolved external file provenance is absent from the manifest");

  int status =
      take(__func__, executeExternalToolInvocationBundle(bundle.string()));
  require(__func__, status == 0,
          "bundle rejected an unchanged resolved external file");
  require(__func__, readFile(bundle / output) == "literal; $(touch never)",
          "bundle with an external file did not execute the tool");

  ExternalToolInvocationImportExpectation expected = importExpectation(spec);
  take(__func__, importExternalToolInvocationBundle(bundle.string(), expected));
  ExternalToolInvocationImportExpectation wrong = expected;
  wrong.externalInputs.front().fingerprint = fingerprint("different-bytes");
  requireFailure(__func__,
                 importExternalToolInvocationBundle(bundle.string(), wrong),
                 "a wrong external input fingerprint was accepted");

  writeText(external, "vendor-library-mutated\n");
  const std::filesystem::path changedBundle = root / "changed-external-bundle";
  take(__func__,
       finalizeExternalToolInvocationBundle(changedBundle.string(), spec));
  status = take(__func__,
                executeExternalToolInvocationBundle(changedBundle.string()));
  require(__func__, status != 0,
          "bundle accepted a changed resolved external file");
  InvocationCompletion completion = take(
      __func__, loadExternalToolInvocationCompletion(changedBundle.string()));
  require(__func__,
          completion.status == InvocationCompletionStatus::BundleContentMismatch,
          "changed external content was not distinguished in completion");
  require(__func__, !std::filesystem::exists(changedBundle / output),
          "the tool ran after external content verification failed");
}

void missingOutputIsRecorded(const std::filesystem::path &root,
                             const std::filesystem::path &tool) {
  const std::filesystem::path bundle = root / "missing-output";
  ExternalToolInvocationBundleSpec spec =
      baseSpec(tool, "outputs/required.txt");
  spec.commands = {{tool.string(), "no-output"}};
  take(__func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
  const int status =
      take(__func__, executeExternalToolInvocationBundle(bundle.string()));
  require(__func__, status != 0,
          "bundle with a missing output returned success");
  InvocationCompletion completion =
      take(__func__, loadExternalToolInvocationCompletion(bundle.string()));
  require(__func__,
          completion.status == InvocationCompletionStatus::MissingOutput,
          "missing output was not distinguished in completion");
  require(__func__, completion.outputDigests.empty(),
          "failed completion retained output digests");

  // A planted stale output never survives as a fresh result: the rerun-safe
  // prelude removes declared outputs before execution, so the tool's missing
  // production is reported rather than the foreign bytes being adopted.
  const std::filesystem::path staleBundle = root / "stale-output";
  ExternalToolInvocationBundleSpec staleSpec =
      baseSpec(tool, "outputs/required.txt");
  staleSpec.commands = {{tool.string(), "no-output"}};
  take(__func__,
       finalizeExternalToolInvocationBundle(staleBundle.string(), staleSpec));
  writeText(staleBundle / staleSpec.declaredOutputs.front(), "stale");
  require(__func__,
          take(__func__,
               executeExternalToolInvocationBundle(staleBundle.string())) != 0,
          "a preexisting output was accepted as a fresh tool result");
  completion = take(__func__,
                    loadExternalToolInvocationCompletion(staleBundle.string()));
  require(__func__,
          completion.status == InvocationCompletionStatus::MissingOutput,
          "a planted stale output was not removed before execution");
  requireFailure(__func__,
                 importExternalToolInvocationBundle(
                     staleBundle.string(), importExpectation(staleSpec)),
                 "a stale planted output was imported as a fresh result");
}

void versionNormalizationMatchesDiscovery(const std::filesystem::path &root) {
  const std::filesystem::path tool = root / "version-tool" / "dc_shell";
  writeExecutable(tool, "#!/usr/bin/env bash\n"
                        "set -u\n"
                        "if [[ \"${1-}\" == --version ]]; then\n"
                        "  printf '%s\\n' 'host: runtime-specific'\n"
                        "  printf '%s\\n' 'dc_shell version - Y-2026.03-SP2'\n"
                        "  printf '%s\\n' 'time: invocation-specific'\n"
                        "  exit 1\n"
                        "fi\n"
                        "[[ \"${1-}\" == run ]] || exit 64\n"
                        "printf '%s' \"$2\" >\"$3\"\n");
  const std::filesystem::path bundle = root / "normalized-version-bundle";
  const std::filesystem::path output = "outputs/version-result.txt";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, output);
  spec.tool.version = "dc_shell version - Y-2026.03-SP2";
  spec.toolVersionProbe = ToolVersionProbe{
      {"--version"}, "dc_shell version", {0, 1}, "dc_shell version"};
  take(__func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
  const std::string manifest = readFile(bundle / "tool-invocation.json");
  require(__func__,
          manifest.find("\"accepted_exit_codes\"") != std::string::npos &&
              manifest.find("\"selected_output_line_substring\"") !=
                  std::string::npos,
          "version normalization contract is absent from the manifest");
  const int status =
      take(__func__, executeExternalToolInvocationBundle(bundle.string()));
  require(__func__, status == 0,
          "bundle rejected the version accepted during discovery");
  require(__func__, readFile(bundle / output) == "literal; $(touch never)",
          "normalized version bundle did not execute the tool command");
}

void invalidPathLeavesNoBundle(const std::filesystem::path &root,
                               const std::filesystem::path &tool) {
  const std::filesystem::path bundle = root / "invalid-bundle";
  ExternalToolInvocationBundleSpec spec =
      baseSpec(tool, "outputs/required.txt");
  spec.files.front().relativePath = "inputs/../escape";
  llvm::Error error =
      finalizeExternalToolInvocationBundle(bundle.string(), spec);
  require(__func__, static_cast<bool>(error), "escaping path was accepted");
  llvm::consumeError(std::move(error));
  require(__func__, !std::filesystem::exists(bundle),
          "failed finalization published a bundle");
}

void conflictingPathLeavesNoBundle(const std::filesystem::path &root,
                                   const std::filesystem::path &tool) {
  const std::filesystem::path bundle = root / "conflicting-bundle";
  ExternalToolInvocationBundleSpec spec =
      baseSpec(tool, "outputs/completion.json/child");
  llvm::Error error =
      finalizeExternalToolInvocationBundle(bundle.string(), spec);
  require(__func__, static_cast<bool>(error),
          "a path below the completion record was accepted");
  llvm::consumeError(std::move(error));
  require(__func__, !std::filesystem::exists(bundle),
          "conflicting finalization published a bundle");
}

void internalVersionPathCannotBeDeclared(const std::filesystem::path &root,
                                         const std::filesystem::path &tool) {
  const std::filesystem::path bundle = root / "reserved-version-output";
  ExternalToolInvocationBundleSpec spec =
      baseSpec(tool, "outputs/.loom-tool-version");
  llvm::Error error =
      finalizeExternalToolInvocationBundle(bundle.string(), spec);
  require(__func__, static_cast<bool>(error),
          "the internal version output path was accepted");
  llvm::consumeError(std::move(error));
  require(__func__, !std::filesystem::exists(bundle),
          "reserved-path finalization published a bundle");
}

void independentBundlesExecuteInParallel(const std::filesystem::path &root,
                                         const std::filesystem::path &tool) {
  const std::filesystem::path first = root / "parallel-a";
  const std::filesystem::path second = root / "parallel-b";
  ExternalToolInvocationBundleSpec firstSpec =
      baseSpec(tool, "outputs/result.txt");
  firstSpec.commands = {{tool.string(), "block", "outputs/tool-entry.log",
                         "outputs/release", "outputs/result.txt"}};
  take(__func__, finalizeExternalToolInvocationBundle(first.string(),
                                                      firstSpec));
  ExternalToolInvocationBundleSpec secondSpec =
      baseSpec(tool, "outputs/result.txt");
  take(__func__, finalizeExternalToolInvocationBundle(second.string(),
                                                      secondSpec));

  const pid_t child = ::fork();
  require(__func__, child >= 0, "could not fork the first bundle execution");
  if (child == 0) {
    llvm::Expected<int> status =
        executeExternalToolInvocationBundle(first.string());
    if (!status) {
      llvm::consumeError(status.takeError());
      ::_exit(254);
    }
    ::_exit(*status == 0 ? 0 : 253);
  }

  const std::filesystem::path entered = first / "outputs" / "tool-entry.log";
  bool toolEntered = false;
  for (unsigned attempt = 0; attempt != 5000; ++attempt) {
    if (std::filesystem::exists(entered)) {
      toolEntered = true;
      break;
    }
    ::usleep(1000);
  }
  if (!toolEntered) {
    writeText(first / "outputs" / "release", "");
    waitForChild(__func__, child);
    fail(__func__, "the first execution did not enter the tool");
  }

  // A second independent bundle root executes to completion while the first
  // is still inside the tool: no shared mutable state exists between roots.
  require(__func__,
          take(__func__,
               executeExternalToolInvocationBundle(second.string())) == 0,
          "an independent bundle could not execute while another ran");
  writeText(first / "outputs" / "release", "");
  const int firstStatus = waitForChild(__func__, child);
  require(__func__, WIFEXITED(firstStatus) && WEXITSTATUS(firstStatus) == 0,
          "the blocked execution did not complete after release");
  take(__func__, importExternalToolInvocationBundle(
                     first.string(), importExpectation(firstSpec)));
  take(__func__, importExternalToolInvocationBundle(
                     second.string(), importExpectation(secondSpec)));
}

void unremovableDeclaredOutputFailsBeforeToolEntry(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  const std::filesystem::path bundle = root / "unremovable-output";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, "outputs/result.txt");
  spec.commands = {{tool.string(), "block", "outputs/tool-entry.log",
                    "outputs/release", "outputs/result.txt"}};
  take(__func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
  // The release marker lets the blocking tool proceed immediately if it is
  // ever entered; the directory makes the stale-output removal fail.
  writeText(bundle / "outputs" / "release", "");
  std::filesystem::create_directories(bundle / "outputs" / "result.txt");

  const int status =
      take(__func__, executeExternalToolInvocationBundle(bundle.string()));
  require(__func__, status != 0,
          "an unremovable declared output did not fail the execution");
  require(__func__,
          !std::filesystem::exists(bundle / "outputs" / "tool-entry.log"),
          "the external tool was entered despite the failed removal");
  InvocationCompletion completion =
      take(__func__, loadExternalToolInvocationCompletion(bundle.string()));
  require(__func__,
          completion.status == InvocationCompletionStatus::BundleContentMismatch,
          "an unremovable declared output did not fail integrity validation");
  requireFailure(__func__,
                 importExternalToolInvocationBundle(
                     bundle.string(), importExpectation(spec)),
                 "stale bytes were imported after a failed removal");
}

void sequentialReexecutionIsCallerOwned(const std::filesystem::path &root,
                                        const std::filesystem::path &tool) {
  const std::filesystem::path bundle = root / "reexecuted";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, "outputs/result.txt");
  take(__func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));

  require(__func__,
          take(__func__,
               executeExternalToolInvocationBundle(bundle.string())) == 0,
          "the first execution failed");
  take(__func__, importExternalToolInvocationBundle(
                     bundle.string(), importExpectation(spec)));

  // The caller chooses to execute the same prepared bundle again: the
  // completion and declared outputs are republished by the new execution.
  require(__func__,
          take(__func__,
               executeExternalToolInvocationBundle(bundle.string())) == 0,
          "a caller-chosen sequential re-execution was refused");
  take(__func__, importExternalToolInvocationBundle(
                     bundle.string(), importExpectation(spec)));

  // An interrupted attempt leaves no completion; re-execution still works,
  // and the incomplete state stays import-rejected in between.
  std::filesystem::remove(bundle / "outputs" / "completion.json");
  requireFailure(__func__,
                 importExternalToolInvocationBundle(
                     bundle.string(), importExpectation(spec)),
                 "an interrupted bundle remained importable");
  require(__func__,
          take(__func__,
               executeExternalToolInvocationBundle(bundle.string())) == 0,
          "re-execution after an interrupted attempt failed");
  take(__func__, importExternalToolInvocationBundle(
                     bundle.string(), importExpectation(spec)));

  // No Loom-owned claim or retry authority remains in the generated script.
  const std::string script = readFile(bundle / "run.sh");
  require(__func__, script.find(".loom-execution-started") ==
                            std::string::npos &&
                        script.find("exit 120") == std::string::npos,
          "run.sh retained an execution claim or retry authority");
}

void successfulImportIsExactAndOutputSafe(const std::filesystem::path &root,
                                          const std::filesystem::path &tool) {
  const std::filesystem::path bundle = root / "strict-import";
  const std::filesystem::path output = "outputs/imported.txt";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, output);
  take(__func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
  require(
      __func__,
      take(__func__, executeExternalToolInvocationBundle(bundle.string())) == 0,
      "strict-import bundle did not execute");

  ExternalToolInvocationImportExpectation expected = importExpectation(spec);
  ImportedExternalToolInvocationBundle imported = take(
      __func__, importExternalToolInvocationBundle(bundle.string(), expected));
  require(__func__,
          take(__func__, readExternalToolInvocationDeclaredOutput(
                             imported, output.generic_string())) ==
              "literal; $(touch never)",
          "declared output bytes were not imported exactly");
  requireFailure(__func__,
                 readExternalToolInvocationDeclaredOutput(
                     imported, "outputs/not-declared.txt"),
                 "an undeclared output was readable");

  ExternalToolInvocationImportExpectation wrong = expected;
  wrong.providerIdentity = "other-provider@1";
  requireFailure(__func__,
                 importExternalToolInvocationBundle(bundle.string(), wrong),
                 "a wrong provider identity was accepted");
  wrong = expected;
  wrong.semanticClosure = SemanticInvocationClosure(loom::ArtifactRootReference{
      "loom.test_request",
      {1, 0},
      take(__func__, loom::parseArtifactIdentityHex(std::string(64, '3')))});
  requireFailure(__func__,
                 importExternalToolInvocationBundle(bundle.string(), wrong),
                 "a wrong semantic closure was accepted");
  wrong = expected;
  wrong.resultImporterIdentity = "other-importer@1";
  requireFailure(__func__,
                 importExternalToolInvocationBundle(bundle.string(), wrong),
                 "a wrong result importer identity was accepted");
  wrong = expected;
  wrong.semanticInputs.front().sourceArtifact = loom::ArtifactRootReference{
      "loom.test_input",
      {1, 0},
      take(__func__, loom::parseArtifactIdentityHex(std::string(64, '2')))};
  requireFailure(__func__,
                 importExternalToolInvocationBundle(bundle.string(), wrong),
                 "a wrong semantic input Artifact reference was accepted");
  wrong = expected;
  wrong.semanticInputs.front().contentDigest = blobDigest("other-input-bytes");
  requireFailure(__func__,
                 importExternalToolInvocationBundle(bundle.string(), wrong),
                 "a wrong semantic input content digest was accepted");

  const std::filesystem::path substituted = root / "strict-import-substituted";
  ExternalToolInvocationBundleSpec substitutedSpec = spec;
  for (MaterializedBundleFile &file : substitutedSpec.files)
    if (file.sourceArtifact)
      file.contents = "substituted-input-bytes";
  take(__func__, finalizeExternalToolInvocationBundle(substituted.string(),
                                                      substitutedSpec));
  require(__func__,
          take(__func__,
               executeExternalToolInvocationBundle(substituted.string())) == 0,
          "content-substituted bundle did not execute");
  requireFailure(
      __func__,
      importExternalToolInvocationBundle(substituted.string(), expected),
      "same-ref semantic input content substitution was accepted");

  wrong = expected;
  wrong.declaredOutputs.push_back("outputs/unexpected.txt");
  requireFailure(__func__,
                 importExternalToolInvocationBundle(bundle.string(), wrong),
                 "wrong declared output membership was accepted");

  const std::string completionBefore =
      readFile(bundle / "outputs" / "completion.json");
  require(
      __func__,
      take(__func__, executeExternalToolInvocationBundle(bundle.string())) == 0,
      "a caller-chosen rerun of a completed bundle was refused");
  require(__func__,
          readFile(bundle / "outputs" / "completion.json") == completionBefore,
          "a deterministic rerun changed the completion record");

  const std::filesystem::path original = root / "strict-import-original";
  std::filesystem::rename(bundle, original);
  std::filesystem::create_directories((bundle / output).parent_path());
  writeText(bundle / output, "replacement");
  require(__func__,
          take(__func__, readExternalToolInvocationDeclaredOutput(
                             imported, output.generic_string())) ==
              "literal; $(touch never)",
          "an imported output was reopened through a replaced bundle root");
}

void unsafeOutputsCannotBeImported(const std::filesystem::path &root,
                                   const std::filesystem::path &tool) {
  const auto makeExecutedBundle = [&](llvm::StringRef name) {
    const std::filesystem::path bundle = root / name.str();
    ExternalToolInvocationBundleSpec spec =
        baseSpec(tool, "outputs/result.txt");
    take(__func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
    require(__func__,
            take(__func__,
                 executeExternalToolInvocationBundle(bundle.string())) == 0,
            "unsafe-output bundle did not execute");
    return std::pair(bundle, std::move(spec));
  };
  const std::filesystem::path outside = root / "outside-output.txt";
  writeText(outside, "outside");

  auto [symlinkBundle, symlinkSpec] = makeExecutedBundle("symlink-output");
  std::filesystem::remove(symlinkBundle / symlinkSpec.declaredOutputs.front());
  std::filesystem::create_symlink(
      outside, symlinkBundle / symlinkSpec.declaredOutputs.front());
  requireFailure(__func__,
                 importExternalToolInvocationBundle(
                     symlinkBundle.string(), importExpectation(symlinkSpec)),
                 "a symlinked declared output was imported");

  auto [directoryBundle, directorySpec] =
      makeExecutedBundle("directory-output");
  std::filesystem::remove(directoryBundle /
                          directorySpec.declaredOutputs.front());
  std::filesystem::create_directory(directoryBundle /
                                    directorySpec.declaredOutputs.front());
  requireFailure(
      __func__,
      importExternalToolInvocationBundle(directoryBundle.string(),
                                         importExpectation(directorySpec)),
      "a directory declared output was imported");

  auto [changedBundle, changedSpec] = makeExecutedBundle("changed-output");
  writeText(changedBundle / changedSpec.declaredOutputs.front(), "changed");
  requireFailure(__func__,
                 importExternalToolInvocationBundle(
                     changedBundle.string(), importExpectation(changedSpec)),
                 "an output changed after completion was imported");
}

void strictImportRejectsAttemptTampering(const std::filesystem::path &root,
                                         const std::filesystem::path &tool) {
  const auto makeExecutedBundle = [&](llvm::StringRef name) {
    const std::filesystem::path bundle = root / name.str();
    ExternalToolInvocationBundleSpec spec =
        baseSpec(tool, "outputs/result.txt");
    take(__func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
    require(__func__,
            take(__func__,
                 executeExternalToolInvocationBundle(bundle.string())) == 0,
            "import-negative bundle did not execute");
    return std::pair(bundle, std::move(spec));
  };

  auto [noncanonicalBundle, noncanonicalSpec] =
      makeExecutedBundle("noncanonical-manifest");
  const std::filesystem::path noncanonicalManifest =
      noncanonicalBundle / "tool-invocation.json";
  writeText(noncanonicalManifest, " " + readFile(noncanonicalManifest));
  requireFailure(
      __func__,
      importExternalToolInvocationBundle(noncanonicalBundle.string(),
                                         importExpectation(noncanonicalSpec)),
      "a noncanonical manifest was accepted");

  auto [unknownBundle, unknownSpec] = makeExecutedBundle("unknown-manifest");
  const std::filesystem::path unknownManifest =
      unknownBundle / "tool-invocation.json";
  std::string unknownText = readFile(unknownManifest);
  const std::size_t version = unknownText.find("  \"version\": \"2.0\",");
  require(__func__, version != std::string::npos,
          "cannot locate manifest version field");
  unknownText.insert(version, "  \"unknown\": true,\n");
  writeText(unknownManifest, unknownText);
  requireFailure(__func__,
                 importExternalToolInvocationBundle(
                     unknownBundle.string(), importExpectation(unknownSpec)),
                 "a manifest with an unknown field was accepted");

  auto [firstBundle, firstSpec] = makeExecutedBundle("manifest-binding-a");
  ExternalToolInvocationBundleSpec secondSpec = firstSpec;
  secondSpec.tool.source = ToolBindingSource::EnvironmentPath;
  const std::filesystem::path secondDistinct =
      root / "manifest-binding-distinct";
  take(__func__, finalizeExternalToolInvocationBundle(secondDistinct.string(),
                                                      secondSpec));
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(
                             secondDistinct.string())) == 0,
          "distinct manifest bundle did not execute");
  writeText(firstBundle / "tool-invocation.json",
            readFile(secondDistinct / "tool-invocation.json"));
  requireFailure(__func__,
                 importExternalToolInvocationBundle(
                     firstBundle.string(), importExpectation(firstSpec)),
                 "completion was accepted with a different manifest");

  auto [malformedBundle, malformedSpec] =
      makeExecutedBundle("malformed-nested-manifest");
  const std::filesystem::path malformedManifest =
      malformedBundle / "tool-invocation.json";
  std::string malformedText = readFile(malformedManifest);
  for (llvm::StringRef field : {"requested_modules", "loaded_modules"}) {
    const std::string original = ("    \"" + field + "\": []").str();
    const std::size_t position = malformedText.find(original);
    require(__func__, position != std::string::npos,
            "cannot locate nested manifest field");
    malformedText.replace(position, original.size(),
                          ("    \"" + field + "\": true").str());
  }
  writeText(malformedManifest, malformedText);
  requireFailure(
      __func__,
      importExternalToolInvocationBundle(malformedBundle.string(),
                                         importExpectation(malformedSpec)),
      "malformed nested manifest fields were accepted");

  const std::filesystem::path incomplete = root / "incomplete-import";
  ExternalToolInvocationBundleSpec incompleteSpec =
      baseSpec(tool, "outputs/result.txt");
  take(__func__, finalizeExternalToolInvocationBundle(incomplete.string(),
                                                      incompleteSpec));
  requireFailure(__func__,
                 importExternalToolInvocationBundle(
                     incomplete.string(), importExpectation(incompleteSpec)),
                 "an invocation without completion was imported");
  const std::filesystem::path failed = root / "failed-import";
  ExternalToolInvocationBundleSpec failedSpec =
      baseSpec(tool, "outputs/result.txt");
  failedSpec.commands = {{tool.string(), "no-output"}};
  take(__func__,
       finalizeExternalToolInvocationBundle(failed.string(), failedSpec));
  require(
      __func__,
      take(__func__, executeExternalToolInvocationBundle(failed.string())) != 0,
      "failed-import bundle unexpectedly succeeded");
  requireFailure(__func__,
                 importExternalToolInvocationBundle(
                     failed.string(), importExpectation(failedSpec)),
                 "a non-success completion was imported");
}

void typedClosureIsExactAndLegacyManifestIsRejected(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  // The Evaluation closure form round-trips through execution and strict
  // import.
  const std::filesystem::path bundle = root / "evaluation-closure";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, "outputs/result.txt");
  spec.semanticClosure = SemanticInvocationClosure(loom::ArtifactRootReference{
      "loom.test_request",
      {1, 0},
      take(__func__, loom::parseArtifactIdentityHex(std::string(64, '7')))});
  take(__func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(bundle.string())) ==
              0,
          "evaluation-closure bundle did not execute");
  take(__func__, importExternalToolInvocationBundle(bundle.string(),
                                                    importExpectation(spec)));

  // A candidate-generator closure without owner bytes fails closed at
  // finalization.
  ExternalToolInvocationBundleSpec emptyClosure =
      baseSpec(tool, "outputs/result.txt");
  emptyClosure.semanticClosure =
      SemanticInvocationClosure(CandidateGeneratorInvocationClosure{});
  requireFailureContains(
      __func__,
      finalizeExternalToolInvocationBundle(
          (root / "empty-closure").string(), emptyClosure),
      "empty owner bytes");

  // A legacy 1.0 manifest is rejected by name; no upgrade path exists.
  const std::filesystem::path legacy = root / "legacy-manifest";
  take(__func__, finalizeExternalToolInvocationBundle(legacy.string(), spec));
  const std::filesystem::path legacyManifest = legacy / "tool-invocation.json";
  std::string legacyText = readFile(legacyManifest);
  const std::size_t version = legacyText.find("\"version\": \"2.0\"");
  require(__func__, version != std::string::npos,
          "cannot locate the 2.0 manifest version");
  legacyText.replace(version, std::string("\"version\": \"2.0\"").size(),
                     "\"version\": \"1.0\"");
  writeText(legacyManifest, legacyText);
  requireFailureContains(
      __func__,
      importExternalToolInvocationBundle(legacy.string(),
                                         importExpectation(spec)),
      "1.0");

  // An unknown closure form is rejected by the strict parser.
  const std::filesystem::path unknownForm = root / "unknown-closure-form";
  take(__func__,
       finalizeExternalToolInvocationBundle(unknownForm.string(), spec));
  const std::filesystem::path unknownManifest =
      unknownForm / "tool-invocation.json";
  std::string unknownText = readFile(unknownManifest);
  const std::size_t form = unknownText.find("\"form\": \"evaluation\"");
  require(__func__, form != std::string::npos,
          "cannot locate the closure form");
  unknownText.replace(form, std::string("\"form\": \"evaluation\"").size(),
                      "\"form\": \"diagnostic\"");
  writeText(unknownManifest, unknownText);
  requireFailureContains(
      __func__,
      importExternalToolInvocationBundle(unknownForm.string(),
                                         importExpectation(spec)),
      "unknown closure form");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one test-directory argument");
  const std::filesystem::path root =
      std::filesystem::absolute(argv[1]).lexically_normal();
  std::filesystem::create_directories(root);
  const std::filesystem::path tool = root / "tool bin; data" / "fake tool";
  writeExecutable(tool, "#!/usr/bin/env bash\n"
                        "set -u\n"
                        "case \"${1-}\" in\n"
                        "  --version) printf '%s\\n' 'Fake EDA 1.2' ;;\n"
                        "  run) printf '%s' \"$2\" >\"$3\" ;;\n"
                        "  block)\n"
                        "    printf '%s\\n' entered >>\"$2\"\n"
                        "    while [[ ! -e \"$3\" ]]; do sleep 0.01; done\n"
                        "    printf '%s' completed >\"$4\"\n"
                        "    ;;\n"
                        "  no-output) : ;;\n"
                        "  *) exit 64 ;;\n"
                        "esac\n");
  const std::filesystem::path container = root / "container bin" / "container";
  writeExecutable(container, "#!/usr/bin/env bash\n"
                             "set -u\n"
                             "if [[ \"${1-}\" == --version ]]; then\n"
                             "  printf '%s\\n' 'PolyArch container v0.1.0'\n"
                             "  exit 0\n"
                             "fi\n"
                             "[[ \"${1-}\" == run ]] || exit 64\n"
                             "shift\n"
                             "workdir=''\n"
                             "while (( $# )); do\n"
                             "  case \"$1\" in\n"
                             "    --workdir) workdir=$2; shift 2 ;;\n"
                             "    --os|--env) shift 2 ;;\n"
                             "    --) shift; break ;;\n"
                             "    *) exit 64 ;;\n"
                             "  esac\n"
                             "done\n"
                             "[[ -n \"$workdir\" ]] || exit 64\n"
                             "container_home=\"$workdir/outputs/fake-home\"\n"
                             "mkdir -p \"$container_home\"\n"
                             "ln -s \"$workdir\" \"$container_home/work\"\n"
                             "export HOME=$container_home\n"
                             "cd \"$HOME\"\n"
                             "printf '%s\\n' 'container wrapper chatter'\n"
                             "exec \"$@\"\n");
  require("main",
          ::setenv("LOOM_BUNDLE_TEST_LICENSE", "license-secret-value", 1) == 0,
          "could not set test environment");
  deterministicHostBundleExecutes(root, tool);
  containerBundleExecutes(root, tool, container);
  externalFileIsRevalidated(root, tool);
  missingOutputIsRecorded(root, tool);
  versionNormalizationMatchesDiscovery(root);
  invalidPathLeavesNoBundle(root, tool);
  conflictingPathLeavesNoBundle(root, tool);
  internalVersionPathCannotBeDeclared(root, tool);
  independentBundlesExecuteInParallel(root, tool);
  unremovableDeclaredOutputFailsBeforeToolEntry(root, tool);
  sequentialReexecutionIsCallerOwned(root, tool);
  successfulImportIsExactAndOutputSafe(root, tool);
  unsafeOutputsCannotBeImported(root, tool);
  strictImportRejectsAttemptTampering(root, tool);
  typedClosureIsExactAndLegacyManifestIsRejected(root, tool);
  return 0;
}
