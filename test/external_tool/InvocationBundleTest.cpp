#include "ExternalTool/InvocationBundle.h"
#include "ExternalTool/ExternalFile.h"

#include "Common/ArtifactText.h"
#include "Common/DiagnosticVerbosity.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SHA256.h"

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <tuple>
#include <unistd.h>
#include <utility>

using namespace loom::external_tool;

namespace {

constexpr int launcherExitCode(InvocationLauncherExitCode code) {
  return static_cast<int>(code);
}

constexpr int kFixtureToolExitCode = 93;
constexpr llvm::StringLiteral kCompatibleTypedClosureManifestVersion = "2.0";
constexpr llvm::StringLiteral kUnsupportedLegacyManifestVersion = "1.0";

std::string jsonVersionMember(llvm::StringRef version) {
  return "\"version\": \"" + version.str() + "\"";
}

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
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(contents.data());
  return take(__func__,
              ExternalFileFingerprint::fromBytes(llvm::SHA256::hash(
                  llvm::ArrayRef<std::uint8_t>(bytes, contents.size()))));
}

loom::BlobDigest blobDigest(llvm::StringRef contents) {
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(contents.data());
  return loom::computeBlobDigest(
      llvm::ArrayRef<std::uint8_t>(bytes, contents.size()));
}

void resultImporterIdentityUsesCanonicalFraming() {
  const std::array<std::uint8_t, 2> descriptorReference = {0x01, 0x02};
  const std::string identity =
      take(__func__,
           deriveExternalToolResultImporterIdentity(
               descriptorReference, loom::ProviderForm::ExternalPrepareImport));
  require(
      __func__,
      identity ==
          "a64d161a4e75675a8338e952ac0d199d9f31863e9f2aa881227c57a316c9389b",
      "result-importer digest framing changed");
  requireFailureContains(
      __func__,
      deriveExternalToolResultImporterIdentity(descriptorReference,
                                               loom::ProviderForm::InProcess),
      "ExternalPrepareImport");
  requireFailureContains(__func__,
                         deriveExternalToolResultImporterIdentity(
                             {}, loom::ProviderForm::ExternalPrepareImport),
                         "descriptor reference");
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
  spec.semanticContract.providerIdentity = "fake_eda@1";
  spec.semanticContract.semanticClosure =
      SemanticInvocationClosure(CandidateGeneratorInvocationClosure{
          {0x01, 0x02},
          {0x03, 0x04},
          blobDigest("fake-model-binding").bytes()});
  spec.semanticContract.resultImporterIdentity = std::string(64, 'a');
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
  expectation.semanticContract = spec.semanticContract;
  for (const MaterializedBundleFile &file : spec.files)
    if (file.sourceArtifact)
      expectation.semanticInputs.push_back(
          {file.relativePath, *file.sourceArtifact, blobDigest(file.contents)});
  for (const ResolvedExternalFile &file : spec.externalFiles)
    expectation.externalInputs.push_back(
        {file.providerInputSlot, file.fingerprint});
  for (const ResolvedExternalFileTree &tree : spec.externalFileTrees)
    expectation.externalFileTrees.push_back(
        {tree.providerInputSlot, tree.members});
  expectation.declaredOutputs = spec.declaredOutputs;
  return expectation;
}

void executionResourceBindingIsExact(const std::filesystem::path &root,
                                     const std::filesystem::path &tool) {
  ExternalToolInvocationBundleSpec spec =
      baseSpec(tool, "outputs/resource-result.txt");
  const loom::BlobDigest host =
      take(__func__,
           deriveExternalToolExecutionBindingDigest(spec.tool, spec.runtime));
  const PreparedExternalToolInvocation prepared =
      take(__func__, finalizeExternalToolInvocationBundle(
                         (root / "resource-binding").string(), spec));
  require(__func__,
          take(__func__, deriveExternalToolExecutionBindingDigest(prepared)) ==
              host,
          "prepared and resolved binding projections differ");

  InvocationRuntimeBinding provenanceOnly = spec.runtime;
  provenanceOnly.rejectedCompositions = {"unselected alternative"};
  require(__func__,
          take(__func__, deriveExternalToolExecutionBindingDigest(
                             spec.tool, provenanceOnly)) == host,
          "rejected fallback provenance changed the execution resource");

  ResolvedToolBinding changedTool = spec.tool;
  changedTool.version = "Fake EDA 2.0";
  require(__func__,
          take(__func__, deriveExternalToolExecutionBindingDigest(
                             changedTool, spec.runtime)) != host,
          "a changed exact tool binding retained the same resource key");

  InvocationRuntimeBinding invalid = spec.runtime;
  invalid.kind = static_cast<InvocationRuntimeKind>(99);
  requireFailureContains(
      __func__, deriveExternalToolExecutionBindingDigest(spec.tool, invalid),
      "runtime binding kind");
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

  const PreparedExternalToolInvocation preparedFirst = take(
      __func__, finalizeExternalToolInvocationBundle(first.string(), spec));
  const PreparedExternalToolInvocation preparedSecond = take(
      __func__, finalizeExternalToolInvocationBundle(second.string(), spec));
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
      take(__func__, executeExternalToolInvocationBundle(preparedFirst));
  require(__func__, status == 0, "successful bundle returned a failure status");
  require(__func__, readFile(first / output) == "literal; $(touch never)",
          "command arguments were not preserved as data");
  require(__func__, !std::filesystem::exists(first / "never"),
          "command data was evaluated by the shell");
  InvocationCompletion completion =
      take(__func__, loadExternalToolInvocationCompletion(preparedFirst));
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
      loadExternalToolInvocationCompletion(preparedFirst);
  require(__func__, !noncanonical,
          "noncanonical completion record was accepted");
  llvm::consumeError(noncanonical.takeError());

  writeExecutable(second / "drivers" / "driver with ' quote.tcl", "tampered\n");
  const int tamperedStatus =
      take(__func__, executeExternalToolInvocationBundle(preparedSecond));
  require(__func__, tamperedStatus != 0,
          "bundle with tampered materialized content returned success");
  InvocationCompletion tampered =
      take(__func__, loadExternalToolInvocationCompletion(preparedSecond));
  require(__func__,
          tampered.status == InvocationCompletionStatus::BundleContentMismatch,
          "tampered bundle content was not distinguished in completion");
}

void toolProducedExecutableLifecycle(const std::filesystem::path &root,
                                     const std::filesystem::path &tool) {
  const std::string produced = "work/generated/simulator";
  const std::string output = "outputs/generated-result.txt";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, output);
  spec.commands = {{tool.string(), "compile", produced},
                   {produced, output, "fresh"}};
  spec.toolProducedExecutables = {produced};
  const std::filesystem::path bundle = root / "produced-executable";
  const PreparedExternalToolInvocation prepared = take(
      __func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
  require(__func__,
          std::filesystem::is_directory((bundle / produced).parent_path()),
          "tool-produced executable parent was not materialized");
  writeExecutable(bundle / produced,
                  "#!/usr/bin/env bash\nprintf stale >\"$1\"\n");
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(prepared)) == 0,
          "fresh tool-produced executable did not run");
  require(__func__, readFile(bundle / output) == "fresh",
          "launcher executed a stale generated program");
  const std::string manifest = readFile(bundle / "tool-invocation.json");
  const std::string currentVersion =
      jsonVersionMember(externalToolInvocationManifestVersion);
  require(__func__,
          manifest.find(currentVersion) != std::string::npos &&
              manifest.find("\"tool_produced_executables\"") !=
                  std::string::npos,
          "produced executable closure is absent from the current manifest");
  take(__func__,
       importExternalToolInvocationBundle(prepared, importExpectation(spec)));

  ExternalToolInvocationBundleSpec noTool = spec;
  noTool.commands = {{produced, output, "invalid"},
                     {tool.string(), "compile", produced}};
  requireFailureContains(__func__,
                         finalizeExternalToolInvocationBundle(
                             (root / "produced-before-tool").string(), noTool),
                         "no preceding frozen-tool command");

  ExternalToolInvocationBundleSpec unused = spec;
  unused.commands = {{tool.string(), "no-output"}};
  requireFailureContains(
      __func__,
      finalizeExternalToolInvocationBundle(
          (root / "unused-produced-executable").string(), unused),
      "must be used");

  ExternalToolInvocationBundleSpec unlisted = spec;
  unlisted.toolProducedExecutables.clear();
  requireFailureContains(
      __func__,
      finalizeExternalToolInvocationBundle(
          (root / "unlisted-produced-executable").string(), unlisted),
      "manifest-listed");

  const std::array<std::string, 3> invalidPaths{
      "/absolute/simulator", "work/../escape", "drivers/simulator"};
  for (std::size_t index = 0; index != invalidPaths.size(); ++index) {
    const std::string &invalidPath = invalidPaths[index];
    ExternalToolInvocationBundleSpec invalid = spec;
    invalid.commands = {{tool.string(), "compile", invalidPath},
                        {invalidPath, output, "invalid"}};
    invalid.toolProducedExecutables = {invalidPath};
    requireFailure(
        __func__,
        finalizeExternalToolInvocationBundle(
            (root / ("invalid-produced-" + std::to_string(index))).string(),
            invalid),
        "invalid tool-produced executable path was accepted");
  }

  ExternalToolInvocationBundleSpec nonExecutable = spec;
  nonExecutable.commands = {{tool.string(), "compile-nonexec", produced},
                            {produced, output, "invalid"}};
  const std::filesystem::path nonExecutableBundle = root / "produced-nonexec";
  const PreparedExternalToolInvocation nonExecutablePrepared =
      take(__func__, finalizeExternalToolInvocationBundle(
                         nonExecutableBundle.string(), nonExecutable));
  require(
      __func__,
      take(__func__,
           executeExternalToolInvocationBundle(nonExecutablePrepared)) ==
          launcherExitCode(
              InvocationLauncherExitCode::ToolProducedExecutableUnavailable),
      "non-executable generated program was not rejected");
  InvocationCompletion nonExecutableCompletion = take(
      __func__, loadExternalToolInvocationCompletion(nonExecutablePrepared));
  require(__func__,
          nonExecutableCompletion.status ==
              InvocationCompletionStatus::ToolExit,
          "non-executable generated program has the wrong completion status");

  ExternalToolInvocationBundleSpec symlink = spec;
  symlink.commands = {{tool.string(), "compile-symlink", produced},
                      {produced, output, "invalid"}};
  const std::filesystem::path symlinkBundle = root / "produced-symlink";
  const PreparedExternalToolInvocation symlinkPrepared = take(
      __func__,
      finalizeExternalToolInvocationBundle(symlinkBundle.string(), symlink));
  require(
      __func__,
      take(__func__, executeExternalToolInvocationBundle(symlinkPrepared)) ==
          launcherExitCode(InvocationLauncherExitCode::BundleContentMismatch),
      "symlink generated program was not rejected");
  InvocationCompletion symlinkCompletion =
      take(__func__, loadExternalToolInvocationCompletion(symlinkPrepared));
  require(__func__,
          symlinkCompletion.status ==
              InvocationCompletionStatus::BundleContentMismatch,
          "symlink generated program was not an integrity failure");
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

  const PreparedExternalToolInvocation prepared = take(
      __func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
  const int status =
      take(__func__, executeExternalToolInvocationBundle(prepared));
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
  const ExternalFileRequirement requirement{"asic.liberty",
                                            fingerprint(contents)};
  std::vector<ResolvedExternalFile> resolved =
      take(__func__, resolveExternalFiles({requirement}, config));

  const std::filesystem::path bundle = root / "external-file-bundle";
  const std::filesystem::path output = "outputs/external-file-result.txt";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, output);
  spec.externalFiles = resolved;
  const PreparedExternalToolInvocation prepared = take(
      __func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));

  const std::string manifest = readFile(bundle / "tool-invocation.json");
  require(__func__,
          manifest.find("\"provider_input_slot\": \"asic.liberty\"") !=
                  std::string::npos &&
              manifest.find("\"local_file_key\": \"asic_typical_lib\"") !=
                  std::string::npos &&
              manifest.find(external.string()) != std::string::npos &&
              manifest.find(formatExternalFileFingerprint(
                  resolved.front().fingerprint)) != std::string::npos,
          "resolved external file provenance is absent from the manifest");

  int status = take(__func__, executeExternalToolInvocationBundle(prepared));
  require(__func__, status == 0,
          "bundle rejected an unchanged resolved external file");
  require(__func__, readFile(bundle / output) == "literal; $(touch never)",
          "bundle with an external file did not execute the tool");

  ExternalToolInvocationImportExpectation expected = importExpectation(spec);
  take(__func__, importExternalToolInvocationBundle(prepared, expected));
  ExternalToolInvocationImportExpectation wrong = expected;
  wrong.externalInputs.front().fingerprint = fingerprint("different-bytes");
  requireFailure(__func__, importExternalToolInvocationBundle(prepared, wrong),
                 "a wrong external input fingerprint was accepted");

  writeText(external, "vendor-library-mutated\n");
  const std::filesystem::path changedBundle = root / "changed-external-bundle";
  const PreparedExternalToolInvocation preparedChanged =
      take(__func__,
           finalizeExternalToolInvocationBundle(changedBundle.string(), spec));
  status = take(__func__, executeExternalToolInvocationBundle(preparedChanged));
  require(__func__, status != 0,
          "bundle accepted a changed resolved external file");
  InvocationCompletion completion =
      take(__func__, loadExternalToolInvocationCompletion(preparedChanged));
  require(__func__,
          completion.status ==
              InvocationCompletionStatus::BundleContentMismatch,
          "changed external content was not distinguished in completion");
  require(__func__, !std::filesystem::exists(changedBundle / output),
          "the tool ran after external content verification failed");
}

void externalFileTreeIsRevalidated(const std::filesystem::path &root,
                                   const std::filesystem::path &tool) {
  const std::filesystem::path tree = root / "external-tree" / "reference.ndm";
  writeText(tree / "pcat", "catalog");
  writeText(tree / "parts" / "p0", "payload");
  const ExternalFileTreeRequirement requirement{
      "reference_library",
      {{"parts/p0", fingerprint("payload")}, {"pcat", fingerprint("catalog")}}};
  LocalToolConfig config;
  config.externalFileTrees.emplace("saed_reference", tree.string());
  std::vector<ResolvedExternalFileTree> resolved =
      take(__func__, resolveExternalFileTrees({requirement}, config));

  const std::filesystem::path bundle = root / "external-tree-bundle";
  ExternalToolInvocationBundleSpec spec =
      baseSpec(tool, "outputs/external-tree-result.txt");
  spec.externalFileTrees = resolved;
  const PreparedExternalToolInvocation prepared = take(
      __func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
  const std::string manifest = readFile(bundle / "tool-invocation.json");
  const std::string currentVersion =
      jsonVersionMember(externalToolInvocationManifestVersion);
  require(__func__,
          manifest.find(currentVersion) != std::string::npos &&
              manifest.find("\"external_file_trees\"") != std::string::npos &&
              manifest.find("\"path\": \"parts/p0\"") != std::string::npos,
          "resolved external file tree is absent from the current manifest");
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(prepared)) == 0,
          "bundle rejected an unchanged external file tree");
  take(__func__,
       importExternalToolInvocationBundle(prepared, importExpectation(spec)));

  ExternalToolInvocationImportExpectation wrong = importExpectation(spec);
  wrong.externalFileTrees.front().members.front().fingerprint =
      fingerprint("wrong");
  requireFailureContains(__func__,
                         importExternalToolInvocationBundle(prepared, wrong),
                         "external file trees");

  writeText(tree / "extra", "extra");
  const std::filesystem::path changed = root / "changed-external-tree-bundle";
  const PreparedExternalToolInvocation changedPrepared = take(
      __func__, finalizeExternalToolInvocationBundle(changed.string(), spec));
  require(
      __func__,
      take(__func__, executeExternalToolInvocationBundle(changedPrepared)) != 0,
      "bundle accepted changed external file tree membership");
  require(__func__,
          take(__func__, loadExternalToolInvocationCompletion(changedPrepared))
                  .status == InvocationCompletionStatus::BundleContentMismatch,
          "changed external file tree was not an integrity failure");

  std::filesystem::remove(tree / "extra");
  writeText(tree / "pcat", "changed");
  const std::filesystem::path changedMember =
      root / "changed-external-tree-member-bundle";
  const PreparedExternalToolInvocation changedMemberPrepared =
      take(__func__,
           finalizeExternalToolInvocationBundle(changedMember.string(), spec));
  require(__func__,
          take(__func__,
               executeExternalToolInvocationBundle(changedMemberPrepared)) != 0,
          "bundle accepted changed external file tree member content");
  require(__func__,
          take(__func__,
               loadExternalToolInvocationCompletion(changedMemberPrepared))
                  .status == InvocationCompletionStatus::BundleContentMismatch,
          "changed external file tree member was not an integrity failure");
}

void missingOutputIsRecorded(const std::filesystem::path &root,
                             const std::filesystem::path &tool) {
  const std::filesystem::path bundle = root / "missing-output";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, "outputs/present.txt");
  spec.declaredOutputs.push_back("outputs/missing.txt");
  const PreparedExternalToolInvocation prepared = take(
      __func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
  const int status =
      take(__func__, executeExternalToolInvocationBundle(prepared));
  require(__func__, status != 0,
          "bundle with a missing output returned success");
  InvocationCompletion completion =
      take(__func__, loadExternalToolInvocationCompletion(prepared));
  require(__func__,
          completion.status == InvocationCompletionStatus::MissingOutput,
          "missing output was not distinguished in completion");
  require(__func__, completion.outputDigests.empty(),
          "failed completion retained output digests");
  require(__func__,
          std::filesystem::exists(bundle / spec.declaredOutputs.front()),
          "the partial-output fixture did not produce its first output");
  ExternalToolInvocationAttemptOutcome partial = take(
      __func__,
      importExternalToolInvocationAttempt(prepared, importExpectation(spec)));
  const auto *failed =
      std::get_if<FailedExternalToolInvocationAttempt>(&partial);
  require(__func__,
          failed &&
              failed->status == InvocationCompletionStatus::MissingOutput &&
              failed->exitCode == status,
          "partial output was not returned as the exact failed attempt");
  require(
      __func__,
      !std::holds_alternative<ImportedExternalToolInvocationBundle>(partial),
      "partial output exposed a readable declared-output snapshot");

  // A planted stale output never survives as a fresh result: the rerun-safe
  // prelude removes declared outputs before execution, so the tool's missing
  // production is reported rather than the foreign bytes being adopted.
  const std::filesystem::path staleBundle = root / "stale-output";
  ExternalToolInvocationBundleSpec staleSpec =
      baseSpec(tool, "outputs/required.txt");
  staleSpec.commands = {{tool.string(), "no-output"}};
  const PreparedExternalToolInvocation preparedStale =
      take(__func__, finalizeExternalToolInvocationBundle(staleBundle.string(),
                                                          staleSpec));
  writeText(staleBundle / staleSpec.declaredOutputs.front(), "stale");
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(preparedStale)) !=
              0,
          "a preexisting output was accepted as a fresh tool result");
  completion =
      take(__func__, loadExternalToolInvocationCompletion(preparedStale));
  require(__func__,
          completion.status == InvocationCompletionStatus::MissingOutput,
          "a planted stale output was not removed before execution");
  requireFailure(__func__,
                 importExternalToolInvocationBundle(
                     preparedStale, importExpectation(staleSpec)),
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
  const PreparedExternalToolInvocation prepared = take(
      __func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
  const std::string manifest = readFile(bundle / "tool-invocation.json");
  require(__func__,
          manifest.find("\"accepted_exit_codes\"") != std::string::npos &&
              manifest.find("\"selected_output_line_substring\"") !=
                  std::string::npos,
          "version normalization contract is absent from the manifest");
  const int status =
      take(__func__, executeExternalToolInvocationBundle(prepared));
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
  llvm::Expected<PreparedExternalToolInvocation> result =
      finalizeExternalToolInvocationBundle(bundle.string(), spec);
  require(__func__, !result, "escaping path was accepted");
  llvm::consumeError(result.takeError());
  require(__func__, !std::filesystem::exists(bundle),
          "failed finalization published a bundle");
}

void malformedSemanticContractLeavesNoBundle(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  const std::filesystem::path bundle = root / "malformed-contract";
  ExternalToolInvocationBundleSpec spec =
      baseSpec(tool, "outputs/required.txt");
  spec.semanticContract.resultImporterIdentity = "not-a-digest";
  requireFailureContains(
      __func__, finalizeExternalToolInvocationBundle(bundle.string(), spec),
      "result importer identity");
  require(__func__, !std::filesystem::exists(bundle),
          "malformed semantic contract published a bundle");
}

void conflictingPathLeavesNoBundle(const std::filesystem::path &root,
                                   const std::filesystem::path &tool) {
  const std::filesystem::path bundle = root / "conflicting-bundle";
  ExternalToolInvocationBundleSpec spec =
      baseSpec(tool, "outputs/completion.json/child");
  llvm::Expected<PreparedExternalToolInvocation> result =
      finalizeExternalToolInvocationBundle(bundle.string(), spec);
  require(__func__, !result, "a path below the completion record was accepted");
  llvm::consumeError(result.takeError());
  require(__func__, !std::filesystem::exists(bundle),
          "conflicting finalization published a bundle");
}

void internalVersionPathCannotBeDeclared(const std::filesystem::path &root,
                                         const std::filesystem::path &tool) {
  const std::filesystem::path bundle = root / "reserved-version-output";
  ExternalToolInvocationBundleSpec spec =
      baseSpec(tool, "outputs/.loom-tool-version");
  llvm::Expected<PreparedExternalToolInvocation> result =
      finalizeExternalToolInvocationBundle(bundle.string(), spec);
  require(__func__, !result, "the internal version output path was accepted");
  llvm::consumeError(result.takeError());
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
  const PreparedExternalToolInvocation preparedFirst =
      take(__func__,
           finalizeExternalToolInvocationBundle(first.string(), firstSpec));
  ExternalToolInvocationBundleSpec secondSpec =
      baseSpec(tool, "outputs/result.txt");
  const PreparedExternalToolInvocation preparedSecond =
      take(__func__,
           finalizeExternalToolInvocationBundle(second.string(), secondSpec));

  const pid_t child = ::fork();
  require(__func__, child >= 0, "could not fork the first bundle execution");
  if (child == 0) {
    llvm::Expected<int> status =
        executeExternalToolInvocationBundle(preparedFirst);
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
          take(__func__, executeExternalToolInvocationBundle(preparedSecond)) ==
              0,
          "an independent bundle could not execute while another ran");
  writeText(first / "outputs" / "release", "");
  const int firstStatus = waitForChild(__func__, child);
  require(__func__, WIFEXITED(firstStatus) && WEXITSTATUS(firstStatus) == 0,
          "the blocked execution did not complete after release");
  take(__func__, importExternalToolInvocationBundle(
                     preparedFirst, importExpectation(firstSpec)));
  take(__func__, importExternalToolInvocationBundle(
                     preparedSecond, importExpectation(secondSpec)));
}

void unremovableDeclaredOutputFailsBeforeToolEntry(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  const std::filesystem::path bundle = root / "unremovable-output";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, "outputs/result.txt");
  spec.commands = {{tool.string(), "block", "outputs/tool-entry.log",
                    "outputs/release", "outputs/result.txt"}};
  const PreparedExternalToolInvocation prepared = take(
      __func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
  // The release marker lets the blocking tool proceed immediately if it is
  // ever entered; the directory makes the stale-output removal fail.
  writeText(bundle / "outputs" / "release", "");
  std::filesystem::create_directories(bundle / "outputs" / "result.txt");

  const int status =
      take(__func__, executeExternalToolInvocationBundle(prepared));
  require(__func__, status != 0,
          "an unremovable declared output did not fail the execution");
  require(__func__,
          !std::filesystem::exists(bundle / "outputs" / "tool-entry.log"),
          "the external tool was entered despite the failed removal");
  InvocationCompletion completion =
      take(__func__, loadExternalToolInvocationCompletion(prepared));
  require(__func__,
          completion.status ==
              InvocationCompletionStatus::BundleContentMismatch,
          "an unremovable declared output did not fail integrity validation");
  requireFailure(
      __func__,
      importExternalToolInvocationBundle(prepared, importExpectation(spec)),
      "stale bytes were imported after a failed removal");
}

void sequentialReexecutionIsCallerOwned(const std::filesystem::path &root,
                                        const std::filesystem::path &tool) {
  const std::filesystem::path bundle = root / "reexecuted";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, "outputs/result.txt");
  const PreparedExternalToolInvocation prepared = take(
      __func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));

  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(prepared)) == 0,
          "the first execution failed");
  take(__func__,
       importExternalToolInvocationBundle(prepared, importExpectation(spec)));

  // The caller chooses to execute the same prepared bundle again: the
  // completion and declared outputs are republished by the new execution.
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(prepared)) == 0,
          "a caller-chosen sequential re-execution was refused");
  take(__func__,
       importExternalToolInvocationBundle(prepared, importExpectation(spec)));

  // An interrupted attempt leaves no completion; re-execution still works,
  // and the incomplete state stays import-rejected in between.
  std::filesystem::remove(bundle / "outputs" / "completion.json");
  requireFailure(
      __func__,
      importExternalToolInvocationBundle(prepared, importExpectation(spec)),
      "an interrupted bundle remained importable");
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(prepared)) == 0,
          "re-execution after an interrupted attempt failed");
  take(__func__,
       importExternalToolInvocationBundle(prepared, importExpectation(spec)));

  // No Loom-owned claim or retry authority remains in the generated script.
  const std::string script = readFile(bundle / "run.sh");
  require(
      __func__,
      script.find(".loom-execution-started") == std::string::npos &&
          script.find("exit " + std::to_string(launcherExitCode(
                                    InvocationLauncherExitCode::
                                        ToolProducedExecutableUnavailable))) ==
              std::string::npos,
      "run.sh retained an execution claim or retry authority");
}

void successfulImportIsExactAndOutputSafe(const std::filesystem::path &root,
                                          const std::filesystem::path &tool) {
  const std::filesystem::path bundle = root / "strict-import";
  const std::filesystem::path output = "outputs/imported.txt";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, output);
  const PreparedExternalToolInvocation prepared = take(
      __func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(prepared)) == 0,
          "strict-import bundle did not execute");

  ExternalToolInvocationImportExpectation expected = importExpectation(spec);
  ExternalToolInvocationAttemptOutcome attempt =
      take(__func__, importExternalToolInvocationAttempt(prepared, expected));
  const auto *snapshot =
      std::get_if<ImportedExternalToolInvocationBundle>(&attempt);
  require(__func__, snapshot != nullptr,
          "successful attempt did not return an imported output snapshot");
  require(__func__,
          take(__func__, readExternalToolInvocationDeclaredOutput(
                             *snapshot, output.generic_string())) ==
              "literal; $(touch never)",
          "declared output bytes were not imported exactly");
  requireFailure(__func__,
                 readExternalToolInvocationDeclaredOutput(
                     *snapshot, "outputs/not-declared.txt"),
                 "an undeclared output was readable");

  ImportedExternalToolInvocationBundle imported =
      take(__func__, importExternalToolInvocationBundle(prepared, expected));
  require(__func__,
          take(__func__, readExternalToolInvocationDeclaredOutput(
                             imported, output.generic_string())) ==
              "literal; $(touch never)",
          "the success-only import wrapper changed successful output bytes");

  ExternalToolInvocationImportExpectation wrong = expected;
  wrong.semanticContract.providerIdentity = "other-provider@1";
  requireFailure(__func__, importExternalToolInvocationBundle(prepared, wrong),
                 "a wrong provider identity was accepted");
  wrong = expected;
  wrong.semanticContract.semanticClosure = SemanticInvocationClosure(
      loom::ArtifactRootReference{"loom.test_request",
                                  {1, 0},
                                  take(__func__, loom::parseArtifactIdentityHex(
                                                     std::string(64, '3')))});
  requireFailure(__func__, importExternalToolInvocationBundle(prepared, wrong),
                 "a wrong semantic closure was accepted");
  wrong = expected;
  wrong.semanticContract.resultImporterIdentity = std::string(64, 'b');
  requireFailure(__func__, importExternalToolInvocationBundle(prepared, wrong),
                 "a wrong result importer identity was accepted");
  wrong = expected;
  wrong.semanticInputs.front().sourceArtifact = loom::ArtifactRootReference{
      "loom.test_input",
      {1, 0},
      take(__func__, loom::parseArtifactIdentityHex(std::string(64, '2')))};
  requireFailure(__func__, importExternalToolInvocationBundle(prepared, wrong),
                 "a wrong semantic input Artifact reference was accepted");
  wrong = expected;
  wrong.semanticInputs.front().contentDigest = blobDigest("other-input-bytes");
  requireFailure(__func__, importExternalToolInvocationBundle(prepared, wrong),
                 "a wrong semantic input content digest was accepted");

  const std::filesystem::path substituted = root / "strict-import-substituted";
  ExternalToolInvocationBundleSpec substitutedSpec = spec;
  for (MaterializedBundleFile &file : substitutedSpec.files)
    if (file.sourceArtifact)
      file.contents = "substituted-input-bytes";
  const PreparedExternalToolInvocation preparedSubstituted =
      take(__func__, finalizeExternalToolInvocationBundle(substituted.string(),
                                                          substitutedSpec));
  require(__func__,
          take(__func__,
               executeExternalToolInvocationBundle(preparedSubstituted)) == 0,
          "content-substituted bundle did not execute");
  requireFailure(
      __func__,
      importExternalToolInvocationBundle(preparedSubstituted, expected),
      "same-ref semantic input content substitution was accepted");

  wrong = expected;
  wrong.declaredOutputs.push_back("outputs/unexpected.txt");
  requireFailure(__func__, importExternalToolInvocationBundle(prepared, wrong),
                 "wrong declared output membership was accepted");

  const std::string completionBefore =
      readFile(bundle / "outputs" / "completion.json");
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(prepared)) == 0,
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
    const PreparedExternalToolInvocation prepared = take(
        __func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
    require(__func__,
            take(__func__, executeExternalToolInvocationBundle(prepared)) == 0,
            "unsafe-output bundle did not execute");
    return std::tuple(bundle, std::move(spec), prepared);
  };
  const std::filesystem::path outside = root / "outside-output.txt";
  writeText(outside, "outside");

  auto [symlinkBundle, symlinkSpec, symlinkPrepared] =
      makeExecutedBundle("symlink-output");
  std::filesystem::remove(symlinkBundle / symlinkSpec.declaredOutputs.front());
  std::filesystem::create_symlink(
      outside, symlinkBundle / symlinkSpec.declaredOutputs.front());
  requireFailure(__func__,
                 importExternalToolInvocationAttempt(
                     symlinkPrepared, importExpectation(symlinkSpec)),
                 "a symlinked declared output was imported");

  auto [directoryBundle, directorySpec, directoryPrepared] =
      makeExecutedBundle("directory-output");
  std::filesystem::remove(directoryBundle /
                          directorySpec.declaredOutputs.front());
  std::filesystem::create_directory(directoryBundle /
                                    directorySpec.declaredOutputs.front());
  requireFailure(__func__,
                 importExternalToolInvocationAttempt(
                     directoryPrepared, importExpectation(directorySpec)),
                 "a directory declared output was imported");

  auto [changedBundle, changedSpec, changedPrepared] =
      makeExecutedBundle("changed-output");
  writeText(changedBundle / changedSpec.declaredOutputs.front(), "changed");
  requireFailure(__func__,
                 importExternalToolInvocationAttempt(
                     changedPrepared, importExpectation(changedSpec)),
                 "an output changed after completion was imported");
}

void strictImportRejectsAttemptTampering(const std::filesystem::path &root,
                                         const std::filesystem::path &tool) {
  const auto makeExecutedBundle = [&](llvm::StringRef name) {
    const std::filesystem::path bundle = root / name.str();
    ExternalToolInvocationBundleSpec spec =
        baseSpec(tool, "outputs/result.txt");
    const PreparedExternalToolInvocation prepared = take(
        __func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
    require(__func__,
            take(__func__, executeExternalToolInvocationBundle(prepared)) == 0,
            "import-negative bundle did not execute");
    return std::tuple(bundle, std::move(spec), prepared);
  };

  auto [noncanonicalBundle, noncanonicalSpec, noncanonicalPrepared] =
      makeExecutedBundle("noncanonical-manifest");
  const std::filesystem::path noncanonicalManifest =
      noncanonicalBundle / "tool-invocation.json";
  writeText(noncanonicalManifest, " " + readFile(noncanonicalManifest));
  // The handle must bind the tampered bytes exactly, or the import stops at
  // the prepared-handle digest check instead of the canonical parser.
  const PreparedExternalToolInvocation noncanonicalHandle{
      noncanonicalBundle.string(), blobDigest(readFile(noncanonicalManifest))};
  requireFailure(__func__,
                 importExternalToolInvocationBundle(
                     noncanonicalHandle, importExpectation(noncanonicalSpec)),
                 "a noncanonical manifest was accepted");

  auto [unknownBundle, unknownSpec, unknownPrepared] =
      makeExecutedBundle("unknown-manifest");
  const std::filesystem::path unknownManifest =
      unknownBundle / "tool-invocation.json";
  std::string unknownText = readFile(unknownManifest);
  const std::size_t version = unknownText.find(
      "  " + jsonVersionMember(externalToolInvocationManifestVersion) + ",");
  require(__func__, version != std::string::npos,
          "cannot locate manifest version field");
  unknownText.insert(version, "  \"unknown\": true,\n");
  writeText(unknownManifest, unknownText);
  const PreparedExternalToolInvocation unknownHandle{
      unknownBundle.string(), blobDigest(readFile(unknownManifest))};
  requireFailure(__func__,
                 importExternalToolInvocationBundle(
                     unknownHandle, importExpectation(unknownSpec)),
                 "a manifest with an unknown field was accepted");

  auto [firstBundle, firstSpec, firstPrepared] =
      makeExecutedBundle("manifest-binding-a");
  ExternalToolInvocationBundleSpec secondSpec = firstSpec;
  secondSpec.tool.source = ToolBindingSource::EnvironmentPath;
  const std::filesystem::path secondDistinct =
      root / "manifest-binding-distinct";
  const PreparedExternalToolInvocation preparedSecondDistinct =
      take(__func__, finalizeExternalToolInvocationBundle(
                         secondDistinct.string(), secondSpec));
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(
                             preparedSecondDistinct)) == 0,
          "distinct manifest bundle did not execute");
  writeText(firstBundle / "tool-invocation.json",
            readFile(secondDistinct / "tool-invocation.json"));
  // The handle binds the exact manifest bytes now present in the directory,
  // so the import reaches the completion-to-manifest binding check.
  const PreparedExternalToolInvocation swappedHandle{
      firstBundle.string(), preparedSecondDistinct.manifestDigest};
  requireFailure(__func__,
                 importExternalToolInvocationBundle(
                     swappedHandle, importExpectation(firstSpec)),
                 "completion was accepted with a different manifest");

  auto [malformedBundle, malformedSpec, malformedPrepared] =
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
  const PreparedExternalToolInvocation malformedHandle{
      malformedBundle.string(), blobDigest(readFile(malformedManifest))};
  requireFailure(__func__,
                 importExternalToolInvocationBundle(
                     malformedHandle, importExpectation(malformedSpec)),
                 "malformed nested manifest fields were accepted");

  const std::filesystem::path incomplete = root / "incomplete-import";
  ExternalToolInvocationBundleSpec incompleteSpec =
      baseSpec(tool, "outputs/result.txt");
  const PreparedExternalToolInvocation preparedIncomplete =
      take(__func__, finalizeExternalToolInvocationBundle(incomplete.string(),
                                                          incompleteSpec));
  requireFailure(__func__,
                 importExternalToolInvocationBundle(
                     preparedIncomplete, importExpectation(incompleteSpec)),
                 "an invocation without completion was imported");
  const std::filesystem::path failed = root / "failed-import";
  ExternalToolInvocationBundleSpec failedSpec =
      baseSpec(tool, "outputs/result.txt");
  failedSpec.commands = {{tool.string(), "no-output"}};
  const PreparedExternalToolInvocation preparedFailed =
      take(__func__,
           finalizeExternalToolInvocationBundle(failed.string(), failedSpec));
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(preparedFailed)) !=
              0,
          "failed-import bundle unexpectedly succeeded");
  requireFailure(__func__,
                 importExternalToolInvocationBundle(
                     preparedFailed, importExpectation(failedSpec)),
                 "a non-success completion was imported");
}

void typedClosureIsExactAndLegacyManifestIsRejected(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  // The Evaluation closure form round-trips through execution and strict
  // import.
  const std::filesystem::path bundle = root / "evaluation-closure";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, "outputs/result.txt");
  spec.semanticContract.semanticClosure = SemanticInvocationClosure(
      loom::ArtifactRootReference{"loom.test_request",
                                  {1, 0},
                                  take(__func__, loom::parseArtifactIdentityHex(
                                                     std::string(64, '7')))});
  const PreparedExternalToolInvocation prepared = take(
      __func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(prepared)) == 0,
          "evaluation-closure bundle did not execute");
  take(__func__,
       importExternalToolInvocationBundle(prepared, importExpectation(spec)));

  // A candidate-generator closure without owner bytes fails closed at
  // finalization.
  ExternalToolInvocationBundleSpec emptyClosure =
      baseSpec(tool, "outputs/result.txt");
  emptyClosure.semanticContract.semanticClosure =
      SemanticInvocationClosure(CandidateGeneratorInvocationClosure{});
  requireFailureContains(__func__,
                         finalizeExternalToolInvocationBundle(
                             (root / "empty-closure").string(), emptyClosure),
                         "empty owner bytes");

  // Compatible typed-closure manifests have neither extension field and
  // remain importable under the current reader.
  const std::filesystem::path compatible = root / "compatible-manifest";
  take(__func__,
       finalizeExternalToolInvocationBundle(compatible.string(), spec));
  const std::filesystem::path compatibleManifest =
      compatible / "tool-invocation.json";
  std::string compatibleText = readFile(compatibleManifest);
  const std::string currentVersion =
      jsonVersionMember(externalToolInvocationManifestVersion);
  const std::size_t compatibleVersion = compatibleText.find(currentVersion);
  const std::string treeField = "  \"external_file_trees\": [],\n";
  const std::size_t treeFieldOffset = compatibleText.find(treeField);
  const std::string producedField = "  \"tool_produced_executables\": [],\n";
  const std::size_t producedFieldOffset = compatibleText.find(producedField);
  require(__func__,
          compatibleVersion != std::string::npos &&
              treeFieldOffset != std::string::npos &&
              producedFieldOffset != std::string::npos,
          "cannot derive the compatible typed-closure manifest fixture");
  compatibleText.replace(
      compatibleVersion, currentVersion.size(),
      jsonVersionMember(kCompatibleTypedClosureManifestVersion));
  compatibleText.erase(treeFieldOffset, treeField.size());
  compatibleText.erase(producedFieldOffset, producedField.size());
  writeText(compatibleManifest, compatibleText);
  const PreparedExternalToolInvocation compatibleHandle{
      compatible.string(), blobDigest(compatibleText)};
  ExternalToolInvocationAttemptOutcome compatibleAttempt =
      take(__func__, importExternalToolInvocationAttempt(
                         compatibleHandle, importExpectation(spec)));
  require(__func__,
          std::holds_alternative<IncompleteExternalToolInvocationAttempt>(
              compatibleAttempt),
          "compatible typed-closure manifest did not reach incomplete import");

  // The unsupported legacy manifest is rejected by name; no upgrade exists.
  const std::filesystem::path legacy = root / "legacy-manifest";
  take(__func__, finalizeExternalToolInvocationBundle(legacy.string(), spec));
  const std::filesystem::path legacyManifest = legacy / "tool-invocation.json";
  std::string legacyText = readFile(legacyManifest);
  const std::size_t version = legacyText.find(currentVersion);
  require(__func__, version != std::string::npos,
          "cannot locate the current manifest version");
  legacyText.replace(version, currentVersion.size(),
                     jsonVersionMember(kUnsupportedLegacyManifestVersion));
  writeText(legacyManifest, legacyText);
  // The handle must bind the tampered bytes exactly, or the import stops at
  // the prepared-handle digest check instead of the version rejection.
  const PreparedExternalToolInvocation legacyHandle{
      legacy.string(), blobDigest(readFile(legacyManifest))};
  requireFailureContains(
      __func__,
      importExternalToolInvocationBundle(legacyHandle, importExpectation(spec)),
      kUnsupportedLegacyManifestVersion.str());

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
  const PreparedExternalToolInvocation unknownFormHandle{
      unknownForm.string(), blobDigest(readFile(unknownManifest))};
  requireFailureContains(__func__,
                         importExternalToolInvocationBundle(
                             unknownFormHandle, importExpectation(spec)),
                         "unknown closure form");
}

void preparedHandleBindsTheExactManifest(const std::filesystem::path &root,
                                         const std::filesystem::path &tool) {
  const std::filesystem::path bundle = root / "prepared-anchor";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, "outputs/result.txt");
  const PreparedExternalToolInvocation prepared = take(
      __func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
  require(__func__,
          prepared.bundleRoot == bundle.string() &&
              prepared.manifestDigest ==
                  blobDigest(readFile(bundle / "tool-invocation.json")),
          "the prepared handle did not bind the exact manifest bytes");
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(prepared)) == 0,
          "prepared-anchor bundle did not execute");
  take(__func__,
       importExternalToolInvocationBundle(prepared, importExpectation(spec)));

  // A handle carrying any other manifest digest never imports the bundle.
  const PreparedExternalToolInvocation wrong{bundle.string(),
                                             blobDigest("wrong")};
  requireFailureContains(
      __func__,
      importExternalToolInvocationBundle(wrong, importExpectation(spec)),
      "prepared handle");

  // A handle bound to one bundle never imports another bundle's directory.
  const std::filesystem::path other = root / "prepared-anchor-other";
  ExternalToolInvocationBundleSpec otherSpec =
      baseSpec(tool, "outputs/other-result.txt");
  const PreparedExternalToolInvocation preparedOther =
      take(__func__,
           finalizeExternalToolInvocationBundle(other.string(), otherSpec));
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(preparedOther)) ==
              0,
          "second prepared-anchor bundle did not execute");
  const PreparedExternalToolInvocation swapped{other.string(),
                                               prepared.manifestDigest};
  requireFailureContains(
      __func__,
      importExternalToolInvocationBundle(swapped, importExpectation(otherSpec)),
      "prepared handle");
}

void executionRejectsSubstitutedBundle(const std::filesystem::path &root,
                                       const std::filesystem::path &tool) {
  const std::filesystem::path bundle = root / "execute-anchor";
  const std::filesystem::path replacement = root / "execute-anchor-other";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, "outputs/result.txt");
  const PreparedExternalToolInvocation prepared = take(
      __func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
  ExternalToolInvocationBundleSpec otherSpec =
      baseSpec(tool, "outputs/other.txt");
  take(__func__,
       finalizeExternalToolInvocationBundle(replacement.string(), otherSpec));
  // A whole self-consistent bundle replacement before execution is rejected
  // against the original prepared handle.
  std::filesystem::remove_all(bundle);
  std::filesystem::rename(replacement, bundle);
  requireFailureContains(__func__,
                         executeExternalToolInvocationBundle(prepared),
                         "prepared handle");
}

void finalizeRejectsNonNormalizedBundleRoot(const std::filesystem::path &root,
                                            const std::filesystem::path &tool) {
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, "outputs/result.txt");
  const std::filesystem::path nonNormalized =
      root / "nested" / ".." / "non-normalized";
  auto result =
      finalizeExternalToolInvocationBundle(nonNormalized.string(), spec);
  require(__func__, !result, "a non-normalized bundle root was finalized");
  llvm::consumeError(result.takeError());
  require(__func__, !std::filesystem::exists(root / "non-normalized"),
          "a rejected finalization published a bundle");
}

void incompleteAttemptHasATypedImportError(const std::filesystem::path &root,
                                           const std::filesystem::path &tool) {
  const std::filesystem::path bundle = root / "incomplete-attempt";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, "outputs/result.txt");
  const PreparedExternalToolInvocation prepared = take(
      __func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));

  writeText(bundle / spec.declaredOutputs.front(),
            "foreign-output-without-completion");
  ExternalToolInvocationImportExpectation wrong = importExpectation(spec);
  wrong.semanticContract.providerIdentity = "wrong-provider@1";
  requireFailureContains(__func__,
                         importExternalToolInvocationAttempt(prepared, wrong),
                         "provider identity");
  ExternalToolInvocationAttemptOutcome attempt = take(
      __func__,
      importExternalToolInvocationAttempt(prepared, importExpectation(spec)));
  require(
      __func__,
      std::holds_alternative<IncompleteExternalToolInvocationAttempt>(attempt),
      "an absent completion was not returned as an incomplete attempt");
  require(
      __func__,
      !std::holds_alternative<ImportedExternalToolInvocationBundle>(attempt),
      "an incomplete attempt exposed a readable declared-output snapshot");

  // A prepared bundle that was never executed has no completion record: the
  // import failure is the one typed incomplete-attempt error.
  llvm::Expected<ImportedExternalToolInvocationBundle> incomplete =
      importExternalToolInvocationBundle(prepared, importExpectation(spec));
  require(__func__, !incomplete, "an unexecuted bundle was importable");
  bool typedIncomplete = false;
  llvm::Error remainder =
      llvm::handleErrors(incomplete.takeError(),
                         [&](const IncompleteExternalToolInvocationError &) {
                           typedIncomplete = true;
                         });
  require(__func__, typedIncomplete,
          "an absent completion record was not typed as incomplete");
  llvm::consumeError(std::move(remainder));

  // A present but malformed completion record is a bundle integrity failure,
  // not an incomplete attempt.
  writeText(bundle / "outputs" / "completion.json", "not a completion\n");
  llvm::Expected<ImportedExternalToolInvocationBundle> malformed =
      importExternalToolInvocationBundle(prepared, importExpectation(spec));
  require(__func__, !malformed, "a malformed completion was imported");
  typedIncomplete = false;
  remainder =
      llvm::handleErrors(malformed.takeError(),
                         [&](const IncompleteExternalToolInvocationError &) {
                           typedIncomplete = true;
                         });
  require(__func__, !typedIncomplete,
          "a malformed completion was typed as an incomplete attempt");
  llvm::consumeError(std::move(remainder));
}

void failedAttemptImportIsExpectationBound(const std::filesystem::path &root,
                                           const std::filesystem::path &tool) {
  const std::filesystem::path bundle = root / "failed-attempt";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, "outputs/result.txt");
  spec.commands = {{tool.string(), "timeout"}};
  const PreparedExternalToolInvocation prepared = take(
      __func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(prepared)) ==
              kFixtureToolExitCode,
          "the timeout-like fixture did not preserve its tool exit status");

  std::filesystem::create_directory(bundle / spec.declaredOutputs.front());
  const ExternalToolInvocationImportExpectation expected =
      importExpectation(spec);
  ExternalToolInvocationAttemptOutcome attempt =
      take(__func__, importExternalToolInvocationAttempt(prepared, expected));
  const auto *failed =
      std::get_if<FailedExternalToolInvocationAttempt>(&attempt);
  require(__func__,
          failed && failed->status == InvocationCompletionStatus::ToolExit &&
              failed->exitCode == kFixtureToolExitCode,
          "ToolExit did not preserve its exact status and exit reason");
  require(
      __func__,
      !std::holds_alternative<ImportedExternalToolInvocationBundle>(attempt),
      "a failed attempt exposed a readable declared-output snapshot");

  ExternalToolInvocationImportExpectation wrong = expected;
  wrong.semanticContract.providerIdentity = "wrong-provider@1";
  requireFailureContains(__func__,
                         importExternalToolInvocationAttempt(prepared, wrong),
                         "provider identity");
  wrong = expected;
  wrong.semanticContract.semanticClosure = SemanticInvocationClosure(
      loom::ArtifactRootReference{"loom.test_request",
                                  {1, 0},
                                  take(__func__, loom::parseArtifactIdentityHex(
                                                     std::string(64, '3')))});
  requireFailureContains(__func__,
                         importExternalToolInvocationAttempt(prepared, wrong),
                         "semantic closure");
  wrong = expected;
  wrong.semanticContract.resultImporterIdentity = "wrong-importer@1";
  requireFailureContains(__func__,
                         importExternalToolInvocationAttempt(prepared, wrong),
                         "result importer identity");
  wrong = expected;
  wrong.semanticInputs.front().contentDigest = blobDigest("wrong-input");
  requireFailureContains(__func__,
                         importExternalToolInvocationAttempt(prepared, wrong),
                         "semantic inputs");

  const std::filesystem::path substitute = root / "failed-attempt-substitute";
  ExternalToolInvocationBundleSpec substituteSpec = spec;
  substituteSpec.tool.source = ToolBindingSource::EnvironmentPath;
  const PreparedExternalToolInvocation substitutePrepared =
      take(__func__, finalizeExternalToolInvocationBundle(substitute.string(),
                                                          substituteSpec));
  writeText(bundle / "tool-invocation.json",
            readFile(substitute / "tool-invocation.json"));
  const PreparedExternalToolInvocation substitutedHandle{
      bundle.string(), substitutePrepared.manifestDigest};
  requireFailureContains(
      __func__,
      importExternalToolInvocationAttempt(substitutedHandle,
                                          importExpectation(substituteSpec)),
      "completion does not bind the imported manifest");
}

void persistentResultCacheIsExact(const std::filesystem::path &root,
                                  const std::filesystem::path &tool) {
  const std::filesystem::path cache = root / "persistent-result-cache";
  const std::filesystem::path counter = root / "tool-entry-count";
  require(__func__,
          ::setenv("LOOM_EXTERNAL_TOOL_CACHE_ROOT", cache.c_str(), 1) == 0,
          "could not enable the result cache");

  const std::string output = "outputs/cache-result.txt";
  auto countedSpec = [&](const std::filesystem::path &executable,
                         llvm::StringRef value) {
    ExternalToolInvocationBundleSpec spec = baseSpec(executable, output);
    spec.files.push_back({"drivers/binary-config.bin", std::string("\xff\0", 2),
                          std::nullopt, false});
    spec.commands = {{executable.string(), "counted-run", counter.string(),
                      value.str(), output}};
    return spec;
  };
  ExternalToolInvocationBundleSpec firstSpec = countedSpec(tool, "cached");
  const PreparedExternalToolInvocation first =
      take(__func__, finalizeExternalToolInvocationBundle(
                         (root / "cache-first").string(), firstSpec));
  const ExternalToolInvocationExecutionObservation population =
      take(__func__, executeExternalToolInvocationBundleObserved(first));
  require(__func__,
          population.exitCode == 0 &&
              population.cacheAvailability ==
                  ExternalToolResultCacheAvailability::Available &&
              population.cacheLookup == ExternalToolResultCacheLookup::Miss &&
              population.cacheDiscard ==
                  ExternalToolResultCacheDiscard::NotAttempted &&
              population.cachePublication ==
                  ExternalToolResultCachePublication::Published &&
              !population.waitedForCacheKeyLock &&
              population.invokedExternalTool,
          "the cache population invocation failed");
  require(__func__, readFile(counter) == "1",
          "cache population did not enter the tool exactly once");

  const std::filesystem::path relocatedTool = root / "relocated" / "fake tool";
  writeExecutable(relocatedTool, readFile(tool));
  ExternalToolInvocationBundleSpec relocatedSpec =
      countedSpec(relocatedTool, "cached");
  const PreparedExternalToolInvocation relocated =
      take(__func__, finalizeExternalToolInvocationBundle(
                         (root / "cache-relocated").string(), relocatedSpec));
  const ExternalToolResultCacheKey firstKey =
      take(__func__, deriveExternalToolResultCacheKey(first));
  ExternalToolInvocationBundleSpec diagnosticBase = baseSpec(tool, output);
  diagnosticBase.commands = {
      {tool.string(), "compile", "work/diagnostic-runner"},
      {"work/diagnostic-runner", output, "diagnostic"}};
  diagnosticBase.toolProducedExecutables = {"work/diagnostic-runner"};
  const PreparedExternalToolInvocation diagnosticBaseInvocation = take(
      __func__, finalizeExternalToolInvocationBundle(
                    (root / "cache-diagnostic-base").string(), diagnosticBase));
  ExternalToolInvocationBundleSpec diagnosticSpec = diagnosticBase;
  diagnosticSpec.diagnosticCommandOrdinals = {1};
  const PreparedExternalToolInvocation diagnosticInvocation =
      take(__func__, finalizeExternalToolInvocationBundle(
                         (root / "cache-diagnostic").string(), diagnosticSpec));
  const ExternalToolResultCacheKey diagnosticBaseKey = take(
      __func__, deriveExternalToolResultCacheKey(diagnosticBaseInvocation));
  const ExternalToolResultCacheKey diagnosticKey =
      take(__func__, deriveExternalToolResultCacheKey(diagnosticInvocation));
  require(__func__, diagnosticKey == diagnosticBaseKey,
          "diagnostic verbosity changed the result cache key");
  const std::string diagnosticManifest =
      readFile(std::filesystem::path(diagnosticInvocation.bundleRoot) /
               "tool-invocation.json");
  const bool expectsArgument =
      loom::diagnosticVerbosity() != loom::DiagnosticVerbosity::Disabled;
  require(
      __func__,
      (diagnosticManifest.find(loom::diagnosticVerbosityArgumentPrefix.str()) !=
       std::string::npos) == expectsArgument,
      "diagnostic projection disagrees with the Common-owned level");
  const ExternalToolResultCacheKey relocatedKey =
      take(__func__, deriveExternalToolResultCacheKey(relocated));
  require(__func__, firstKey == relocatedKey,
          "equivalent local tool and bundle paths changed the cache key");
  require(__func__, first.manifestDigest != relocated.manifestDigest,
          "the path-relocated fixtures unexpectedly share one manifest");
  const ExternalToolInvocationExecutionObservation cacheHit =
      take(__func__, executeExternalToolInvocationBundleObserved(relocated));
  require(__func__,
          cacheHit.exitCode == 0 &&
              cacheHit.cacheAvailability ==
                  ExternalToolResultCacheAvailability::Available &&
              cacheHit.cacheLookup == ExternalToolResultCacheLookup::Hit &&
              cacheHit.cacheDiscard ==
                  ExternalToolResultCacheDiscard::NotAttempted &&
              cacheHit.cachePublication ==
                  ExternalToolResultCachePublication::NotAttempted &&
              !cacheHit.invokedExternalTool,
          "the path-relocated cache lookup failed");
  require(__func__, readFile(counter) == "1",
          "a cache hit re-entered the external tool");
  const InvocationCompletion relocatedCompletion =
      take(__func__, loadExternalToolInvocationCompletion(relocated));
  require(__func__,
          relocatedCompletion.manifestDigest == relocated.manifestDigest,
          "a cache hit reused another bundle's completion authority");
  const ImportedExternalToolInvocationBundle relocatedImport =
      take(__func__, importExternalToolInvocationBundle(
                         relocated, importExpectation(relocatedSpec)));
  require(__func__,
          take(__func__, readExternalToolInvocationDeclaredOutput(
                             relocatedImport, output)) == "cached",
          "strict import did not accept the restored declared output");

  const std::filesystem::path changedLauncherTool =
      root / "changed-launcher" / "fake tool";
  writeExecutable(changedLauncherTool, readFile(tool) + "\n");
  ExternalToolInvocationBundleSpec changedLauncher =
      countedSpec(changedLauncherTool, "cached");
  const PreparedExternalToolInvocation launcherInvocation =
      take(__func__,
           finalizeExternalToolInvocationBundle(
               (root / "cache-changed-launcher").string(), changedLauncher));
  const ExternalToolResultCacheKey launcherKey =
      take(__func__, deriveExternalToolResultCacheKey(launcherInvocation));
  require(__func__,
          launcherKey.inputMaterialDigest == firstKey.inputMaterialDigest &&
              launcherKey.executionConfigurationDigest ==
                  firstKey.executionConfigurationDigest &&
              launcherKey.toolVersionDigest != firstKey.toolVersionDigest,
          "launcher mutation did not invalidate only the tool key domain");
  require(__func__,
          take(__func__,
               executeExternalToolInvocationBundle(launcherInvocation)) == 0,
          "changed launcher did not execute as a cache miss");
  require(__func__, readFile(counter) == "2",
          "changed launcher bytes were incorrectly reused");

  const std::filesystem::path entry =
      cache / "entries" /
      loom::formatBlobDigestHex(firstKey.inputMaterialDigest) /
      loom::formatBlobDigestHex(firstKey.executionConfigurationDigest) /
      loom::formatBlobDigestHex(firstKey.toolVersionDigest);
  writeText(entry / "payload" / output, "corrupt");
  const PreparedExternalToolInvocation afterCorruption = take(
      __func__, finalizeExternalToolInvocationBundle(
                    (root / "cache-after-corruption").string(), firstSpec));
  const ExternalToolInvocationExecutionObservation recovered = take(
      __func__, executeExternalToolInvocationBundleObserved(afterCorruption));
  require(__func__,
          recovered.exitCode == 0 &&
              recovered.cacheAvailability ==
                  ExternalToolResultCacheAvailability::Available &&
              recovered.cacheLookup == ExternalToolResultCacheLookup::Miss &&
              recovered.cacheDiscard ==
                  ExternalToolResultCacheDiscard::Discarded &&
              recovered.cachePublication ==
                  ExternalToolResultCachePublication::Published &&
              recovered.invokedExternalTool,
          "a corrupt cache entry did not fall back to real execution");
  require(__func__, readFile(counter) == "3",
          "a corrupt cache entry was adopted as a hit");

  ExternalToolInvocationBundleSpec changedInput = firstSpec;
  for (MaterializedBundleFile &file : changedInput.files)
    if (file.sourceArtifact)
      file.contents = "changed-input-bytes";
  const PreparedExternalToolInvocation inputInvocation = take(
      __func__, finalizeExternalToolInvocationBundle(
                    (root / "cache-changed-input").string(), changedInput));
  const ExternalToolResultCacheKey inputKey =
      take(__func__, deriveExternalToolResultCacheKey(inputInvocation));
  require(__func__,
          inputKey.inputMaterialDigest != firstKey.inputMaterialDigest &&
              inputKey.executionConfigurationDigest ==
                  firstKey.executionConfigurationDigest &&
              inputKey.toolVersionDigest == firstKey.toolVersionDigest,
          "input mutation did not invalidate only the input key domain");
  require(
      __func__,
      take(__func__, executeExternalToolInvocationBundle(inputInvocation)) == 0,
      "changed input did not execute as a cache miss");

  ExternalToolInvocationBundleSpec changedConfiguration =
      countedSpec(tool, "changed-configuration");
  const PreparedExternalToolInvocation configurationInvocation =
      take(__func__, finalizeExternalToolInvocationBundle(
                         (root / "cache-changed-configuration").string(),
                         changedConfiguration));
  const ExternalToolResultCacheKey configurationKey =
      take(__func__, deriveExternalToolResultCacheKey(configurationInvocation));
  require(__func__,
          configurationKey.inputMaterialDigest ==
                  firstKey.inputMaterialDigest &&
              configurationKey.executionConfigurationDigest !=
                  firstKey.executionConfigurationDigest &&
              configurationKey.toolVersionDigest == firstKey.toolVersionDigest,
          "configuration mutation did not invalidate only its key domain");
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(
                             configurationInvocation)) == 0,
          "changed configuration did not execute as a cache miss");

  const std::filesystem::path changedVersionTool =
      root / "changed-version" / "fake tool";
  std::string changedVersionBody = readFile(tool);
  const std::size_t versionOffset = changedVersionBody.find("Fake EDA 1.2");
  require(__func__, versionOffset != std::string::npos,
          "could not derive the changed-version fixture");
  changedVersionBody.replace(versionOffset, std::string("Fake EDA 1.2").size(),
                             "Fake EDA 2.0");
  writeExecutable(changedVersionTool, changedVersionBody);
  ExternalToolInvocationBundleSpec changedVersion =
      countedSpec(changedVersionTool, "cached");
  changedVersion.tool.version = "Fake EDA 2.0";
  const PreparedExternalToolInvocation versionInvocation = take(
      __func__, finalizeExternalToolInvocationBundle(
                    (root / "cache-changed-version").string(), changedVersion));
  const ExternalToolResultCacheKey versionKey =
      take(__func__, deriveExternalToolResultCacheKey(versionInvocation));
  require(__func__,
          versionKey.inputMaterialDigest == firstKey.inputMaterialDigest &&
              versionKey.executionConfigurationDigest ==
                  firstKey.executionConfigurationDigest &&
              versionKey.toolVersionDigest != firstKey.toolVersionDigest,
          "tool-version mutation did not invalidate only its key domain");
  require(__func__,
          take(__func__,
               executeExternalToolInvocationBundle(versionInvocation)) == 0,
          "changed tool version did not execute as a cache miss");
  require(__func__, readFile(counter) == "6",
          "one of the three key-domain changes was incorrectly reused");

  const std::filesystem::path failureCounter = root / "failure-entry-count";
  ExternalToolInvocationBundleSpec failed = baseSpec(tool, output);
  failed.commands = {{tool.string(), "counted-fail", failureCounter.string()}};
  failed.declaredOutputs.clear();
  const PreparedExternalToolInvocation failedInvocation =
      take(__func__, finalizeExternalToolInvocationBundle(
                         (root / "cache-failed").string(), failed));
  for (unsigned attempt = 0; attempt != 2; ++attempt)
    require(__func__,
            take(__func__, executeExternalToolInvocationBundle(
                               failedInvocation)) == kFixtureToolExitCode,
            "failed invocation did not preserve its exact status");
  require(__func__, readFile(failureCounter) == "2",
          "a failed attempt was reused from the result cache");

  const std::filesystem::path undeclaredCounter =
      root / "undeclared-entry-count";
  ExternalToolInvocationBundleSpec undeclared = baseSpec(tool, output);
  undeclared.commands = {{tool.string(), "counted-run-extra",
                          undeclaredCounter.string(), "declared", output,
                          "outputs/undeclared.txt"}};
  for (llvm::StringRef name :
       {"cache-undeclared-first", "cache-undeclared-second"}) {
    const std::filesystem::path bundle = root / name.str();
    const PreparedExternalToolInvocation invocation =
        take(__func__,
             finalizeExternalToolInvocationBundle(bundle.string(), undeclared));
    require(__func__,
            take(__func__, executeExternalToolInvocationBundle(invocation)) ==
                0,
            "an undeclared-output attempt did not preserve tool success");
    require(
        __func__,
        std::filesystem::is_regular_file(bundle / "outputs" / "undeclared.txt"),
        "the undeclared-output fixture did not expose its extra entry");
  }
  require(__func__, readFile(undeclaredCounter) == "2",
          "an output-open attempt was reused from the result cache");

  constexpr llvm::StringLiteral midflightContents = "stable-external-input\n";
  const std::filesystem::path midflightExternal =
      root / "midflight-external.lib";
  writeText(midflightExternal, midflightContents);
  LocalToolConfig midflightConfig;
  midflightConfig.externalFiles.emplace("midflight_external",
                                        midflightExternal.string());
  ExternalToolInvocationBundleSpec midflight =
      baseSpec(tool, "outputs/midflight-result.txt");
  midflight.externalFiles = take(
      __func__, resolveExternalFiles(
                    {ExternalFileRequirement{"midflight.liberty",
                                             fingerprint(midflightContents)}},
                    midflightConfig));
  midflight.commands = {{tool.string(), "block", "outputs/tool-entry.log",
                         "outputs/release", "outputs/midflight-result.txt"}};
  const PreparedExternalToolInvocation midflightPrepared =
      take(__func__, finalizeExternalToolInvocationBundle(
                         (root / "cache-midflight").string(), midflight));
  const ExternalToolResultCacheKey midflightKey =
      take(__func__, deriveExternalToolResultCacheKey(midflightPrepared));
  const pid_t midflightChild = ::fork();
  require(__func__, midflightChild >= 0,
          "could not fork the midflight cache fixture");
  if (midflightChild == 0) {
    auto status = executeExternalToolInvocationBundle(midflightPrepared);
    if (!status) {
      llvm::consumeError(status.takeError());
      ::_exit(254);
    }
    ::_exit(*status == 0 ? 0 : 253);
  }
  const std::filesystem::path midflightEntered =
      root / "cache-midflight" / "outputs" / "tool-entry.log";
  for (unsigned attempt = 0;
       attempt != 5000 && !std::filesystem::exists(midflightEntered); ++attempt)
    ::usleep(1000);
  if (!std::filesystem::exists(midflightEntered)) {
    writeText(root / "cache-midflight" / "outputs" / "release", "");
    waitForChild(__func__, midflightChild);
    fail(__func__, "midflight cache fixture did not enter the tool");
  }
  writeText(midflightExternal, "changed-during-execution\n");
  writeText(root / "cache-midflight" / "outputs" / "release", "");
  const int midflightStatus = waitForChild(__func__, midflightChild);
  require(__func__,
          WIFEXITED(midflightStatus) && WEXITSTATUS(midflightStatus) == 0,
          "midflight input mutation changed the real execution result");
  take(__func__, importExternalToolInvocationBundle(
                     midflightPrepared, importExpectation(midflight)));
  const std::filesystem::path midflightEntry =
      cache / "entries" /
      loom::formatBlobDigestHex(midflightKey.inputMaterialDigest) /
      loom::formatBlobDigestHex(midflightKey.executionConfigurationDigest) /
      loom::formatBlobDigestHex(midflightKey.toolVersionDigest);
  require(__func__, !std::filesystem::exists(midflightEntry),
          "a result was cached after its external input changed");

  writeText(midflightExternal, midflightContents);
  const std::filesystem::path replayRoot = root / "cache-midflight-replay";
  const PreparedExternalToolInvocation midflightReplay =
      take(__func__, finalizeExternalToolInvocationBundle(replayRoot.string(),
                                                          midflight));
  writeText(replayRoot / "outputs" / "release", "");
  require(
      __func__,
      take(__func__, executeExternalToolInvocationBundle(midflightReplay)) == 0,
      "midflight cache fixture could not execute after input restoration");
  require(__func__,
          std::filesystem::exists(replayRoot / "outputs" / "tool-entry.log"),
          "an unpublished midflight result was incorrectly reused");
  require(__func__, ::unsetenv("LOOM_EXTERNAL_TOOL_CACHE_ROOT") == 0,
          "could not disable the result cache");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one test-directory argument");
  const std::filesystem::path root =
      std::filesystem::absolute(argv[1]).lexically_normal();
  std::filesystem::create_directories(root);
  const std::filesystem::path tool = root / "tool bin; data" / "fake tool";
  const std::string toolBody =
      "#!/usr/bin/env bash\n"
      "set -u\n"
      "increment_counter() {\n"
      "  local value=0\n"
      "  if [[ -f \"$1\" ]]; then IFS= read -r value <\"$1\"; fi\n"
      "  printf '%s' \"$((value + 1))\" >\"$1\"\n"
      "}\n"
      "case \"${1-}\" in\n"
      "  --version) printf '%s\\n' 'Fake EDA 1.2' ;;\n"
      "  run) printf '%s' \"$2\" >\"$3\" ;;\n"
      "  counted-run) increment_counter \"$2\"; printf '%s' \"$3\" >\"$4\" ;;\n"
      "  counted-run-extra) increment_counter \"$2\"; printf '%s' \"$3\" "
      ">\"$4\"; printf '%s' extra >\"$5\" ;;\n"
      "  counted-fail) increment_counter \"$2\"; exit " +
      std::to_string(kFixtureToolExitCode) +
      " ;;\n"
      "  block)\n"
      "    printf '%s\\n' entered >>\"$2\"\n"
      "    while [[ ! -e \"$3\" ]]; do sleep 0.01; done\n"
      "    printf '%s' completed >\"$4\"\n"
      "    ;;\n"
      "  no-output) : ;;\n"
      "  compile|compile-nonexec|compile-symlink)\n"
      "    mkdir -p -- \"$(dirname -- \"$2\")\"\n"
      "    if [[ \"$1\" == compile-symlink ]]; then\n"
      "      ln -s /bin/true \"$2\"\n"
      "    else\n"
      "      cat >\"$2\" <<'LOOM_GENERATED'\n"
      "#!/usr/bin/env bash\n"
      "set -eu\n"
      "printf '%s' \"$2\" >\"$1\"\n"
      "LOOM_GENERATED\n"
      "      if [[ \"$1\" == compile ]]; then chmod u+x \"$2\"; fi\n"
      "    fi\n"
      "    ;;\n"
      "  timeout) exit " +
      std::to_string(kFixtureToolExitCode) +
      " ;;\n"
      "  *) exit 64 ;;\n"
      "esac\n";
  writeExecutable(tool, toolBody);
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
  require("main", ::unsetenv("LOOM_EXTERNAL_TOOL_CACHE_ROOT") == 0,
          "could not isolate the cache fixture");
  resultImporterIdentityUsesCanonicalFraming();
  executionResourceBindingIsExact(root, tool);
  deterministicHostBundleExecutes(root, tool);
  toolProducedExecutableLifecycle(root, tool);
  containerBundleExecutes(root, tool, container);
  externalFileIsRevalidated(root, tool);
  externalFileTreeIsRevalidated(root, tool);
  missingOutputIsRecorded(root, tool);
  versionNormalizationMatchesDiscovery(root);
  invalidPathLeavesNoBundle(root, tool);
  malformedSemanticContractLeavesNoBundle(root, tool);
  conflictingPathLeavesNoBundle(root, tool);
  internalVersionPathCannotBeDeclared(root, tool);
  independentBundlesExecuteInParallel(root, tool);
  unremovableDeclaredOutputFailsBeforeToolEntry(root, tool);
  sequentialReexecutionIsCallerOwned(root, tool);
  successfulImportIsExactAndOutputSafe(root, tool);
  unsafeOutputsCannotBeImported(root, tool);
  strictImportRejectsAttemptTampering(root, tool);
  typedClosureIsExactAndLegacyManifestIsRejected(root, tool);
  preparedHandleBindsTheExactManifest(root, tool);
  executionRejectsSubstitutedBundle(root, tool);
  finalizeRejectsNonNormalizedBundleRoot(root, tool);
  incompleteAttemptHasATypedImportError(root, tool);
  failedAttemptImportIsExpectationBound(root, tool);
  persistentResultCacheIsExact(root, tool);
  return 0;
}
