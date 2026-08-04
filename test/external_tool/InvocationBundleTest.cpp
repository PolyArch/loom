#include "ExternalTool/InvocationBundle.h"
#include "ExternalTool/ExternalFile.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SHA256.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
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

void take(const char *test, llvm::Error error) {
  if (error)
    fail(test, llvm::toString(std::move(error)));
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
  spec.semanticBindingIdentity = "fake_model@1";
  spec.resultImporterIdentity = "fake_importer@1";
  spec.tool = toolBinding(tool, ToolBindingSource::Explicit);
  spec.toolVersionProbe = ToolVersionProbe{{"--version"}, "Fake EDA"};
  spec.runtime.kind = InvocationRuntimeKind::Host;
  spec.commands = {{tool.string(), "run", "literal; $(touch never)",
                    output.generic_string()}};
  spec.inheritEnvironment = {"LOOM_BUNDLE_TEST_LICENSE"};
  spec.declaredOutputs = {output.generic_string()};
  spec.files = {
      {"drivers/driver with ' quote.tcl", "puts {driver}\n", std::nullopt,
       false},
      {"inputs/input.bin", "input-bytes", "artifact:input:01", false}};
  return spec;
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
              completion.exitCode == 0,
          "successful completion was not imported");

  writeExecutable(first / "outputs" / "completion.json",
                  " {\"schema\":\"loom.external_tool_completion\","
                  "\"version\":\"1.0\",\"status\":\"success\","
                  "\"exit_code\":0}\n");
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

  std::filesystem::remove(bundle / output);
  writeText(external, "vendor-library-mutated\n");
  status = take(__func__,
                executeExternalToolInvocationBundle(bundle.string()));
  require(__func__, status != 0,
          "bundle accepted a changed resolved external file");
  InvocationCompletion completion =
      take(__func__, loadExternalToolInvocationCompletion(bundle.string()));
  require(__func__,
          completion.status == InvocationCompletionStatus::BundleContentMismatch,
          "changed external content was not distinguished in completion");
  require(__func__, !std::filesystem::exists(bundle / output),
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
  return 0;
}
