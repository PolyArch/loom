#include "EDA/Adapters/OpenSource/Verilator.h"
#include "ExternalTool/LocalConfig.h"
#include "ExternalTool/Provider.h"
#include "ExternalTool/RuntimeBinding.h"
#include "ExternalTool/ShellProbe.h"
#include "Hardware/RTL/CirctConformance.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

llvm::cl::OptionCategory probeCategory("Loom hardware probe options");

llvm::cl::opt<std::string>
    bundleRoot("bundle",
               llvm::cl::desc("Directory to publish the invocation bundle"),
               llvm::cl::value_desc("path"), llvm::cl::Required,
               llvm::cl::cat(probeCategory));

llvm::cl::opt<std::string>
    localConfigPath("loom-local-config",
                    llvm::cl::desc("Machine-local external-tool configuration"),
                    llvm::cl::value_desc("path"), llvm::cl::init(""),
                    llvm::cl::cat(probeCategory));

llvm::cl::opt<bool> executeBundle(
    "execute",
    llvm::cl::desc("Execute the generated run.sh and import completion"),
    llvm::cl::init(false), llvm::cl::cat(probeCategory));

llvm::Error probeError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "hardware_probe_failed: " + message);
}

llvm::Expected<loom::external_tool::LocalToolConfig> loadConfiguration() {
  if (localConfigPath.empty())
    return loom::external_tool::defaultLocalToolConfig();
  return loom::external_tool::loadLocalToolConfig(localConfigPath);
}

llvm::Expected<llvm::SmallString<256>> absoluteBundleRoot() {
  llvm::SmallString<256> path(bundleRoot);
  if (std::error_code error = llvm::sys::fs::make_absolute(path))
    return probeError("could not make bundle path absolute: " +
                      error.message());
  llvm::sys::path::remove_dots(path, true);
  return path;
}

llvm::Error rejectUnsupportedProviderOptions(
    const loom::external_tool::LocalToolConfig &config) {
  auto tool = config.tools.find("verilator");
  if (tool != config.tools.end() && !tool->second.providerOptions.empty())
    return probeError(
        "Verilator conformance adapter accepts no provider_options");
  if (!config.polyArchContainer.providerOptions.empty())
    return probeError(
        "PolyArch/container conformance adapter accepts no provider_options");
  return llvm::Error::success();
}

std::vector<std::string>
inheritedEnvironment(const loom::external_tool::LocalToolConfig &config,
                     loom::external_tool::InvocationRuntimeKind runtimeKind) {
  std::vector<std::string> names;
  auto append = [&](const std::vector<std::string> &values) {
    for (const std::string &value : values)
      if (std::find(names.begin(), names.end(), value) == names.end())
        names.push_back(value);
  };
  auto tool = config.tools.find("verilator");
  if (tool != config.tools.end())
    append(tool->second.inheritEnvironment);
  if (runtimeKind ==
      loom::external_tool::InvocationRuntimeKind::PolyArchContainer)
    append(config.polyArchContainer.inheritEnvironment);
  return names;
}

struct TemporaryDirectoryCleanup {
  std::filesystem::path path;

  ~TemporaryDirectoryCleanup() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

llvm::Expected<std::optional<std::string>> preflightVerilatorContainer(
    llvm::StringRef scratchDirectory,
    const loom::external_tool::ResolvedToolBinding &tool,
    const loom::external_tool::ResolvedToolBinding &container,
    llvm::StringRef os, llvm::ArrayRef<std::string> inheritEnvironment) {
  llvm::SmallString<256> prefix(scratchDirectory);
  llvm::sys::path::append(prefix, "loom-container-preflight");
  llvm::SmallString<256> temporaryDirectory;
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory(prefix, temporaryDirectory))
    return probeError("could not create composition preflight directory: " +
                      error.message());
  TemporaryDirectoryCleanup cleanup{
      std::filesystem::path(temporaryDirectory.str().str())};

  loom::external_tool::InvocationRuntimeBinding runtime;
  runtime.kind = loom::external_tool::InvocationRuntimeKind::PolyArchContainer;
  runtime.polyArchContainer = container;
  runtime.os = os.str();

  const auto &containerProvider =
      loom::external_tool::polyArchContainerProvider();
  const auto &verilatorProvider = loom::external_tool::verilatorProvider();
  loom::external_tool::ExternalToolInvocationBundleSpec bundle;
  bundle.providerIdentity = "verilator.runtime-preflight@1";
  bundle.semanticBindingIdentity = "verilator.version-probe@1";
  bundle.resultImporterIdentity = "external-tool.completion@1";
  bundle.tool = tool;
  bundle.toolVersionProbe = verilatorProvider.versionProbe;
  bundle.runtime = std::move(runtime);
  bundle.containerVersionProbe = containerProvider.versionProbe;
  bundle.commands = {{tool.executable}};
  bundle.commands.front().insert(
      bundle.commands.front().end(),
      verilatorProvider.versionProbe.arguments.begin(),
      verilatorProvider.versionProbe.arguments.end());
  bundle.inheritEnvironment.assign(inheritEnvironment.begin(),
                                   inheritEnvironment.end());

  llvm::SmallString<256> preflightBundle(temporaryDirectory);
  llvm::sys::path::append(preflightBundle, "bundle");
  if (llvm::Error error =
          loom::external_tool::finalizeExternalToolInvocationBundle(
              preflightBundle, bundle))
    return error;
  llvm::Expected<int> status =
      loom::external_tool::executeExternalToolInvocationBundle(preflightBundle);
  if (!status)
    return std::optional<std::string>(
        "generated composition preflight could not execute: " +
        llvm::toString(status.takeError()));
  llvm::Expected<loom::external_tool::InvocationCompletion> completion =
      loom::external_tool::loadExternalToolInvocationCompletion(
          preflightBundle);
  if (!completion)
    return std::optional<std::string>(
        "generated composition preflight has no valid completion: " +
        llvm::toString(completion.takeError()));
  if (*status != 0 ||
      completion->status !=
          loom::external_tool::InvocationCompletionStatus::Success)
    return std::optional<std::string>(
        "containerized Verilator version probe exited with status " +
        std::to_string(*status));
  return std::optional<std::string>{};
}

llvm::Error run() {
  llvm::Expected<loom::external_tool::LocalToolConfig> config =
      loadConfiguration();
  if (!config)
    return config.takeError();
  if (llvm::Error error = rejectUnsupportedProviderOptions(*config))
    return error;

  llvm::Expected<llvm::SmallString<256>> root = absoluteBundleRoot();
  if (!root)
    return root.takeError();
  llvm::SmallString<256> probeDirectory(*root);
  llvm::sys::path::remove_filename(probeDirectory);
  if (probeDirectory.empty())
    probeDirectory = "/";

  const auto &verilator = loom::external_tool::verilatorProvider();
  loom::external_tool::ShellToolBindingProbe toolProbe(
      probeDirectory.str().str(), verilator.versionProbe);
  const loom::external_tool::ToolEnvironment toolEnvironment =
      loom::external_tool::captureToolEnvironment(verilator.binding);
  llvm::Expected<loom::external_tool::ResolvedToolBinding> tool =
      loom::external_tool::resolveToolBinding(verilator.binding, *config,
                                              toolEnvironment, toolProbe);
  if (!tool)
    return tool.takeError();

  const auto &container = loom::external_tool::polyArchContainerProvider();
  loom::external_tool::ShellToolBindingProbe containerProbe(
      probeDirectory.str().str(), container.versionProbe);
  const loom::external_tool::ToolEnvironment containerEnvironment =
      loom::external_tool::captureToolEnvironment(container.binding);
  llvm::Expected<loom::external_tool::InvocationRuntimeBinding> runtime =
      loom::external_tool::resolveInvocationRuntime(
          *tool, *config, container.binding, containerEnvironment,
          containerProbe, verilator.runtimeCompatibility,
          [&](const loom::external_tool::ResolvedToolBinding &resolvedTool,
              const loom::external_tool::ResolvedToolBinding &resolvedContainer,
              llvm::StringRef os)
              -> llvm::Expected<std::optional<std::string>> {
            return preflightVerilatorContainer(
                probeDirectory, resolvedTool, resolvedContainer, os,
                inheritedEnvironment(
                    *config, loom::external_tool::InvocationRuntimeKind::
                                 PolyArchContainer));
          });
  if (!runtime)
    return runtime.takeError();

  llvm::Expected<std::string> systemVerilog =
      loom::hardware::rtl::emitCirctConformanceSystemVerilog();
  if (!systemVerilog)
    return systemVerilog.takeError();
  llvm::Expected<loom::external_tool::ExternalToolInvocationBundleSpec> bundle =
      loom::eda::open_source::makeVerilatorLintBundle(
          *systemVerilog, "circt-api-conformance@1", *tool, *runtime,
          inheritedEnvironment(*config, runtime->kind));
  if (!bundle)
    return bundle.takeError();
  if (llvm::Error error =
          loom::external_tool::finalizeExternalToolInvocationBundle(root->str(),
                                                                    *bundle))
    return error;

  llvm::outs() << "bundle: " << root->str() << '\n';
  if (!executeBundle)
    return llvm::Error::success();
  llvm::Expected<int> status =
      loom::external_tool::executeExternalToolInvocationBundle(root->str());
  if (!status)
    return status.takeError();
  llvm::Expected<loom::external_tool::InvocationCompletion> completion =
      loom::external_tool::loadExternalToolInvocationCompletion(root->str());
  if (!completion)
    return completion.takeError();
  if (*status != 0 ||
      completion->status !=
          loom::external_tool::InvocationCompletionStatus::Success)
    return probeError("generated bundle did not complete successfully");
  llvm::outs() << "completion: success\n";
  return llvm::Error::success();
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initialization(argc, argv);
  llvm::cl::HideUnrelatedOptions(probeCategory);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Loom CIRCT and Verilator probe\n");
  if (llvm::Error error = run()) {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    return 1;
  }
  return 0;
}
