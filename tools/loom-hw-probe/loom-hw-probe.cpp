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
    probeDirectoryOption(
        "probe-dir",
        llvm::cl::desc("Directory for temporary tool probe scripts"),
        llvm::cl::value_desc("path"), llvm::cl::init(""),
        llvm::cl::cat(probeCategory));

llvm::cl::opt<std::string>
    localConfigPath("loom-local-config",
                    llvm::cl::desc("Machine-local external-tool configuration"),
                    llvm::cl::value_desc("path"), llvm::cl::init(""),
                    llvm::cl::cat(probeCategory));

llvm::Error probeError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "hardware_probe_failed: " + message);
}

llvm::Expected<loom::external_tool::LocalToolConfig> loadConfiguration() {
  if (localConfigPath.empty())
    return loom::external_tool::defaultLocalToolConfig();
  return loom::external_tool::loadLocalToolConfig(localConfigPath);
}

llvm::Expected<llvm::SmallString<256>> absoluteProbeDirectory() {
  llvm::SmallString<256> path(probeDirectoryOption);
  if (path.empty())
    path = ".";
  if (std::error_code error = llvm::sys::fs::make_absolute(path))
    return probeError("could not make probe directory absolute: " +
                      error.message());
  llvm::sys::path::remove_dots(path, true);
  std::error_code directoryError;
  if (!std::filesystem::is_directory(
          std::filesystem::path(path.str().str()), directoryError) ||
      directoryError)
    return probeError("probe directory does not exist: " + path.str().str());
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

llvm::Error run() {
  llvm::Expected<loom::external_tool::LocalToolConfig> config =
      loadConfiguration();
  if (!config)
    return config.takeError();
  if (llvm::Error error = rejectUnsupportedProviderOptions(*config))
    return error;

  llvm::Expected<llvm::SmallString<256>> probeDirectory =
      absoluteProbeDirectory();
  if (!probeDirectory)
    return probeDirectory.takeError();

  const auto &verilator = loom::external_tool::verilatorProvider();
  loom::external_tool::ShellToolBindingProbe toolProbe(
      probeDirectory->str().str(), verilator.versionProbe);
  const loom::external_tool::ToolEnvironment toolEnvironment =
      loom::external_tool::captureToolEnvironment(verilator.binding);
  llvm::Expected<loom::external_tool::ResolvedToolBinding> tool =
      loom::external_tool::resolveToolBinding(verilator.binding, *config,
                                              toolEnvironment, toolProbe);
  if (!tool)
    return tool.takeError();

  const auto &container = loom::external_tool::polyArchContainerProvider();
  loom::external_tool::ShellToolBindingProbe containerProbe(
      probeDirectory->str().str(), container.versionProbe);
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
            return loom::external_tool::probeContainerToolComposition(
                probeDirectory->str(), resolvedTool, verilator.versionProbe,
                resolvedContainer, os,
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
  if (!llvm::StringRef(*systemVerilog)
           .contains("module loom_circt_conformance"))
    return probeError("CIRCT conformance emission lost the expected module");

  llvm::outs() << "tool: " << tool->executable << '\n';
  llvm::outs() << "version: " << tool->version << '\n';
  if (runtime->kind == loom::external_tool::InvocationRuntimeKind::Host) {
    llvm::outs() << "runtime: host\n";
  } else {
    llvm::outs() << "runtime: polyarch_container " << *runtime->os << '\n';
  }
  llvm::outs() << "emission: ok\n";
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
