#include "YosysSynthesisInvocation.h"

#include "Common/BlobDigest.h"
#include "EDA/Adapters/AsicStandardCellContracts.h"
#include "ExternalTool/ExternalFile.h"
#include "ExternalTool/RuntimeBinding.h"
#include "ExternalTool/ShellProbe.h"

#include <filesystem>
#include <set>

namespace loom::eda::open_source {
namespace {

using namespace external_tool;
using namespace hardware;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "open_source_yosys_invalid: " + message);
}

llvm::Error rejectUndeclaredOutputs(llvm::StringRef bundleRoot) {
  const std::filesystem::path outputs =
      std::filesystem::path(bundleRoot.str()) / "outputs";
  const std::set<std::string> allowed{
      "completion.json",        "netlist.v",  "rtl-structure.json",
      "netlist-structure.json", "stderr.log", "stdout.log"};
  std::set<std::string> found;
  std::error_code error;
  const std::filesystem::file_status rootStatus =
      std::filesystem::symlink_status(outputs, error);
  if (error || !std::filesystem::is_directory(rootStatus) ||
      std::filesystem::is_symlink(rootStatus))
    return invalid("outputs directory is missing or not an ordinary directory");
  for (std::filesystem::directory_iterator iterator(outputs, error), end;
       !error && iterator != end; iterator.increment(error)) {
    const std::filesystem::path path = iterator->path();
    const std::filesystem::file_status status =
        std::filesystem::symlink_status(path, error);
    if (error)
      break;
    const std::string name = path.filename().string();
    if (!std::filesystem::is_regular_file(status) ||
        std::filesystem::is_symlink(status) || !allowed.count(name))
      return invalid("outputs directory contains undeclared output '" + name +
                     "'");
    found.insert(name);
  }
  if (error)
    return invalid("could not enumerate outputs directory: " + error.message());
  if (found != allowed)
    return invalid("outputs directory omits a lifecycle or declared output");
  return llvm::Error::success();
}

} // namespace

using namespace external_tool;
using namespace hardware;

ExternalToolInvocationImportExpectation yosysSynthesisInvocationExpectation(
    const ExternalToolSemanticContract &contract,
    llvm::ArrayRef<MaterializedBundleFile> semanticInputs,
    const ResolvedYosysGateNetlistConfigView &config) {
  ExternalToolInvocationImportExpectation result;
  result.semanticContract = contract;
  for (const MaterializedBundleFile &file : semanticInputs) {
    result.semanticInputs.push_back(ExternalToolInvocationSemanticInput{
        file.relativePath, *file.sourceArtifact,
        computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
            reinterpret_cast<const std::uint8_t *>(file.contents.data()),
            file.contents.size()))});
  }
  result.externalInputs.push_back(ExternalToolInvocationExternalInput{
      asicStandardCellLibertyInputSlot.str(), config.standardCellLiberty()});
  result.declaredOutputs.push_back(yosysNetlistOutputPath.str());
  result.declaredOutputs.push_back(yosysRtlStructureOutputPath.str());
  result.declaredOutputs.push_back(yosysNetlistStructureOutputPath.str());
  return result;
}

llvm::Expected<PreparedExternalToolInvocation> prepareYosysSynthesisInvocation(
    const ResolvedYosysGateNetlistConfigView &config,
    const ExternalToolSemanticContract &contract,
    llvm::ArrayRef<MaterializedBundleFile> semanticInputs, llvm::StringRef top,
    llvm::ArrayRef<std::string> rtlPaths, const YosysMappedChildren *children,
    const ExternalToolPreparationContext &context) {
  auto externalFiles = resolveExternalFiles(
      {{asicStandardCellLibertyInputSlot.str(), config.standardCellLiberty()}},
      context.localConfig);
  if (!externalFiles)
    return externalFiles.takeError();

  const ExternalToolProviderDescriptor &toolProvider = yosysProvider();
  const std::filesystem::path destination(context.bundleDestination);
  const std::filesystem::path probeRoot = destination.parent_path();
  ShellToolBindingProbe toolProbe(probeRoot.string(),
                                  toolProvider.versionProbe);
  const ToolEnvironment toolEnvironment =
      captureToolEnvironment(toolProvider.binding);
  auto tool = resolveToolBinding(toolProvider.binding, context.localConfig,
                                 toolEnvironment, toolProbe);
  if (!tool)
    return tool.takeError();
  if (tool->version != config.stableProviderBuildIdentity())
    return invalid(llvm::Twine("resolved Yosys build '") + tool->version +
                   "' does not match semantic build '" +
                   config.stableProviderBuildIdentity() + "'");

  std::vector<std::string> inheritEnvironment;
  const auto configured =
      context.localConfig.tools.find(toolProvider.binding.key);
  if (configured != context.localConfig.tools.end())
    inheritEnvironment = configured->second.inheritEnvironment;

  const ExternalToolProviderDescriptor &containerProvider =
      polyArchContainerProvider();
  ShellToolBindingProbe containerProbe(probeRoot.string(),
                                       containerProvider.versionProbe);
  const ToolEnvironment containerEnvironment =
      captureToolEnvironment(containerProvider.binding);
  auto runtime = resolveInvocationRuntime(
      *tool, context.localConfig, containerProvider.binding,
      containerEnvironment, containerProbe, toolProvider.runtimeCompatibility,
      [&](const ResolvedToolBinding &resolvedTool,
          const ResolvedToolBinding &container,
          llvm::StringRef os) -> llvm::Expected<std::optional<std::string>> {
        return probeContainerToolComposition(probeRoot.string(), resolvedTool,
                                             toolProvider.versionProbe,
                                             container, os, inheritEnvironment);
      });
  if (!runtime)
    return runtime.takeError();

  auto driver =
      children
          ? renderYosysBlockSynthesisDriver(
                top, rtlPaths, externalFiles->front().absolutePath, *children)
          : renderYosysSynthesisDriver(top, rtlPaths,
                                       externalFiles->front().absolutePath);
  if (!driver)
    return driver.takeError();
  std::vector<MaterializedBundleFile> files{
      {"drivers/synthesize.ys", std::move(*driver), std::nullopt, false}};
  files.insert(files.end(), semanticInputs.begin(), semanticInputs.end());
  const std::string executable = tool->executable;
  ExternalToolInvocationBundleSpec specification{
      contract,
      std::move(*tool),
      toolProvider.versionProbe,
      std::move(*runtime),
      containerProvider.versionProbe,
      {{executable, "-q", "-s", "drivers/synthesize.ys"}},
      std::move(inheritEnvironment),
      {yosysNetlistOutputPath.str(), yosysRtlStructureOutputPath.str(),
       yosysNetlistStructureOutputPath.str()},
      std::move(files),
      std::move(*externalFiles),
      {}};
  return finalizeExternalToolInvocationBundle(context.bundleDestination,
                                              specification);
}

llvm::Expected<YosysSynthesisOutput>
readYosysSynthesisOutput(const PreparedExternalToolInvocation &prepared,
                         const ImportedExternalToolInvocationBundle &imported,
                         llvm::StringRef top) {
  if (llvm::Error error = rejectUndeclaredOutputs(prepared.bundleRoot))
    return std::move(error);
  auto netlist = readExternalToolInvocationDeclaredOutput(
      imported, yosysNetlistOutputPath);
  if (!netlist)
    return netlist.takeError();
  auto rtlStructureText = readExternalToolInvocationDeclaredOutput(
      imported, yosysRtlStructureOutputPath);
  if (!rtlStructureText)
    return rtlStructureText.takeError();
  auto netlistStructureText = readExternalToolInvocationDeclaredOutput(
      imported, yosysNetlistStructureOutputPath);
  if (!netlistStructureText)
    return netlistStructureText.takeError();
  auto rtlStructure = parseYosysStructureFacts(*rtlStructureText);
  if (!rtlStructure)
    return rtlStructure.takeError();
  auto netlistStructure = parseYosysStructureFacts(*netlistStructureText);
  if (!netlistStructure)
    return netlistStructure.takeError();
  if (llvm::Error error =
          validateYosysSynthesizedStructure(*netlistStructure, top))
    return std::move(error);
  if (llvm::Error error =
          compareYosysTopPortGeometry(*rtlStructure, *netlistStructure, top))
    return std::move(error);
  return YosysSynthesisOutput{std::move(*netlist),
                              std::move(*netlistStructure)};
}

std::string renderYosysStandardCellBlackBoxContract(
    const ExternalFileFingerprint &library,
    llvm::ArrayRef<RepresentationLocator> unresolved) {
  std::string contract =
      "loom.open_source.yosys.standard_cell_contract.1.0\nliberty_sha256=" +
      formatExternalFileFingerprint(library) + "\n";
  for (const RepresentationLocator &locator : unresolved)
    contract += "module=" + locator.canonicalName + "\n";
  return contract;
}

} // namespace loom::eda::open_source
