#include "EDA/Adapters/OpenSource/Verilator.h"

#include "ExternalTool/Provider.h"

#include "llvm/ADT/Twine.h"

#include <string>
#include <utility>

namespace loom::eda::open_source {
namespace {

llvm::Error adapterError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "verilator_adapter_invalid: " + message);
}

} // namespace

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeVerilatorLintBundle(llvm::StringRef systemVerilog,
                        llvm::StringRef semanticBindingIdentity,
                        const external_tool::ResolvedToolBinding &tool,
                        const external_tool::InvocationRuntimeBinding &runtime,
                        llvm::ArrayRef<std::string> inheritEnvironment) {
  if (systemVerilog.empty())
    return adapterError("SystemVerilog input is empty");
  if (semanticBindingIdentity.empty())
    return adapterError("semantic binding identity is empty");
  if (tool.toolKey != "verilator")
    return adapterError("tool binding is not Verilator");

  const external_tool::ExternalToolProviderDescriptor &provider =
      external_tool::verilatorProvider();
  external_tool::ExternalToolInvocationBundleSpec bundle;
  bundle.providerIdentity = "verilator.lint@1";
  bundle.semanticBindingIdentity = semanticBindingIdentity.str();
  bundle.resultImporterIdentity = "verilator.lint.completion@1";
  bundle.tool = tool;
  bundle.toolVersionProbe = provider.versionProbe;
  bundle.runtime = runtime;
  bundle.containerVersionProbe =
      external_tool::polyArchContainerProvider().versionProbe;
  bundle.commands = {
      {tool.executable, "--lint-only", "--Wno-fatal", "--Wall",
       "drivers/loom_circt_conformance.sv"},
  };
  bundle.inheritEnvironment.assign(inheritEnvironment.begin(),
                                   inheritEnvironment.end());
  bundle.files = {{"drivers/loom_circt_conformance.sv", systemVerilog.str(),
                   std::nullopt, false}};
  return bundle;
}

} // namespace loom::eda::open_source
