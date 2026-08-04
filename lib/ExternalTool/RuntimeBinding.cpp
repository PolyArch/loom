#include "ExternalTool/RuntimeBinding.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::external_tool {
namespace {

llvm::Error runtimeError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "runtime_binding_unavailable: " + message);
}

llvm::Expected<InvocationRuntimeBinding> hostOrError(RuntimePolicy policy,
                                                     llvm::StringRef reason) {
  if (policy == RuntimePolicy::PolyArchContainer)
    return runtimeError(reason);
  InvocationRuntimeBinding binding;
  binding.rejectedCompositions.push_back(reason.str());
  return binding;
}

} // namespace

llvm::Expected<InvocationRuntimeBinding> resolveInvocationRuntime(
    const ResolvedToolBinding &tool, const LocalToolConfig &config,
    const ToolProviderDescriptor &containerProvider,
    const ToolEnvironment &environment, ToolBindingProbe &probe,
    const ToolRuntimeCompatibility &compatibility,
    RuntimeCompositionPreflight preflight) {
  if (config.runtimePolicy == RuntimePolicy::Host)
    return InvocationRuntimeBinding{};

  if (!compatibility.supportsPolyArchContainer)
    return hostOrError(config.runtimePolicy,
                       "tool provider does not support PolyArch/container");

  llvm::Expected<std::optional<ResolvedToolBinding>> container =
      resolvePolyArchContainerBinding(containerProvider, config, environment,
                                      probe);
  if (!container)
    return container.takeError();
  if (!*container)
    return hostOrError(config.runtimePolicy,
                       "PolyArch/container binding is unavailable");

  std::vector<std::string> operatingSystems;
  if (config.polyArchContainer.os) {
    if (std::find(compatibility.preferredOperatingSystems.begin(),
                  compatibility.preferredOperatingSystems.end(),
                  *config.polyArchContainer.os) ==
        compatibility.preferredOperatingSystems.end())
      return hostOrError(config.runtimePolicy,
                         "configured PolyArch/container OS is incompatible");
    operatingSystems.push_back(*config.polyArchContainer.os);
  } else {
    operatingSystems = compatibility.preferredOperatingSystems;
  }
  if (operatingSystems.empty())
    return hostOrError(config.runtimePolicy,
                       "tool provider declares no compatible container OS");

  InvocationRuntimeBinding binding;
  for (const std::string &os : operatingSystems) {
    if (os.empty())
      return runtimeError("tool provider declares an empty container OS");
    llvm::Expected<std::optional<std::string>> rejection =
        preflight(tool, **container, os);
    if (!rejection)
      return rejection.takeError();
    if (!*rejection) {
      binding.kind = InvocationRuntimeKind::PolyArchContainer;
      binding.polyArchContainer = std::move(**container);
      binding.os = os;
      return binding;
    }
    if ((*rejection)->empty())
      return runtimeError("composition preflight returned an empty rejection");
    binding.rejectedCompositions.push_back(os + ": " + **rejection);
  }

  if (config.runtimePolicy == RuntimePolicy::PolyArchContainer)
    return runtimeError("PolyArch/container composition rejected: " +
                        binding.rejectedCompositions.back());
  binding.kind = InvocationRuntimeKind::Host;
  return binding;
}

} // namespace loom::external_tool
