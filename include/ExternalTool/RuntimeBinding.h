#ifndef LOOM_EXTERNALTOOL_RUNTIMEBINDING_H
#define LOOM_EXTERNALTOOL_RUNTIMEBINDING_H

#include "ExternalTool/Binding.h"
#include "ExternalTool/LocalConfig.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>
#include <vector>

namespace loom::external_tool {

enum class InvocationRuntimeKind {
  Host,
  PolyArchContainer,
};

struct InvocationRuntimeBinding {
  InvocationRuntimeKind kind = InvocationRuntimeKind::Host;
  std::optional<ResolvedToolBinding> polyArchContainer;
  std::optional<std::string> os;
  std::vector<std::string> rejectedCompositions;
};

struct ToolRuntimeCompatibility {
  bool supportsPolyArchContainer = false;
  std::vector<std::string> preferredOperatingSystems;
};

using RuntimeCompositionPreflight =
    llvm::function_ref<llvm::Expected<std::optional<std::string>>(
        const ResolvedToolBinding &tool,
        const ResolvedToolBinding &polyArchContainer, llvm::StringRef os)>;

llvm::Expected<InvocationRuntimeBinding> resolveInvocationRuntime(
    const ResolvedToolBinding &tool, const LocalToolConfig &config,
    const ToolProviderDescriptor &containerProvider,
    const ToolEnvironment &environment, ToolBindingProbe &probe,
    const ToolRuntimeCompatibility &compatibility,
    RuntimeCompositionPreflight preflight);

} // namespace loom::external_tool

#endif // LOOM_EXTERNALTOOL_RUNTIMEBINDING_H
