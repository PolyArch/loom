#ifndef LOOM_EXTERNALTOOL_SHELLPROBE_H
#define LOOM_EXTERNALTOOL_SHELLPROBE_H

#include "ExternalTool/Binding.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::external_tool {

struct ToolVersionProbe {
  ToolVersionProbe() = default;
  ToolVersionProbe(
      std::vector<std::string> arguments,
      std::optional<std::string> requiredOutputSubstring,
      std::vector<int> acceptedExitCodes = {0},
      std::optional<std::string> selectedOutputLineSubstring = std::nullopt)
      : arguments(std::move(arguments)),
        requiredOutputSubstring(std::move(requiredOutputSubstring)),
        acceptedExitCodes(std::move(acceptedExitCodes)),
        selectedOutputLineSubstring(std::move(selectedOutputLineSubstring)) {}

  std::vector<std::string> arguments;
  std::optional<std::string> requiredOutputSubstring;
  std::vector<int> acceptedExitCodes{0};
  std::optional<std::string> selectedOutputLineSubstring;
};

llvm::ArrayRef<llvm::StringLiteral> defaultModuleInitializationPaths();

/// Probes one already-resolved PolyArch/container and tool composition by
/// executing the provider-declared tool version probe inside the container.
/// The result carries the normalized selected version output match: an empty
/// optional means the composition executed and matched the resolved tool
/// version, and a string is the human-readable composition rejection reason.
/// Every name in inheritEnvironment is a fail-closed required environment
/// variable: an absent variable rejects the composition before any
/// execution, and the container inherits the process environment through the
/// shared run protocol.
llvm::Expected<std::optional<std::string>> probeContainerToolComposition(
    llvm::StringRef probeDirectory, const ResolvedToolBinding &tool,
    const ToolVersionProbe &toolVersionProbe,
    const ResolvedToolBinding &polyArchContainer, llvm::StringRef os,
    llvm::ArrayRef<std::string> inheritEnvironment);

class ShellToolBindingProbe final : public ToolBindingProbe {
public:
  ShellToolBindingProbe(std::string probeDirectory,
                        ToolVersionProbe versionProbe);

  llvm::Expected<std::optional<ProbedToolBinding>>
  probeExecutable(llvm::StringRef path) override;

  llvm::Expected<std::optional<ProbedToolBinding>>
  probeModules(const ModuleProbeRequest &request) override;

private:
  std::string probeDirectory_;
  ToolVersionProbe versionProbe_;
};

} // namespace loom::external_tool

#endif // LOOM_EXTERNALTOOL_SHELLPROBE_H
