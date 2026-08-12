#ifndef LOOM_EXTERNALTOOL_BINDING_H
#define LOOM_EXTERNALTOOL_BINDING_H

#include "ExternalTool/LocalConfig.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace loom::external_tool {

struct ToolEnvironmentCandidate {
  std::string variable;
  std::string relativeExecutable;
};

struct ToolProviderDescriptor {
  std::string key;
  std::vector<std::string> executableNames;
  std::vector<ToolEnvironmentCandidate> environmentCandidates;
  std::vector<std::string> moduleAliases;
};

struct ToolEnvironment {
  std::string path;
  std::map<std::string, std::string> variables;
};

ToolEnvironment captureToolEnvironment(const ToolProviderDescriptor &provider);

struct ModuleProbeRequest {
  std::optional<std::string> initScript;
  std::vector<std::string> modules;
  std::vector<std::string> executableNames;
  std::vector<ToolEnvironmentCandidate> environmentCandidates;
};

struct ProbedToolBinding {
  std::string executable;
  std::string version;
  std::vector<std::string> loadedModules;
  std::optional<std::string> moduleInit;
};

class ToolBindingProbe {
public:
  virtual ~ToolBindingProbe() = default;

  virtual llvm::Expected<std::optional<ProbedToolBinding>>
  probeExecutable(llvm::StringRef path) = 0;

  virtual llvm::Expected<std::optional<ProbedToolBinding>>
  probeModules(const ModuleProbeRequest &request) = 0;
};

enum class ToolBindingSource {
  Explicit,
  EnvironmentPath,
  EnvironmentRoot,
  Module,
};

struct ResolvedToolBinding {
  std::string toolKey;
  ToolBindingSource source;
  std::string executable;
  std::string version;
  std::vector<std::string> requestedModules;
  std::vector<std::string> loadedModules;
  std::optional<std::string> moduleInit;
  std::optional<std::string> environmentVariable;
};

llvm::Expected<std::optional<ResolvedToolBinding>>
resolveEnvironmentToolBinding(const ToolProviderDescriptor &provider,
                              const ToolEnvironment &environment,
                              ToolBindingProbe &probe);

llvm::Expected<ResolvedToolBinding>
resolveToolBinding(const ToolProviderDescriptor &provider,
                   const LocalToolConfig &config,
                   const ToolEnvironment &environment, ToolBindingProbe &probe);

llvm::Expected<std::optional<ResolvedToolBinding>>
resolvePolyArchContainerBinding(const ToolProviderDescriptor &provider,
                                const LocalToolConfig &config,
                                const ToolEnvironment &environment,
                                ToolBindingProbe &probe);

} // namespace loom::external_tool

#endif // LOOM_EXTERNALTOOL_BINDING_H
