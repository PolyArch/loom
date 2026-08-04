#include "ExternalTool/Binding.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"

#include <filesystem>
#include <set>
#include <utility>

namespace loom::external_tool {
namespace {

llvm::Error bindingError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "tool_binding_unavailable: " + message);
}

llvm::Expected<ResolvedToolBinding>
resolvedBinding(const ToolProviderDescriptor &provider,
                ToolBindingSource source, ProbedToolBinding probed,
                std::vector<std::string> requestedModules = {},
                std::optional<std::string> environmentVariable = std::nullopt) {
  if (probed.executable.empty() ||
      !llvm::sys::path::is_absolute(probed.executable))
    return bindingError(
        "provider probe returned a non-absolute executable for '" +
        provider.key + "'");
  if (probed.version.empty())
    return bindingError("provider probe returned an empty version for '" +
                        provider.key + "'");
  if (!requestedModules.empty() && probed.loadedModules.empty())
    return bindingError(
        "provider module probe returned an empty loaded-module closure for '" +
        provider.key + "'");
  return ResolvedToolBinding{
      provider.key,
      source,
      std::move(probed.executable),
      std::move(probed.version),
      std::move(requestedModules),
      std::move(probed.loadedModules),
      std::move(probed.moduleInit),
      std::move(environmentVariable),
  };
}

llvm::Expected<std::optional<ResolvedToolBinding>>
probeExecutable(const ToolProviderDescriptor &provider,
                ToolBindingSource source, llvm::StringRef path,
                ToolBindingProbe &probe,
                std::optional<std::string> environmentVariable = std::nullopt) {
  auto result = probe.probeExecutable(path);
  if (!result)
    return result.takeError();
  if (!*result)
    return std::optional<ResolvedToolBinding>{};
  auto resolved = resolvedBinding(provider, source, std::move(**result), {},
                                  std::move(environmentVariable));
  if (!resolved)
    return resolved.takeError();
  return std::optional<ResolvedToolBinding>{std::move(*resolved)};
}

llvm::Expected<std::optional<ResolvedToolBinding>>
probeModules(const ToolProviderDescriptor &provider, ToolBindingSource source,
             const std::optional<std::string> &moduleInit,
             llvm::ArrayRef<std::string> modules, ToolBindingProbe &probe) {
  ModuleProbeRequest request;
  request.initScript = moduleInit;
  request.modules.assign(modules.begin(), modules.end());
  request.executableNames = provider.executableNames;
  request.environmentCandidates = provider.environmentCandidates;
  auto result = probe.probeModules(request);
  if (!result)
    return result.takeError();
  if (!*result)
    return std::optional<ResolvedToolBinding>{};
  auto resolved =
      resolvedBinding(provider, source, std::move(**result), request.modules);
  if (!resolved)
    return resolved.takeError();
  return std::optional<ResolvedToolBinding>{std::move(*resolved)};
}

llvm::Error validateDescriptor(const ToolProviderDescriptor &provider) {
  auto isInvalidRelativePath = [](llvm::StringRef spelling) {
    const std::filesystem::path path(spelling.str());
    if (spelling.empty() || spelling.find('\0') != llvm::StringRef::npos ||
        path.is_absolute() ||
        path.lexically_normal().generic_string() != spelling)
      return true;
    for (const std::filesystem::path &component : path)
      if (component == "..")
        return true;
    return false;
  };

  if (provider.key.empty())
    return bindingError("provider key is empty");
  if (provider.executableNames.empty())
    return bindingError("provider '" + provider.key +
                        "' has no executable names");
  std::set<std::string> executableNames;
  for (const std::string &name : provider.executableNames) {
    if (isInvalidRelativePath(name) || !executableNames.insert(name).second)
      return bindingError("provider '" + provider.key +
                          "' has an invalid executable name");
  }
  std::set<std::pair<std::string, std::string>> environmentCandidates;
  for (const ToolEnvironmentCandidate &candidate :
       provider.environmentCandidates) {
    if (!isValidEnvironmentName(candidate.variable))
      return bindingError("provider '" + provider.key +
                          "' has an invalid environment variable name");
    if (isInvalidRelativePath(candidate.relativeExecutable) ||
        !environmentCandidates
             .insert({candidate.variable, candidate.relativeExecutable})
             .second)
      return bindingError("provider '" + provider.key +
                          "' has an invalid relative executable");
  }
  std::set<std::string> moduleAliases;
  for (const std::string &alias : provider.moduleAliases)
    if (alias.empty() || alias.find('\0') != std::string::npos ||
        !moduleAliases.insert(alias).second)
      return bindingError("provider '" + provider.key +
                          "' has an invalid module alias");
  return llvm::Error::success();
}

llvm::Expected<std::optional<ResolvedToolBinding>>
resolveBinding(const ToolProviderDescriptor &provider,
               const LocalExplicitBinding *explicitBinding,
               const std::optional<std::string> &moduleInit,
               const ToolEnvironment &environment, ToolBindingProbe &probe) {
  if (llvm::Error error = validateDescriptor(provider))
    return std::move(error);

  if (explicitBinding && explicitBinding->isConfigured()) {
    if (explicitBinding->executable) {
      auto result = probeExecutable(provider, ToolBindingSource::Explicit,
                                    *explicitBinding->executable, probe);
      if (!result)
        return result.takeError();
      if (*result)
        return std::move(**result);
    } else {
      auto result = probeModules(provider, ToolBindingSource::Explicit,
                                 moduleInit, explicitBinding->modules, probe);
      if (!result)
        return result.takeError();
      if (*result)
        return std::move(**result);
    }
    return bindingError("explicit binding for '" + provider.key +
                        "' is unavailable or incompatible");
  }

  llvm::SmallVector<llvm::StringRef, 16> pathEntries;
  llvm::StringRef(environment.path)
      .split(pathEntries, llvm::sys::EnvPathSeparator, -1, false);
  for (const std::string &name : provider.executableNames) {
    for (llvm::StringRef directory : pathEntries) {
      llvm::SmallString<256> path(directory);
      llvm::sys::path::append(path, name);
      auto result = probeExecutable(
          provider, ToolBindingSource::EnvironmentPath, path, probe);
      if (!result)
        return result.takeError();
      if (*result)
        return std::move(**result);
    }
  }

  for (const ToolEnvironmentCandidate &candidate :
       provider.environmentCandidates) {
    auto value = environment.variables.find(candidate.variable);
    if (value == environment.variables.end() || value->second.empty())
      continue;
    llvm::SmallString<256> path(value->second);
    llvm::sys::path::append(path, candidate.relativeExecutable);
    auto result = probeExecutable(provider, ToolBindingSource::EnvironmentRoot,
                                  path, probe, candidate.variable);
    if (!result)
      return result.takeError();
    if (*result)
      return std::move(**result);
  }

  for (const std::string &alias : provider.moduleAliases) {
    const std::vector<std::string> modules{alias};
    auto result = probeModules(provider, ToolBindingSource::Module, moduleInit,
                               modules, probe);
    if (!result)
      return result.takeError();
    if (*result)
      return std::move(**result);
  }

  return std::optional<ResolvedToolBinding>{};
}

} // namespace

ToolEnvironment captureToolEnvironment(const ToolProviderDescriptor &provider) {
  ToolEnvironment environment;
  if (std::optional<std::string> path = llvm::sys::Process::GetEnv("PATH"))
    environment.path = std::move(*path);
  for (const ToolEnvironmentCandidate &candidate :
       provider.environmentCandidates)
    if (std::optional<std::string> value =
            llvm::sys::Process::GetEnv(candidate.variable))
      environment.variables.emplace(candidate.variable, std::move(*value));
  return environment;
}

llvm::Expected<ResolvedToolBinding> resolveToolBinding(
    const ToolProviderDescriptor &provider, const LocalToolConfig &config,
    const ToolEnvironment &environment, ToolBindingProbe &probe) {
  auto configured = config.tools.find(provider.key);
  const LocalExplicitBinding *binding =
      configured == config.tools.end() ? nullptr : &configured->second.binding;
  auto resolved =
      resolveBinding(provider, binding, config.moduleInit, environment, probe);
  if (!resolved)
    return resolved.takeError();
  if (!*resolved)
    return bindingError("no configured, environment, or module binding for '" +
                        provider.key + "'");
  return std::move(**resolved);
}

llvm::Expected<std::optional<ResolvedToolBinding>>
resolvePolyArchContainerBinding(const ToolProviderDescriptor &provider,
                                const LocalToolConfig &config,
                                const ToolEnvironment &environment,
                                ToolBindingProbe &probe) {
  return resolveBinding(provider, &config.polyArchContainer.binding,
                        config.moduleInit, environment, probe);
}

} // namespace loom::external_tool
