#ifndef LOOM_EXTERNALTOOL_LOCALCONFIG_H
#define LOOM_EXTERNALTOOL_LOCALCONFIG_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace loom::external_tool {

bool isValidEnvironmentName(llvm::StringRef name);

enum class RuntimePolicy {
  Auto,
  Host,
  PolyArchContainer,
};

struct LocalExplicitBinding {
  std::optional<std::string> executable;
  std::vector<std::string> modules;

  bool isConfigured() const { return executable || !modules.empty(); }
};

struct LocalProviderConfig {
  LocalExplicitBinding binding;
  std::vector<std::string> inheritEnvironment;
  llvm::json::Object providerOptions;
};

struct LocalContainerConfig : LocalProviderConfig {
  std::optional<std::string> os;
};

struct LocalToolConfig {
  std::optional<std::string> moduleInit;
  std::map<std::string, std::string> externalFiles;
  RuntimePolicy runtimePolicy = RuntimePolicy::Auto;
  LocalContainerConfig polyArchContainer;
  std::map<std::string, LocalProviderConfig> tools;
};

LocalToolConfig defaultLocalToolConfig();

llvm::Expected<LocalToolConfig>
parseLocalToolConfig(llvm::StringRef body, llvm::StringRef sourceName);

llvm::Expected<LocalToolConfig> loadLocalToolConfig(llvm::StringRef path);

} // namespace loom::external_tool

#endif // LOOM_EXTERNALTOOL_LOCALCONFIG_H
