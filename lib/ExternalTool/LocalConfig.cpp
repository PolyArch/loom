#include "ExternalTool/LocalConfig.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"

#include <cctype>
#include <initializer_list>
#include <set>
#include <system_error>
#include <utility>

namespace loom::external_tool {
namespace {

llvm::Error configError(llvm::StringRef source, const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "local_config_invalid: " + source + ": " +
                                     message);
}

bool containsNull(llvm::StringRef value) {
  return value.find('\0') != llvm::StringRef::npos;
}

llvm::Error rejectDuplicateKeys(llvm::yaml::Node *node,
                                llvm::StringRef source) {
  if (!node)
    return llvm::Error::success();
  if (auto *mapping = llvm::dyn_cast<llvm::yaml::MappingNode>(node)) {
    std::set<std::string> keys;
    for (llvm::yaml::KeyValueNode &entry : *mapping) {
      auto *keyNode = llvm::dyn_cast<llvm::yaml::ScalarNode>(entry.getKey());
      if (!keyNode)
        return configError(source, "object keys must be strings");
      llvm::SmallString<64> storage;
      const llvm::StringRef key = keyNode->getValue(storage);
      if (!keys.insert(key.str()).second)
        return configError(source, "duplicate key '" + key + "'");
      if (llvm::Error error = rejectDuplicateKeys(entry.getValue(), source))
        return error;
    }
    return llvm::Error::success();
  }
  if (auto *sequence = llvm::dyn_cast<llvm::yaml::SequenceNode>(node)) {
    for (llvm::yaml::Node &entry : *sequence)
      if (llvm::Error error = rejectDuplicateKeys(&entry, source))
        return error;
  }
  return llvm::Error::success();
}

llvm::Error rejectDuplicateKeys(llvm::StringRef body, llvm::StringRef source) {
  llvm::SourceMgr sourceManager;
  llvm::yaml::Stream stream(body, sourceManager);
  auto document = stream.begin();
  if (document == stream.end())
    return llvm::Error::success();
  if (llvm::Error error = rejectDuplicateKeys(document->getRoot(), source))
    return error;
  ++document;
  if (document != stream.end())
    return configError(source, "multiple documents are not supported");
  return llvm::Error::success();
}

llvm::Error requireOnlyKeys(const llvm::json::Object &object,
                            std::initializer_list<llvm::StringRef> allowed,
                            llvm::StringRef source, llvm::StringRef field) {
  for (const auto &[key, value] : object) {
    const llvm::StringRef keyRef = key;
    bool found = false;
    for (llvm::StringRef candidate : allowed)
      found |= keyRef == candidate;
    if (!found) {
      const llvm::Twine prefix = field.empty() ? llvm::Twine() : field + ".";
      return configError(source, "unknown key '" + prefix + keyRef + "'");
    }
  }
  return llvm::Error::success();
}

llvm::Expected<std::vector<std::string>>
parseStringArray(const llvm::json::Value *value, llvm::StringRef source,
                 llvm::StringRef field) {
  if (!value)
    return std::vector<std::string>{};
  const llvm::json::Array *array = value->getAsArray();
  if (!array)
    return configError(source, field + " must be an array");
  std::vector<std::string> result;
  result.reserve(array->size());
  for (const llvm::json::Value &entry : *array) {
    std::optional<llvm::StringRef> string = entry.getAsString();
    if (!string)
      return configError(source, field + " entries must be strings");
    result.push_back(string->str());
  }
  return result;
}

llvm::Expected<LocalExplicitBinding>
parseBinding(const llvm::json::Value *value, llvm::StringRef source,
             llvm::StringRef field) {
  LocalExplicitBinding binding;
  if (!value)
    return binding;
  const llvm::json::Object *object = value->getAsObject();
  if (!object)
    return configError(source, field + " must be an object");
  if (llvm::Error error =
          requireOnlyKeys(*object, {"executable", "modules"}, source, field))
    return std::move(error);
  const bool hasExecutable = object->get("executable") != nullptr;
  const bool hasModules = object->get("modules") != nullptr;
  if (hasExecutable == hasModules)
    return configError(source, field +
                                   " must contain exactly one of executable "
                                   "or modules");
  if (hasExecutable) {
    std::optional<llvm::StringRef> executable = object->getString("executable");
    if (!executable)
      return configError(source, field + ".executable must be a string");
    if (containsNull(*executable))
      return configError(source, field + ".executable contains NUL");
    if (executable->empty() || !llvm::sys::path::is_absolute(*executable))
      return configError(source,
                         field + ".executable must be an absolute path");
    binding.executable = executable->str();
  } else {
    auto modules = parseStringArray(object->get("modules"), source,
                                    (field + ".modules").str());
    if (!modules)
      return modules.takeError();
    if (modules->empty())
      return configError(source, field + ".modules must be a nonempty array");
    for (const std::string &module : *modules) {
      if (containsNull(module))
        return configError(source, field + ".modules entry contains NUL");
      if (module.empty())
        return configError(source, field + ".modules entries must be nonempty");
    }
    binding.modules = std::move(*modules);
  }
  return binding;
}

llvm::Expected<LocalProviderConfig>
parseProviderConfig(const llvm::json::Object &object, llvm::StringRef source,
                    llvm::StringRef field, bool isContainer = false) {
  if (llvm::Error error = requireOnlyKeys(
          object,
          isContainer
              ? std::initializer_list<llvm::StringRef>{"binding",
                                                       "inherit_environment",
                                                       "provider_options", "os"}
              : std::initializer_list<llvm::StringRef>{"binding",
                                                       "inherit_environment",
                                                       "provider_options"},
          source, field))
    return std::move(error);
  LocalProviderConfig config;
  auto binding =
      parseBinding(object.get("binding"), source, (field + ".binding").str());
  if (!binding)
    return binding.takeError();
  config.binding = std::move(*binding);
  auto environment = parseStringArray(object.get("inherit_environment"), source,
                                      (field + ".inherit_environment").str());
  if (!environment)
    return environment.takeError();
  config.inheritEnvironment = std::move(*environment);
  std::set<std::string> environmentNames;
  for (const std::string &name : config.inheritEnvironment) {
    if (!isValidEnvironmentName(name))
      return configError(source, field +
                                     ".inherit_environment contains an invalid "
                                     "environment variable name");
    if (!environmentNames.insert(name).second)
      return configError(source,
                         field + ".inherit_environment contains a duplicate "
                                 "environment variable name");
  }
  if (const llvm::json::Value *options = object.get("provider_options")) {
    const llvm::json::Object *optionsObject = options->getAsObject();
    if (!optionsObject)
      return configError(source, field + ".provider_options must be an object");
    config.providerOptions = *optionsObject;
  }
  return config;
}

llvm::Expected<RuntimePolicy> parseRuntimePolicy(llvm::StringRef policy,
                                                 llvm::StringRef source) {
  if (policy == "auto")
    return RuntimePolicy::Auto;
  if (policy == "host")
    return RuntimePolicy::Host;
  if (policy == "polyarch_container")
    return RuntimePolicy::PolyArchContainer;
  return configError(source, "runtime.policy has an unknown value");
}

} // namespace

bool isValidEnvironmentName(llvm::StringRef name) {
  if (name.empty() ||
      !(std::isalpha(static_cast<unsigned char>(name.front())) ||
        name.front() == '_'))
    return false;
  for (char character : name.drop_front())
    if (!(std::isalnum(static_cast<unsigned char>(character)) ||
          character == '_'))
      return false;
  return true;
}

LocalToolConfig defaultLocalToolConfig() { return LocalToolConfig{}; }

llvm::Expected<LocalToolConfig>
parseLocalToolConfig(llvm::StringRef body, llvm::StringRef sourceName) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(body);
  if (!parsed)
    return configError(sourceName, llvm::toString(parsed.takeError()));
  if (llvm::Error error = rejectDuplicateKeys(body, sourceName))
    return std::move(error);
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return configError(sourceName, "the top-level value must be an object");
  if (llvm::Error error = requireOnlyKeys(
          *root,
          {"schema", "version", "experiment_root", "module", "external_files",
           "external_file_trees", "runtime", "tools"},
          sourceName, ""))
    return std::move(error);
  if (root->getString("schema") != "loom.local_tool_config")
    return configError(sourceName, "schema must be loom.local_tool_config");
  const std::optional<llvm::StringRef> version = root->getString("version");
  if (!version || (*version != "1.0" && *version != "1.1"))
    return configError(sourceName, "version must be 1.0 or 1.1");
  if (*version == "1.0" && root->get("external_file_trees"))
    return configError(sourceName, "external_file_trees requires version 1.1");

  LocalToolConfig config;
  if (const llvm::json::Value *rootValue = root->get("experiment_root")) {
    std::optional<llvm::StringRef> path = rootValue->getAsString();
    if (!path)
      return configError(sourceName, "experiment_root must be a string");
    if (containsNull(*path))
      return configError(sourceName, "experiment_root contains NUL");
    if (path->empty() || !llvm::sys::path::is_absolute(*path))
      return configError(sourceName,
                         "experiment_root must be an absolute path");
    config.experimentRoot = path->str();
  }
  if (const llvm::json::Value *moduleValue = root->get("module")) {
    const llvm::json::Object *module = moduleValue->getAsObject();
    if (!module)
      return configError(sourceName, "module must be an object");
    if (llvm::Error error =
            requireOnlyKeys(*module, {"init"}, sourceName, "module"))
      return std::move(error);
    std::optional<llvm::StringRef> init = module->getString("init");
    if (init && containsNull(*init))
      return configError(sourceName, "module.init contains NUL");
    if (!init || init->empty() || !llvm::sys::path::is_absolute(*init))
      return configError(sourceName, "module.init must be an absolute path");
    config.moduleInit = init->str();
  }

  if (const llvm::json::Value *filesValue = root->get("external_files")) {
    const llvm::json::Object *files = filesValue->getAsObject();
    if (!files)
      return configError(sourceName, "external_files must be an object");
    for (const auto &[key, value] : *files) {
      const llvm::StringRef keyRef = key;
      if (containsNull(keyRef))
        return configError(sourceName, "external file key contains NUL");
      if (keyRef.empty())
        return configError(sourceName, "external file key must be nonempty");
      const std::string field = (llvm::Twine("external_files.") + keyRef).str();
      std::optional<llvm::StringRef> path = value.getAsString();
      if (!path)
        return configError(sourceName, field + " must be a string");
      if (containsNull(*path))
        return configError(sourceName, field + " contains NUL");
      if (path->empty() || !llvm::sys::path::is_absolute(*path))
        return configError(sourceName, field + " must be an absolute path");
      config.externalFiles.emplace(keyRef.str(), path->str());
    }
  }

  if (const llvm::json::Value *treesValue = root->get("external_file_trees")) {
    const llvm::json::Object *trees = treesValue->getAsObject();
    if (!trees)
      return configError(sourceName, "external_file_trees must be an object");
    for (const auto &[key, value] : *trees) {
      const llvm::StringRef keyRef = key;
      if (containsNull(keyRef))
        return configError(sourceName, "external file tree key contains NUL");
      if (keyRef.empty())
        return configError(sourceName,
                           "external file tree key must be nonempty");
      const std::string field =
          (llvm::Twine("external_file_trees.") + keyRef).str();
      std::optional<llvm::StringRef> path = value.getAsString();
      if (!path)
        return configError(sourceName, field + " must be a string");
      if (containsNull(*path))
        return configError(sourceName, field + " contains NUL");
      if (path->empty() || !llvm::sys::path::is_absolute(*path))
        return configError(sourceName, field + " must be an absolute path");
      config.externalFileTrees.emplace(keyRef.str(), path->str());
    }
  }

  if (const llvm::json::Value *runtimeValue = root->get("runtime")) {
    const llvm::json::Object *runtime = runtimeValue->getAsObject();
    if (!runtime)
      return configError(sourceName, "runtime must be an object");
    if (llvm::Error error = requireOnlyKeys(
            *runtime, {"policy", "polyarch_container"}, sourceName, "runtime"))
      return std::move(error);
    if (const llvm::json::Value *policyValue = runtime->get("policy")) {
      std::optional<llvm::StringRef> policy = policyValue->getAsString();
      if (!policy)
        return configError(sourceName, "runtime.policy must be a string");
      auto parsedPolicy = parseRuntimePolicy(*policy, sourceName);
      if (!parsedPolicy)
        return parsedPolicy.takeError();
      config.runtimePolicy = *parsedPolicy;
    }
    if (const llvm::json::Value *containerValue =
            runtime->get("polyarch_container")) {
      const llvm::json::Object *container = containerValue->getAsObject();
      if (!container)
        return configError(sourceName,
                           "runtime.polyarch_container must be an object");
      auto provider = parseProviderConfig(*container, sourceName,
                                          "runtime.polyarch_container", true);
      if (!provider)
        return provider.takeError();
      static_cast<LocalProviderConfig &>(config.polyArchContainer) =
          std::move(*provider);
      if (const llvm::json::Value *osValue = container->get("os")) {
        std::optional<llvm::StringRef> os = osValue->getAsString();
        if (os && containsNull(*os))
          return configError(sourceName,
                             "runtime.polyarch_container.os contains NUL");
        if (!os || os->empty())
          return configError(sourceName,
                             "runtime.polyarch_container.os must be a "
                             "nonempty string");
        config.polyArchContainer.os = os->str();
      }
    }
  }

  if (const llvm::json::Value *toolsValue = root->get("tools")) {
    const llvm::json::Object *tools = toolsValue->getAsObject();
    if (!tools)
      return configError(sourceName, "tools must be an object");
    for (const auto &[key, value] : *tools) {
      const llvm::StringRef keyRef = key;
      if (containsNull(keyRef))
        return configError(sourceName, "tool key contains NUL");
      if (keyRef.empty())
        return configError(sourceName, "tool key must be nonempty");
      const std::string field = (llvm::Twine("tools.") + keyRef).str();
      const llvm::json::Object *object = value.getAsObject();
      if (!object)
        return configError(sourceName, field + " must be an object");
      auto provider = parseProviderConfig(*object, sourceName, field);
      if (!provider)
        return provider.takeError();
      config.tools.emplace(keyRef.str(), std::move(*provider));
    }
  }
  return config;
}

llvm::Expected<LocalToolConfig> loadLocalToolConfig(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (std::error_code error = buffer.getError())
    return configError(path, error.message());
  return parseLocalToolConfig((*buffer)->getBuffer(), path);
}

} // namespace loom::external_tool
