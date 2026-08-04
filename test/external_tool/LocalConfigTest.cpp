#include "ExternalTool/LocalConfig.h"

#include "llvm/Support/Error.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

using namespace loom::external_tool;

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  std::cerr << test << ": " << message << '\n';
  std::exit(1);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectErrorContains(const char *test, llvm::Expected<T> value,
                         llvm::StringRef expected) {
  if (value)
    fail(test, "expected an error");
  std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected),
          "unexpected error: " + message);
}

void parsesExplicitBindings() {
  const char *body = R"json({
    "schema": "loom.local_tool_config",
    "version": "1.0",
    "module": {"init": "/opt/modules/init/bash"},
    "external_files": {
      "saed14_tt_liberty": "/opt/pdk/saed14/lib/tt.lib",
      "sram_logical_view": "/opt/ip/sram/sram.db"
    },
    "runtime": {
      "policy": "polyarch_container",
      "polyarch_container": {
        "binding": {"modules": ["container/1.0"]},
        "os": "almalinux9",
        "inherit_environment": ["DISPLAY"],
        "provider_options": {"network": "host"}
      }
    },
    "tools": {
      "vcs": {
        "binding": {"executable": "/opt/vcs/bin/vcs"},
        "inherit_environment": ["SNPSLMD_LICENSE_FILE"],
        "provider_options": {"queue": true}
      }
    }
  })json";

  LocalToolConfig config = take(__func__, parseLocalToolConfig(body, "test"));
  require(__func__, config.moduleInit == "/opt/modules/init/bash",
          "module initialization path was not preserved");
  require(__func__,
          config.externalFiles ==
              std::map<std::string, std::string>{
                  {"saed14_tt_liberty", "/opt/pdk/saed14/lib/tt.lib"},
                  {"sram_logical_view", "/opt/ip/sram/sram.db"}},
          "external files were not parsed");
  require(__func__, config.runtimePolicy == RuntimePolicy::PolyArchContainer,
          "runtime policy was not parsed");
  require(__func__,
          config.polyArchContainer.binding.modules ==
              std::vector<std::string>{"container/1.0"},
          "container module binding was not parsed");
  require(__func__, config.polyArchContainer.os == "almalinux9",
          "container OS was not parsed");
  require(__func__,
          config.polyArchContainer.inheritEnvironment ==
              std::vector<std::string>{"DISPLAY"},
          "container environment names were not parsed");

  auto tool = config.tools.find("vcs");
  require(__func__, tool != config.tools.end(), "tool entry is missing");
  require(__func__, tool->second.binding.executable == "/opt/vcs/bin/vcs",
          "tool executable binding was not parsed");
  require(__func__,
          tool->second.inheritEnvironment ==
              std::vector<std::string>{"SNPSLMD_LICENSE_FILE"},
          "tool environment names were not parsed");
  require(__func__, tool->second.providerOptions.getBoolean("queue") == true,
          "tool provider option was not preserved");
}

void rejectsDuplicateKeys() {
  const char *body = R"json({
    "schema": "wrong",
    "schema": "loom.local_tool_config",
    "version": "1.0"
  })json";
  expectErrorContains(__func__, parseLocalToolConfig(body, "duplicate.json"),
                      "duplicate key");
}

void rejectsUnknownFields() {
  expectErrorContains(
      __func__,
      parseLocalToolConfig(
          R"json({"schema":"loom.local_tool_config","version":"1.0","extra":1})json",
          "unknown.json"),
      "unknown key");
  expectErrorContains(
      __func__,
      parseLocalToolConfig(
          R"json({"schema":"loom.local_tool_config","version":"1.0","module":"wrong"})json",
          "module.json"),
      "module must be an object");
  expectErrorContains(
      __func__,
      parseLocalToolConfig(
          R"json({"schema":"loom.local_tool_config","version":"1.0","tools":{"vcs":{"mystery":true}}})json",
          "tool.json"),
      "unknown key");
}

void rejectsInvalidBindingsAndNames() {
  expectErrorContains(
      __func__,
      parseLocalToolConfig(
          R"json({"schema":"loom.local_tool_config","version":"1.0","module":{"init":"relative/init"}})json",
          "relative.json"),
      "absolute");
  expectErrorContains(
      __func__,
      parseLocalToolConfig(
          R"json({"schema":"loom.local_tool_config","version":"1.0","tools":{"vcs":{"binding":{"executable":"/opt/vcs","modules":["vcs"]}}}})json",
          "union.json"),
      "exactly one");
  expectErrorContains(
      __func__,
      parseLocalToolConfig(
          R"json({"schema":"loom.local_tool_config","version":"1.0","tools":{"vcs":{"binding":{"modules":[]}}}})json",
          "modules.json"),
      "nonempty");
  expectErrorContains(
      __func__,
      parseLocalToolConfig(
          R"json({"schema":"loom.local_tool_config","version":"1.0","tools":{"vcs":{"inherit_environment":["BAD=VALUE"]}}})json",
          "environment.json"),
      "environment variable");
  expectErrorContains(
      __func__,
      parseLocalToolConfig(
          R"json({"schema":"loom.local_tool_config","version":"1.0","tools":{"":{"binding":{"executable":"/opt/tool"}}}})json",
          "tool-key.json"),
      "tool key");
  expectErrorContains(
      __func__,
      parseLocalToolConfig(
          R"json({"schema":"loom.local_tool_config","version":"1.0","tools":{"vcs":{"binding":{"executable":"/opt/vcs\u0000suffix"}}}})json",
          "nul-executable.json"),
      "NUL");
  expectErrorContains(
      __func__,
      parseLocalToolConfig(
          R"json({"schema":"loom.local_tool_config","version":"1.0","tools":{"vcs":{"binding":{"modules":["vcs\u0000suffix"]}}}})json",
          "nul-module.json"),
      "NUL");
  expectErrorContains(
      __func__,
      parseLocalToolConfig(
          R"json({"schema":"loom.local_tool_config","version":"1.0","external_files":{"liberty":"relative/tt.lib"}})json",
          "relative-external-file.json"),
      "external_files.liberty must be an absolute path");
  expectErrorContains(
      __func__,
      parseLocalToolConfig(
          R"json({"schema":"loom.local_tool_config","version":"1.0","external_files":{"":"/opt/pdk/tt.lib"}})json",
          "empty-external-file-key.json"),
      "external file key must be nonempty");
  expectErrorContains(
      __func__,
      parseLocalToolConfig(
          R"json({"schema":"loom.local_tool_config","version":"1.0","external_files":{"liberty":42}})json",
          "external-file-type.json"),
      "external_files.liberty must be a string");
  expectErrorContains(
      __func__,
      parseLocalToolConfig(
          R"json({"schema":"loom.local_tool_config","version":"1.0","external_files":{"liberty":"/opt/pdk/tt.lib\u0000suffix"}})json",
          "nul-external-file.json"),
      "external_files.liberty contains NUL");

  expectErrorContains(
      __func__,
      parseLocalToolConfig(
          R"json({"schema":"loom.local_tool_config","version":"1.0","platform_payload_roots":{"legacy":"/opt/platform"}})json",
          "legacy-platform-payload-root.json"),
      "unknown key 'platform_payload_roots'");
}

void defaultsDoNotLoadMachineState() {
  const LocalToolConfig config = defaultLocalToolConfig();
  require(__func__, config.runtimePolicy == RuntimePolicy::Auto,
          "default runtime policy is not auto");
  require(__func__,
          !config.moduleInit && config.externalFiles.empty() &&
              config.tools.empty() &&
              !config.polyArchContainer.binding.isConfigured() &&
              !config.polyArchContainer.os,
          "default local configuration contains machine state");
}

} // namespace

int main(int argc, char **argv) {
  parsesExplicitBindings();
  rejectsDuplicateKeys();
  rejectsUnknownFields();
  rejectsInvalidBindingsAndNames();
  defaultsDoNotLoadMachineState();
  require(__func__, argc == 2, "expected the example config path");
  LocalToolConfig example = take(__func__, loadLocalToolConfig(argv[1]));
  require(__func__, example.runtimePolicy == RuntimePolicy::Auto,
          "the example must preserve automatic runtime selection");
  require(__func__,
          example.externalFiles ==
              std::map<std::string, std::string>{
                  {"asic_liberty", "/path/to/pdk/library/typical.lib"}},
          "the example must show one placeholder external file");
  require(__func__, example.tools.count("verilator") == 1,
          "the example must include a Verilator binding");
  return 0;
}
