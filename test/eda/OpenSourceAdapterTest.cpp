#include "EDA/Adapters/OpenSource/Verilator.h"

#include "llvm/Support/Error.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace loom::external_tool;
using namespace loom::eda::open_source;

namespace {

[[noreturn]] void fail(const std::string &message) {
  std::cerr << message << '\n';
  std::exit(1);
}

void require(bool condition, const std::string &message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

} // namespace

int main() {
  ResolvedToolBinding tool{"verilator",
                           ToolBindingSource::Module,
                           "/tools/verilator",
                           "Verilator 5.050",
                           {"verilator"},
                           {"verilator/5.050"},
                           "/etc/profile.d/modules.sh",
                           std::nullopt};
  InvocationRuntimeBinding runtime;
  ExternalToolInvocationBundleSpec bundle = take(makeVerilatorLintBundle(
      "module loom_circt_conformance; endmodule\n", "circt-api@1", tool,
      runtime, {"LM_LICENSE_FILE"}));
  require(bundle.providerIdentity == "verilator.lint@1" &&
              bundle.semanticBindingIdentity == "circt-api@1" &&
              bundle.resultImporterIdentity == "verilator.lint.completion@1",
          "adapter identities are incomplete");
  require(bundle.files.size() == 1 &&
              bundle.files.front().relativePath ==
                  "drivers/loom_circt_conformance.sv" &&
              bundle.files.front().contents.find("module") != std::string::npos,
          "adapter did not materialize generated SystemVerilog");
  require(bundle.commands ==
                  std::vector<std::vector<std::string>>{
                      {"/tools/verilator", "--lint-only", "--Wno-fatal",
                       "--Wall", "drivers/loom_circt_conformance.sv"}} &&
              bundle.inheritEnvironment ==
                  std::vector<std::string>{"LM_LICENSE_FILE"},
          "adapter command projection is not structured or deterministic");
  return 0;
}
