#include "Hardware/RTL/CirctConformance.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

using namespace loom::hardware::rtl;

namespace {

[[noreturn]] void fail(const std::string &message) {
  std::cerr << message << '\n';
  std::exit(1);
}

std::string take(llvm::Expected<std::string> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

} // namespace

int main() {
  const std::string first = take(emitCirctConformanceSystemVerilog());
  const std::string second = take(emitCirctConformanceSystemVerilog());
  if (first != second)
    fail("identical CIRCT construction produced different SystemVerilog");
  const llvm::StringRef output(first);
  const llvm::StringRef requiredNames[] = {
      "module loom_circt_conformance",
      "config_subtract",
      "lhs_valid",
      "lhs_ready",
      "rhs_valid",
      "rhs_ready",
      "result_valid",
      "result_ready",
      "result_data",
      "result_valid_reg",
      "result_data_reg",
      "memory_command_valid",
      "memory_command_ready",
      "memory_request_valid",
      "memory_request_ready",
      "memory_response_valid",
      "memory_response_ready",
      "memory_result_valid",
      "memory_result_ready",
  };
  for (llvm::StringRef name : requiredNames)
    if (!output.contains(name))
      fail("CIRCT conformance output lacks '" + name.str() + "':\n" + first);
  if (!output.contains("always_ff") ||
      !output.contains("posedge async_reset") ||
      !output.contains("if (sync_reset)") || !output.contains("lhs_data +") ||
      !output.contains("lhs_data -"))
    fail("CIRCT conformance output lacks state, reset, or configured compute "
         "logic:\n" +
         first);
  return 0;
}
