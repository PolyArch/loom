#include "EDA/Adapters/OpenSource/Verilator.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <optional>
#include <string>

namespace loom::eda::open_source {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "verilator_adapter_invalid: " + message);
}

bool isPortableIdentifier(llvm::StringRef value) {
  const auto isFirst = [](char character) {
    return (character >= 'A' && character <= 'Z') ||
           (character >= 'a' && character <= 'z') || character == '_';
  };
  const auto isRest = [&](char character) {
    return isFirst(character) || (character >= '0' && character <= '9') ||
           character == '$';
  };
  return !value.empty() && isFirst(value.front()) &&
         llvm::all_of(value.drop_front(), isRest);
}

std::optional<std::uint64_t>
getUnsignedInteger(const llvm::json::Object &object, llvm::StringRef field) {
  const llvm::json::Value *value = object.get(field);
  return value ? value->getAsUINT64() : std::nullopt;
}

std::string serializeResult(const VerilatorFunctionalResult &result) {
  std::string bytes = "{\"schema\":\"loom.verilator_functional_result\","
                      "\"version\":\"1.0\",\"status\":\"";
  bytes +=
      result.status == VerilatorFunctionalStatus::Passed ? "passed" : "failed";
  bytes += "\",\"completed_transactions\":" +
           std::to_string(result.completedTransactions);
  if (result.firstFailingTransaction)
    bytes += ",\"first_failing_transaction\":" +
             std::to_string(*result.firstFailingTransaction);
  bytes += "}\n";
  return bytes;
}

} // namespace

llvm::Expected<std::string>
renderVerilatorFunctionalDriver(llvm::StringRef testbenchTop) {
  if (!isPortableIdentifier(testbenchTop))
    return invalid("testbench top is not a portable HDL identifier");
  return "--binary\n"
         "--timing\n"
         "--assert\n"
         "--no-skip-identical\n"
         "-CFLAGS\n"
         "-std=gnu++20\n"
         "--top-module\n" +
         testbenchTop.str() +
         "\n--Mdir\n"
         "outputs/verilator\n"
         "-o\n"
         "simulation\n"
         "inputs/design.sv\n"
         "inputs/testbench.sv\n";
}

llvm::Expected<VerilatorFunctionalResult>
parseVerilatorFunctionalResult(llvm::StringRef contents) {
  auto parsed = llvm::json::parse(contents);
  if (!parsed)
    return invalid("functional result is malformed JSON: " +
                   llvm::toString(parsed.takeError()));
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object)
    return invalid("functional result root is not an object");

  const std::optional<llvm::StringRef> schema = object->getString("schema");
  const std::optional<llvm::StringRef> version = object->getString("version");
  const std::optional<llvm::StringRef> status = object->getString("status");
  const std::optional<std::uint64_t> completed =
      getUnsignedInteger(*object, "completed_transactions");
  if (!schema || *schema != "loom.verilator_functional_result" || !version ||
      *version != "1.0" || !status || !completed || *completed == 0)
    return invalid("functional result fields are invalid");

  VerilatorFunctionalResult result;
  result.completedTransactions = *completed;
  const std::optional<std::uint64_t> firstFailing =
      getUnsignedInteger(*object, "first_failing_transaction");
  if (*status == "passed") {
    if (object->size() != 4 || firstFailing)
      return invalid("passed result contains inconsistent fields");
    result.status = VerilatorFunctionalStatus::Passed;
  } else if (*status == "failed") {
    if (object->size() != 5 || !firstFailing || *firstFailing >= *completed)
      return invalid("failed result contains inconsistent fields");
    result.status = VerilatorFunctionalStatus::Failed;
    result.firstFailingTransaction = *firstFailing;
  } else {
    return invalid("functional result has an unknown status");
  }

  if (contents != serializeResult(result))
    return invalid("functional result is not canonically encoded");
  return result;
}

} // namespace loom::eda::open_source
