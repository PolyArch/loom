#include "EDA/Adapters/OpenSource/Verilator.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

namespace {

using loom::eda::open_source::parseVerilatorFunctionalResult;
using loom::eda::open_source::renderVerilatorFunctionalDriver;
using loom::eda::open_source::VerilatorFunctionalStatus;

[[noreturn]] void fail(llvm::StringRef test, llvm::StringRef message) {
  std::cerr << test.str() << ": " << message.str() << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, llvm::StringRef message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectInvalid(llvm::StringRef test, llvm::Expected<T> value) {
  if (value)
    fail(test, "invalid input was accepted");
  llvm::consumeError(value.takeError());
}

void driverBytesAreDeterministicAndExact() {
  const std::string driver =
      take(__func__, renderVerilatorFunctionalDriver("loom_testbench"));
  const std::string expected = R"args(--binary
--timing
--assert
--no-skip-identical
-CFLAGS
-std=gnu++20
--top-module
loom_testbench
--Mdir
outputs/verilator
-o
simulation
inputs/design.sv
inputs/testbench.sv
)args";
  require(__func__, driver == expected, "driver bytes changed");
  require(__func__,
          take(__func__, renderVerilatorFunctionalDriver("loom_testbench")) ==
              driver,
          "driver render is not deterministic");
  require(__func__,
          take(__func__, renderVerilatorFunctionalDriver("other_top")) !=
              driver,
          "driver ignores the exact testbench top");
}

void unsafeTopsAreRejected() {
  for (llvm::StringRef top :
       {"", "9bad", "has space", "quo\"te", "semi;colon", "new\nline",
        "carriage\r", "back\\slash", "\\escaped", "unicode-\xc3\xa9"})
    expectInvalid(__func__, renderVerilatorFunctionalDriver(top));
  require(__func__,
          take(__func__, renderVerilatorFunctionalDriver("dollar$1"))
                  .find("dollar$1\n") != std::string::npos,
          "identifier grammar rejected a legal continuation dollar");
}

void canonicalResultsAreParsedExactly() {
  const auto passed =
      take(__func__, parseVerilatorFunctionalResult(
                         "{\"schema\":\"loom.verilator_functional_result\","
                         "\"version\":\"1.0\",\"status\":\"passed\","
                         "\"completed_transactions\":3}\n"));
  require(__func__, passed.status == VerilatorFunctionalStatus::Passed,
          "passed result changed status");
  require(__func__, passed.completedTransactions == 3,
          "passed result changed transaction count");
  require(__func__, !passed.firstFailingTransaction,
          "passed result acquired a failing transaction");

  const auto failed =
      take(__func__, parseVerilatorFunctionalResult(
                         "{\"schema\":\"loom.verilator_functional_result\","
                         "\"version\":\"1.0\",\"status\":\"failed\","
                         "\"completed_transactions\":2,"
                         "\"first_failing_transaction\":1}\n"));
  require(__func__, failed.status == VerilatorFunctionalStatus::Failed,
          "failed result changed status");
  require(__func__, failed.completedTransactions == 2,
          "failed result changed transaction count");
  require(__func__, failed.firstFailingTransaction == 1,
          "failed result changed the first failing transaction");

  const auto fullRange = take(
      __func__, parseVerilatorFunctionalResult(
                    "{\"schema\":\"loom.verilator_functional_result\","
                    "\"version\":\"1.0\",\"status\":\"failed\","
                    "\"completed_transactions\":18446744073709551615,"
                    "\"first_failing_transaction\":18446744073709551614}\n"));
  require(__func__,
          fullRange.completedTransactions == UINT64_C(18446744073709551615) &&
              fullRange.firstFailingTransaction ==
                  UINT64_C(18446744073709551614),
          "full uint64 result range was not preserved");
}

void malformedAndInconsistentResultsAreRejected() {
  for (llvm::StringRef contents : {
           "not json",
           "[]",
           "{}",
           "{\"schema\":\"loom.verilator_functional_result\","
           "\"version\":\"1.0\",\"status\":\"passed\","
           "\"completed_transactions\":0}\n",
           "{\"schema\":\"loom.verilator_functional_result\","
           "\"version\":\"1.0\",\"status\":\"failed\","
           "\"completed_transactions\":1}\n",
           "{\"schema\":\"loom.verilator_functional_result\","
           "\"version\":\"1.0\",\"status\":\"passed\","
           "\"completed_transactions\":1,"
           "\"first_failing_transaction\":0}\n",
           "{\"schema\":\"loom.verilator_functional_result\","
           "\"version\":\"1.0\",\"status\":\"failed\","
           "\"completed_transactions\":1,"
           "\"first_failing_transaction\":1}\n",
           "{\"schema\":\"loom.verilator_functional_result\","
           "\"version\":\"1.0\",\"status\":\"unknown\","
           "\"completed_transactions\":1}\n",
           "{\"schema\":\"loom.verilator_functional_result\","
           "\"version\":\"1.0\",\"status\":\"passed\","
           "\"completed_transactions\":1,\"extra\":false}\n",
           "{\"version\":\"1.0\","
           "\"schema\":\"loom.verilator_functional_result\","
           "\"status\":\"passed\",\"completed_transactions\":1}\n",
           "{\"schema\":\"loom.verilator_functional_result\","
           "\"version\":\"1.0\",\"status\":\"passed\","
           "\"completed_transactions\":1}",
           "{\"schema\":\"loom.verilator_functional_result\","
           "\"version\":\"1.0\",\"status\":\"passed\","
           "\"completed_transactions\":-1}\n",
           "{\"schema\":\"loom.verilator_functional_result\","
           "\"version\":\"1.0\",\"status\":\"passed\","
           "\"completed_transactions\":18446744073709551616}\n",
       })
    expectInvalid(__func__, parseVerilatorFunctionalResult(contents));
}

const std::string kGoodDesign = R"sv(module add9(
  input logic [7:0] lhs,
  input logic [7:0] rhs,
  output logic [8:0] sum
);
  assign sum = {1'b0, lhs} + {1'b0, rhs};
endmodule
)sv";

const std::string kBadDesign = R"sv(module add9(
  input logic [7:0] lhs,
  input logic [7:0] rhs,
  output logic [8:0] sum
);
  assign sum = {1'b0, lhs} - {1'b0, rhs};
endmodule
)sv";

const std::string kTestbench = R"sv(module loom_testbench;
  logic [7:0] lhs;
  logic [7:0] rhs;
  logic [8:0] sum;
  integer result_file;

  add9 subject(.lhs(lhs), .rhs(rhs), .sum(sum));

  task automatic check_sum(
    input logic [7:0] next_lhs,
    input logic [7:0] next_rhs,
    input logic [8:0] expected,
    input integer ordinal,
    input integer completed
  );
    begin
      lhs = next_lhs;
      rhs = next_rhs;
      #1;
      if (sum !== expected) begin
        result_file = $fopen("outputs/verilator-functional-result.json", "w");
        if (result_file == 0)
          $fatal(1, "could not open functional result");
        $fwrite(result_file, "{\"schema\":\"loom.verilator_functional_result\",\"version\":\"1.0\",\"status\":\"failed\",\"completed_transactions\":%0d,\"first_failing_transaction\":%0d}\n", completed, ordinal);
        $fclose(result_file);
        $finish;
      end
    end
  endtask

  initial begin
    check_sum(8'd0, 8'd0, 9'd0, 0, 1);
    check_sum(8'd7, 8'd9, 9'd16, 1, 2);
    check_sum(8'd255, 8'd2, 9'd257, 2, 3);
    result_file = $fopen("outputs/verilator-functional-result.json", "w");
    if (result_file == 0)
      $fatal(1, "could not open functional result");
    $fwrite(result_file, "{\"schema\":\"loom.verilator_functional_result\",\"version\":\"1.0\",\"status\":\"passed\",\"completed_transactions\":3}\n");
    $fclose(result_file);
    $finish;
  end
endmodule
)sv";

void writeFile(const std::filesystem::path &path, llvm::StringRef contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!stream)
    fail("emit", "could not write materialized file");
}

std::string readFile(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  std::ostringstream contents;
  contents << stream.rdbuf();
  if (!stream)
    fail("verify", "could not read materialized file");
  return contents.str();
}

int emit(const std::filesystem::path &root, llvm::StringRef designKind) {
  if (designKind != "good" && designKind != "bad")
    fail("emit", "unknown design kind");
  writeFile(root / "inputs" / "design.sv",
            designKind == "good" ? kGoodDesign : kBadDesign);
  writeFile(root / "inputs" / "testbench.sv", kTestbench);
  writeFile(root / "drivers" / "verilator.args",
            take("emit", renderVerilatorFunctionalDriver("loom_testbench")));
  std::filesystem::create_directories(root / "outputs");
  return EXIT_SUCCESS;
}

int verify(const std::filesystem::path &root, llvm::StringRef expected) {
  const auto result = take(
      "verify", parseVerilatorFunctionalResult(readFile(
                    root / "outputs" / "verilator-functional-result.json")));
  if (expected == "passed") {
    require("verify",
            result.status == VerilatorFunctionalStatus::Passed &&
                result.completedTransactions == 3 &&
                !result.firstFailingTransaction,
            "real passed result changed");
  } else if (expected == "failed") {
    require("verify",
            result.status == VerilatorFunctionalStatus::Failed &&
                result.completedTransactions == 2 &&
                result.firstFailingTransaction == 1,
            "real adverse result changed");
  } else {
    fail("verify", "unknown expected status");
  }
  return EXIT_SUCCESS;
}

int compare(const std::filesystem::path &lhs,
            const std::filesystem::path &rhs) {
  require("compare",
          readFile(lhs / "drivers" / "verilator.args") ==
              readFile(rhs / "drivers" / "verilator.args"),
          "fresh-root driver bytes diverged");
  require("compare",
          readFile(lhs / "outputs" / "verilator-functional-result.json") ==
              readFile(rhs / "outputs" / "verilator-functional-result.json"),
          "fresh-root result bytes diverged");
  return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 4 && llvm::StringRef(argv[1]) == "--emit")
    return emit(argv[2], argv[3]);
  if (argc == 4 && llvm::StringRef(argv[1]) == "--verify")
    return verify(argv[2], argv[3]);
  if (argc == 4 && llvm::StringRef(argv[1]) == "--compare")
    return compare(argv[2], argv[3]);
  if (argc != 1)
    fail("main", "unexpected arguments");
  driverBytesAreDeterministicAndExact();
  unsafeTopsAreRejected();
  canonicalResultsAreParsedExactly();
  malformedAndInconsistentResultsAreRejected();
  return EXIT_SUCCESS;
}
