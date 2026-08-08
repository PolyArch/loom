#ifndef LOOM_EDA_ADAPTERS_OPENSOURCE_VERILATOR_H
#define LOOM_EDA_ADAPTERS_OPENSOURCE_VERILATOR_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>

namespace loom::eda::open_source {

enum class VerilatorFunctionalStatus : std::uint8_t {
  Passed = 0,
  Failed = 1,
};

/// The ephemeral result of one completed self-checking Verilator execution.
/// The exact EvaluationRequest owner interprets these facts before publishing
/// any Evidence; this value is not an Artifact or an Evidence substitute.
struct VerilatorFunctionalResult final {
  VerilatorFunctionalStatus status;
  std::uint64_t completedTransactions;
  std::optional<std::uint64_t> firstFailingTransaction;

  friend bool operator==(const VerilatorFunctionalResult &lhs,
                         const VerilatorFunctionalResult &rhs) {
    return lhs.status == rhs.status &&
           lhs.completedTransactions == rhs.completedTransactions &&
           lhs.firstFailingTransaction == rhs.firstFailingTransaction;
  }
};

/// Renders the byte-deterministic Verilator 5.050 response file for a
/// self-checking functional run. It consumes only inputs/design.sv and
/// inputs/testbench.sv and builds outputs/verilator/simulation. The caller
/// owns execution of that binary; the testbench writes the fixed
/// outputs/verilator-functional-result.json result.
llvm::Expected<std::string>
renderVerilatorFunctionalDriver(llvm::StringRef testbenchTop);

/// Strictly parses the canonical authored result protocol. A passed run has
/// no failing transaction. A failed run identifies its first failing ordinal,
/// which must be inside the nonempty completed prefix.
llvm::Expected<VerilatorFunctionalResult>
parseVerilatorFunctionalResult(llvm::StringRef contents);

} // namespace loom::eda::open_source

#endif // LOOM_EDA_ADAPTERS_OPENSOURCE_VERILATOR_H
