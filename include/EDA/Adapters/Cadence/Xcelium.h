#ifndef LOOM_EDA_ADAPTERS_CADENCE_XCELIUM_H
#define LOOM_EDA_ADAPTERS_CADENCE_XCELIUM_H

#include "EDA/Adapters/Cadence/Common.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace loom::eda::cadence {

enum class XceliumFunctionalStatus : std::uint8_t { Passed, Failed };

struct XceliumFunctionalResult final {
  XceliumFunctionalStatus status;
  std::uint64_t completedTransactions;
  std::optional<std::uint64_t> firstFailingTransaction;
};

const CadenceInvocationDescriptor &xceliumFunctionalDescriptor();

llvm::Expected<std::vector<std::string>>
renderXceliumFunctionalCommand(llvm::StringRef executable,
                               llvm::StringRef testbenchTop,
                               llvm::ArrayRef<std::string> sourcePaths);

llvm::Expected<XceliumFunctionalResult>
parseXceliumFunctionalResult(llvm::StringRef contents);

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeXceliumFunctionalBundleSpec(const CadenceBundleInputs &inputs,
                                llvm::StringRef testbenchTop,
                                llvm::ArrayRef<std::string> sourcePaths);

llvm::Expected<XceliumFunctionalResult> importXceliumFunctionalResult(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const CadenceBundleInputs &inputs);

} // namespace loom::eda::cadence

#endif // LOOM_EDA_ADAPTERS_CADENCE_XCELIUM_H
