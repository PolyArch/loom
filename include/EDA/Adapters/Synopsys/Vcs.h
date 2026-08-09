#ifndef LOOM_EDA_ADAPTERS_SYNOPSYS_VCS_H
#define LOOM_EDA_ADAPTERS_SYNOPSYS_VCS_H

#include "EDA/Adapters/Synopsys/Common.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace loom::eda::synopsys {

enum class VcsFunctionalStatus : std::uint8_t { Passed, Failed };

struct VcsFunctionalResult final {
  VcsFunctionalStatus status;
  std::uint64_t completedTransactions;
  std::optional<std::uint64_t> firstFailingTransaction;
};

const SynopsysInvocationDescriptor &vcsFunctionalDescriptor();

llvm::Expected<std::vector<std::string>>
renderVcsFunctionalCommand(llvm::StringRef executable,
                           llvm::StringRef testbenchTop,
                           llvm::ArrayRef<std::string> sourcePaths);

llvm::Expected<VcsFunctionalResult>
parseVcsFunctionalResult(llvm::StringRef contents);

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeVcsFunctionalBundleSpec(const SynopsysBundleInputs &inputs,
                            llvm::StringRef testbenchTop,
                            llvm::ArrayRef<std::string> sourcePaths);

llvm::Expected<VcsFunctionalResult> importVcsFunctionalResult(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const SynopsysBundleInputs &inputs);

} // namespace loom::eda::synopsys

#endif // LOOM_EDA_ADAPTERS_SYNOPSYS_VCS_H
