#ifndef LOOM_EDA_ADAPTERS_SYNOPSYS_DESIGNCOMPILER_H
#define LOOM_EDA_ADAPTERS_SYNOPSYS_DESIGNCOMPILER_H

#include "EDA/Adapters/Synopsys/Common.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace loom::eda::synopsys {

struct DesignCompilerGateNetlist final {
  std::string verilog;
};

const SynopsysInvocationDescriptor &designCompilerDescriptor();

llvm::Expected<std::string> renderDesignCompilerDriver(
    llvm::StringRef top, llvm::ArrayRef<std::string> rtlSources,
    llvm::StringRef generationConstraint, llvm::StringRef targetLibrary);

llvm::Expected<DesignCompilerGateNetlist>
parseDesignCompilerGateNetlist(llvm::StringRef contents, llvm::StringRef top);

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeDesignCompilerBundleSpec(const SynopsysBundleInputs &inputs,
                             llvm::StringRef top,
                             llvm::ArrayRef<std::string> rtlSources,
                             llvm::StringRef generationConstraint);

llvm::Expected<DesignCompilerGateNetlist> importDesignCompilerGateNetlist(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const SynopsysBundleInputs &inputs, llvm::StringRef top);

} // namespace loom::eda::synopsys

#endif // LOOM_EDA_ADAPTERS_SYNOPSYS_DESIGNCOMPILER_H
