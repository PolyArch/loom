#ifndef LOOM_EDA_ADAPTERS_OPENSOURCE_VERILATOR_H
#define LOOM_EDA_ADAPTERS_OPENSOURCE_VERILATOR_H

#include "ExternalTool/InvocationBundle.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace loom::eda::open_source {

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeVerilatorLintBundle(llvm::StringRef systemVerilog,
                        llvm::StringRef semanticBindingIdentity,
                        const external_tool::ResolvedToolBinding &tool,
                        const external_tool::InvocationRuntimeBinding &runtime,
                        llvm::ArrayRef<std::string> inheritEnvironment = {});

} // namespace loom::eda::open_source

#endif // LOOM_EDA_ADAPTERS_OPENSOURCE_VERILATOR_H
