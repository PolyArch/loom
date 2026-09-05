#ifndef LOOM_EDA_OPENSOURCE_YOSYSSYNTHESISINVOCATION_H
#define LOOM_EDA_OPENSOURCE_YOSYSSYNTHESISINVOCATION_H

#include "EDA/Adapters/OpenSource/Yosys.h"
#include "EDA/Adapters/OpenSource/YosysGateNetlist.h"

namespace loom::eda::open_source {

// Provider-local execution plumbing shared by whole and reusable block flows.
llvm::Expected<external_tool::PreparedExternalToolInvocation>
prepareYosysSynthesisInvocation(
    const ResolvedYosysGateNetlistConfigView &config,
    const external_tool::ExternalToolSemanticContract &contract,
    llvm::ArrayRef<external_tool::MaterializedBundleFile> semanticInputs,
    llvm::StringRef top, llvm::ArrayRef<std::string> rtlPaths,
    const YosysMappedChildren *children,
    const external_tool::ExternalToolPreparationContext &context);

external_tool::ExternalToolInvocationImportExpectation
yosysSynthesisInvocationExpectation(
    const external_tool::ExternalToolSemanticContract &contract,
    llvm::ArrayRef<external_tool::MaterializedBundleFile> semanticInputs,
    const ResolvedYosysGateNetlistConfigView &config);

struct YosysSynthesisOutput final {
  std::string netlist;
  YosysStructureFacts structure;
};

llvm::Expected<YosysSynthesisOutput> readYosysSynthesisOutput(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const external_tool::ImportedExternalToolInvocationBundle &imported,
    llvm::StringRef top);

std::string renderYosysStandardCellBlackBoxContract(
    const ExternalFileFingerprint &library,
    llvm::ArrayRef<hardware::RepresentationLocator> unresolved);

} // namespace loom::eda::open_source

#endif // LOOM_EDA_OPENSOURCE_YOSYSSYNTHESISINVOCATION_H
