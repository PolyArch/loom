#ifndef LOOM_FRONTEND_EXECUTABLE_COMPILERTARGETBINDINGINTERNAL_H
#define LOOM_FRONTEND_EXECUTABLE_COMPILERTARGETBINDINGINTERNAL_H

#include "Frontend/Executable/CompilerTargetBinding.h"

#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <vector>

namespace llvm {
class TargetMachine;
}

namespace loom::detail {

struct DecodedCompilerTargetBindingFields final {
  CompilerProcessorArchitectureRef processorArchitecture;
  ArchitectureFingerprint architectureFingerprint;
  LlvmProviderIdentity provider;
  std::string targetTriple;
  std::string dataLayout;
  fabric::RiscVAbi backendAbi;
  CompilerObjectFormat objectFormat;
  fabric::RiscVCodeModel codeModel;
  fabric::RelocationModel relocationModel;
  std::string backendCpu;
  std::vector<std::string> backendFeatures;
  std::vector<TargetScopeBinding> targetScopeBindings;
  std::vector<CompilerSupportComponent> supportComponents;
};

struct ReconstructedCompilerTarget final {
  LlvmProviderIdentity provider;
  std::string targetTriple;
  std::string dataLayout;
  CompilerObjectFormat objectFormat;
  std::vector<std::string> backendFeatures;
  std::vector<TargetScopeBinding> targetScopeBindings;
};

llvm::Expected<fabric::InstructionCoreArchitecturalContract>
resolveProcessorArchitecture(const CompilerProcessorArchitectureRef &processor,
                             const ArtifactStore &store);

llvm::Expected<ReconstructedCompilerTarget> reconstructCompilerTarget(
    const fabric::InstructionCoreArchitecturalContract &architecture,
    fabric::RiscVAbi backendAbi, fabric::RiscVCodeModel codeModel,
    fabric::RelocationModel relocationModel, llvm::StringRef backendCpu);

llvm::Expected<std::unique_ptr<llvm::TargetMachine>>
createCompilerTargetMachine(llvm::StringRef targetTriple,
                            fabric::RiscVAbi backendAbi,
                            fabric::RiscVCodeModel codeModel,
                            fabric::RelocationModel relocationModel,
                            llvm::StringRef backendCpu,
                            llvm::ArrayRef<std::string> backendFeatures);

std::string
serializeCompilerTargetBinding(const CompilerTargetBinding &binding);

llvm::Expected<DecodedCompilerTargetBindingFields>
parseCompilerTargetBindingFields(llvm::StringRef jsonText);

llvm::Error validateSupportComponents(
    llvm::ArrayRef<CompilerSupportComponent> supportComponents);
llvm::Expected<std::vector<CompilerSupportComponent>>
canonicalizeSupportComponents(
    llvm::ArrayRef<CompilerSupportComponent> supportComponents);

} // namespace loom::detail

#endif // LOOM_FRONTEND_EXECUTABLE_COMPILERTARGETBINDINGINTERNAL_H
