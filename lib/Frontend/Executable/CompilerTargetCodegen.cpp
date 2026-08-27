#include "CompilerTargetBindingInternal.h"
#include "Common/InvocationDiagnosticLog.h"
#include "Frontend/Executable/CompilerTargetBinding.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace loom {
namespace {

llvm::Error codegenError(llvm::StringRef marker, const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 marker + ": " + message);
}

} // namespace

llvm::Expected<std::vector<std::uint8_t>>
emitCompilerTargetObject(std::unique_ptr<llvm::Module> module,
                         const CompilerTargetBinding &binding) {
  if (!module)
    return codegenError("compiler_target_object_invalid",
                        "the LLVM module is null");
  if (binding.objectFormat() != CompilerObjectFormat::Elf)
    return codegenError("compiler_target_object_format_unavailable",
                        "the exact binding does not select ELF");

  std::string verificationMessage;
  llvm::raw_string_ostream verificationStream(verificationMessage);
  if (llvm::verifyModule(*module, &verificationStream))
    return codegenError("compiler_target_object_invalid",
                        verificationStream.str());
  if (llvm::Error error = validateModuleCompilerTarget(*module, binding))
    return std::move(error);

  auto machine = detail::createCompilerTargetMachine(
      binding.targetTriple(), binding.backendAbi(), binding.codeModel(),
      binding.relocationModel(), binding.backendCpu(),
      binding.backendFeatures());
  if (!machine)
    return machine.takeError();
  if ((*machine)->createDataLayout().getStringRepresentation() !=
      binding.dataLayout())
    return codegenError("compiler_target_reconstruction_mismatch",
                        "the target machine changed the binding DataLayout");

  llvm::SmallVector<char, 0> object;
  llvm::raw_svector_ostream objectStream(object);
  llvm::legacy::PassManager passes;
  if ((*machine)->addPassesToEmitFile(passes, objectStream, nullptr,
                                      llvm::CodeGenFileType::ObjectFile))
    return codegenError("compiler_target_object_provider_unavailable",
                        "the selected target cannot emit an ELF object");
  std::uint64_t functionCount = 0;
  std::uint64_t totalInstructions = 0;
  std::uint64_t largestFunctionInstructions = 0;
  llvm::StringRef largestFunction;
  for (const llvm::Function &function : *module) {
    if (function.isDeclaration())
      continue;
    ++functionCount;
    const std::uint64_t instructions = function.getInstructionCount();
    totalInstructions += instructions;
    if (instructions > largestFunctionInstructions) {
      largestFunctionInstructions = instructions;
      largestFunction = function.getName();
    }
  }
  const auto begin = std::chrono::steady_clock::now();
  passes.run(*module);
  const std::uint64_t elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - begin)
          .count();
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary, InvocationDiagnosticStage::Deployment,
      InvocationDiagnosticEvent::Statistics, [&] {
        llvm::json::Object payload;
        payload["statistics_kind"] = "compiler_target_codegen";
        payload["target_triple"] = binding.targetTriple();
        payload["duration_ns"] = elapsed;
        payload["function_count"] = functionCount;
        payload["total_instructions"] = totalInstructions;
        payload["largest_function_instructions"] =
            largestFunctionInstructions;
        payload["largest_function"] = largestFunction;
        return llvm::json::Value(std::move(payload));
      });
  if (object.empty())
    return codegenError("compiler_target_object_execution_failed",
                        "the target emitted an empty object");
  return std::vector<std::uint8_t>(object.begin(), object.end());
}

} // namespace loom
