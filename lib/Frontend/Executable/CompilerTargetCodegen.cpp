#include "CompilerTargetBindingInternal.h"
#include "Frontend/Executable/CompilerTargetBinding.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

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
  passes.run(*module);
  if (object.empty())
    return codegenError("compiler_target_object_execution_failed",
                        "the target emitted an empty object");
  return std::vector<std::uint8_t>(object.begin(), object.end());
}

} // namespace loom
