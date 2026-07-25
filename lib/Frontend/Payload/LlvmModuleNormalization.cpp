#include "Frontend/Payload/LlvmModuleNormalization.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace loom {
namespace {

llvm::Error rejected(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

} // namespace

llvm::Expected<std::unique_ptr<llvm::Module>>
parseCompleteLlvmModule(llvm::ArrayRef<std::uint8_t> bitcode,
                        llvm::LLVMContext &context) {
  const llvm::MemoryBufferRef buffer(
      llvm::StringRef(reinterpret_cast<const char *>(bitcode.data()),
                      bitcode.size()),
      "loom.relocatable_accelerator_payload");

  llvm::Expected<std::unique_ptr<llvm::Module>> parsed =
      llvm::parseBitcodeFile(buffer, context);
  if (!parsed)
    return rejected("llvm_module_unparsable: " +
                    llvm::toString(parsed.takeError()));

  // parseBitcodeFile already reads every function body. Materializing again
  // states the complete-module requirement explicitly and surfaces any deferred
  // failure before the module is treated as whole.
  if (llvm::Error error = (*parsed)->materializeAll())
    return rejected("llvm_module_materialization_failed: " +
                    llvm::toString(std::move(error)));

  std::string verifierReport;
  llvm::raw_string_ostream verifierStream(verifierReport);
  if (llvm::verifyModule(**parsed, &verifierStream))
    return rejected("llvm_module_invalid: " + verifierReport);
  return std::move(*parsed);
}

llvm::Expected<NormalizedLlvmModule>
normalizeLlvmModule(llvm::ArrayRef<std::uint8_t> sourceBitcode) {
  llvm::LLVMContext context;
  llvm::Expected<std::unique_ptr<llvm::Module>> parsed =
      parseCompleteLlvmModule(sourceBitcode, context);
  if (!parsed)
    return parsed.takeError();
  llvm::Module &module = **parsed;

  const std::string sourceTriple = module.getTargetTriple().str();
  if (sourceTriple.empty())
    return rejected("target_triple_absent: the module declares no target "
                    "triple");

  // Triple::normalize is the pinned canonical printer for target triples: it
  // accepts equivalent spellings and returns exactly one canonical form.
  const std::string canonicalTargetTriple =
      llvm::Triple::normalize(sourceTriple);
  const llvm::Triple triple(canonicalTargetTriple);
  if (triple.getArch() == llvm::Triple::UnknownArch)
    return rejected("target_triple_unsupported: the pinned LLVM provider does "
                    "not recognize target triple '" +
                    canonicalTargetTriple + "'");

  const std::string sourceDataLayout = module.getDataLayoutStr();
  if (sourceDataLayout.empty())
    return rejected("data_layout_absent: the module declares no data layout");

  llvm::Expected<llvm::DataLayout> sourceLayout =
      llvm::DataLayout::parse(sourceDataLayout);
  if (!sourceLayout)
    return rejected("data_layout_invalid: " +
                    llvm::toString(sourceLayout.takeError()));

  // Triple::computeDataLayout is the pinned provider's sole authority for data
  // layout strings: clang derives every module data layout from it, and so do
  // the code generators. DataLayout::getStringRepresentation only echoes the
  // spelling a layout was parsed from and is documented as unusable for
  // comparison, so it cannot canonicalize equivalent spellings. Structural
  // equality through DataLayout::operator== is LLVM's own equivalence test, so
  // an equivalent input spelling is accepted here and the canonical spelling is
  // the one stored. A layout the pinned provider does not print for this triple
  // is rejected rather than echoed back in its source spelling.
  const std::string canonicalDataLayout = triple.computeDataLayout();
  if (canonicalDataLayout.empty())
    return rejected("data_layout_not_canonical: the pinned LLVM provider "
                    "defines no data layout for target triple '" +
                    canonicalTargetTriple + "'");
  llvm::Expected<llvm::DataLayout> canonicalLayout =
      llvm::DataLayout::parse(canonicalDataLayout);
  if (!canonicalLayout)
    return rejected("data_layout_invalid: " +
                    llvm::toString(canonicalLayout.takeError()));
  if (*sourceLayout != *canonicalLayout)
    return rejected("data_layout_not_canonical: the module data layout is not "
                    "equivalent to the pinned canonical layout '" +
                    canonicalDataLayout + "' for target triple '" +
                    canonicalTargetTriple + "'");

  module.setTargetTriple(triple);
  module.setDataLayout(*canonicalLayout);

  llvm::SmallVector<char, 0> written;
  llvm::raw_svector_ostream stream(written);
  llvm::WriteBitcodeToFile(module, stream, /*ShouldPreserveUseListOrder=*/false,
                           /*Index=*/nullptr, /*GenerateHash=*/false);

  NormalizedLlvmModule normalized;
  normalized.canonicalTargetTriple = canonicalTargetTriple;
  normalized.canonicalDataLayout = canonicalDataLayout;
  normalized.bitcode.assign(written.begin(), written.end());
  normalized.bitcodeDigest = llvm::SHA256::hash(normalized.bitcode);
  return normalized;
}

} // namespace loom
