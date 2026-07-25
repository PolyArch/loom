#ifndef LOOM_FRONTEND_PAYLOAD_LLVMMODULENORMALIZATION_H
#define LOOM_FRONTEND_PAYLOAD_LLVMMODULENORMALIZATION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace loom {

/// The result of the one deterministic LLVM parser and bitcode writer contract
/// that payload version 1.0 owns.
///
/// The bitcode bytes are the module authority. The canonical target facts are
/// mechanical projections of that same module, and the digest is a mechanical
/// projection of exactly those bytes.
struct NormalizedLlvmModule {
  std::string canonicalTargetTriple;
  std::string canonicalDataLayout;
  std::vector<std::uint8_t> bitcode;
  std::array<std::uint8_t, 32> bitcodeDigest = {};
};

/// Parses bitcode with the pinned LLVM provider into `context`, fully
/// materializes it, and runs the LLVM verifier.
///
/// This is the sole parse contract payload version 1.0 owns. Normalization
/// applies it when producing canonical bytes, and the final link applies it
/// again in the linking context, so a module is never treated as whole before
/// the pinned provider has read all of it and accepted it.
llvm::Expected<std::unique_ptr<llvm::Module>>
parseCompleteLlvmModule(llvm::ArrayRef<std::uint8_t> bitcode,
                        llvm::LLVMContext &context);

/// Parses the source bitcode with the pinned LLVM provider, fully materializes
/// it, runs the LLVM verifier, canonicalizes the target triple and data layout
/// through the pinned LLVM parsers and printers, and writes the complete module
/// back through the fixed writer contract.
///
/// LLVM's non-semantic use-list order is the only detail dropped. Nothing is
/// sorted, renamed, stripped, optimized, or summarized, and no wrapper or
/// LLVM-generated module hash is added. Normalizing already normalized bytes
/// reproduces them exactly, which is what makes stored payload bytes checkable.
///
/// Every failure is a typed error: the payload is never repaired or guessed.
llvm::Expected<NormalizedLlvmModule>
normalizeLlvmModule(llvm::ArrayRef<std::uint8_t> sourceBitcode);

} // namespace loom

#endif // LOOM_FRONTEND_PAYLOAD_LLVMMODULENORMALIZATION_H
