#ifndef LOOM_FRONTEND_PAYLOAD_ACCELERATORFINALLINK_H
#define LOOM_FRONTEND_PAYLOAD_ACCELERATORFINALLINK_H

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <memory>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace loom {

/// Imports the exact linked LLVM module and selected-input report produced by
/// one pinned LLD invocation.
///
/// LLD remains the sole owner of archive extraction, symbol resolution,
/// COMDAT/ODR handling, LTO configuration, and whole-program optimization.
/// Loom uses the resolution report only to locate the exact payload carriers
/// LLD selected, validates that cohort, and verifies that the linked module
/// carries exactly the same payload projections before removing those
/// non-semantic carrier globals. The returned module is therefore the unique
/// Part 1 hand-off rather than a second independently linked approximation.
///
/// The resolution and linked bitcode are ephemeral outputs of the same LLD
/// attempt. Malformed reports, unreadable selections, incompatible payloads,
/// stale coupling, or malformed linked bitcode fail closed.
llvm::Expected<std::unique_ptr<llvm::Module>>
importLldAcceleratorFinalLink(llvm::MemoryBufferRef resolution,
                              llvm::MemoryBufferRef linkedBitcode,
                              llvm::LLVMContext &context);

} // namespace loom

#endif // LOOM_FRONTEND_PAYLOAD_ACCELERATORFINALLINK_H
