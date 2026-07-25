#ifndef LOOM_FRONTEND_PAYLOAD_ACCELERATORFINALLINK_H
#define LOOM_FRONTEND_PAYLOAD_ACCELERATORFINALLINK_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace loom {

/// The ordinary linker's selected-input result.
///
/// The ordinary linker is the sole authority for archive search, symbol
/// resolution, and which object files and archive members participate in a
/// link. Loom records that result and consults nothing else. An archive member
/// is named only by its exact child offset in its archive, so collection has no
/// way to reach a member the ordinary linker did not select; member names are
/// not unique inside an archive and are therefore diagnostics, never identity.
class LinkerSelectedInputs {
public:
  /// One relocatable input the ordinary linker selected.
  struct Selection {
    std::string path;
    /// The selected archive member's exact child offset, or no value when the
    /// selection is a complete object file.
    std::optional<std::uint64_t> memberChildOffset;
  };

  void selectObjectFile(llvm::StringRef objectPath);
  void selectArchiveMember(llvm::StringRef archivePath,
                           std::uint64_t memberChildOffset);

  llvm::ArrayRef<Selection> selections() const { return selections_; }

private:
  std::vector<Selection> selections_;
};

/// Collects the payload carried by exactly the selected inputs, checks the raw
/// cohort through the production preflight, and forms one linked LLVM module.
///
/// The pinned LLVM Linker is the sole module-linking authority: symbol
/// resolution, COMDAT and ODR handling, and module-flag validation across the
/// collected modules are all its decisions, and none of them are reimplemented
/// or inferred from a copied manifest here.
///
/// Internalization and whole-program optimization belong to the same LLVM
/// authority, and LLVM derives both from the ordinary linker's symbol-resolution
/// result: which collected symbol is still referenced from outside this payload
/// cohort. A selected-input list does not carry that result, so neither runs
/// here. Loom performs no substitute for them rather than inventing the
/// resolution they need.
///
/// The result is the ordinary Part 1 hand-off toward S0. It is null exactly
/// when no selected input carries a payload, which is a valid link implying no
/// accelerator compilation rather than a failure. A selected input whose
/// carrier or payload is unreadable, malformed, stale, unsupported, or
/// incompatible is a typed error naming that exact object or archive member; it
/// is never silently discarded.
///
/// The whole operation is failure-atomic. Nothing is returned unless every
/// selected payload was decoded, parsed, fully materialized, verified, and
/// linked, and the linked module itself verified.
llvm::Expected<std::unique_ptr<llvm::Module>>
linkSelectedAcceleratorPayloads(const LinkerSelectedInputs &selected,
                                llvm::LLVMContext &context);

} // namespace loom

#endif // LOOM_FRONTEND_PAYLOAD_ACCELERATORFINALLINK_H
