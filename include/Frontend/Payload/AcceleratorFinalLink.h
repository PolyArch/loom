#ifndef LOOM_FRONTEND_PAYLOAD_ACCELERATORFINALLINK_H
#define LOOM_FRONTEND_PAYLOAD_ACCELERATORFINALLINK_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/LTO/LTO.h"
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
/// The ordinary linker is the sole authority for archive search and for which
/// object files and archive members participate in a link. Loom records that
/// result and consults nothing else. An archive member is named only by its
/// exact child offset in its archive, so collection has no way to reach a
/// member the ordinary linker did not select. Member names are not unique
/// inside an archive and never carry identity here; a diagnostic quotes the
/// name for readability alongside the child offset that identifies the member.
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

/// The ordinary linker's resolution of one symbol of one selected input.
///
/// The input is named by the exact Selection that linker reported, so two
/// archive members sharing a name stay distinct inputs that can be given
/// different facts for the same symbol. The symbol and the resolution are the
/// pinned LTO API's own types. The ordinary linker owns whether a definition
/// prevails, whether it is final in this linkage unit, and whether anything
/// outside the payload cohort can see it. Loom carries those facts from that
/// linker to LTO unchanged: it derives no resolution, keeps no symbol table,
/// and never answers for a symbol itself.
using LinkerSymbolResolver = llvm::function_ref<llvm::lto::SymbolResolution(
    const LinkerSelectedInputs::Selection &selection,
    const llvm::lto::InputFile::Symbol &symbol)>;

/// Collects the payload carried by exactly the selected inputs, checks the raw
/// cohort through the production preflight, and runs the pinned LTO pipeline
/// over them with the ordinary linker's resolutions.
///
/// LTO is the sole whole-program authority here. Symbol resolution, COMDAT and
/// ODR handling, module-flag validation, internalization, and whole-program
/// optimization are all its work, driven by the resolutions the ordinary linker
/// supplied. The pipeline stops at its own last pre-code-generation hook, so
/// what this produces is the unique post-LTO linked module rather than native
/// objects.
///
/// That module is the ordinary Part 1 hand-off to Part 2. It is null exactly
/// when no selected input carries a payload, which is a valid link implying no
/// accelerator compilation rather than a failure. A selected input whose
/// carrier or payload is unreadable, malformed, stale, unsupported, or
/// incompatible is a typed error naming that exact object or archive member; it
/// is never silently discarded.
///
/// The whole operation is failure-atomic. Nothing is returned unless every
/// selected payload was decoded, admitted to LTO, linked, optimized, and
/// verified.
llvm::Expected<std::unique_ptr<llvm::Module>>
linkSelectedAcceleratorPayloads(const LinkerSelectedInputs &selected,
                                LinkerSymbolResolver resolveSymbol,
                                llvm::LLVMContext &context);

} // namespace loom

#endif // LOOM_FRONTEND_PAYLOAD_ACCELERATORFINALLINK_H
