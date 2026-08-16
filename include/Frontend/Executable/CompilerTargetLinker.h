#ifndef LOOM_FRONTEND_EXECUTABLE_COMPILERTARGETLINKER_H
#define LOOM_FRONTEND_EXECUTABLE_COMPILERTARGETLINKER_H

#include "Frontend/Executable/CompilerTargetBinding.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {

struct CompilerTargetLinkWorkspace final {
  std::string temporaryDirectory;
};

/// Links one exact target object into a static ELF executable through the
/// pinned in-process LLD provider. Temporary paths are invocation state and
/// are removed before return; they never enter the executable contract.
llvm::Expected<std::vector<std::uint8_t>> linkCompilerTargetExecutable(
    llvm::ArrayRef<std::uint8_t> objectBytes,
    const CompilerTargetBinding &binding, llvm::StringRef entrySymbol,
    std::uint64_t imageBase, const CompilerTargetLinkWorkspace &workspace);

} // namespace loom

#endif // LOOM_FRONTEND_EXECUTABLE_COMPILERTARGETLINKER_H
