#ifndef LOOM_FRONTEND_COMPILATION_STATICGLOBALMEMORY_H
#define LOOM_FRONTEND_COMPILATION_STATICGLOBALMEMORY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llvm {
class Module;
} // namespace llvm

namespace loom::frontend {

/// The logical permissions of one program global. Placement does not change
/// them: a read-only image remains read-only in local SRAM or external memory.
enum class StaticMemoryPermissions : std::uint32_t {
  ReadOnly = 0,
  ReadWrite = 1,
};

/// Whether the linked LLVM module provides relocation-free bytes that can be
/// copied into a Dataflow-visible local memory image. External declarations,
/// thread-local storage, externally initialized storage, non-default address
/// spaces, and relocation-bearing initializers remain runtime-provided.
enum class StaticGlobalProvision : std::uint32_t {
  Image = 0,
  ExternalRuntime = 1,
};

/// One ephemeral projection of a linked LLVM global. The symbol is only a
/// lookup key inside the current compiler invocation; persistent Deployment
/// records replace it with a LogicalMemoryRootRef and never serialize it.
struct StaticGlobalMemory {
  std::string symbol;
  StaticGlobalProvision provision = StaticGlobalProvision::ExternalRuntime;
  StaticMemoryPermissions permissions = StaticMemoryPermissions::ReadWrite;
  std::uint64_t sizeBytes = 0;
  std::uint64_t alignmentBytes = 0;
  std::vector<std::uint8_t> bytes;
};

/// The exact module DataLayout plus a symbol-sorted, total inventory of
/// addressable LLVM globals. It is a compiler-internal projection, not an
/// Artifact family or a second owner of LLVM semantics.
struct StaticGlobalMemoryCatalog {
  std::string dataLayout;
  std::vector<StaticGlobalMemory> globals;

  const StaticGlobalMemory *lookup(llvm::StringRef symbol) const;
};

/// Projects the linked LLVM module's static storage contract. Defined globals
/// receive an Image only when their initializer has a complete,
/// relocation-free byte representation under the module-owned DataLayout.
/// All other valid globals are retained as ExternalRuntime instead of being
/// silently dropped or assigned guessed bytes.
llvm::Expected<StaticGlobalMemoryCatalog>
projectStaticGlobalMemory(const llvm::Module &module);

} // namespace loom::frontend

#endif // LOOM_FRONTEND_COMPILATION_STATICGLOBALMEMORY_H
