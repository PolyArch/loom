#ifndef LOOM_FRONTEND_EXECUTABLE_EXECUTABLEELF_H
#define LOOM_FRONTEND_EXECUTABLE_EXECUTABLEELF_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llvm::object {
class ELFObjectFileBase;
}

namespace loom {

class CompilerTargetBinding;

struct ExecutableLoadSegment final {
  std::uint64_t ordinal = 0;
  std::uint64_t virtualAddress = 0;
  std::uint64_t fileOffset = 0;
  std::uint64_t fileSize = 0;
  std::uint64_t memorySize = 0;
  std::uint64_t alignment = 0;
  bool readable = false;
  bool writable = false;
  bool executable = false;

  friend bool operator==(const ExecutableLoadSegment &lhs,
                         const ExecutableLoadSegment &rhs) {
    return lhs.ordinal == rhs.ordinal &&
           lhs.virtualAddress == rhs.virtualAddress &&
           lhs.fileOffset == rhs.fileOffset && lhs.fileSize == rhs.fileSize &&
           lhs.memorySize == rhs.memorySize && lhs.alignment == rhs.alignment &&
           lhs.readable == rhs.readable && lhs.writable == rhs.writable &&
           lhs.executable == rhs.executable;
  }
  friend bool operator!=(const ExecutableLoadSegment &lhs,
                         const ExecutableLoadSegment &rhs) {
    return !(lhs == rhs);
  }
};

struct ExecutableLoadRange final {
  std::uint64_t begin = 0;
  std::uint64_t end = 0;
};

llvm::Error validateElfTarget(const llvm::object::ELFObjectFileBase &object,
                              const CompilerTargetBinding &target);

llvm::Expected<std::vector<ExecutableLoadSegment>>
projectExecutableLoadSegments(const llvm::object::ELFObjectFileBase &object,
                              std::size_t blobSize);

llvm::Expected<std::vector<ExecutableLoadSegment>>
projectCompilerTargetExecutableLoadSegments(
    llvm::ArrayRef<std::uint8_t> bytes, const CompilerTargetBinding &target);

llvm::Expected<ExecutableLoadRange>
projectCompilerTargetExecutableLoadRange(llvm::ArrayRef<std::uint8_t> bytes,
                                         const CompilerTargetBinding &target);

} // namespace loom

#endif // LOOM_FRONTEND_EXECUTABLE_EXECUTABLEELF_H
