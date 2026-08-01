#ifndef LOOM_FRONTEND_EXECUTABLE_INSTRUCTIONCOREBINARY_H
#define LOOM_FRONTEND_EXECUTABLE_INSTRUCTIONCOREBINARY_H

#include "Common/Artifact.h"
#include "Common/BlobDigest.h"
#include "Dataflow/IR/DataflowCanonicalEntity.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom {

class ArtifactStore;
class BlobStore;
namespace detail {
class InstructionCoreBinaryBuilder;
}

inline constexpr ArtifactSchemaDescriptor instructionCoreBinarySchema{
    "loom.instruction_core_binary", SchemaVersion{1, 0}};

struct InstructionLoadSegment final {
  std::uint64_t ordinal;
  std::uint64_t virtualAddress;
  std::uint64_t fileOffset;
  std::uint64_t fileSize;
  std::uint64_t memorySize;
  std::uint64_t alignment;
  bool readable;
  bool writable;
  bool executable;

  friend bool operator==(const InstructionLoadSegment &lhs,
                         const InstructionLoadSegment &rhs) {
    return lhs.ordinal == rhs.ordinal &&
           lhs.virtualAddress == rhs.virtualAddress &&
           lhs.fileOffset == rhs.fileOffset && lhs.fileSize == rhs.fileSize &&
           lhs.memorySize == rhs.memorySize && lhs.alignment == rhs.alignment &&
           lhs.readable == rhs.readable && lhs.writable == rhs.writable &&
           lhs.executable == rhs.executable;
  }
  friend bool operator!=(const InstructionLoadSegment &lhs,
                         const InstructionLoadSegment &rhs) {
    return !(lhs == rhs);
  }
};

struct ThreadEntryBinding final {
  dataflow::RootThreadLaunchRef rootThreadLaunch;
  std::uint64_t entryOrdinal;

  friend bool operator==(const ThreadEntryBinding &lhs,
                         const ThreadEntryBinding &rhs) {
    return lhs.rootThreadLaunch == rhs.rootThreadLaunch &&
           lhs.entryOrdinal == rhs.entryOrdinal;
  }
};

struct RuntimeImport final {
  std::uint64_t supportComponentOrdinal;
  std::string abiSymbol;
  std::optional<std::string> abiSymbolVersion;

  friend bool operator==(const RuntimeImport &lhs, const RuntimeImport &rhs) {
    return lhs.supportComponentOrdinal == rhs.supportComponentOrdinal &&
           lhs.abiSymbol == rhs.abiSymbol &&
           lhs.abiSymbolVersion == rhs.abiSymbolVersion;
  }
};

struct InstructionCoreBinaryDraft final {
  ArtifactRootReference canonicalDataflow;
  ArtifactRootReference compilerTargetBinding;
  std::vector<std::uint8_t> executableBytes;
  std::vector<ThreadEntryBinding> threadEntryTable;
  std::vector<RuntimeImport> runtimeImports;
};

class InstructionCoreBinary final {
public:
  const ArtifactRootReference &canonicalDataflow() const {
    return canonicalDataflow_;
  }
  const ArtifactRootReference &compilerTargetBinding() const {
    return compilerTargetBinding_;
  }
  const BlobDigest &codeBlob() const { return codeBlob_; }
  llvm::ArrayRef<InstructionLoadSegment> loadSegments() const {
    return loadSegments_;
  }
  llvm::ArrayRef<ThreadEntryBinding> threadEntryTable() const {
    return threadEntryTable_;
  }
  llvm::ArrayRef<RuntimeImport> runtimeImports() const {
    return runtimeImports_;
  }

  llvm::Expected<std::uint64_t>
  threadEntry(dataflow::RootThreadLaunchRef root) const;

private:
  InstructionCoreBinary(ArtifactRootReference canonicalDataflow,
                        ArtifactRootReference compilerTargetBinding,
                        BlobDigest codeBlob,
                        std::vector<InstructionLoadSegment> loadSegments,
                        std::vector<ThreadEntryBinding> threadEntryTable,
                        std::vector<RuntimeImport> runtimeImports)
      : canonicalDataflow_(std::move(canonicalDataflow)),
        compilerTargetBinding_(std::move(compilerTargetBinding)),
        codeBlob_(codeBlob), loadSegments_(std::move(loadSegments)),
        threadEntryTable_(std::move(threadEntryTable)),
        runtimeImports_(std::move(runtimeImports)) {}

  ArtifactRootReference canonicalDataflow_;
  ArtifactRootReference compilerTargetBinding_;
  BlobDigest codeBlob_;
  std::vector<InstructionLoadSegment> loadSegments_;
  std::vector<ThreadEntryBinding> threadEntryTable_;
  std::vector<RuntimeImport> runtimeImports_;

  friend llvm::Expected<InstructionCoreBinary>
  decodeInstructionCoreBinary(llvm::StringRef, const ArtifactStore &,
                              const BlobStore &);
  friend llvm::Expected<class FinalizedInstructionCoreBinary>
  finalizeInstructionCoreBinary(InstructionCoreBinaryDraft,
                                const ArtifactStore &, const BlobStore &);
  friend class detail::InstructionCoreBinaryBuilder;
};

class FinalizedInstructionCoreBinary final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }
  const InstructionCoreBinary &binary() const { return binary_; }

private:
  FinalizedInstructionCoreBinary(ArtifactRootReference reference,
                                 CanonicalSemanticBytes canonicalBytes,
                                 InstructionCoreBinary binary)
      : reference_(std::move(reference)),
        canonicalBytes_(std::move(canonicalBytes)), binary_(std::move(binary)) {
  }

  ArtifactRootReference reference_;
  CanonicalSemanticBytes canonicalBytes_;
  InstructionCoreBinary binary_;

  friend llvm::Expected<FinalizedInstructionCoreBinary>
  finalizeInstructionCoreBinary(InstructionCoreBinaryDraft,
                                const ArtifactStore &, const BlobStore &);
  friend llvm::Expected<FinalizedInstructionCoreBinary>
  importInstructionCoreBinary(const ArtifactRootReference &,
                              const ArtifactStore &, const BlobStore &);
};

llvm::Expected<FinalizedInstructionCoreBinary>
finalizeInstructionCoreBinary(InstructionCoreBinaryDraft draft,
                              const ArtifactStore &artifacts,
                              const BlobStore &blobs);

llvm::Expected<FinalizedInstructionCoreBinary>
importInstructionCoreBinary(const ArtifactRootReference &reference,
                            const ArtifactStore &artifacts,
                            const BlobStore &blobs);

} // namespace loom

#endif // LOOM_FRONTEND_EXECUTABLE_INSTRUCTIONCOREBINARY_H
