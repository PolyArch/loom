#ifndef LOOM_FRONTEND_EXECUTABLE_INSTRUCTIONCOREBINARYINTERNAL_H
#define LOOM_FRONTEND_EXECUTABLE_INSTRUCTIONCOREBINARYINTERNAL_H

#include "Frontend/Executable/InstructionCoreBinary.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {
class CompilerTargetBinding;
}

namespace loom::detail {

struct DecodedInstructionCoreBinaryFields final {
  ArtifactRootReference canonicalDataflow;
  ArtifactRootReference compilerTargetBinding;
  BlobDigest codeBlob;
  std::vector<InstructionLoadSegment> loadSegments;
  std::vector<ThreadEntryBinding> threadEntryTable;
  std::vector<RuntimeImport> runtimeImports;
};

struct ParsedInstructionElf final {
  std::vector<InstructionLoadSegment> loadSegments;
  std::uint64_t entryCount;
  std::vector<std::pair<std::string, std::optional<std::string>>>
      unresolvedImports;
};

class InstructionCoreBinaryBuilder final {
public:
  static InstructionCoreBinary
  create(ArtifactRootReference canonicalDataflow,
         ArtifactRootReference compilerTargetBinding, BlobDigest codeBlob,
         std::vector<InstructionLoadSegment> loadSegments,
         std::vector<ThreadEntryBinding> threadEntryTable,
         std::vector<RuntimeImport> runtimeImports);
};

llvm::Expected<ParsedInstructionElf>
parseInstructionElf(llvm::ArrayRef<std::uint8_t> bytes,
                    const CompilerTargetBinding &target);

llvm::Expected<std::vector<std::pair<std::string, std::optional<std::string>>>>
parseInstructionDynamicExports(llvm::ArrayRef<std::uint8_t> bytes,
                               const CompilerTargetBinding &target);

llvm::Expected<std::vector<ThreadEntryBinding>>
canonicalizeThreadEntries(llvm::ArrayRef<ThreadEntryBinding> entries,
                          const ArtifactIdentity &dataflowArtifact);

llvm::Expected<std::vector<RuntimeImport>>
canonicalizeRuntimeImports(llvm::ArrayRef<RuntimeImport> imports,
                           const CompilerTargetBinding &target);

std::string serializeInstructionCoreBinary(const InstructionCoreBinary &binary);

llvm::Expected<DecodedInstructionCoreBinaryFields>
parseInstructionCoreBinaryFields(llvm::StringRef jsonText);

} // namespace loom::detail

#endif // LOOM_FRONTEND_EXECUTABLE_INSTRUCTIONCOREBINARYINTERNAL_H
