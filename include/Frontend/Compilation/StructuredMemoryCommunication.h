#ifndef LOOM_FRONTEND_COMPILATION_STRUCTUREDMEMORYCOMMUNICATION_H
#define LOOM_FRONTEND_COMPILATION_STRUCTUREDMEMORYCOMMUNICATION_H

#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::frontend {

enum class StructuredMemoryCommunicationDecisionKind : std::uint32_t {
  StageConstantGlobal = 0,
};

struct StructuredMemoryCommunicationDecision final {
  StructuredEntityRef memoryInput;
  StructuredMemoryCommunicationDecisionKind kind;

  friend bool operator==(const StructuredMemoryCommunicationDecision &lhs,
                         const StructuredMemoryCommunicationDecision &rhs) {
    return lhs.memoryInput == rhs.memoryInput && lhs.kind == rhs.kind;
  }
};

struct MaterializedStructuredMemoryCommunicationCandidate final {
  StructuredProgramCandidate structuredProgram;
  std::vector<StructuredOperationSourceProvenance> sourceProvenance;
};

struct StructuredMemoryCommunicationDecisionDomain final {
  std::vector<StructuredMemoryCommunicationDecision> decisions;
  std::uint64_t inspectedMemoryScopes = 0;
};

llvm::ArrayRef<std::uint8_t> structuredMemoryCommunicationDecisionSchemaBytes();
llvm::Expected<std::vector<std::uint8_t>>
encodeStructuredMemoryCommunicationDecision(
    const StructuredMemoryCommunicationDecision &decision);
llvm::Expected<StructuredMemoryCommunicationDecision>
adoptStructuredMemoryCommunicationDecision(
    llvm::ArrayRef<std::uint8_t> canonicalBytes);

llvm::Expected<StructuredMemoryCommunicationDecisionDomain>
enumerateStructuredMemoryCommunicationDecisions(
    const StructuredProgramCandidate &parent,
    std::uint64_t scopeExpansionLimit);

llvm::Expected<MaterializedStructuredMemoryCommunicationCandidate>
materializeStructuredMemoryCommunicationDecision(
    const StructuredProgramCandidate &parent,
    const StructuredMemoryCommunicationDecision &decision);

} // namespace loom::frontend

#endif // LOOM_FRONTEND_COMPILATION_STRUCTUREDMEMORYCOMMUNICATION_H
