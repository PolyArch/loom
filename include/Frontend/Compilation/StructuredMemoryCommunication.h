#ifndef LOOM_FRONTEND_COMPILATION_STRUCTUREDMEMORYCOMMUNICATION_H
#define LOOM_FRONTEND_COMPILATION_STRUCTUREDMEMORYCOMMUNICATION_H

#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace loom::frontend {

enum class StructuredMemoryCommunicationDecisionKind : std::uint32_t {
  StageConstantGlobal = 0,
  PermuteLocalBufferLayout = 1,
  PipelineStagedLoop = 2,
  PromoteOrderedBufferToChannel = 3,
};

struct StageConstantGlobalDecision final {
  StructuredEntityRef anchor;

  friend bool operator==(const StageConstantGlobalDecision &lhs,
                         const StageConstantGlobalDecision &rhs) {
    return lhs.anchor == rhs.anchor;
  }
};

struct PermuteLocalBufferLayoutDecision final {
  StructuredEntityRef anchor;
  std::uint64_t adjacentStoragePosition = 0;

  friend bool operator==(const PermuteLocalBufferLayoutDecision &lhs,
                         const PermuteLocalBufferLayoutDecision &rhs) {
    return lhs.anchor == rhs.anchor &&
           lhs.adjacentStoragePosition == rhs.adjacentStoragePosition;
  }
};

struct PipelineStagedLoopDecision final {
  StructuredEntityRef anchor;

  friend bool operator==(const PipelineStagedLoopDecision &lhs,
                         const PipelineStagedLoopDecision &rhs) {
    return lhs.anchor == rhs.anchor;
  }
};

struct PromoteOrderedBufferToChannelDecision final {
  StructuredEntityRef anchor;

  friend bool operator==(const PromoteOrderedBufferToChannelDecision &lhs,
                         const PromoteOrderedBufferToChannelDecision &rhs) {
    return lhs.anchor == rhs.anchor;
  }
};

using StructuredMemoryCommunicationDecision =
    std::variant<StageConstantGlobalDecision, PermuteLocalBufferLayoutDecision,
                 PipelineStagedLoopDecision,
                 PromoteOrderedBufferToChannelDecision>;

inline bool operator!=(const StructuredMemoryCommunicationDecision &lhs,
                       const StructuredMemoryCommunicationDecision &rhs) {
  return !(lhs == rhs);
}

StructuredMemoryCommunicationDecisionKind
structuredMemoryCommunicationDecisionKind(
    const StructuredMemoryCommunicationDecision &decision);
const StructuredEntityRef &structuredMemoryCommunicationDecisionAnchor(
    const StructuredMemoryCommunicationDecision &decision);

struct MaterializedStructuredMemoryCommunicationCandidate final {
  StructuredProgramCandidate structuredProgram;
  std::optional<StructuredEntityRef> trackedSpatialRegion;
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
    const StructuredMemoryCommunicationDecision &decision,
    std::optional<StructuredEntityRef> trackedSpatialRegion = std::nullopt);

} // namespace loom::frontend

#endif // LOOM_FRONTEND_COMPILATION_STRUCTUREDMEMORYCOMMUNICATION_H
