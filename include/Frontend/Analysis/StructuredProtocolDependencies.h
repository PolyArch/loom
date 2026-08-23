#ifndef LOOM_FRONTEND_ANALYSIS_STRUCTUREDPROTOCOLDEPENDENCIES_H
#define LOOM_FRONTEND_ANALYSIS_STRUCTUREDPROTOCOLDEPENDENCIES_H

#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::frontend::analysis {

/// One exact producer-consumer relation between two selected protocol
/// callables. A relation is present only when direct calls in one block pass
/// the same derived SSA memory root from an earlier writer to a later reader.
/// It is an immutable Structured-program projection, not channel legality or
/// candidate identity.
struct StructuredProtocolDependency final {
  StructuredEntityRef producer;
  StructuredEntityRef consumer;
  std::uint64_t sharedMemoryObjectCount = 0;
  std::uint64_t knownSharedMemoryBytes = 0;
  std::uint64_t unknownSharedMemoryObjectCount = 0;

  friend bool operator==(const StructuredProtocolDependency &lhs,
                         const StructuredProtocolDependency &rhs) {
    return lhs.producer == rhs.producer && lhs.consumer == rhs.consumer &&
           lhs.sharedMemoryObjectCount == rhs.sharedMemoryObjectCount &&
           lhs.knownSharedMemoryBytes == rhs.knownSharedMemoryBytes &&
           lhs.unknownSharedMemoryObjectCount ==
               rhs.unknownSharedMemoryObjectCount;
  }
};

enum class StructuredProtocolDependencyKnowledge : std::uint8_t {
  ProvenPresent,
  ProvenAbsent,
  Unknown,
};

/// One ordered pair in the complete protocol-root relation. Presence is
/// backed by the exact direct-call memory projection above. Absence is
/// published only when the analyzed call and memory-effect domain proves that
/// no written object can be read by the consumer. Every other pair remains
/// Unknown; callers must not interpret it as a zero-cost cut.
struct StructuredProtocolDependencyRelation final {
  StructuredEntityRef producer;
  StructuredEntityRef consumer;
  StructuredProtocolDependencyKnowledge knowledge =
      StructuredProtocolDependencyKnowledge::Unknown;
  std::optional<StructuredProtocolDependency> dependency;

  friend bool
  operator==(const StructuredProtocolDependencyRelation &lhs,
             const StructuredProtocolDependencyRelation &rhs) {
    return lhs.producer == rhs.producer && lhs.consumer == rhs.consumer &&
           lhs.knowledge == rhs.knowledge &&
           lhs.dependency == rhs.dependency;
  }
};

/// Candidate-independent, immutable protocol relation projected in caller
/// root order. The relation contains exactly one entry for every distinct
/// ordered pair. It is neither channel legality nor a Mapping feasibility
/// result.
struct StructuredProtocolDependencyProjection final {
  std::vector<StructuredProtocolDependencyRelation> relations;

  std::vector<StructuredProtocolDependency> presentDependencies() const;

  friend bool
  operator==(const StructuredProtocolDependencyProjection &lhs,
             const StructuredProtocolDependencyProjection &rhs) {
    return lhs.relations == rhs.relations;
  }
};

llvm::Expected<StructuredProtocolDependencyProjection>
projectStructuredProtocolDependencyProjection(
    const StructuredProgramCandidate &program,
    llvm::ArrayRef<StructuredEntityRef> protocolRoots);

/// Compatibility projection containing only exact present relations. Unknown
/// aliasing, unresolved effects, indirect calls, and cross-block control are
/// omitted, so callers that must distinguish absence from unknown must consume
/// `projectStructuredProtocolDependencyProjection` instead. Mapping remains
/// the sole owner of placement, channel promotion, and routing legality.
llvm::Expected<std::vector<StructuredProtocolDependency>>
projectStructuredProtocolDependencies(
    const StructuredProgramCandidate &program,
    llvm::ArrayRef<StructuredEntityRef> protocolRoots);

} // namespace loom::frontend::analysis

#endif // LOOM_FRONTEND_ANALYSIS_STRUCTUREDPROTOCOLDEPENDENCIES_H
