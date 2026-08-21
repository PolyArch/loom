#ifndef LOOM_FRONTEND_ANALYSIS_STRUCTUREDPROTOCOLDEPENDENCIES_H
#define LOOM_FRONTEND_ANALYSIS_STRUCTUREDPROTOCOLDEPENDENCIES_H

#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
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

/// Derives the finite exact-memory dependency graph for protocol roots in
/// caller order. Unknown aliasing, unresolved effects, indirect calls, and
/// cross-block control remain absent rather than being guessed. Downstream DSE
/// may use the projection for traversal and ranking, but Mapping remains the
/// sole owner of placement, channel promotion, and routing legality.
llvm::Expected<std::vector<StructuredProtocolDependency>>
projectStructuredProtocolDependencies(
    const StructuredProgramCandidate &program,
    llvm::ArrayRef<StructuredEntityRef> protocolRoots);

} // namespace loom::frontend::analysis

#endif // LOOM_FRONTEND_ANALYSIS_STRUCTUREDPROTOCOLDEPENDENCIES_H
