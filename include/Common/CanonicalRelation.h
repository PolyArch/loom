#ifndef LOOM_COMMON_CANONICALRELATION_H
#define LOOM_COMMON_CANONICALRELATION_H

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {

/// One directed, typed relation between two vertices. `label` is an
/// owner-defined canonical binary encoding of the relation kind and all
/// semantic ordinals. It must contain no source-local identity.
struct CanonicalRelationEdge {
  std::uint32_t source = 0;
  std::uint32_t target = 0;
  std::string label;
};

/// Exact canonicalization result for one finite semantic relation graph.
/// `canonicalOrder` maps canonical positions to input vertex ordinals.
struct CanonicalRelationResult {
  CanonicalSemanticBytes bytes;
  std::vector<std::uint32_t> canonicalOrder;
};

/// Computes the lexicographically least serialization over all graph
/// isomorphisms. Vertex intrinsic bytes and edge labels are opaque owner-owned
/// semantic encodings. Input ordinals, container order, and pointers never
/// enter the result.
llvm::Expected<CanonicalRelationResult>
canonicalizeRelationGraph(llvm::ArrayRef<std::string> vertexIntrinsics,
                          llvm::ArrayRef<CanonicalRelationEdge> edges);

} // namespace loom

#endif // LOOM_COMMON_CANONICALRELATION_H
