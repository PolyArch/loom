#ifndef FABRIC_TECH_PARTITIONER_CANDIDATECACHE_H
#define FABRIC_TECH_PARTITIONER_CANDIDATECACHE_H

#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/Tech/TemplateLibrary.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace fabric {

// Per-op set of FU template ids that could plausibly cover a subgraph
// rooted at the op. Computed once per `dataflow.graph` and cached.
struct CandidateSet {
  // The op this candidate set is for. Borrowed; valid as long as the source
  // graph has not been mutated.
  ::mlir::Operation *root = nullptr;
  // Template ids (indices into `TemplateLibrary::templates()`), sorted
  // ascending for deterministic iteration.
  ::llvm::SmallVector<unsigned, 8> templateIds;
};

// Deterministic per-graph candidate cache.
//
// Construction is parallelizable: a worker pool is spawned per the
// `threadCount` argument (0 -> hardware_concurrency()). Each worker writes
// its result into a pre-allocated slot keyed by op program-position so the
// resulting vector is order-stable regardless of thread count. After the
// workers join, each `CandidateSet`'s `templateIds` is sorted ascending.
//
// The library is borrowed; the cache stores raw `Operation *`s into the
// graph's body. It is the caller's responsibility to discard the cache
// before mutating the graph.
class CandidateCache {
public:
  // Build the cache for the given graph. For each non-terminator op in the
  // graph body that is fabric-supported, collect the ids of every template
  // in `lib` whose `rootOpName` matches the op's name.
  //
  // Multi-op coverage (matching against larger template bodies) is left to
  // the partitioner search; the cache only records the singleton-rooted
  // admissibility under the current minimal policy.
  static CandidateCache build(::dataflow::GraphOp graph,
                              const TemplateLibrary &lib,
                              unsigned threadCount = 0);

  // Look up candidates for an op. Returns an empty range if `op` was not
  // recorded (terminator, non-fabric-supported, or not in the graph used
  // to build this cache).
  ::llvm::ArrayRef<unsigned> templatesForOp(::mlir::Operation *op) const;

  // All cached candidate sets in graph-body program order. Stable across
  // runs and across thread counts.
  const ::llvm::SmallVector<CandidateSet> &all() const { return cache; }

private:
  ::llvm::SmallVector<CandidateSet> cache;
  ::llvm::DenseMap<::mlir::Operation *, unsigned> opIndex;
};

} // namespace fabric

#endif // FABRIC_TECH_PARTITIONER_CANDIDATECACHE_H
