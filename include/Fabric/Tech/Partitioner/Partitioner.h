#ifndef FABRIC_TECH_PARTITIONER_PARTITIONER_H
#define FABRIC_TECH_PARTITIONER_PARTITIONER_H

#include "Common/Config.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/Tech/TemplateLibrary.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <memory>

namespace fabric {

// One block in the partitioner's output. A block groups together a set of
// non-terminator ops from the dataflow.graph body that will be materialized
// as a single dataflow.subgraph (when `tpl != nullptr`) or that remain in
// place at the graph level (when `tpl == nullptr`).
struct Block {
  // Stable per-PartitionResult id, assigned in creation order.
  unsigned id = 0;
  // Pointers to ops in the original dataflow.graph body. Pointers are not
  // owning; they remain valid until the partitioner's caller mutates the IR.
  ::llvm::SmallVector<::mlir::Operation *> ops;
  // Bound FU template, if any. nullptr means "this block stays at graph
  // level" (no FU implements it under the current library).
  const FuTemplate *tpl = nullptr;
};

// Result of running a partitioner on a single dataflow.graph.
struct PartitionResult {
  ::llvm::SmallVector<Block> blocks;
};

// Partitioner interface. Implementations are stateless across runs; any
// scratch state must be local to `run`.
class IPartitioner {
public:
  virtual ~IPartitioner() = default;

  virtual PartitionResult run(::dataflow::GraphOp graph,
                              const TemplateLibrary &lib,
                              const ::loom::TechMapConfig &cfg) = 0;
};

// Factory: produce a partitioner for the named algorithm. `algorithm` must
// be one of "greedy", "list", "beam", "sa". The current implementations all
// fall back to a singleton partition (one block per op); algorithm-specific
// behavior is intentionally deferred.
std::unique_ptr<IPartitioner> createPartitioner(::llvm::StringRef algorithm);

// Shared baseline: produce one Block per non-terminator op in the graph
// body. Each op that names a template root in `lib` is bound to the first
// matching template; everything else gets `tpl == nullptr` and stays at the
// graph level when materialized.
PartitionResult buildSingletonPartition(::dataflow::GraphOp graph,
                                        const TemplateLibrary &lib);

} // namespace fabric

#endif // FABRIC_TECH_PARTITIONER_PARTITIONER_H
