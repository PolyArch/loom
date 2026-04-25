#ifndef FABRIC_TECH_SUBGRAPHMATCHER_H
#define FABRIC_TECH_SUBGRAPHMATCHER_H

#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"

#include <string>

namespace fabric {

struct FuMatchResult {
  // Whether the pattern subgraph is implementable by `fu`.
  bool matched = false;
  // Reference to the matching FU.
  FuOp fu;
  // Human-readable description of the matched configuration.
  std::string configDescription;
  // Per-fabric-op sw_configs to apply to `fu` to realize the matched
  // pattern subgraph. Keys are pointers into `fu`'s body.
  ::llvm::DenseMap<::mlir::Operation *, ::mlir::DictionaryAttr> swConfigsByOp;
};

// Whether two dataflow.subgraph instances describe the same software-level
// computation. Block argument counts and types must agree; the body op
// sequence must match by name, attribute values (excluding `loom.*`
// annotations) and SSA connectivity (operand-position references resolved
// via deterministic value numbering).
bool subgraphsStructurallyEqual(::dataflow::SubgraphOp a,
                                 ::dataflow::SubgraphOp b);

// Try to find a software configuration of `fu` that implements `pattern`.
// `tempModule` is used as scratch space for the FU's enumerated candidates;
// its body is cleared before each FU is queried so the same scratch module
// can be reused across calls. Returns a default-constructed FuMatchResult
// (matched == false) when no configuration matches.
FuMatchResult mapPatternToFu(::dataflow::SubgraphOp pattern, FuOp fu,
                             ::mlir::ModuleOp tempModule);

} // namespace fabric

#endif // FABRIC_TECH_SUBGRAPHMATCHER_H
