#ifndef FABRIC_TECH_SUBGRAPHENUMERATOR_H
#define FABRIC_TECH_SUBGRAPHENUMERATOR_H

#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace fabric {

struct FuSubgraphCandidate {
  // The wrapping function (newly created in `module`). Its signature is the
  // FU's lifted signature (fabric.bits<N> -> iN, with bits<0> -> none and
  // bits<1> -> i1) and its body holds a single dataflow.subgraph plus a
  // func.return that propagates the subgraph's results.
  ::mlir::func::FuncOp wrapper;
  // The dataflow.subgraph nested inside `wrapper`.
  ::dataflow::SubgraphOp subgraph;
  // Human-readable description of the FU configuration that produced this
  // candidate (e.g. "op#0=arith.subi; demux#0.sel=1; mux#0.sel=1").
  std::string configDescription;
};

// Enumerate every dataflow.subgraph that some software configuration of
// `fu` can implement. Each candidate is created as a fresh
// func.func in `module` (appended at module end), wrapping one
// dataflow.subgraph that mirrors the chosen configuration. The wrapper's
// name is `<baseName>_<idx>` for monotonically increasing idx.
//
// Configurations that route a "dead" value (no token) into a fabric.op
// input or into a yield position are silently dropped.
//
// V1 limitations:
//   * Only supports a curated integer-flavored sw op set
//     (arith.{addi,subi,muli,divsi,divui,remsi,remui,shli,shrsi,shrui,
//     andi,ori,xori,minsi,maxsi,minui,maxui}).
//   * fabric.mux / fabric.demux iterate over their sel values only;
//     the discard / disconnect modes are ignored.
//   * Predicate-bearing or float ops are not yet supported. If `fu`
//     contains any unsupported op_list entry the function returns an
//     empty vector and, when `unsupported` is non-null, writes the
//     offending op symbol to it.
llvm::SmallVector<FuSubgraphCandidate>
enumerateFuSubgraphs(FuOp fu, ::mlir::ModuleOp module,
                     ::llvm::StringRef baseName,
                     ::llvm::StringRef *unsupported = nullptr);

} // namespace fabric

#endif // FABRIC_TECH_SUBGRAPHENUMERATOR_H
