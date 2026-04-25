#ifndef FABRIC_TECH_SUBGRAPHENUMERATOR_H
#define FABRIC_TECH_SUBGRAPHENUMERATOR_H

#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace fabric {

struct FuSubgraphCandidate {
  // The wrapping function created in the parent module. Its signature is
  // the FU's lifted signature (fabric.bits<N> -> iN, with bits<0> -> none
  // and bits<1> -> i1) and its body holds a single dataflow.subgraph plus
  // a func.return that propagates the subgraph's results.
  ::mlir::func::FuncOp wrapper;
  // The dataflow.subgraph nested inside `wrapper`.
  ::dataflow::SubgraphOp subgraph;
  // Human-readable description of the FU configuration that produced this
  // candidate.
  std::string configDescription;
  // Tech-mapping output: for every configurable fabric op (fabric.op /
  // fabric.mux / fabric.demux) inside the originating FU, the
  // sw_configs dictionary that, when written back to that op, realizes
  // this candidate. Keys are the canonical sw_config attribute names,
  // e.g. "op_sel", "predicate", "step_op", "cont_cond", "sel".
  ::llvm::DenseMap<::mlir::Operation *, ::mlir::DictionaryAttr> swConfigsByOp;
};

// Enumerate every dataflow.subgraph that some software configuration of
// `fu` can implement. Each candidate is created as a fresh private
// func.func in `module` (appended at module end), wrapping one
// dataflow.subgraph that mirrors the chosen configuration. The wrapper's
// name is `<baseName>_<idx>` for monotonically increasing idx.
//
// Capability matrix:
//
//   * fabric.op support
//       - integer arith {addi,subi,muli,divsi,divui,remsi,remui,
//         shli,shrsi,shrui,andi,ori,xori,minsi,maxsi,minui,maxui}
//       - integer compare arith.cmpi (predicate iterated from
//         hw_params["predicate"])
//       - float arith {addf,subf,mulf,divf,remf,minimumf,maximumf}
//       - float compare arith.cmpf (predicate iterated)
//       - int/float casts {sitofp,uitofp,fptosi,fptoui}
//       - math unary {sin,cos,tan,sinh,cosh,tanh,exp,exp2,expm1,
//         log,log2,log10,log1p,floor,ceil,round,trunc,roundeven,
//         sqrt,rsqrt,absf,absi,erf}
//       - dataflow.constant (const_hex_value parsed into an
//         IntegerAttr / FloatAttr depending on the result port flavor)
//       - dataflow.stream (step_op, cont_cond iterated)
//       - dataflow.{carry,invariant,gate} (no extra attrs needed)
//   * fabric.mux / fabric.demux iterate sel and the discard / disconnect
//     modes (each emits an explicit per-op sw_configs combination).
//   * Variadic dataflow.{sync,mux,demux} fabric.ops are not yet
//     materialized.
//
// If `fu` references any op outside the supported set the function
// returns an empty vector and, when `unsupported` is non-null, writes the
// offending op symbol to it.
::llvm::SmallVector<FuSubgraphCandidate>
enumerateFuSubgraphs(FuOp fu, ::mlir::ModuleOp module,
                     ::llvm::StringRef baseName,
                     ::llvm::StringRef *unsupported = nullptr);

} // namespace fabric

#endif // FABRIC_TECH_SUBGRAPHENUMERATOR_H
