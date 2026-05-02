// RUN: not loom %s -loom-generalize-subgraphs-to-fu 2>&1 | FileCheck %s

// Per spec section "Failure reasons (closed enumeration)":
//   `unsupported_op` -- an input subgraph contains a software op not
//   supported by `fabric.op` (per `opSchemas()` in
//   `lib/Fabric/IR/FabricOps.cpp`); for example `dataflow.load`,
//   `dataflow.store`, `dataflow.graph`, `arith.constant`, `ub.poison`.
//
// Reachability note: the dataflow.subgraph dialect verifier shares the
// same allowlist (`isAllowedInDataflowSubgraph` -> `isFabricOpSupported`)
// and rejects any forbidden op at MLIR parse / verify time, before the
// synthesizer pass ever sees the IR. Demonstrate this by feeding a
// `dataflow.load` (the spec's canonical "unsupported in fabric.op"
// example) inside a `dataflow.subgraph`: parsing fails with the
// dataflow-dialect's own diagnostic and `loom` exits non-zero. The
// synthesizer's `unsupported_op` enum value remains as a closed-set
// guard for the day a future strategy produces lifted IR that contains
// an op outside the allowlist; it is currently dead-code-defensive.
//
// CHECK: error: {{.*}}'dataflow.load' op is not allowed inside dataflow.subgraph

func.func @pat_load(%mem: memref<10xi32>, %addr: index, %ctrl: none) -> i32
    attributes {loom.synth_group = "g_unsup"} {
  %r = dataflow.subgraph(%a = %addr : index, %c = %ctrl : none) -> i32 {
    %d, %done = "dataflow.load"(%mem, %a, %c)
        : (memref<10xi32>, index, none) -> (i32, none)
    dataflow.yield %d : i32
  }
  return %r : i32
}
