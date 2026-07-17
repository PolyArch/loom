// RUN: not loom %s -loom-synthesize-configured-functions='fail-as-error=true' 2>&1 | FileCheck %s

// Per spec section "Failure Reasons":
//   `unsupported_op` -- a configured function contains a software op not
//   supported by `fabric.op`; for example `dataflow.load`,
//   `dataflow.store`, or a nested `dataflow.graph`.
//
// The configured-function adapter accepts registered leaf operations and the
// synthesis strategy owns the target capability check.
//
// CHECK: error: {{.*}}synthesis failed: unsupported_op
// CHECK: anchor: unsupported operation dataflow.load

func.func @pat_load(%mem: memref<10xi32>, %addr: index, %ctrl: none) -> i32
    attributes {loom.synth_group = "g_unsup"} {
  %d, %done = "dataflow.load"(%mem, %addr, %ctrl)
      : (memref<10xi32>, index, none) -> (i32, none)
  return %d : i32
}
