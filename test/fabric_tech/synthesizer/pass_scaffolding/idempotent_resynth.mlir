// RUN: loom %s -loom-generalize-subgraphs-to-fu 2>&1 | FileCheck %s

// The module already contains a top-level `func.func @fu_y` tagged with
// `loom.synthesized_for = "y"` -- evidence that a previous run of this
// pass produced that function for group `y`. Re-running the pass is a
// no-op for that group: the precheck detects the marker, emits a
// `remark: skipping idempotent re-synth`, and the input func.func is
// neither annotated with `loom.synth_failed` nor stripped.

// CHECK: remark: {{.*}}group "y": skipping idempotent re-synth
// CHECK-NOT: loom.synth_failed
// CHECK-DAG: func.func @fu_y
// CHECK-DAG: loom.synthesized_for = "y"

func.func @fu_y() attributes {loom.synthesized_for = "y"} {
  return
}

func.func @pat_addi(%a: i32, %b: i32) -> i32 attributes {loom.synth_group = "y"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
