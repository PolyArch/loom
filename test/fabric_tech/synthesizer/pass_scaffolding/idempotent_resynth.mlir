// RUN: loom %s -loom-generalize-subgraphs-to-fu 2>&1 | FileCheck %s

// The module already contains a top-level `func.func @fu_y` tagged with
// `loom.synthesized_for = "y"` that is a real synthesized wrapper:
//   * body shape: exactly one inner `fabric.fu` plus a `func.return`
//     terminator
//   * inner fabric.fu passes its own verifier
//   * signature matches the lift of the input subgraph's block-arg
//     types (i32, i32) and yield types (i32) to fabric.bits<32>
// Re-running the pass is a no-op for that group: the precheck detects
// the marker, validates the body shape and signature, and emits a
// `remark: skipping idempotent re-synth`. The input func.func is
// neither annotated with `loom.synth_failed` nor stripped.

// CHECK: remark: {{.*}}group "y": skipping idempotent re-synth
// CHECK-NOT: loom.synth_failed
// CHECK-DAG: func.func @fu_y
// CHECK-DAG: loom.synthesized_for = "y"

func.func @fu_y(%a: !fabric.bits<32>, %b: !fabric.bits<32>) -> !fabric.bits<32>
    attributes {loom.synthesized_for = "y"} {
  %r = fabric.fu(%aa = %a : !fabric.bits<32>, %bb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    %x = fabric.op [@arith.addi] (%aa, %bb) {hw_params = [{}]}
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %x : !fabric.bits<32>
  }
  return %r : !fabric.bits<32>
}

func.func @pat_addi(%a: i32, %b: i32) -> i32 attributes {loom.synth_group = "y"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
