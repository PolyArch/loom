// RUN: loom %s -loom-generalize-subgraphs-to-fu 2>&1 | FileCheck %s

// The module's `fabric.module @fu_y` carries `loom.synthesized_for = "y"`
// but its body is empty (just the implicit `fabric.yield` terminator).
// The tightened idempotence precheck requires marker-tagged wrappers
// to contain exactly one `fabric.pe` (and the `fabric.module`'s
// `fabric.yield` terminator), with the inner `fabric.pe` body holding
// exactly one `fabric.fu`. With zero `fabric.pe` ops the precheck
// rejects this as `symbol_conflict` rather than silently honoring it
// as a no-op idempotent re-synth.
//
// The diagnostic carries the `[wrapper-body-shape]` tag so debugging
// is easy; the input func.func picks up
// `loom.synth_failed = "symbol_conflict"`.

// CHECK: warning: {{.*}}group "y": symbol_conflict
// CHECK-SAME: [wrapper-body-shape]
// CHECK-DAG: fabric.module @fu_y
// CHECK-DAG: loom.synthesized_for = "y"
// CHECK: func.func @pat_addi
// CHECK-SAME: loom.synth_failed = "symbol_conflict"

fabric.module @fu_y() attributes {loom.synthesized_for = "y"} {
  fabric.yield
}

func.func @pat_addi(%a: i32, %b: i32) -> i32 attributes {loom.synth_group = "y"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
