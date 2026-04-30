// RUN: loom %s -loom-generalize-subgraphs-to-fu 2>&1 | FileCheck %s

// The module's `func.func @fu_y` carries `loom.synthesized_for = "y"`
// but its body is empty (just a `func.return`). The tightened
// idempotence precheck requires marker-tagged wrappers to contain
// exactly one `fabric.fu` plus a `func.return` terminator (B1). With
// zero `fabric.fu` ops the precheck rejects this as `symbol_conflict`
// rather than silently honoring it as a no-op idempotent re-synth.
//
// The diagnostic message names the failing check (B1) so debugging is
// easy; the input func.func picks up
// `loom.synth_failed = "symbol_conflict"`.

// CHECK: warning: {{.*}}group "y": symbol_conflict
// CHECK-SAME: [B1]
// CHECK-DAG: func.func @fu_y
// CHECK-DAG: loom.synthesized_for = "y"
// CHECK: func.func @pat_addi
// CHECK-SAME: loom.synth_failed = "symbol_conflict"

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
