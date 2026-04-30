// RUN: loom %s -loom-generalize-subgraphs-to-fu 2>&1 | FileCheck %s

// The module's `func.func @fu_y` carries `loom.synthesized_for = "y"`
// and contains a real fabric.fu (so B1 + B2 are satisfied), but its
// signature `(!fabric.bits<64>) -> !fabric.bits<64>` does not match the
// expected signature derived from the input subgraph
// `(!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>`. The
// tightened idempotence precheck rejects this as `symbol_conflict`
// rather than silently honoring it as an idempotent no-op (the
// placeholder is clearly incompatible with the input pattern).
//
// The diagnostic message names the failing check (B3) and prints both
// the expected and actual signatures.

// CHECK: warning: {{.*}}group "y": symbol_conflict
// CHECK-SAME: signature mismatch
// CHECK-SAME: [B3]
// CHECK: func.func @pat_addi
// CHECK-SAME: loom.synth_failed = "symbol_conflict"

func.func @fu_y(%a: !fabric.bits<64>) -> !fabric.bits<64>
    attributes {loom.synthesized_for = "y"} {
  %r = fabric.fu(%aa = %a : !fabric.bits<64>) -> !fabric.bits<64> {
    %x = fabric.op [@arith.addi] (%aa, %aa) {hw_params = [{}]}
         : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
    fabric.yield %x : !fabric.bits<64>
  }
  return %r : !fabric.bits<64>
}

func.func @pat_addi(%a: i32, %b: i32) -> i32 attributes {loom.synth_group = "y"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
