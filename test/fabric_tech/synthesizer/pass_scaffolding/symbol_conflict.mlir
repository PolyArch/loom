// RUN: loom %s -loom-generalize-subgraphs-to-fu 2>&1 | FileCheck %s

// The module already contains a top-level `func.func @fu_x` that does
// NOT carry the `loom.synthesized_for` tag. The synthesizer's wrapper
// symbol name `@fu_<sanitized("x")>` collides with that pre-existing
// function. The pass detects this before running synthesis, marks every
// input func.func in the group with `loom.synth_failed = "symbol_conflict"`,
// emits a warning, and never attempts synthesis for the group.

// CHECK: warning: {{.*}}group "x": symbol_conflict
// CHECK-DAG: loom.synth_failed = "symbol_conflict"
// CHECK-DAG: func.func @fu_x

func.func @fu_x() {
  return
}

func.func @pat_addi(%a: i32, %b: i32) -> i32 attributes {loom.synth_group = "x"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
