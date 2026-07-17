// RUN: loom %s -loom-synthesize-configured-functions 2>&1 | FileCheck %s

// The module already contains a top-level `func.func private @fu_x` that
// does NOT carry the `loom.synthesized_for` tag. Private visibility
// makes it unambiguously a non-input symbol (a helper / library
// placeholder) rather than a user pattern; input validation skips it,
// while the synthesizer's wrapper symbol name `@fu_<sanitized("x")>`
// still collides with it. The pass detects this before running
// synthesis, marks the input func.func in the group with
// `loom.synth_failed = "symbol_conflict"`, emits a warning, and never
// attempts synthesis for the group.
//
// Negative checks ensure the placeholder `@fu_x` is NOT bucketed as an
// input candidate (would otherwise pick up `loom.synth_failed =
// "invalid_input"` from the validator since it has zero subgraphs).

// CHECK: warning: {{.*}}group "x": symbol_conflict
// CHECK: func.func private @fu_x() {
// CHECK-NOT: loom.synth_failed
// CHECK: func.func @pat_addi
// CHECK-SAME: loom.synth_failed = "symbol_conflict"

func.func private @fu_x() -> () {
  return
}

func.func @pat_addi(%a: i32, %b: i32) -> i32 attributes {loom.synth_group = "x"} {
  %s = arith.addi %a, %b : i32
  return %s : i32
}
