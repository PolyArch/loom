// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Mismatched input/output count.
func.func @sync_count_mismatch(%a: i32, %b: i32) -> i32 {
  // expected-error @+1 {{number of inputs (2) must equal number of outputs (1)}}
  %0 = "dataflow.sync"(%a, %b) : (i32, i32) -> i32
  return %0 : i32
}

// -----
// Positional type mismatch.
func.func @sync_type_mismatch(%a: i32, %b: f32) -> (i32, i32) {
  // expected-error @+1 {{input #1 type 'f32' must match output #1 type 'i32'}}
  %0:2 = "dataflow.sync"(%a, %b) : (i32, f32) -> (i32, i32)
  return %0#0, %0#1 : i32, i32
}
