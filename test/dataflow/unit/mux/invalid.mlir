// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Fewer than 2 inputs.
func.func @mux_too_few(%sel: i1, %a: i32) -> i32 {
  // expected-error @+1 {{requires at least 2 inputs}}
  %0 = "dataflow.mux"(%sel, %a) : (i1, i32) -> i32
  return %0 : i32
}

// -----
// 2 inputs must use i1 sel.
func.func @mux_2_with_index(%sel: index, %a: i32, %b: i32) -> i32 {
  // expected-error @+1 {{with 2 inputs, 'sel' must be 'i1'}}
  %0 = "dataflow.mux"(%sel, %a, %b) : (index, i32, i32) -> i32
  return %0 : i32
}

// -----
// >2 inputs must use index sel.
func.func @mux_3_with_i1(%sel: i1, %a: i32, %b: i32, %c: i32) -> i32 {
  // expected-error @+1 {{with more than 2 inputs, 'sel' must be 'index'}}
  %0 = "dataflow.mux"(%sel, %a, %b, %c) : (i1, i32, i32, i32) -> i32
  return %0 : i32
}

// -----
// Input type must match output type.
func.func @mux_type_mismatch(%sel: i1, %a: i32, %b: f32) -> i32 {
  // expected-error @+1 {{input #1 type 'f32' must match output type 'i32'}}
  %0 = "dataflow.mux"(%sel, %a, %b) : (i1, i32, f32) -> i32
  return %0 : i32
}
