// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Fewer than 2 outputs.
func.func @demux_too_few(%sel: i1, %in: i32) -> i32 {
  // expected-error @+1 {{requires at least 2 outputs}}
  %0 = "dataflow.demux"(%sel, %in) : (i1, i32) -> i32
  return %0 : i32
}

// -----
// 2 outputs must use i1 sel.
func.func @demux_2_with_index(%sel: index, %in: i32) -> (i32, i32) {
  // expected-error @+1 {{with 2 outputs, 'sel' must be 'i1'}}
  %0:2 = "dataflow.demux"(%sel, %in) : (index, i32) -> (i32, i32)
  return %0#0, %0#1 : i32, i32
}

// -----
// >2 outputs must use index sel.
func.func @demux_3_with_i1(%sel: i1, %in: i32) -> (i32, i32, i32) {
  // expected-error @+1 {{with more than 2 outputs, 'sel' must be 'index'}}
  %0:3 = "dataflow.demux"(%sel, %in) : (i1, i32) -> (i32, i32, i32)
  return %0#0, %0#1, %0#2 : i32, i32, i32
}

// -----
// Output type must match input type.
func.func @demux_type_mismatch(%sel: i1, %in: i32) -> (i32, f32) {
  // expected-error @+1 {{output #1 type 'f32' must match input type 'i32'}}
  %0:2 = "dataflow.demux"(%sel, %in) : (i1, i32) -> (i32, f32)
  return %0#0, %0#1 : i32, f32
}
