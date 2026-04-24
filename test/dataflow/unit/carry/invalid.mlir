// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// cond must be i1.
func.func @carry_bad_cond(%cond: i8, %init: i32, %carry: i32) -> i32 {
  // expected-error @+1 {{op operand #0 must be 1-bit signless integer}}
  %0 = "dataflow.carry"(%cond, %init, %carry) : (i8, i32, i32) -> i32
  return %0 : i32
}

// -----
// init/carry/output types must match each other.
func.func @carry_mismatched_types(%cond: i1, %init: i32, %carry: i64) -> i32 {
  // expected-error @+1 {{all of {init, carry, output} have same type}}
  %0 = "dataflow.carry"(%cond, %init, %carry) : (i1, i32, i64) -> i32
  return %0 : i32
}

// -----
// output type must match init/carry.
func.func @carry_bad_output_type(%cond: i1, %init: i32, %carry: i32) -> i64 {
  // expected-error @+1 {{all of {init, carry, output} have same type}}
  %0 = "dataflow.carry"(%cond, %init, %carry) : (i1, i32, i32) -> i64
  return %0 : i64
}
