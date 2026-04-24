// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// cond must be i1.
func.func @invariant_bad_cond(%cond: i8, %init: i32) -> i32 {
  // expected-error @+1 {{op operand #0 must be 1-bit signless integer}}
  %0 = "dataflow.invariant"(%cond, %init) : (i8, i32) -> i32
  return %0 : i32
}

// -----
// output type must match init.
func.func @invariant_bad_output_type(%cond: i1, %init: i32) -> i64 {
  // expected-error @+1 {{all of {init, output} have same type}}
  %0 = "dataflow.invariant"(%cond, %init) : (i1, i32) -> i64
  return %0 : i64
}
