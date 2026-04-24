// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// before_cond must be i1.
func.func @gate_bad_before_cond(%bc: i8, %bv: i32) -> (i1, i32) {
  // expected-error @+1 {{op operand #0 must be 1-bit signless integer}}
  %ac, %av = "dataflow.gate"(%bc, %bv) : (i8, i32) -> (i1, i32)
  return %ac, %av : i1, i32
}

// -----
// after_cond must be i1.
func.func @gate_bad_after_cond(%bc: i1, %bv: i32) -> (i8, i32) {
  // expected-error @+1 {{op result #0 must be 1-bit signless integer}}
  %ac, %av = "dataflow.gate"(%bc, %bv) : (i1, i32) -> (i8, i32)
  return %ac, %av : i8, i32
}

// -----
// before_value and after_value types must match.
func.func @gate_mismatched_value(%bc: i1, %bv: i32) -> (i1, i64) {
  // expected-error @+1 {{all of {before_value, after_value} have same type}}
  %ac, %av = "dataflow.gate"(%bc, %bv) : (i1, i32) -> (i1, i64)
  return %ac, %av : i1, i64
}
