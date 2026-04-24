// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// ctrl must be 'none' type.
func.func @const_bad_ctrl(%ctrl: i32) -> i32 {
  // expected-error @+1 {{op operand #0 must be none type}}
  %0 = "dataflow.constant"(%ctrl) {const_value = 1 : i32} : (i32) -> i32
  return %0 : i32
}

// -----
// const_value type must match result type.
func.func @const_type_mismatch(%ctrl: none) -> i32 {
  // expected-error @+1 {{'const_value' type 'i64' must match result type 'i32'}}
  %0 = "dataflow.constant"(%ctrl) {const_value = 1 : i64} : (none) -> i32
  return %0 : i32
}

// -----
// const_value must be a typed attribute (unit attr has no type).
func.func @const_untyped(%ctrl: none) -> i32 {
  // expected-error @+1 {{'const_value' must be a typed attribute}}
  %0 = "dataflow.constant"(%ctrl) {const_value = unit} : (none) -> i32
  return %0 : i32
}
