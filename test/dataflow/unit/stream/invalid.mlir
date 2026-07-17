// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Unknown step_op.
func.func @stream_bad_step_op(%lb: i32, %ub: i32, %step: i32) -> (i32, i1) {
  // expected-error @+1 {{'step_op' must be one of}}
  %idx, %rwc = dataflow.stream %lb, %ub, %step {step_op = "%=", cont_cond = "<"} : i32
  return %idx, %rwc : i32, i1
}

// -----
// Unknown cont_cond.
func.func @stream_bad_cont_cond(%lb: i32, %ub: i32, %step: i32) -> (i32, i1) {
  // expected-error @+1 {{'cont_cond' must be one of}}
  %idx, %rwc = dataflow.stream %lb, %ub, %step {step_op = "+=", cont_cond = "=="} : i32
  return %idx, %rwc : i32, i1
}

// -----
// Operand types must match each other and the index result type.
func.func @stream_mismatched_operands(%lb: i32, %ub: i64, %step: i32) -> (i32, i1) {
  // expected-error @+1 {{all of {init, limit, step, iv} have same type}}
  %idx, %rwc = "dataflow.stream"(%lb, %ub, %step) {step_op = "+=", cont_cond = "<"} : (i32, i64, i32) -> (i32, i1)
  return %idx, %rwc : i32, i1
}
