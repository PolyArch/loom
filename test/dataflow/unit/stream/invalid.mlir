// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Unknown step kind.
func.func @stream_bad_step_kind(%init: i32, %limit: i32, %step: i32) -> (i32, i1) {
  // expected-error @+1 {{expected dataflow.stream step kind}}
  %iv, %phase = dataflow.stream %init, %limit, %step step rem while slt : i32
  return %iv, %phase : i32, i1
}

// -----
// Unknown continuation predicate.
func.func @stream_bad_predicate(%init: i32, %limit: i32, %step: i32) -> (i32, i1) {
  // expected-error @+1 {{expected integer comparison predicate}}
  %iv, %phase = dataflow.stream %init, %limit, %step step add while ordered : i32
  return %iv, %phase : i32, i1
}

// -----
// Custom syntax owns step_kind and predicate.
func.func @stream_duplicate_configuration(%init: i32, %limit: i32, %step: i32) -> (i32, i1) {
  // expected-error @+1 {{attribute 'step_kind' occurs more than once}}
  %iv, %phase = dataflow.stream %init, %limit, %step step add while slt {step_kind = 3 : i32} : i32
  return %iv, %phase : i32, i1
}

// -----
// Raw step enum values outside the typed domain are invalid.
func.func @stream_bad_raw_step(%init: i32, %limit: i32, %step: i32) -> (i32, i1) {
  // expected-error @+1 {{attribute 'step_kind' failed to satisfy constraint}}
  %iv, %phase = "dataflow.stream"(%init, %limit, %step) {predicate = 2 : i64, step_kind = 99 : i32} : (i32, i32, i32) -> (i32, i1)
  return %iv, %phase : i32, i1
}

// -----
// Raw predicate enum values outside the typed domain are invalid.
func.func @stream_bad_raw_predicate(%init: i32, %limit: i32, %step: i32) -> (i32, i1) {
  // expected-error @+1 {{attribute 'predicate' failed to satisfy constraint}}
  %iv, %phase = "dataflow.stream"(%init, %limit, %step) {predicate = 99 : i64, step_kind = 0 : i32} : (i32, i32, i32) -> (i32, i1)
  return %iv, %phase : i32, i1
}

// -----
// Stream induction values are scalar integers.
func.func @stream_vector_bounds(%init: vector<2xi32>,
                                %limit: vector<2xi32>,
                                %step: vector<2xi32>)
    -> (vector<2xi32>, i1) {
  // expected-error @+1 {{operand #0 must be signless integer}}
  %iv, %phase = dataflow.stream %init, %limit, %step step add while slt
      : vector<2xi32>
  return %iv, %phase : vector<2xi32>, i1
}

// -----
// Operand types must match each other and the induction result type.
func.func @stream_mismatched_operands(%init: i32, %limit: i64, %step: i32) -> (i32, i1) {
  // expected-error @+1 {{all of {init, limit, step, iv} have same type}}
  %iv, %phase = "dataflow.stream"(%init, %limit, %step) {predicate = 2 : i64, step_kind = 0 : i32} : (i32, i64, i32) -> (i32, i1)
  return %iv, %phase : i32, i1
}
