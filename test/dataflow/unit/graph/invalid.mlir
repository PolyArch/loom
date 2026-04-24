// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Block argument type mismatch (the declared block arg type must match the
// outer operand's type; the parser catches this via SSA value type unification).
func.func @graph_bad_bbarg_type(%x: i32) {
  // expected-note @-1 {{prior use here}}
  // expected-error @+1 {{use of value '%x' expects different type than prior uses: 'f32' vs 'i32'}}
  dataflow.graph(%a = %x : f32) -> () {
    dataflow.yield
  }
  return
}

// -----
// Yield value count mismatch.
func.func @graph_bad_yield_count(%x: i32) -> (i32, i32) {
  %r:2 = dataflow.graph(%a = %x : i32) -> (i32, i32) {
    // expected-error @+1 {{yield value count (1) must match parent graph result count (2)}}
    dataflow.yield %a : i32
  }
  return %r#0, %r#1 : i32, i32
}

// -----
// Yield value type mismatch.
func.func @graph_bad_yield_type(%x: i32) -> f32 {
  %r = dataflow.graph(%a = %x : i32) -> f32 {
    // expected-error @+1 {{yield value #0 type 'i32' must match parent graph result type 'f32'}}
    dataflow.yield %a : i32
  }
  return %r : f32
}

// -----
// yield outside of graph.
func.func @yield_outside() {
  // expected-error @+1 {{expects parent op 'dataflow.graph'}}
  dataflow.yield
}
