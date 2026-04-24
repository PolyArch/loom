// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// data type must match memref element type.
func.func @load_bad_data_type(%mem: memref<10xi32>, %addr: index, %ctrl: none) -> (i64, none) {
  // expected-error @+1 {{failed to verify that 'data' type matches memref element type}}
  %data, %done = "dataflow.load"(%mem, %addr, %ctrl) : (memref<10xi32>, index, none) -> (i64, none)
  return %data, %done : i64, none
}

// -----
// addr must be index.
func.func @load_bad_addr(%mem: memref<10xi32>, %addr: i32, %ctrl: none) -> (i32, none) {
  // expected-error @+1 {{operand #1 must be index}}
  %data, %done = "dataflow.load"(%mem, %addr, %ctrl) : (memref<10xi32>, i32, none) -> (i32, none)
  return %data, %done : i32, none
}

// -----
// ctrl must be none.
func.func @load_bad_ctrl(%mem: memref<10xi32>, %addr: index, %ctrl: i1) -> (i32, none) {
  // expected-error @+1 {{operand #2 must be none type}}
  %data, %done = "dataflow.load"(%mem, %addr, %ctrl) : (memref<10xi32>, index, i1) -> (i32, none)
  return %data, %done : i32, none
}

// -----
// done must be none.
func.func @load_bad_done(%mem: memref<10xi32>, %addr: index, %ctrl: none) -> (i32, i1) {
  // expected-error @+1 {{result #1 must be none type}}
  %data, %done = "dataflow.load"(%mem, %addr, %ctrl) : (memref<10xi32>, index, none) -> (i32, i1)
  return %data, %done : i32, i1
}
