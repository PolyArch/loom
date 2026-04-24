// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// data type must match memref element type.
func.func @store_bad_data_type(%mem: memref<10xi32>, %addr: index, %data: i64, %ctrl: none) -> none {
  // expected-error @+1 {{failed to verify that 'data' type matches memref element type}}
  %done = "dataflow.store"(%mem, %addr, %data, %ctrl) : (memref<10xi32>, index, i64, none) -> none
  return %done : none
}

// -----
// addr must be index.
func.func @store_bad_addr(%mem: memref<10xi32>, %addr: i32, %data: i32, %ctrl: none) -> none {
  // expected-error @+1 {{operand #1 must be index}}
  %done = "dataflow.store"(%mem, %addr, %data, %ctrl) : (memref<10xi32>, i32, i32, none) -> none
  return %done : none
}

// -----
// ctrl must be none.
func.func @store_bad_ctrl(%mem: memref<10xi32>, %addr: index, %data: i32, %ctrl: i1) -> none {
  // expected-error @+1 {{operand #3 must be none type}}
  %done = "dataflow.store"(%mem, %addr, %data, %ctrl) : (memref<10xi32>, index, i32, i1) -> none
  return %done : none
}

// -----
// done must be none.
func.func @store_bad_done(%mem: memref<10xi32>, %addr: index, %data: i32, %ctrl: none) -> i1 {
  // expected-error @+1 {{result #0 must be none type}}
  %done = "dataflow.store"(%mem, %addr, %data, %ctrl) : (memref<10xi32>, index, i32, none) -> i1
  return %done : i1
}
