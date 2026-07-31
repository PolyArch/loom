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

// -----
// Vector element type must match the memory element type.
func.func @store_bad_vector_element(
    %mem: memref<10xi32>, %addr: index, %data: vector<4xi16>, %ctrl: none)
    -> none {
  // expected-error @+1 {{data vector element type i16 must match memory element type i32}}
  %done = "dataflow.store"(%mem, %addr, %data, %ctrl)
      : (memref<10xi32>, index, vector<4xi16>, none) -> none
  return %done : none
}

// -----
// Mask element type must be i1.
func.func @store_bad_mask_element(
    %mem: memref<10xi32>, %addr: index, %data: vector<4xi32>,
    %mask: vector<4xi8>, %ctrl: none) -> none {
  // expected-error @+1 {{mask vector element type must be 'i1'}}
  %done = "dataflow.store"(%mem, %addr, %data, %ctrl, %mask)
      : (memref<10xi32>, index, vector<4xi32>, none, vector<4xi8>) -> none
  return %done : none
}

// -----
// A scatter address vector has the complete data-vector shape.
func.func @store_bad_scatter_address_shape(
    %mem: memref<10xi32>, %addr: vector<2xindex>,
    %data: vector<4xi32>, %ctrl: none) -> none {
  // expected-error @+1 {{address vector shape 'vector<2xindex>' must match data vector shape 'vector<4xi32>'}}
  %done = dataflow.store %mem[%addr] %data %ctrl
      : memref<10xi32>, vector<2xindex>, vector<4xi32>
  return %done : none
}
