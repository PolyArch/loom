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

// -----
// Vector data must have a positive fixed rank.
func.func @load_rank_zero_vector(
    %mem: memref<10xi32>, %addr: index, %ctrl: none)
    -> (vector<i32>, none) {
  // expected-error @+1 {{data vector must be a fixed-size vector}}
  %data, %done = "dataflow.load"(%mem, %addr, %ctrl)
      : (memref<10xi32>, index, none) -> (vector<i32>, none)
  return %data, %done : vector<i32>, none
}

// -----
// A vector-valued memory element stays one element access, but it is still a
// semantic vector, so a rank-zero element type is rejected.
func.func @load_rank_zero_element(
    %mem: memref<10xvector<i32>>, %addr: index, %ctrl: none)
    -> (vector<i32>, none) {
  // expected-error @+1 {{data vector must be a fixed-size vector}}
  %data, %done = dataflow.load %mem[%addr] %ctrl : memref<10xvector<i32>>
  return %data, %done : vector<i32>, none
}

// -----
// A scalable element type is likewise outside the semantic vector contract.
func.func @load_scalable_element(
    %mem: memref<10xvector<[4]xi32>>, %addr: index, %ctrl: none)
    -> (vector<[4]xi32>, none) {
  // expected-error @+1 {{data vector must be a fixed-size vector}}
  %data, %done = dataflow.load %mem[%addr] %ctrl : memref<10xvector<[4]xi32>>
  return %data, %done : vector<[4]xi32>, none
}

// -----
// A mask is illegal on a scalar access.
func.func @load_scalar_mask(
    %mem: memref<10xi32>, %addr: index, %mask: vector<4xi1>, %ctrl: none)
    -> (i32, none) {
  // expected-error @+1 {{mask is only valid for a vector memory access}}
  %data, %done = "dataflow.load"(%mem, %addr, %ctrl, %mask)
      : (memref<10xi32>, index, none, vector<4xi1>) -> (i32, none)
  return %data, %done : i32, none
}

// -----
// Mask shape must match vector data shape.
func.func @load_bad_mask_shape(
    %mem: memref<10xi32>, %addr: index, %mask: vector<2xi1>, %ctrl: none)
    -> (vector<4xi32>, none) {
  // expected-error @+1 {{mask vector shape 'vector<2xi1>' must match data vector shape 'vector<4xi32>'}}
  %data, %done = "dataflow.load"(%mem, %addr, %ctrl, %mask)
      : (memref<10xi32>, index, none, vector<2xi1>) -> (vector<4xi32>, none)
  return %data, %done : vector<4xi32>, none
}

// -----
// Gather address lanes must match result lanes.
func.func @load_bad_gather_address_shape(
    %mem: memref<10xi32>, %addr: vector<2xindex>, %ctrl: none)
    -> (vector<4xi32>, none) {
  // expected-error @+1 {{address vector shape 'vector<2xindex>' must match data vector shape 'vector<4xi32>'}}
  %data, %done = "dataflow.load"(%mem, %addr, %ctrl)
      : (memref<10xi32>, vector<2xindex>, none) -> (vector<4xi32>, none)
  return %data, %done : vector<4xi32>, none
}

// -----
// Gather addresses must contain index elements.
func.func @load_bad_gather_address_element(
    %mem: memref<10xi32>, %addr: vector<4xi32>, %ctrl: none)
    -> (vector<4xi32>, none) {
  // expected-error @+1 {{address vector element type must be 'index'}}
  %data, %done = "dataflow.load"(%mem, %addr, %ctrl)
      : (memref<10xi32>, vector<4xi32>, none) -> (vector<4xi32>, none)
  return %data, %done : vector<4xi32>, none
}

// -----
// Gather mask lanes must match result lanes.
func.func @load_bad_gather_mask_shape(
    %mem: memref<10xi32>, %addr: vector<4xindex>,
    %mask: vector<2xi1>, %ctrl: none) -> (vector<4xi32>, none) {
  // expected-error @+1 {{mask vector shape 'vector<2xi1>' must match data vector shape 'vector<4xi32>'}}
  %data, %done = "dataflow.load"(%mem, %addr, %ctrl, %mask)
      : (memref<10xi32>, vector<4xindex>, none, vector<2xi1>)
          -> (vector<4xi32>, none)
  return %data, %done : vector<4xi32>, none
}
