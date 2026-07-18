// RUN: loom %s -split-input-file -verify-diagnostics

// -----
func.func @parallelize_rejects_rank_two(%data: i8, %phase: i1) {
  // expected-error @+1 {{data vector must be a fixed-size rank-1 vector}}
  %vector, %mask, %group_phase = dataflow.parallelize %data, %phase
    : (i8, i1) -> (vector<2x2xi8>, vector<4xi1>, i1)
  return
}

// -----
func.func @parallelize_rejects_scalable(%data: i8, %phase: i1) {
  // expected-error @+1 {{data vector must be a fixed-size rank-1 vector}}
  %vector, %mask, %group_phase = dataflow.parallelize %data, %phase
    : (i8, i1) -> (vector<[4]xi8>, vector<4xi1>, i1)
  return
}

// -----
func.func @parallelize_rejects_element_mismatch(%data: i8, %phase: i1) {
  // expected-error @+1 {{data vector element type 'i16' must match scalar type 'i8'}}
  %vector, %mask, %group_phase = dataflow.parallelize %data, %phase
    : (i8, i1) -> (vector<3xi16>, vector<3xi1>, i1)
  return
}

// -----
func.func @parallelize_rejects_mask_shape(%data: f32, %phase: i1) {
  // expected-error @+1 {{mask vector shape 'vector<4xi1>' must match data vector shape 'vector<3xf32>'}}
  %vector, %mask, %group_phase = dataflow.parallelize %data, %phase
    : (f32, i1) -> (vector<3xf32>, vector<4xi1>, i1)
  return
}

// -----
func.func @serialize_rejects_mask_element(
    %vector: vector<3xf32>, %mask: vector<3xi8>, %phase: i1) {
  // expected-error @+1 {{mask vector element type must be 'i1'}}
  %data, %scalar_phase = dataflow.serialize %vector, %mask, %phase
    : (vector<3xf32>, vector<3xi8>, i1) -> (f32, i1)
  return
}

// -----
func.func @serialize_rejects_scalar_mismatch(
    %vector: vector<3xf32>, %mask: vector<3xi1>, %phase: i1) {
  // expected-error @+1 {{scalar result type 'f64' must match data vector element type 'f32'}}
  %data, %scalar_phase = dataflow.serialize %vector, %mask, %phase
    : (vector<3xf32>, vector<3xi1>, i1) -> (f64, i1)
  return
}

// -----
func.func @pack_rejects_rank_two(%vector: vector<2x2xi8>) {
  // expected-error @+1 {{data vector must be a fixed-size rank-1 vector}}
  %packed = dataflow.pack %vector : vector<2x2xi8> -> i32
  return
}

// -----
func.func @pack_rejects_index(%vector: vector<3xindex>) {
  // expected-error @+1 {{data vector element type must be a nonzero-width integer or floating-point type}}
  %packed = dataflow.pack %vector : vector<3xindex> -> i192
  return
}

// -----
func.func @pack_rejects_zero_width(%vector: vector<3xi0>) {
  // expected-error @+1 {{data vector element type must be a nonzero-width integer or floating-point type}}
  %packed = dataflow.pack %vector : vector<3xi0> -> i0
  return
}

// -----
func.func @pack_rejects_packed_width(%vector: vector<3xf32>) {
  // expected-error @+1 {{packed integer width 64 must equal vector bit width 96}}
  %packed = dataflow.pack %vector : vector<3xf32> -> i64
  return
}

// -----
func.func @unpack_rejects_packed_width(%packed: i65) {
  // expected-error @+1 {{packed integer width 65 must equal vector bit width 96}}
  %vector = dataflow.unpack %packed : i65 -> vector<3xi32>
  return
}
