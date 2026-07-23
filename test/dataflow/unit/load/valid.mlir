// RUN: loom %s | loom | FileCheck %s

// CHECK-LABEL: @load_static_i32
func.func @load_static_i32(%mem: memref<10xi32>, %addr: index, %ctrl: none) -> (i32, none) {
  // CHECK: %{{.*}}, %{{.*}} = dataflow.load %{{.*}}[%{{.*}}] %{{.*}} : memref<10xi32>
  %data, %done = dataflow.load %mem[%addr] %ctrl : memref<10xi32>
  return %data, %done : i32, none
}

// CHECK-LABEL: @load_dynamic_f32
func.func @load_dynamic_f32(%mem: memref<?xf32>, %addr: index, %ctrl: none) -> (f32, none) {
  // CHECK: dataflow.load %{{.*}}[%{{.*}}] %{{.*}} : memref<?xf32>
  %data, %done = dataflow.load %mem[%addr] %ctrl : memref<?xf32>
  return %data, %done : f32, none
}

// CHECK-LABEL: @load_2d_i64
func.func @load_2d_i64(%mem: memref<4x4xi64>, %addr: index, %ctrl: none) -> (i64, none) {
  // CHECK: dataflow.load %{{.*}}[%{{.*}}] %{{.*}} : memref<4x4xi64>
  %data, %done = dataflow.load %mem[%addr] %ctrl : memref<4x4xi64>
  return %data, %done : i64, none
}

// CHECK-LABEL: @load_vector_i32
func.func @load_vector_i32(
    %mem: memref<10xi32>, %addr: index, %ctrl: none)
    -> (vector<4xi32>, none) {
  // CHECK: dataflow.load %{{.*}}[%{{.*}}] %{{.*}} : memref<10xi32>, vector<4xi32>
  %data, %done =
      dataflow.load %mem[%addr] %ctrl : memref<10xi32>, vector<4xi32>
  return %data, %done : vector<4xi32>, none
}

// An explicit data type that repeats a vector-valued memory element still
// names one element access, and the printer drops the redundant type.
// CHECK-LABEL: @load_explicit_vector_element_data
func.func @load_explicit_vector_element_data(
    %mem: memref<?xvector<4xi16>>, %addr: index, %ctrl: none)
    -> (vector<4xi16>, none) {
  // CHECK: dataflow.load %{{.*}}[%{{.*}}] %{{.*}} : memref<?xvector<4xi16>>
  %data, %done = dataflow.load %mem[%addr] %ctrl
      : memref<?xvector<4xi16>>, vector<4xi16>
  return %data, %done : vector<4xi16>, none
}

// CHECK-LABEL: @load_masked_vector_i32
func.func @load_masked_vector_i32(
    %mem: memref<10xi32>, %addr: index, %mask: vector<4xi1>, %ctrl: none)
    -> (vector<4xi32>, none) {
  // CHECK: dataflow.load %{{.*}}[%{{.*}}] %{{.*}} mask %{{.*}} : memref<10xi32>, vector<4xi32>
  %data, %done = dataflow.load %mem[%addr] %ctrl mask %mask
      : memref<10xi32>, vector<4xi32>
  return %data, %done : vector<4xi32>, none
}

// CHECK-LABEL: @load_gather_i32
func.func @load_gather_i32(
    %mem: memref<10xi32>, %addr: vector<4xindex>, %mask: vector<4xi1>,
    %ctrl: none) -> (vector<4xi32>, none) {
  // CHECK: dataflow.load %{{.*}}[%{{.*}}] %{{.*}} mask %{{.*}} : memref<10xi32>, vector<4xindex>, vector<4xi32>
  %data, %done = dataflow.load %mem[%addr] %ctrl mask %mask
      : memref<10xi32>, vector<4xindex>, vector<4xi32>
  return %data, %done : vector<4xi32>, none
}

// CHECK-LABEL: @load_multi_rank_contiguous
func.func @load_multi_rank_contiguous(
    %mem: memref<10xi32>, %addr: index, %ctrl: none)
    -> (vector<2x3xi32>, none) {
  // CHECK: dataflow.load %{{.*}}[%{{.*}}] %{{.*}} : memref<10xi32>, vector<2x3xi32>
  %data, %done =
      dataflow.load %mem[%addr] %ctrl : memref<10xi32>, vector<2x3xi32>
  return %data, %done : vector<2x3xi32>, none
}

// CHECK-LABEL: @load_multi_rank_gather
func.func @load_multi_rank_gather(
    %mem: memref<10xi32>, %addr: vector<2x3xindex>, %mask: vector<2x3xi1>,
    %ctrl: none) -> (vector<2x3xi32>, none) {
  // CHECK: dataflow.load %{{.*}}[%{{.*}}] %{{.*}} mask %{{.*}} : memref<10xi32>, vector<2x3xindex>, vector<2x3xi32>
  %data, %done = dataflow.load %mem[%addr] %ctrl mask %mask
      : memref<10xi32>, vector<2x3xindex>, vector<2x3xi32>
  return %data, %done : vector<2x3xi32>, none
}

// CHECK-LABEL: @load_scalar_vector_element
func.func @load_scalar_vector_element(
    %mem: memref<10xvector<2xi32>>, %addr: index, %ctrl: none)
    -> (vector<2xi32>, none) {
  // CHECK: dataflow.load %{{.*}}[%{{.*}}] %{{.*}} : memref<10xvector<2xi32>>
  %data, %done =
      dataflow.load %mem[%addr] %ctrl : memref<10xvector<2xi32>>
  return %data, %done : vector<2xi32>, none
}
