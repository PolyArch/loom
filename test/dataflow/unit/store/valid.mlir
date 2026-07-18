// RUN: loom %s | loom | FileCheck %s

// CHECK-LABEL: @store_static_i32
func.func @store_static_i32(%mem: memref<10xi32>, %addr: index, %data: i32, %ctrl: none) -> none {
  // CHECK: %{{.*}} = dataflow.store %{{.*}}[%{{.*}}] %{{.*}} %{{.*}} : memref<10xi32>
  %done = dataflow.store %mem[%addr] %data %ctrl : memref<10xi32>
  return %done : none
}

// CHECK-LABEL: @store_dynamic_f32
func.func @store_dynamic_f32(%mem: memref<?xf32>, %addr: index, %data: f32, %ctrl: none) -> none {
  // CHECK: dataflow.store %{{.*}}[%{{.*}}] %{{.*}} %{{.*}} : memref<?xf32>
  %done = dataflow.store %mem[%addr] %data %ctrl : memref<?xf32>
  return %done : none
}

// CHECK-LABEL: @store_2d_i64
func.func @store_2d_i64(%mem: memref<4x4xi64>, %addr: index, %data: i64, %ctrl: none) -> none {
  // CHECK: dataflow.store %{{.*}}[%{{.*}}] %{{.*}} %{{.*}} : memref<4x4xi64>
  %done = dataflow.store %mem[%addr] %data %ctrl : memref<4x4xi64>
  return %done : none
}

// CHECK-LABEL: @store_vector_i32
func.func @store_vector_i32(
    %mem: memref<10xi32>, %addr: index, %data: vector<4xi32>, %ctrl: none)
    -> none {
  // CHECK: dataflow.store %{{.*}}[%{{.*}}] %{{.*}} %{{.*}} : memref<10xi32>, vector<4xi32>
  %done =
      dataflow.store %mem[%addr] %data %ctrl : memref<10xi32>, vector<4xi32>
  return %done : none
}

// CHECK-LABEL: @store_explicit_vector_index_data
func.func @store_explicit_vector_index_data(
    %mem: memref<?xvector<4xindex>>, %addr: index,
    %data: vector<4xindex>, %ctrl: none) -> none {
  // CHECK: dataflow.store %{{.*}}[%{{.*}}] %{{.*}} %{{.*}} : memref<?xvector<4xindex>>
  %done = dataflow.store %mem[%addr] %data %ctrl
      : memref<?xvector<4xindex>>, vector<4xindex>
  return %done : none
}

// CHECK-LABEL: @store_masked_vector_i32
func.func @store_masked_vector_i32(
    %mem: memref<10xi32>, %addr: index, %data: vector<4xi32>,
    %mask: vector<4xi1>, %ctrl: none) -> none {
  // CHECK: dataflow.store %{{.*}}[%{{.*}}] %{{.*}} %{{.*}} mask %{{.*}} : memref<10xi32>, vector<4xi32>
  %done = dataflow.store %mem[%addr] %data %ctrl mask %mask
      : memref<10xi32>, vector<4xi32>
  return %done : none
}

// CHECK-LABEL: @store_scalar_vector_element
func.func @store_scalar_vector_element(
    %mem: memref<10xvector<2xi32>>, %addr: index, %data: vector<2xi32>,
    %ctrl: none) -> none {
  // CHECK: dataflow.store %{{.*}}[%{{.*}}] %{{.*}} %{{.*}} : memref<10xvector<2xi32>>
  %done =
      dataflow.store %mem[%addr] %data %ctrl : memref<10xvector<2xi32>>
  return %done : none
}
