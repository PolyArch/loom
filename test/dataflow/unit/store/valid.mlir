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
