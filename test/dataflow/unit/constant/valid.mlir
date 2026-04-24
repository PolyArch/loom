// RUN: loom %s | loom | FileCheck %s

// CHECK-LABEL: @const_i32
func.func @const_i32(%ctrl: none) -> i32 {
  // CHECK: dataflow.constant %{{.*}} {const_value = 42 : i32} : i32
  %0 = dataflow.constant %ctrl {const_value = 42 : i32} : i32
  return %0 : i32
}

// CHECK-LABEL: @const_i64
func.func @const_i64(%ctrl: none) -> i64 {
  // CHECK: dataflow.constant %{{.*}} {const_value = 7 : i64} : i64
  %0 = dataflow.constant %ctrl {const_value = 7 : i64} : i64
  return %0 : i64
}

// CHECK-LABEL: @const_f32
func.func @const_f32(%ctrl: none) -> f32 {
  // CHECK: dataflow.constant %{{.*}} {const_value = 3.140000e+00 : f32} : f32
  %0 = dataflow.constant %ctrl {const_value = 3.14 : f32} : f32
  return %0 : f32
}

// CHECK-LABEL: @const_dense
func.func @const_dense(%ctrl: none) -> vector<4xi32> {
  // CHECK: dataflow.constant %{{.*}} {const_value = dense<[1, 2, 3, 4]> : vector<4xi32>} : vector<4xi32>
  %0 = dataflow.constant %ctrl {const_value = dense<[1, 2, 3, 4]> : vector<4xi32>} : vector<4xi32>
  return %0 : vector<4xi32>
}
