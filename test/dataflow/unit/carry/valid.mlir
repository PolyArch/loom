// RUN: loom %s | loom | FileCheck %s

// CHECK-LABEL: @carry_i32
func.func @carry_i32(%cond: i1, %init: i32, %carry: i32) -> i32 {
  // CHECK: dataflow.carry %{{.*}}, %{{.*}}, %{{.*}} : i32
  %0 = dataflow.carry %cond, %init, %carry : i32
  return %0 : i32
}

// CHECK-LABEL: @carry_i1
func.func @carry_i1(%cond: i1, %init: i1, %carry: i1) -> i1 {
  // CHECK: dataflow.carry %{{.*}}, %{{.*}}, %{{.*}} : i1
  %0 = dataflow.carry %cond, %init, %carry : i1
  return %0 : i1
}

// CHECK-LABEL: @carry_f32
func.func @carry_f32(%cond: i1, %init: f32, %carry: f32) -> f32 {
  // CHECK: dataflow.carry %{{.*}}, %{{.*}}, %{{.*}} : f32
  %0 = dataflow.carry %cond, %init, %carry : f32
  return %0 : f32
}

// CHECK-LABEL: @carry_vector
func.func @carry_vector(%cond: i1, %init: vector<4xi32>, %carry: vector<4xi32>) -> vector<4xi32> {
  // CHECK: dataflow.carry %{{.*}}, %{{.*}}, %{{.*}} : vector<4xi32>
  %0 = dataflow.carry %cond, %init, %carry : vector<4xi32>
  return %0 : vector<4xi32>
}

// CHECK-LABEL: @carry_index
func.func @carry_index(%cond: i1, %init: index, %carry: index) -> index {
  // CHECK: dataflow.carry %{{.*}}, %{{.*}}, %{{.*}} : index
  %0 = dataflow.carry %cond, %init, %carry : index
  return %0 : index
}
