// RUN: loom %s | loom | FileCheck %s

// CHECK-LABEL: @sync_zero
func.func @sync_zero() {
  // CHECK: dataflow.sync : () -> ()
  "dataflow.sync"() : () -> ()
  return
}

// CHECK-LABEL: @sync_one
func.func @sync_one(%a: i32) -> i32 {
  // CHECK: dataflow.sync %{{.*}} : (i32) -> i32
  %0 = dataflow.sync %a : (i32) -> i32
  return %0 : i32
}

// CHECK-LABEL: @sync_two_homogeneous
func.func @sync_two_homogeneous(%a: i32, %b: i32) -> (i32, i32) {
  // CHECK: %{{.*}}:2 = dataflow.sync %{{.*}}, %{{.*}} : (i32, i32) -> (i32, i32)
  %0:2 = dataflow.sync %a, %b : (i32, i32) -> (i32, i32)
  return %0#0, %0#1 : i32, i32
}

// CHECK-LABEL: @sync_mixed
func.func @sync_mixed(%a: i32, %b: f32, %c: vector<4xi8>) -> (i32, f32, vector<4xi8>) {
  // CHECK: %{{.*}}:3 = dataflow.sync %{{.*}}, %{{.*}}, %{{.*}} : (i32, f32, vector<4xi8>) -> (i32, f32, vector<4xi8>)
  %0:3 = dataflow.sync %a, %b, %c : (i32, f32, vector<4xi8>) -> (i32, f32, vector<4xi8>)
  return %0#0, %0#1, %0#2 : i32, f32, vector<4xi8>
}
