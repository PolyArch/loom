// RUN: loom %s | loom | FileCheck %s

// CHECK-LABEL: @invariant_i32
func.func @invariant_i32(%cond: i1, %init: i32) -> i32 {
  // CHECK: dataflow.invariant %{{.*}}, %{{.*}} : i32
  %0 = dataflow.invariant %cond, %init : i32
  return %0 : i32
}

// CHECK-LABEL: @invariant_i1
func.func @invariant_i1(%cond: i1, %init: i1) -> i1 {
  // CHECK: dataflow.invariant %{{.*}}, %{{.*}} : i1
  %0 = dataflow.invariant %cond, %init : i1
  return %0 : i1
}

// CHECK-LABEL: @invariant_f64
func.func @invariant_f64(%cond: i1, %init: f64) -> f64 {
  // CHECK: dataflow.invariant %{{.*}}, %{{.*}} : f64
  %0 = dataflow.invariant %cond, %init : f64
  return %0 : f64
}

// CHECK-LABEL: @invariant_vector
func.func @invariant_vector(%cond: i1, %init: vector<8xi16>) -> vector<8xi16> {
  // CHECK: dataflow.invariant %{{.*}}, %{{.*}} : vector<8xi16>
  %0 = dataflow.invariant %cond, %init : vector<8xi16>
  return %0 : vector<8xi16>
}
