// RUN: loom %s | loom | FileCheck %s

// CHECK-LABEL: @gate_i32
func.func @gate_i32(%bc: i1, %bv: i32) -> (i1, i32) {
  // CHECK: %{{.*}}, %{{.*}} = dataflow.gate %{{.*}}, %{{.*}} : i32
  %ac, %av = dataflow.gate %bc, %bv : i32
  return %ac, %av : i1, i32
}

// CHECK-LABEL: @gate_i1
func.func @gate_i1(%bc: i1, %bv: i1) -> (i1, i1) {
  // CHECK: dataflow.gate %{{.*}}, %{{.*}} : i1
  %ac, %av = dataflow.gate %bc, %bv : i1
  return %ac, %av : i1, i1
}

// CHECK-LABEL: @gate_f32
func.func @gate_f32(%bc: i1, %bv: f32) -> (i1, f32) {
  // CHECK: dataflow.gate %{{.*}}, %{{.*}} : f32
  %ac, %av = dataflow.gate %bc, %bv : f32
  return %ac, %av : i1, f32
}

// CHECK-LABEL: @gate_vector
func.func @gate_vector(%bc: i1, %bv: vector<4xi8>) -> (i1, vector<4xi8>) {
  // CHECK: dataflow.gate %{{.*}}, %{{.*}} : vector<4xi8>
  %ac, %av = dataflow.gate %bc, %bv : vector<4xi8>
  return %ac, %av : i1, vector<4xi8>
}
