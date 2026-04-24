// RUN: loom %s | loom | FileCheck %s

// CHECK-LABEL: @mux_2inputs_i1
func.func @mux_2inputs_i1(%sel: i1, %a: i32, %b: i32) -> i32 {
  // CHECK: dataflow.mux %{{.*}}, %{{.*}}, %{{.*}} : (i1, i32, i32) -> i32
  %0 = dataflow.mux %sel, %a, %b : (i1, i32, i32) -> i32
  return %0 : i32
}

// CHECK-LABEL: @mux_3inputs_index
func.func @mux_3inputs_index(%sel: index, %a: f32, %b: f32, %c: f32) -> f32 {
  // CHECK: dataflow.mux %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (index, f32, f32, f32) -> f32
  %0 = dataflow.mux %sel, %a, %b, %c : (index, f32, f32, f32) -> f32
  return %0 : f32
}

// CHECK-LABEL: @mux_5inputs_index
func.func @mux_5inputs_index(%sel: index, %a: i64, %b: i64, %c: i64, %d: i64, %e: i64) -> i64 {
  // CHECK: dataflow.mux %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (index, i64, i64, i64, i64, i64) -> i64
  %0 = dataflow.mux %sel, %a, %b, %c, %d, %e : (index, i64, i64, i64, i64, i64) -> i64
  return %0 : i64
}
