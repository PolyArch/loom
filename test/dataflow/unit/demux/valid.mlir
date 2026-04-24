// RUN: loom %s | loom | FileCheck %s

// CHECK-LABEL: @demux_2outputs_i1
func.func @demux_2outputs_i1(%sel: i1, %in: i32) -> (i32, i32) {
  // CHECK: %{{.*}}:2 = dataflow.demux %{{.*}}, %{{.*}} : (i1, i32) -> (i32, i32)
  %0:2 = dataflow.demux %sel, %in : (i1, i32) -> (i32, i32)
  return %0#0, %0#1 : i32, i32
}

// CHECK-LABEL: @demux_3outputs_index
func.func @demux_3outputs_index(%sel: index, %in: f32) -> (f32, f32, f32) {
  // CHECK: %{{.*}}:3 = dataflow.demux %{{.*}}, %{{.*}} : (index, f32) -> (f32, f32, f32)
  %0:3 = dataflow.demux %sel, %in : (index, f32) -> (f32, f32, f32)
  return %0#0, %0#1, %0#2 : f32, f32, f32
}
