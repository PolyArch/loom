// RUN: loom %s | loom | FileCheck %s

// CHECK-LABEL: @stream_add_slt
func.func @stream_add_slt(%init: i32, %limit: i32, %step: i32) -> (i32, i1) {
  // CHECK: %{{.*}}, %{{.*}} = dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} step add while slt : i32
  %iv, %phase = dataflow.stream %init, %limit, %step step add while slt : i32
  return %iv, %phase : i32, i1
}

// CHECK-LABEL: @stream_sub_sgt
func.func @stream_sub_sgt(%init: i8, %limit: i8, %step: i8) -> (i8, i1) {
  // CHECK: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} step sub while sgt : i8
  %iv, %phase = dataflow.stream %init, %limit, %step step sub while sgt : i8
  return %iv, %phase : i8, i1
}

// CHECK-LABEL: @stream_mul_sle
func.func @stream_mul_sle(%init: i64, %limit: i64, %step: i64) -> (i64, i1) {
  // CHECK: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} step mul while sle : i64
  %iv, %phase = dataflow.stream %init, %limit, %step step mul while sle : i64
  return %iv, %phase : i64, i1
}

// CHECK-LABEL: @stream_sdiv_slt
func.func @stream_sdiv_slt(%init: i32, %limit: i32, %step: i32) -> (i32, i1) {
  // CHECK: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} step sdiv while slt : i32
  %iv, %phase = dataflow.stream %init, %limit, %step step sdiv while slt : i32
  return %iv, %phase : i32, i1
}

// CHECK-LABEL: @stream_udiv_ult
func.func @stream_udiv_ult(%init: i32, %limit: i32, %step: i32) -> (i32, i1) {
  // CHECK: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} step udiv while ult : i32
  %iv, %phase = dataflow.stream %init, %limit, %step step udiv while ult : i32
  return %iv, %phase : i32, i1
}

// CHECK-LABEL: @stream_shl_ne
func.func @stream_shl_ne(%init: i16, %limit: i16, %step: i16) -> (i16, i1) {
  // CHECK: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} step shl while ne : i16
  %iv, %phase = dataflow.stream %init, %limit, %step step shl while ne : i16
  return %iv, %phase : i16, i1
}

// CHECK-LABEL: @stream_ashr_sge
func.func @stream_ashr_sge(%init: i32, %limit: i32, %step: i32) -> (i32, i1) {
  // CHECK: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} step ashr while sge : i32
  %iv, %phase = dataflow.stream %init, %limit, %step step ashr while sge : i32
  return %iv, %phase : i32, i1
}

// CHECK-LABEL: @stream_lshr_uge
func.func @stream_lshr_uge(%init: i32, %limit: i32, %step: i32) -> (i32, i1) {
  // CHECK: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} step lshr while uge : i32
  %iv, %phase = dataflow.stream %init, %limit, %step step lshr while uge : i32
  return %iv, %phase : i32, i1
}
