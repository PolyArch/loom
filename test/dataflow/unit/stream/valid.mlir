// RUN: loom %s | loom | FileCheck %s

// CHECK-LABEL: @stream_basic
func.func @stream_basic(%lb: i32, %ub: i32, %step: i32) -> (i32, i1) {
  // CHECK: %{{.*}}, %{{.*}} = dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} {cont_cond = "<", step_op = "+="} : i32
  %idx, %rwc = dataflow.stream %lb, %ub, %step {step_op = "+=", cont_cond = "<"} : i32
  return %idx, %rwc : i32, i1
}

// CHECK-LABEL: @stream_i64_mul
func.func @stream_i64_mul(%lb: i64, %ub: i64, %step: i64) -> (i64, i1) {
  // CHECK: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} {cont_cond = "<=", step_op = "*="} : i64
  %idx, %rwc = dataflow.stream %lb, %ub, %step {step_op = "*=", cont_cond = "<="} : i64
  return %idx, %rwc : i64, i1
}

// CHECK-LABEL: @stream_shl_ne
func.func @stream_shl_ne(%lb: i16, %ub: i16, %step: i16) -> (i16, i1) {
  // CHECK: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} {cont_cond = "!=", step_op = "<<="} : i16
  %idx, %rwc = dataflow.stream %lb, %ub, %step {step_op = "<<=", cont_cond = "!="} : i16
  return %idx, %rwc : i16, i1
}

// CHECK-LABEL: @stream_shr_ge
func.func @stream_shr_ge(%lb: i32, %ub: i32, %step: i32) -> (i32, i1) {
  // CHECK: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} {cont_cond = ">=", step_op = ">>="} : i32
  %idx, %rwc = dataflow.stream %lb, %ub, %step {step_op = ">>=", cont_cond = ">="} : i32
  return %idx, %rwc : i32, i1
}

// CHECK-LABEL: @stream_sub_gt
func.func @stream_sub_gt(%lb: i8, %ub: i8, %step: i8) -> (i8, i1) {
  // CHECK: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} {cont_cond = ">", step_op = "-="} : i8
  %idx, %rwc = dataflow.stream %lb, %ub, %step {step_op = "-=", cont_cond = ">"} : i8
  return %idx, %rwc : i8, i1
}

// CHECK-LABEL: @stream_div_lt
func.func @stream_div_lt(%lb: i32, %ub: i32, %step: i32) -> (i32, i1) {
  // CHECK: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} {cont_cond = "<", step_op = "/="} : i32
  %idx, %rwc = dataflow.stream %lb, %ub, %step {step_op = "/=", cont_cond = "<"} : i32
  return %idx, %rwc : i32, i1
}
