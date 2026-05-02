// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/incremental.yaml dump-stats=true' 2>&1 | FileCheck %s --implicit-check-not=unrealized_conversion_cast

// Tier C includes state-bearing dataflow.gate and dataflow.invariant
// heads, and both must be wrapped as fabric.op instances in FU bodies.

// CHECK: remark: {{.*}}synth-stat group=gate_state strategy=incremental reason=success
// CHECK-SAME: covered=2/2
// CHECK: fabric.module @fu_gate_state
// CHECK: fabric.fu
// CHECK-NOT: = dataflow.gate
// CHECK-DAG: fabric.op [@dataflow.gate]
// CHECK-DAG: fabric.op [@arith.addi]
// CHECK-DAG: fabric.op [@arith.xori]
// CHECK-DAG: fabric.mux
// CHECK-NOT: = dataflow.gate
// CHECK: fabric.yield

// CHECK: remark: {{.*}}synth-stat group=invariant_state strategy=incremental reason=success
// CHECK-SAME: covered=2/2
// CHECK: fabric.module @fu_invariant_state
// CHECK: fabric.fu
// CHECK-NOT: = dataflow.invariant
// CHECK-DAG: fabric.op [@dataflow.invariant]
// CHECK-DAG: fabric.op [@arith.addi]
// CHECK-DAG: fabric.op [@arith.xori]
// CHECK-DAG: fabric.mux
// CHECK-NOT: = dataflow.invariant
// CHECK: fabric.yield

func.func @pat_gate_addi(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32
    attributes {loom.synth_group = "gate_state"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32,
                         %s = %step : i32, %in = %init : i32) -> i32 {
    %idx, %rwc = dataflow.stream %l, %u, %s
                 {step_op = "+=", cont_cond = "<"} : i32
    %after_cond, %after_value = dataflow.gate %rwc, %in : i32
    %out = arith.addi %after_value, %idx : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}

func.func @pat_gate_xori(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32
    attributes {loom.synth_group = "gate_state"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32,
                         %s = %step : i32, %in = %init : i32) -> i32 {
    %idx, %rwc = dataflow.stream %l, %u, %s
                 {step_op = "+=", cont_cond = "<"} : i32
    %after_cond, %after_value = dataflow.gate %rwc, %in : i32
    %out = arith.xori %after_value, %idx : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}

func.func @pat_invariant_addi(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32
    attributes {loom.synth_group = "invariant_state"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32,
                         %s = %step : i32, %in = %init : i32) -> i32 {
    %idx, %rwc = dataflow.stream %l, %u, %s
                 {step_op = "+=", cont_cond = "<"} : i32
    %latched = dataflow.invariant %rwc, %in : i32
    %out = arith.addi %latched, %idx : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}

func.func @pat_invariant_xori(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32
    attributes {loom.synth_group = "invariant_state"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32,
                         %s = %step : i32, %in = %init : i32) -> i32 {
    %idx, %rwc = dataflow.stream %l, %u, %s
                 {step_op = "+=", cont_cond = "<"} : i32
    %latched = dataflow.invariant %rwc, %in : i32
    %out = arith.xori %latched, %idx : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}
