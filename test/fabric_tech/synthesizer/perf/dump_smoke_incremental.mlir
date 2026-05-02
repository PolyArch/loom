// Smoke test: drive `loom-synth-fu-dump` end-to-end on a tier-B
// workload. Two inputs share an arith.addi prefix; one extends with
// arith.muli at the yield, one terminates immediately. The incremental
// strategy folds the inputs by inserting a fabric.demux on the addi
// output (one arm to yield, one arm to muli) and a fabric.mux at the
// yield to collapse both branches back into one output port. The
// helper must print the demux/mux pattern in the dumped FU IR plus
// the canonical stats line and a wallclock measurement.

// RUN: loom-synth-fu-dump --config=%p/incremental.yaml %s | FileCheck %s

// CHECK: // --- synthesized FUs ---
// CHECK: fabric.module @fu_tierB_demo
// CHECK-SAME: loom.synthesized_for = "tierB_demo"
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK-DAG: fabric.op [@arith.addi]
// CHECK-DAG: fabric.demux
// CHECK-DAG: fabric.op [@arith.muli]
// CHECK-DAG: fabric.mux
// CHECK: fabric.yield

// CHECK: // --- synth stats ---
// CHECK: synth-stat group=tierB_demo strategy=incremental reason=success

// CHECK: wallclock_us={{[0-9]+}}

func.func @pat_add_only(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "tierB_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %t = arith.addi %x, %y : i32
    dataflow.yield %t : i32
  }
  return %r : i32
}

func.func @pat_add_then_mul(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "tierB_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %t = arith.addi %x, %y : i32
    %m = arith.muli %t, %z : i32
    dataflow.yield %m : i32
  }
  return %r : i32
}
