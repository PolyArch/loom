// Smoke test: drive `loom-synth-fu-dump` end-to-end on a tier-A
// workload. Two i32 subgraphs in the `alu` group share the
// arith.addi/subi hardware-share group, so the anchor strategy folds
// them into one fabric.op whose op_list is the sorted union of the two
// observed names. The helper must print (a) the synthesized FU IR,
// (b) the canonical synth-stat line, and (c) a wallclock measurement
// in microseconds.

// RUN: loom-synth-fu-dump --config=%p/anchor.yaml %s | FileCheck %s

// CHECK: // --- synthesized FUs ---
// CHECK: fabric.module @fu_alu
// CHECK-SAME: loom.synthesized_for = "alu"
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.op [@arith.addi, @arith.subi]
// CHECK: fabric.yield

// CHECK: // --- synth stats ---
// CHECK: synth-stat group=alu strategy=anchor reason=success
// CHECK-SAME: covered=2/2

// CHECK: wallclock_us={{[0-9]+}}

func.func @pat_addi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

func.func @pat_subi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.subi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
