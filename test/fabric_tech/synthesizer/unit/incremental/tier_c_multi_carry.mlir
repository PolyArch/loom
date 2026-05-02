// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/incremental.yaml dump-stats=true' 2>&1 | FileCheck %s --implicit-check-not=unrealized_conversion_cast

// Tier C: two independent carry classes in each input should become two
// separate state slots in the synthesized FU.

// CHECK: remark: {{.*}}synth-stat group=multi_carry strategy=incremental reason=success
// CHECK-SAME: covered=2/2
// CHECK: fabric.module @fu_multi_carry
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.op [@dataflow.stream]
// CHECK: fabric.op [@dataflow.stream]
// CHECK: fabric.op [@dataflow.carry]
// CHECK: fabric.op [@dataflow.carry]
// CHECK-NOT: fabric.op [@dataflow.carry]
// CHECK-DAG: fabric.op [@arith.addi]
// CHECK-DAG: fabric.op [@arith.xori]
// CHECK-DAG: fabric.mux
// CHECK-NOT: fabric.op [@dataflow.carry]
// CHECK: fabric.yield

func.func @pat_multi_carry_add_xor(%lb0: i32, %ub0: i32, %step0: i32,
                                   %lb1: i32, %ub1: i32, %step1: i32,
                                   %init0: i32, %init1: i32) -> (i32, i32)
    attributes {loom.synth_group = "multi_carry"} {
  %p, %q = dataflow.subgraph(%l0 = %lb0 : i32, %u0 = %ub0 : i32,
                             %s0 = %step0 : i32, %l1 = %lb1 : i32,
                             %u1 = %ub1 : i32, %s1 = %step1 : i32,
                             %in0 = %init0 : i32, %in1 = %init1 : i32)
      -> (i32, i32) {
    %idx0, %rwc0 = dataflow.stream %l0, %u0, %s0
                  {step_op = "+=", cont_cond = "<"} : i32
    %idx1, %rwc1 = dataflow.stream %l1, %u1, %s1
                  {step_op = "*=", cont_cond = "<="} : i32
    %c0 = dataflow.carry %rwc0, %in0, %nxt0 : i32
    %c1 = dataflow.carry %rwc1, %in1, %nxt1 : i32
    %nxt0 = arith.addi %c0, %idx0 : i32
    %nxt1 = arith.xori %c1, %idx1 : i32
    dataflow.yield %c0, %c1 : i32, i32
  }
  return %p, %q : i32, i32
}

func.func @pat_multi_carry_xor_add(%lb0: i32, %ub0: i32, %step0: i32,
                                   %lb1: i32, %ub1: i32, %step1: i32,
                                   %init0: i32, %init1: i32) -> (i32, i32)
    attributes {loom.synth_group = "multi_carry"} {
  %p, %q = dataflow.subgraph(%l0 = %lb0 : i32, %u0 = %ub0 : i32,
                             %s0 = %step0 : i32, %l1 = %lb1 : i32,
                             %u1 = %ub1 : i32, %s1 = %step1 : i32,
                             %in0 = %init0 : i32, %in1 = %init1 : i32)
      -> (i32, i32) {
    %idx0, %rwc0 = dataflow.stream %l0, %u0, %s0
                  {step_op = "+=", cont_cond = "<"} : i32
    %idx1, %rwc1 = dataflow.stream %l1, %u1, %s1
                  {step_op = "*=", cont_cond = "<="} : i32
    %c0 = dataflow.carry %rwc0, %in0, %nxt0 : i32
    %c1 = dataflow.carry %rwc1, %in1, %nxt1 : i32
    %nxt0 = arith.xori %c0, %idx0 : i32
    %nxt1 = arith.addi %c1, %idx1 : i32
    dataflow.yield %c0, %c1 : i32, i32
  }
  return %p, %q : i32, i32
}
