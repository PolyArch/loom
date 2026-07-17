// RUN: loom %s -loom-synthesize-configured-functions='config=%p/anchor.yaml dump-stats=true' 2>&1 | FileCheck %s

// Configured functions with two outputs: one is an addi/subi (single share
// group), the other is an andi/ori (different share group). Two inputs
// share the same topology shape; the synthesized FU has two independent
// physical ops. Their complete modes form four explicit legal encodings,
// including the two input functions and two additional combinations.

// CHECK: remark: {{.*}}synth-stat group=multi_yield strategy=anchor reason=success
// CHECK-SAME: covered=2/2 nodes=2/0/0 encodings=4
// CHECK: fabric.module @fu_multi_yield
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK-DAG: fabric.op [@arith.addi, @arith.subi]
// CHECK-DAG: fabric.op [@arith.andi, @arith.ori]
// CHECK: fabric.yield {{.*}} !fabric.bits<32>, !fabric.bits<32>

func.func @pat_addi_andi(%a: i32, %b: i32) -> (i32, i32)
    attributes {loom.synth_group = "multi_yield"} {
  %u = arith.addi %a, %b : i32
  %v = arith.andi %a, %b : i32
  return %u, %v : i32, i32
}

func.func @pat_subi_ori(%a: i32, %b: i32) -> (i32, i32)
    attributes {loom.synth_group = "multi_yield"} {
  %u = arith.subi %a, %b : i32
  %v = arith.ori %a, %b : i32
  return %u, %v : i32, i32
}
