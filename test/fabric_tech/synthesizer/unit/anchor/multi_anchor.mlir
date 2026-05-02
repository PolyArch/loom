// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/anchor.yaml dump-stats=true' 2>&1 | FileCheck %s

// Subgraph with two yield operands: one is an addi/subi (single share
// group), the other is an andi/ori (different share group). Two inputs
// share the same topology shape; the synthesized FU has two output
// ports, each fed by its own fabric.op merging the per-position op
// names.

// CHECK: remark: {{.*}}synth-stat group=multi_yield strategy=anchor reason=success
// CHECK-SAME: covered=2/2 nodes=2/0/0
// CHECK: fabric.module @fu_multi_yield
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK-DAG: fabric.op [@arith.addi, @arith.subi]
// CHECK-DAG: fabric.op [@arith.andi, @arith.ori]
// CHECK: fabric.yield {{.*}} !fabric.bits<32>, !fabric.bits<32>

func.func @pat_addi_andi(%a: i32, %b: i32) -> (i32, i32)
    attributes {loom.synth_group = "multi_yield"} {
  %p, %q = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> (i32, i32) {
    %u = arith.addi %x, %y : i32
    %v = arith.andi %x, %y : i32
    dataflow.yield %u, %v : i32, i32
  }
  return %p, %q : i32, i32
}

func.func @pat_subi_ori(%a: i32, %b: i32) -> (i32, i32)
    attributes {loom.synth_group = "multi_yield"} {
  %p, %q = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> (i32, i32) {
    %u = arith.subi %x, %y : i32
    %v = arith.ori %x, %y : i32
    dataflow.yield %u, %v : i32, i32
  }
  return %p, %q : i32, i32
}
