// ILP partitioner output must be deterministic across thread counts:
// the candidate cache is the only multi-threaded part, and its output is
// already sorted; the MIP itself is run with HiGHS threads=1.

// RUN: echo "fabric_techmap:" > %t.ilp1.yaml
// RUN: echo "  algorithm: ilp" >> %t.ilp1.yaml
// RUN: echo "  threads: 1" >> %t.ilp1.yaml
// RUN: echo "fabric_techmap:" > %t.ilp4.yaml
// RUN: echo "  algorithm: ilp" >> %t.ilp4.yaml
// RUN: echo "  threads: 4" >> %t.ilp4.yaml
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.ilp1.yaml" > %t.ilp1.mlir
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.ilp4.yaml" > %t.ilp4.mlir
// RUN: diff %t.ilp1.mlir %t.ilp4.mlir
// RUN: FileCheck %s < %t.ilp1.mlir

// CHECK-LABEL: @fu_muli
fabric.module @fu_muli(%cast0_fu_muli : !fabric.bits<32>, %cast1_fu_muli : !fabric.bits<32>) {
  fabric.pe [spatial] (%a = %cast0_fu_muli : !fabric.bits<32>, %b = %cast1_fu_muli : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}


// CHECK-LABEL: @fu_addi
fabric.module @fu_addi(%cast0_fu_addi : !fabric.bits<32>, %cast1_fu_addi : !fabric.bits<32>) {
  fabric.pe [spatial] (%a = %cast0_fu_addi : !fabric.bits<32>, %b = %cast1_fu_addi : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}


// CHECK-LABEL: @graph_chain
// CHECK: dataflow.subgraph
// CHECK: dataflow.subgraph
// CHECK: dataflow.subgraph
func.func @graph_chain(%a: i32, %b: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32) -> i32 {
    %p = arith.muli %x, %y : i32
    %q = arith.addi %p, %y : i32
    %s = arith.addi %q, %x : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
