// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Smoke test: a single fabric.fu offering arith.addi, plus a function-wrapped
// dataflow.graph with one arith.addi. The partitioner's singleton-fallback
// behavior should lift the addi into a dataflow.subgraph that yields the
// original result.

// CHECK-LABEL: @fu_addi
func.func @fu_addi(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  // CHECK: fabric.fu
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  return
}

// CHECK-LABEL: @graph_addi
func.func @graph_addi(%a: i32, %b: i32) -> i32 {
  // CHECK: dataflow.graph
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32) -> i32 {
    // CHECK: dataflow.subgraph
    // CHECK: arith.addi
    // CHECK: dataflow.yield
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
