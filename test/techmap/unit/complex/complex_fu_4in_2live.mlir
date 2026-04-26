// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// FU has 4 physical inputs %w, %x, %y, %z but a fabric.mux in the body
// reduces every live configuration to a 2-input compute (the mux selects
// one of (%w, %x) and the addi consumes the result together with %y; %z
// never reaches the live compute). The graph contains a single
// 2-input arith.addi. The partitioner must wrap that addi in a
// dataflow.subgraph backed by one of the FU's 2-input templates.

// CHECK-LABEL: @fu_4in_2live
func.func @fu_4in_2live(%a: !fabric.bits<32>, %b: !fabric.bits<32>,
                         %c: !fabric.bits<32>, %d: !fabric.bits<32>) {
  %r = fabric.fu(%w = %a : !fabric.bits<32>,
                 %x = %b : !fabric.bits<32>,
                 %y = %c : !fabric.bits<32>,
                 %z = %d : !fabric.bits<32>) -> !fabric.bits<32> {
    %m = fabric.mux %w, %x : !fabric.bits<32>
    %r0 = fabric.op [@arith.addi] (%m, %y)
          : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %r0 : !fabric.bits<32>
  }
  return
}

// CHECK-LABEL: @graph_one_addi
// CHECK: dataflow.graph
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.addi
// CHECK-NEXT: dataflow.yield
// CHECK-NOT: dataflow.subgraph
// CHECK: dataflow.yield
func.func @graph_one_addi(%a: i32, %b: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32) -> i32 {
    %p = arith.addi %x, %y : i32
    dataflow.yield %p : i32
  }
  return %r : i32
}
