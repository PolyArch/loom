// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Pins: a chain whose tail feeds back into the head via a dataflow.carry
// must not be split across two dataflow.subgraph blocks (which would
// create a multi-block SSA cycle). The partitioner
// is expected to wrap the carry in its own subgraph and leave the
// loop-closing producer at graph level (greedy default behavior).

// CHECK-LABEL: @fu_carry
func.func @fu_carry(%c: !fabric.bits<1>, %i: !fabric.bits<32>,
                    %k: !fabric.bits<32>) {
  %r = fabric.fu(%cc = %c : !fabric.bits<1>,
                 %ii = %i : !fabric.bits<32>,
                 %kk = %k : !fabric.bits<32>) -> !fabric.bits<32> {
    %o = fabric.op [@dataflow.carry] (%cc, %ii, %kk)
         : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>)
           -> !fabric.bits<32>
    fabric.yield %o : !fabric.bits<32>
  }
  return
}

// CHECK-LABEL: @fu_addsub
func.func @fu_addsub(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi, @arith.subi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  return
}

// The graph contains a carry-addi-subi loop. The partitioner must avoid
// putting the producer chain in a subgraph that points back into the
// carry's subgraph; the carry subgraph is the only one that can close
// the loop.
// CHECK-LABEL: @graph_loop_with_alu
// CHECK: dataflow.graph
// CHECK-NOT: dataflow.subgraph{{.*}}dataflow.subgraph
// CHECK: dataflow.yield
func.func @graph_loop_with_alu(%cond: i1, %init: i32, %step: i32) -> i32 {
  %r = dataflow.graph(%c = %cond : i1, %i = %init : i32, %s = %step : i32) -> i32 {
    %acc = dataflow.carry %c, %i, %next : i32
    %inc = arith.addi %acc, %s : i32
    %next = arith.subi %inc, %s : i32
    dataflow.yield %acc : i32
  }
  return %r : i32
}
