// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Pins: a chain whose tail feeds back into the head via a dataflow.carry
// must not be split across two dataflow.subgraph blocks (which would
// create a multi-block SSA cycle). The partitioner
// is expected to wrap the carry in its own subgraph and leave the
// loop-closing producer at graph level (greedy default behavior).

// CHECK-LABEL: @fu_carry
fabric.module @fu_carry(%c : !fabric.bits<1>, %i : !fabric.bits<1>, %k : !fabric.bits<1>) {
  fabric.spatial_pe(%pc = %c : !fabric.bits<1>,
                    %pi = %i : !fabric.bits<1>,
                    %pk = %k : !fabric.bits<1>) -> !fabric.bits<1> {
    fabric.fu(%cc = %pc : !fabric.bits<1>,
              %ii = %pi : !fabric.bits<1>,
              %kk = %pk : !fabric.bits<1>) -> !fabric.bits<1> {
      %o = fabric.op [@dataflow.carry] (%cc, %ii, %kk)
           : (!fabric.bits<1>, !fabric.bits<1>, !fabric.bits<1>)
             -> !fabric.bits<1>
      fabric.yield %o : !fabric.bits<1>
    }
  }
  fabric.yield
}

// CHECK-LABEL: @fu_addsub
fabric.module @fu_addsub(%cast0_fu_addsub : !fabric.bits<32>, %cast1_fu_addsub : !fabric.bits<32>) {
  fabric.spatial_pe(%a = %cast0_fu_addsub : !fabric.bits<32>, %b = %cast1_fu_addsub : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi, @arith.subi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
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
