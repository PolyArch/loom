// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Stress: an FU with an arith.muli followed by a fabric.demux at the
// output stage. The demux is a 1-of-N selector, so for any given config
// only one of its outputs is live; the other ports do not contribute a
// value to the FU's external behavior. The enumerator therefore emits
// per-config single-result subgraph templates (one for sel=0, one for
// sel=1), each wrapping the muli compute. The partitioner can then bind
// the graph's muli to one of these templates, allowing it to be wrapped
// even though it has only a single SSA result while the FU declares two
// output ports.

// CHECK-LABEL: @fu_muli_demux
fabric.module @fu_muli_demux(%cast0_fu_muli_demux : !fabric.bits<32>, %cast1_fu_muli_demux : !fabric.bits<32>) {
  fabric.spatial_pe(%a = %cast0_fu_muli_demux : !fabric.bits<32>,
                    %b = %cast1_fu_muli_demux : !fabric.bits<32>)
                   -> (!fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%x = %a : !fabric.bits<32>,
              %y = %b : !fabric.bits<32>)
             -> (!fabric.bits<32>, !fabric.bits<32>) {
      %p = fabric.op [@arith.muli] (%x, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %d0, %d1 = fabric.demux %p : !fabric.bits<32> -> 2
      fabric.yield %d0, %d1 : !fabric.bits<32>, !fabric.bits<32>
    }
  }
  fabric.yield
}


// CHECK-LABEL: @fu_addi
fabric.module @fu_addi(%cast0_fu_addi : !fabric.bits<32>, %cast1_fu_addi : !fabric.bits<32>) {
  fabric.spatial_pe(%a = %cast0_fu_addi : !fabric.bits<32>, %b = %cast1_fu_addi : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}


// The muli is now wrapped in its own dataflow.subgraph (bound to one of
// the muli-demux per-config templates). Each downstream addi is also
// wrapped in its own subgraph.
// CHECK-LABEL: @graph_muli_fanout
// CHECK: dataflow.graph
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.muli
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.addi
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.addi
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.addi
// CHECK-NOT: dataflow.subgraph
// CHECK: dataflow.yield
func.func @graph_muli_fanout(%a: i32, %b: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32) -> i32 {
    %p = arith.muli %x, %y : i32
    %u = arith.addi %p, %y : i32
    %v = arith.addi %p, %x : i32
    %w = arith.addi %u, %v : i32
    dataflow.yield %w : i32
  }
  return %r : i32
}
