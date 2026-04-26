// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Stress: an FU with two fabric.mux ops bracketing an arith.addi. The mux
// network sits inside the FU body (not just at I/O), so the enumerator
// emits multiple sw_config combinations of the two mux selectors. The
// graph contains a single addi; the partitioner must pick one of the
// admissible templates and wrap it.
//
// Template count for the FU: with mux#1.sel=0 both mux branches are live
// and 2 mux#0.sel values produce 2 templates; with mux#1.sel=1 the upper
// path is silent so mux#0 must enter discard or disconnect (not both),
// adding 5 more templates. Empirically the enumerator emits 7 templates
// for this shape; the exact count is locked in
// test/fabric/unit/fu_enum/internal_mux_network.mlir, here we only need
// at least one template to cover the graph's addi.

// CHECK-LABEL: @fu_two_mux
func.func @fu_two_mux(%a: !fabric.bits<32>, %b: !fabric.bits<32>,
                      %c: !fabric.bits<32>, %d: !fabric.bits<32>) {
  %r = fabric.fu(%w = %a : !fabric.bits<32>,
                 %x = %b : !fabric.bits<32>,
                 %y = %c : !fabric.bits<32>,
                 %z = %d : !fabric.bits<32>) -> !fabric.bits<32> {
    %m1 = fabric.mux %w, %x : !fabric.bits<32>
    %p = fabric.op [@arith.addi] (%m1, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %m2 = fabric.mux %p, %z : !fabric.bits<32>
    fabric.yield %m2 : !fabric.bits<32>
  }
  return
}

// CHECK-LABEL: @graph_one_addi
// CHECK: dataflow.graph
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.addi
// CHECK-NEXT: dataflow.yield
// Only one subgraph is emitted.
// CHECK-NOT: dataflow.subgraph
// CHECK: dataflow.yield
func.func @graph_one_addi(%a: i32, %b: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32) -> i32 {
    %p = arith.addi %x, %y : i32
    dataflow.yield %p : i32
  }
  return %r : i32
}
