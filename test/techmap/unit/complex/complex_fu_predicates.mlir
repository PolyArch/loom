// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Stress: an arith.cmpi FU with hw_params declaring three predicate values
// {eq, slt, sgt}. The graph has three arith.cmpi ops, one per predicate.
// Each predicate value yields a distinct enumerated template; the
// partitioner must cover each of the three graph cmpi ops with a
// matching template, producing three predicate-distinct subgraphs.
// To satisfy the spatial_pe uniform-W rule we expose the FU at bits<1>
// throughout (cmpi's TypeParam(0) inputs accept any width); the graph
// is correspondingly typed with i1 inputs.

// CHECK-LABEL: @fu_cmpi
fabric.module @fu_cmpi {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<1>
  %b = builtin.unrealized_conversion_cast to !fabric.bits<1>
  fabric.spatial_pe(%pa = %a : !fabric.bits<1>,
                    %pb = %b : !fabric.bits<1>) -> !fabric.bits<1> {
    fabric.fu(%x = %pa : !fabric.bits<1>, %y = %pb : !fabric.bits<1>)
                  -> !fabric.bits<1> {
      %k = fabric.op [@arith.cmpi] (%x, %y)
           {hw_params = [{predicate = ["eq", "slt", "sgt"]}]}
           : (!fabric.bits<1>, !fabric.bits<1>) -> !fabric.bits<1>
      fabric.yield %k : !fabric.bits<1>
    }
  }
  fabric.yield
}


// CHECK-LABEL: @graph_three_cmpi
// CHECK: dataflow.graph
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.cmpi eq
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.cmpi slt
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.cmpi sgt
// CHECK-NOT: dataflow.subgraph
// CHECK: dataflow.yield
func.func @graph_three_cmpi(%a: i1, %b: i1) -> (i1, i1, i1) {
  %r:3 = dataflow.graph(%x = %a : i1, %y = %b : i1) -> (i1, i1, i1) {
    %e = arith.cmpi eq, %x, %y : i1
    %l = arith.cmpi slt, %x, %y : i1
    %g = arith.cmpi sgt, %x, %y : i1
    dataflow.yield %e, %l, %g : i1, i1, i1
  }
  return %r#0, %r#1, %r#2 : i1, i1, i1
}
