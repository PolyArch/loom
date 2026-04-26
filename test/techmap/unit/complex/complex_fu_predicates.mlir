// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Stress: an arith.cmpi FU with hw_params declaring three predicate values
// {eq, slt, sgt}. The graph has three arith.cmpi ops, one per predicate.
// Each predicate value yields a distinct enumerated template; the
// partitioner must cover each of the three graph cmpi ops with a
// matching template, producing three predicate-distinct subgraphs.

// CHECK-LABEL: @fu_cmpi
func.func @fu_cmpi(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<1> {
    %k = fabric.op [@arith.cmpi] (%x, %y)
         {hw_params = [{predicate = ["eq", "slt", "sgt"]}]}
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
    fabric.yield %k : !fabric.bits<1>
  }
  return
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
func.func @graph_three_cmpi(%a: i32, %b: i32) -> (i1, i1, i1) {
  %r:3 = dataflow.graph(%x = %a : i32, %y = %b : i32) -> (i1, i1, i1) {
    %e = arith.cmpi eq, %x, %y : i32
    %l = arith.cmpi slt, %x, %y : i32
    %g = arith.cmpi sgt, %x, %y : i32
    dataflow.yield %e, %l, %g : i1, i1, i1
  }
  return %r#0, %r#1, %r#2 : i1, i1, i1
}
