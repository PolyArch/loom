// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Stress: graph contains dataflow.constant (subgraph-allowed) and
// dataflow.load (graph-only) plus an arith.addi chain. Expected:
// dataflow.load remains at graph level (its load+done multi-result
// shape is not in any FU template here, and dataflow.load itself is
// excluded from dataflow.subgraph by the dialect verifier); the
// constant and the addi each get a singleton subgraph.

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


// (Originally a dataflow.constant FU with a bits<0> control input lived
//  here; that op cannot be wrapped in pe because bits<0> on the
//  PE boundary violates the uniform-W rule (W >= 1). The constant
//  subgraph wrapping behavior is exercised at the IR level by
//  fabric/unit/op/valid.mlir without going through the PE container.)


// CHECK-LABEL: @graph_const_load
// CHECK: dataflow.graph
// dataflow.constant has no covering FU here so it stays at graph level.
// CHECK-DAG: dataflow.constant
// load stays at graph level (graph-only op).
// CHECK-DAG: dataflow.load
// addi gets wrapped.
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.addi
// CHECK-NEXT: dataflow.yield
// CHECK-NOT: dataflow.subgraph
// CHECK: dataflow.yield
func.func @graph_const_load(%mem: memref<16xi32>, %addr: index, %ctrl: none) -> i32 {
  %r = dataflow.graph(%m = %mem : memref<16xi32>, %a = %addr : index, %c = %ctrl : none) -> i32 {
    %k = dataflow.constant %c {const_value = 42 : i32} : i32
    %d, %done = dataflow.load %m[%a] %c : memref<16xi32>
    %s = arith.addi %k, %d : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
