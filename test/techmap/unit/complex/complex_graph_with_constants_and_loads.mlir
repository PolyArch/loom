// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Stress: graph contains dataflow.constant (subgraph-allowed) and
// dataflow.load (graph-only) plus an arith.addi chain. Expected:
// dataflow.load remains at graph level (its load+done multi-result
// shape is not in any FU template here, and dataflow.load itself is
// excluded from dataflow.subgraph by the dialect verifier); the
// constant and the addi each get a singleton subgraph.

// CHECK-LABEL: @fu_addi
func.func @fu_addi(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  return
}

// FU implements dataflow.constant. The control input is bits<0> (none
// flavour) so the candidate cache binds it to the constant's typed
// const_value attribute at materialization time.
// CHECK-LABEL: @fu_const
func.func @fu_const(%ctrl: !fabric.bits<0>) {
  %r = fabric.fu(%c = %ctrl : !fabric.bits<0>) -> !fabric.bits<32> {
    %k = fabric.op [@dataflow.constant] (%c)
         {sw_configs = {const_hex_value = "0000002a"}}
         : (!fabric.bits<0>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  return
}

// CHECK-LABEL: @graph_const_load
// CHECK: dataflow.graph
// constant gets wrapped in a singleton subgraph.
// CHECK: dataflow.subgraph
// CHECK-NEXT: dataflow.constant
// CHECK-NEXT: dataflow.yield
// load stays at graph level (graph-only op).
// CHECK: dataflow.load
// addi gets wrapped after load.
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
