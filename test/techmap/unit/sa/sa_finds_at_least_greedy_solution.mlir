// Simulated annealing must never produce a worse partition than the greedy
// seed: the SA loop initialises `best` from the greedy seed, and only ever
// replaces it with a strictly cheaper neighbour. We validate the contract by
// counting the number of `dataflow.subgraph` regions emitted on a small
// known input where greedy already finds the optimum (one fused two-op
// subgraph). SA must match: any extra block would imply a cost regression
// past the seed.

// RUN: echo "techmap:" > %t.greedy.yaml
// RUN: echo "  algorithm: greedy" >> %t.greedy.yaml
// RUN: echo "techmap:" > %t.sa.yaml
// RUN: echo "  algorithm: sa" >> %t.sa.yaml
// RUN: echo "  sa_steps: 500" >> %t.sa.yaml
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.greedy.yaml" > %t.greedy.mlir
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.sa.yaml" > %t.sa.mlir
// RUN: grep -c "dataflow.subgraph" %t.greedy.mlir > %t.greedy.count
// RUN: grep -c "dataflow.subgraph" %t.sa.mlir > %t.sa.count
// The fabric.fu lowering uses different ops; "dataflow.subgraph" appears
// only inside the partitioned graph body, never in the FU template func.
// On this input greedy fuses muli->addi into one block (one subgraph). SA
// must produce at most that many.
// RUN: diff %t.greedy.count %t.sa.count
// RUN: FileCheck %s < %t.sa.mlir

fabric.module @fu_muli_addi(%cast0_fu_muli_addi : !fabric.bits<32>, %cast1_fu_muli_addi : !fabric.bits<32>) {
  fabric.spatial_pe(%a = %cast0_fu_muli_addi : !fabric.bits<32>, %b = %cast1_fu_muli_addi : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %m = fabric.op [@arith.addi] (%k, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %m : !fabric.bits<32>
  }
  }
  fabric.yield
}


// CHECK-LABEL: @graph_two_op
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.addi
// CHECK-NEXT: dataflow.yield
// Only one subgraph is emitted; the next match is the graph terminator.
// CHECK-NOT: dataflow.subgraph
// CHECK: dataflow.yield
func.func @graph_two_op(%a: i32, %b: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32) -> i32 {
    %p = arith.muli %x, %y : i32
    %q = arith.addi %p, %y : i32
    dataflow.yield %q : i32
  }
  return %r : i32
}
