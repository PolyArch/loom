// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Stress: an FU with an arith.muli followed by a fabric.demux at the
// output stage, producing two outputs from one compute. The graph has a
// single arith.muli whose i32 result fans out to two arith.addi
// consumers. Intuitively the muli could bind to the muli-demux FU with
// the demux providing the two fan-out copies; in practice the matcher
// requires the graph root op to have the same arity as the FU template's
// root, and the muli-demux template root is "arith.muli followed by a
// 2-output demux" (a 2-result subgraph). A single-result arith.muli does
// not currently match it, so the muli is left at graph level and only
// the addi consumers are wrapped.
//
// TODO: when the partitioner learns to absorb single-output graph ops
// into multi-output FU templates by treating the demux as an SSA
// fan-out helper, update the CHECKs below to wrap the muli too.

// CHECK-LABEL: @fu_muli_demux
func.func @fu_muli_demux(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r0, %r1 = fabric.fu(%x = %a : !fabric.bits<32>,
                       %y = %b : !fabric.bits<32>)
                       -> (!fabric.bits<32>, !fabric.bits<32>) {
    %p = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %d0, %d1 = fabric.demux %p : !fabric.bits<32> -> 2
    fabric.yield %d0, %d1 : !fabric.bits<32>, !fabric.bits<32>
  }
  return
}

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

// Lock current behavior: muli stays at graph level, the three downstream
// addi consumers each get their own subgraph. See TODO above.
// CHECK-LABEL: @graph_muli_fanout
// CHECK: dataflow.graph
// CHECK: arith.muli
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
