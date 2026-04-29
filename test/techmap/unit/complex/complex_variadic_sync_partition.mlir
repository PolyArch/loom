// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Stress: graph contains two dataflow.sync rendezvous of differing
// arities (a 2-input/2-output sync and a 3-input/3-output sync).
// The library offers a single FU with M=4 variadic dataflow.sync. The
// enumerator emits one template per popcount (1..4 after dedup), and
// the partitioner binds each user sync to a compatible template.

// FU with M=4 variadic dataflow.sync: covers any active subset
// 1 <= N <= 4.
// CHECK-LABEL: @fu_sync4
fabric.module @fu_sync4(%cast0_fu_sync4 : !fabric.bits<32>, %cast1_fu_sync4 : !fabric.bits<32>, %cast2_fu_sync4 : !fabric.bits<32>, %cast3_fu_sync4 : !fabric.bits<32>) {
  fabric.spatial_pe(%a = %cast0_fu_sync4 : !fabric.bits<32>, %b = %cast1_fu_sync4 : !fabric.bits<32>, %c = %cast2_fu_sync4 : !fabric.bits<32>, %d = %cast3_fu_sync4 : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>,
                      !fabric.bits<32>, !fabric.bits<32>) {
  %r:4 = fabric.fu(%w = %a : !fabric.bits<32>,
                   %x = %b : !fabric.bits<32>,
                   %y = %c : !fabric.bits<32>,
                   %z = %d : !fabric.bits<32>)
                  -> (!fabric.bits<32>, !fabric.bits<32>,
                      !fabric.bits<32>, !fabric.bits<32>) {
    %p, %q, %r0, %s = fabric.op [@dataflow.sync] (%w, %x, %y, %z)
                      : (!fabric.bits<32>, !fabric.bits<32>,
                         !fabric.bits<32>, !fabric.bits<32>)
                        -> (!fabric.bits<32>, !fabric.bits<32>,
                            !fabric.bits<32>, !fabric.bits<32>)
    fabric.yield %p, %q, %r0, %s : !fabric.bits<32>, !fabric.bits<32>,
                                   !fabric.bits<32>, !fabric.bits<32>
  }
  }
  fabric.yield
}


// CHECK-LABEL: @graph_two_syncs
// CHECK: dataflow.graph
// First sync: 2 inputs / 2 outputs.
// CHECK: dataflow.subgraph
// CHECK-NEXT: dataflow.sync
// CHECK-SAME: (i32, i32) -> (i32, i32)
// CHECK-NEXT: dataflow.yield
// Second sync: 3 inputs / 3 outputs.
// CHECK: dataflow.subgraph
// CHECK-NEXT: dataflow.sync
// CHECK-SAME: (i32, i32, i32) -> (i32, i32, i32)
// CHECK-NEXT: dataflow.yield
func.func @graph_two_syncs(%a: i32, %b: i32, %c: i32, %d: i32, %e: i32)
    -> (i32, i32, i32, i32, i32) {
  %r:5 = dataflow.graph(%aa = %a : i32, %bb = %b : i32, %cc = %c : i32,
                        %dd = %d : i32, %ee = %e : i32)
                       -> (i32, i32, i32, i32, i32) {
    %p:2 = dataflow.sync %aa, %bb : (i32, i32) -> (i32, i32)
    %q:3 = dataflow.sync %cc, %dd, %ee : (i32, i32, i32) -> (i32, i32, i32)
    dataflow.yield %p#0, %p#1, %q#0, %q#1, %q#2 : i32, i32, i32, i32, i32
  }
  return %r#0, %r#1, %r#2, %r#3, %r#4 : i32, i32, i32, i32, i32
}
