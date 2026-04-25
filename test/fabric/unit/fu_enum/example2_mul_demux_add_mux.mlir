// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// User Example 2: 3-input 1-output FU.
//   %mul = muli(%x, %y)
//   %d0, %d1 = demux %mul        // demux.sel chooses which output is live
//   %add = addi(%d1, %z)         // addi consumes demux output #1
//   %out = mux(%d0, %add)        // mux.sel chooses demux #0 or addi
//
// Expected supported subgraphs:
//   demux.sel=0, mux.sel=0 -> a*b
//   demux.sel=1, mux.sel=1 -> a*b + c
// (the other two configs are dropped because they yield a dead value.)

// CHECK-LABEL: @fu_mul_or_mac
func.func @fu_mul_or_mac(%a: !fabric.bits<32>, %b: !fabric.bits<32>,
                         %c: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>,
                 %y = %b : !fabric.bits<32>,
                 %z = %c : !fabric.bits<32>) -> !fabric.bits<32> {
    %mul = fabric.op [@arith.muli] (%x, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %d0, %d1 = fabric.demux %mul : !fabric.bits<32> -> 2
    %add = fabric.op [@arith.addi] (%d1, %z)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %out = fabric.mux %d0, %add : !fabric.bits<32>
    fabric.yield %out : !fabric.bits<32>
  }

  // Subgraph for "%a * %b" (demux.sel=0, mux.sel=0):
  // CHECK: dataflow.subgraph
  // CHECK-SAME: demux#0{sel=0,discard=false,disconnect=false}; mux#0{sel=0,discard=false,disconnect=false}
  // CHECK:   %[[M0:.*]] = arith.muli %{{.*}}, %{{.*}} : i32
  // CHECK:   dataflow.yield %[[M0]] : i32

  // Subgraph for "%a * %b + %c" (demux.sel=1, mux.sel=1):
  // CHECK: dataflow.subgraph
  // CHECK-SAME: demux#0{sel=1,discard=false,disconnect=false}; mux#0{sel=1,discard=false,disconnect=false}
  // CHECK:   %[[M1:.*]] = arith.muli %{{.*}}, %{{.*}} : i32
  // CHECK:   %[[A1:.*]] = arith.addi %[[M1]], %{{.*}} : i32
  // CHECK:   dataflow.yield %[[A1]] : i32

  return
}
