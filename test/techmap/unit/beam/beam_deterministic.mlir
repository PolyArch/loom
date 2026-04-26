// Two runs with different worker thread counts must produce byte-identical
// output IR under the beam partitioner. This guards both the candidate
// cache (already covered for dump output) and the beam search itself,
// which must not depend on candidate-cache iteration order or any
// non-deterministic successor sort.
//
// The temporary YAML config files override `algorithm` to `beam` and the
// `threads` knob; cost weights and beam width stay at defaults.

// RUN: echo "techmap:" > %t.cfg1.yaml
// RUN: echo "  algorithm: beam" >> %t.cfg1.yaml
// RUN: echo "  threads: 1" >> %t.cfg1.yaml
// RUN: echo "techmap:" > %t.cfg4.yaml
// RUN: echo "  algorithm: beam" >> %t.cfg4.yaml
// RUN: echo "  threads: 4" >> %t.cfg4.yaml
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.cfg1.yaml" > %t.t1.mlir
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.cfg4.yaml" > %t.t4.mlir
// RUN: diff %t.t1.mlir %t.t4.mlir
// RUN: FileCheck %s < %t.t1.mlir

// Two FUs in the library: one for arith.addi, one for arith.muli. The
// graph is a deterministic mixed chain so the partitioner has multiple
// per-op decisions to make. With single-op templates only, the expected
// partition is one subgraph per op. Beam-search retains all four states
// at each step (default beam_width=4) and must select identically across
// thread configurations.

func.func @fu_addi(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  return
}

func.func @fu_muli(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  return
}

// CHECK-LABEL: @graph_chain
// CHECK: dataflow.subgraph
// CHECK: arith.addi
// CHECK: dataflow.subgraph
// CHECK: arith.muli
// CHECK: dataflow.subgraph
// CHECK: arith.addi
// CHECK: dataflow.subgraph
// CHECK: arith.muli
func.func @graph_chain(%a: i32, %b: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32) -> i32 {
    %v0 = arith.addi %x, %y : i32
    %v1 = arith.muli %v0, %y : i32
    %v2 = arith.addi %v1, %y : i32
    %v3 = arith.muli %v2, %y : i32
    dataflow.yield %v3 : i32
  }
  return %r : i32
}
