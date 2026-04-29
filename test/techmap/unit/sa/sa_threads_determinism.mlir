// Same input, same SA seed, different worker thread counts must produce
// byte-identical output IR. The candidate cache is the only multi-threaded
// component in the pipeline; the SA loop itself is strictly serial.
//
// The temporary YAML configs override only the `threads` knob; the SA
// algorithm and seed are pinned so the only varying axis is thread count.

// RUN: echo "techmap:" > %t.cfg1.yaml
// RUN: echo "  algorithm: sa" >> %t.cfg1.yaml
// RUN: echo "  sa_steps: 200" >> %t.cfg1.yaml
// RUN: echo "  sa_seed: 1234" >> %t.cfg1.yaml
// RUN: echo "  threads: 1" >> %t.cfg1.yaml
// RUN: echo "techmap:" > %t.cfg4.yaml
// RUN: echo "  algorithm: sa" >> %t.cfg4.yaml
// RUN: echo "  sa_steps: 200" >> %t.cfg4.yaml
// RUN: echo "  sa_seed: 1234" >> %t.cfg4.yaml
// RUN: echo "  threads: 4" >> %t.cfg4.yaml
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.cfg1.yaml" > %t.t1.mlir
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.cfg4.yaml" > %t.t4.mlir
// RUN: diff %t.t1.mlir %t.t4.mlir
// RUN: FileCheck %s < %t.t1.mlir

fabric.module @fu_addi(%cast0_fu_addi : !fabric.bits<32>, %cast1_fu_addi : !fabric.bits<32>) {
  fabric.spatial_pe(%a = %cast0_fu_addi : !fabric.bits<32>, %b = %cast1_fu_addi : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}


fabric.module @fu_muli(%cast0_fu_muli : !fabric.bits<32>, %cast1_fu_muli : !fabric.bits<32>) {
  fabric.spatial_pe(%a = %cast0_fu_muli : !fabric.bits<32>, %b = %cast1_fu_muli : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
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
