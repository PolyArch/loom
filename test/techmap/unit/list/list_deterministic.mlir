// Two runs with different worker thread counts must produce byte-identical
// output IR under the list-priority partitioner. This guards both the
// candidate cache (already covered for dump output) and the partitioner's
// serial search (which must not depend on candidate-cache iteration order
// or priority-queue tie-breaking).
//
// The temporary YAML config files override `algorithm` to `list` and the
// `threads` knob; cost weights stay at defaults.

// RUN: echo "techmap:" > %t.cfg1.yaml
// RUN: echo "  algorithm: list" >> %t.cfg1.yaml
// RUN: echo "  threads: 1" >> %t.cfg1.yaml
// RUN: echo "techmap:" > %t.cfg4.yaml
// RUN: echo "  algorithm: list" >> %t.cfg4.yaml
// RUN: echo "  threads: 4" >> %t.cfg4.yaml
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.cfg1.yaml" > %t.t1.mlir
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.cfg4.yaml" > %t.t4.mlir
// RUN: diff %t.t1.mlir %t.t4.mlir
// RUN: FileCheck %s < %t.t1.mlir

// Two FUs in the library: one for arith.addi, one for arith.muli. The
// graph is a deterministic mixed chain so the partitioner has multiple
// per-op decisions to make. With single-op templates the expected
// partition is one subgraph per op.

fabric.module @fu_addi {
  %cast0_fu_addi = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %cast1_fu_addi = builtin.unrealized_conversion_cast to !fabric.bits<32>
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


fabric.module @fu_muli {
  %cast0_fu_muli = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %cast1_fu_muli = builtin.unrealized_conversion_cast to !fabric.bits<32>
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
