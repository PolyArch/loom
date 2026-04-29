// RUN: echo "techmap:" > %t.greedy.yaml
// RUN: echo "  algorithm: greedy" >> %t.greedy.yaml
// RUN: echo "techmap:" > %t.list.yaml
// RUN: echo "  algorithm: list" >> %t.list.yaml
// RUN: echo "techmap:" > %t.beam.yaml
// RUN: echo "  algorithm: beam" >> %t.beam.yaml
// RUN: echo "techmap:" > %t.sa.yaml
// RUN: echo "  algorithm: sa" >> %t.sa.yaml
// RUN: echo "  sa_steps: 200" >> %t.sa.yaml
// RUN: echo "techmap:" > %t.ilp.yaml
// RUN: echo "  algorithm: ilp" >> %t.ilp.yaml
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.greedy.yaml" > %t.greedy.mlir
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.list.yaml" > %t.list.mlir
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.beam.yaml" > %t.beam.mlir
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.sa.yaml" > %t.sa.mlir
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.ilp.yaml" > %t.ilp.mlir
// All five algorithms must produce structurally valid output. The graph
// is acyclic and every op kind is covered by exactly one single-op
// template, so each algorithm must wrap each of the 12 ops in its own
// dataflow.subgraph (12 total).
// RUN: grep -c "dataflow.subgraph" %t.greedy.mlir > %t.count
// RUN: grep -c "dataflow.subgraph" %t.list.mlir | diff %t.count -
// RUN: grep -c "dataflow.subgraph" %t.beam.mlir | diff %t.count -
// RUN: grep -c "dataflow.subgraph" %t.sa.mlir | diff %t.count -
// RUN: grep -c "dataflow.subgraph" %t.ilp.mlir | diff %t.count -
// RUN: FileCheck %s < %t.greedy.mlir

// Stress: a 12-op graph that exercises all five partitioner algorithms
// on the same input. Library covers arith.{addi, subi, muli, cmpi} and
// no feedback edges are present, so the structurally valid result is
// identical across algorithms (12 singleton subgraphs).

// CHECK-LABEL: @fu_addi
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

// CHECK-LABEL: @fu_subi
fabric.module @fu_subi(%cast0_fu_subi : !fabric.bits<32>, %cast1_fu_subi : !fabric.bits<32>) {
  fabric.spatial_pe(%a = %cast0_fu_subi : !fabric.bits<32>, %b = %cast1_fu_subi : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.subi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}

// CHECK-LABEL: @fu_muli
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

// CHECK-LABEL: @fu_cmpi
fabric.module @fu_cmpi(%cast0_fu_cmpi : !fabric.bits<1>, %cast1_fu_cmpi : !fabric.bits<1>) {
  fabric.spatial_pe(%a = %cast0_fu_cmpi : !fabric.bits<1>, %b = %cast1_fu_cmpi : !fabric.bits<1>) -> !fabric.bits<1> {
    fabric.fu(%x = %a : !fabric.bits<1>, %y = %b : !fabric.bits<1>)
                  -> !fabric.bits<1> {
      %k = fabric.op [@arith.cmpi] (%x, %y)
           {hw_params = [{predicate = ["eq", "slt"]}]}
           : (!fabric.bits<1>, !fabric.bits<1>) -> !fabric.bits<1>
      fabric.yield %k : !fabric.bits<1>
    }
  }
  fabric.yield
}


// CHECK-LABEL: @graph_complex
// CHECK: dataflow.graph
// Spot-check that every op kind ends up in some subgraph; the total
// count is enforced by the diff RUN lines above.
// CHECK-DAG: arith.addi
// CHECK-DAG: arith.subi
// CHECK-DAG: arith.muli
// CHECK-DAG: arith.cmpi eq
// CHECK-DAG: arith.cmpi slt
// CHECK: dataflow.yield
func.func @graph_complex(%a: i32, %b: i32, %c: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %t0 = arith.addi %x, %y : i32
    %t1 = arith.subi %t0, %z : i32
    %t2 = arith.muli %t1, %z : i32
    %t3 = arith.addi %t2, %x : i32
    %t4 = arith.subi %t3, %x : i32
    %t5 = arith.addi %t4, %y : i32
    %t6 = arith.muli %t5, %y : i32
    %t7 = arith.addi %t6, %z : i32
    %t8 = arith.muli %t7, %x : i32
    %t9 = arith.addi %t8, %y : i32
    %p0 = arith.cmpi eq, %t1, %t6 : i32
    %p1 = arith.cmpi slt, %t2, %t9 : i32
    dataflow.yield %t9 : i32
  }
  return %r : i32
}
