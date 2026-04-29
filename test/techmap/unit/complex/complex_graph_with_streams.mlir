// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Stress: graph mixes multi-result ops (dataflow.stream, dataflow.gate)
// with subgraph-allowed ops (dataflow.invariant, arith.addi). Both
// stream and gate are in fabric.op's allowlist BUT no FU template here
// covers them, and additionally they have multi-result body roots that
// the single-output graph matcher leaves alone. Expected: stream + gate
// stay at graph level; invariant + arith.addi each get wrapped.

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


// CHECK-LABEL: @fu_invariant
fabric.module @fu_invariant(%c : !fabric.bits<1>, %v : !fabric.bits<1>) {
  fabric.spatial_pe(%pc = %c : !fabric.bits<1>,
                    %pv = %v : !fabric.bits<1>) -> !fabric.bits<1> {
    fabric.fu(%cn = %pc : !fabric.bits<1>,
              %vn = %pv : !fabric.bits<1>) -> !fabric.bits<1> {
      %o = fabric.op [@dataflow.invariant] (%cn, %vn)
           : (!fabric.bits<1>, !fabric.bits<1>) -> !fabric.bits<1>
      fabric.yield %o : !fabric.bits<1>
    }
  }
  fabric.yield
}

// CHECK-LABEL: @graph_streams
// CHECK: dataflow.graph
// dataflow.stream is left at graph level (no covering template).
// CHECK: dataflow.stream
// dataflow.invariant gets wrapped.
// CHECK: dataflow.subgraph
// CHECK-NEXT: dataflow.invariant
// CHECK-NEXT: dataflow.yield
// Two consecutive arith.addi each get their own subgraph.
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.addi
// CHECK-NEXT: dataflow.yield
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.addi
// CHECK-NEXT: dataflow.yield
// dataflow.gate remains at graph level.
// CHECK: dataflow.gate
// CHECK-NOT: dataflow.subgraph
// CHECK: dataflow.yield
func.func @graph_streams(%lb: i32, %ub: i32, %step: i32, %cond: i1, %k: i32) -> i32 {
  %r = dataflow.graph(%l = %lb : i32, %u = %ub : i32, %s = %step : i32,
                      %c = %cond : i1, %kk = %k : i32) -> i32 {
    %idx, %rwc = dataflow.stream %l, %u, %s {step_op = "+=", cont_cond = "<"} : i32
    %inv = dataflow.invariant %c, %kk : i32
    %t1 = arith.addi %idx, %inv : i32
    %t2 = arith.addi %t1, %inv : i32
    %gc, %gv = dataflow.gate %rwc, %t2 : i32
    dataflow.yield %gv : i32
  }
  return %r : i32
}
