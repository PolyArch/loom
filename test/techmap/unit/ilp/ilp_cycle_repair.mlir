// Targeted unit test for the ILP partitioner's post-solve cycle-repair
// path. The single-op MIP has no acyclicity requirement, so on a graph
// containing SSA feedback its optimum binds both endpoints of the cycle
// into separate single-op subgraphs that mutually reference each other,
// which violates the no-multi-block-cycle invariant.
//
// The post-solve pass detects this multi-block SCC, demotes the
// cycle-participating block whose template id is largest (so the
// "cheaper" template stays bound) to graph level, and emits a
// module-level diagnostic. The result is a partition whose bound
// blocks form a DAG.

// RUN: echo "techmap:" > %t.ilp.yaml
// RUN: echo "  algorithm: ilp" >> %t.ilp.yaml
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.ilp.yaml" \
// RUN:   2> %t.diag > %t.mlir
// RUN: FileCheck %s < %t.mlir
// RUN: FileCheck --check-prefix=DIAG %s < %t.diag

// CHECK-LABEL: @fu_addi
fabric.module @fu_addi(%cast0_fu_addi : !fabric.bits<32>, %cast1_fu_addi : !fabric.bits<32>) {
  fabric.pe [spatial] (%a = %cast0_fu_addi : !fabric.bits<32>, %b = %cast1_fu_addi : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}


// CHECK-LABEL: @fu_carry
fabric.module @fu_carry(%cond : !fabric.bits<1>, %init : !fabric.bits<1>, %carry : !fabric.bits<1>) {
  fabric.pe [spatial] (%pcond = %cond : !fabric.bits<1>,
                    %pinit = %init : !fabric.bits<1>,
                    %pcarry = %carry : !fabric.bits<1>) -> !fabric.bits<1> {
    fabric.fu(%c = %pcond : !fabric.bits<1>,
              %i = %pinit : !fabric.bits<1>,
              %k = %pcarry : !fabric.bits<1>) -> !fabric.bits<1> {
      %o = fabric.op [@dataflow.carry] (%c, %i, %k)
           : (!fabric.bits<1>, !fabric.bits<1>, !fabric.bits<1>)
             -> !fabric.bits<1>
      fabric.yield %o : !fabric.bits<1>
    }
  }
  fabric.yield
}

// The bound result must form a DAG: at most one of {carry, addi} may
// remain wrapped, the other must stay at graph level. The repair
// demotes the cycle-participating block whose template id is largest;
// for this library the carry template is registered after addi so the
// carry block gets demoted and the addi block stays bound.
// CHECK-LABEL: @graph_feedback
// CHECK: dataflow.graph
// CHECK: dataflow.carry
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.addi
// CHECK-NEXT: dataflow.yield
// CHECK: dataflow.yield
// CHECK-NOT: dataflow.subgraph
func.func @graph_feedback(%cond: i1, %init: i32, %step: i32) -> i32 {
  %r = dataflow.graph(%c = %cond : i1, %i = %init : i32, %s = %step : i32) -> i32 {
    %acc = dataflow.carry %c, %i, %next : i32
    %next = arith.addi %acc, %s : i32
    dataflow.yield %acc : i32
  }
  return %r : i32
}

// DIAG: warning: loom-ilp-partitioner: HiGHS solution induced a multi-block SSA cycle
// DIAG-SAME: demoting block(s) to graph level to break the cycle
