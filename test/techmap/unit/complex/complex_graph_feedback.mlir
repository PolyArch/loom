// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s
// RUN: echo "techmap:" > %t.ilp.yaml
// RUN: echo "  algorithm: ilp" >> %t.ilp.yaml
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.ilp.yaml" \
// RUN:   2> %t.ilp.diag > %t.ilp.mlir
// RUN: FileCheck --check-prefix=ILP %s < %t.ilp.mlir
// RUN: FileCheck --check-prefix=ILPDIAG %s < %t.ilp.diag

// Stress: a graph with a self-feedback loop. dataflow.carry's third
// operand is the result of an arith.addi which itself consumes carry's
// output. Graph-region semantics permit this intra-block self reference.
// The partitioner library here covers dataflow.carry but does NOT cover
// arith.addi when keeping it inside a subgraph would create an
// inter-block cycle (the addi feeds back into carry in the next block).
//
// Expected current behavior (greedy default): wrap carry in a singleton
// subgraph, leave the feedback addi at graph level so the loop closes
// inside the graph block, not across two subgraph blocks.
//
// The ILP partitioner's MIP has no acyclicity constraint, so its raw
// optimum would wrap both ops into mutually-referencing subgraphs. A
// post-solve cycle-repair pass detects the multi-block SCC and demotes
// one block to graph level; the demotion order prefers to keep the
// "cheaper" template, so for this input the carry block ends up
// demoted while the addi block remains bound. The exact victim differs
// from greedy, but both outputs avoid any multi-block SSA cycle.
//
// Note: the brief originally asked for a self-feedback chain using a
// single arith.addi; that requires graph-region forward-reference of
// the addi's own SSA value, which arith.addi cannot express by itself
// (its result must already exist when the op consumes it as an operand).
// dataflow.carry is the canonical loop-closing primitive and is what
// graph-region tests use (see test/dataflow/unit/graph/valid.mlir
// @graph_self_feedback).

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

// CHECK-LABEL: @fu_carry
func.func @fu_carry(%cond: !fabric.bits<1>,
                    %init: !fabric.bits<32>,
                    %carry: !fabric.bits<32>) {
  %r = fabric.fu(%c = %cond : !fabric.bits<1>,
                 %i = %init : !fabric.bits<32>,
                 %k = %carry : !fabric.bits<32>) -> !fabric.bits<32> {
    %o = fabric.op [@dataflow.carry] (%c, %i, %k)
         : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>)
           -> !fabric.bits<32>
    fabric.yield %o : !fabric.bits<32>
  }
  return
}

// Lock current behavior: carry wrapped, feedback addi stays at graph
// level so the carry+addi cycle is contained in a single graph block.
// Greedy must NOT also wrap %next, otherwise the partition would have
// two subgraph blocks referencing each other (a 2-block cycle).
// CHECK-LABEL: @graph_feedback
// CHECK: dataflow.graph
// CHECK: dataflow.subgraph
// CHECK-NEXT: dataflow.carry
// CHECK-NEXT: dataflow.yield
// CHECK: arith.addi
// CHECK-NOT: dataflow.subgraph
// CHECK: dataflow.yield
func.func @graph_feedback(%cond: i1, %init: i32, %step: i32) -> i32 {
  %r = dataflow.graph(%c = %cond : i1, %i = %init : i32, %s = %step : i32) -> i32 {
    %acc = dataflow.carry %c, %i, %next : i32
    %next = arith.addi %acc, %s : i32
    dataflow.yield %acc : i32
  }
  return %r : i32
}

// ILP after cycle-repair must also be acyclic at the block level.
// On this input the post-solve pass keeps the addi block bound and
// demotes the carry block to graph level. The order may differ from
// greedy, but the result must contain exactly one dataflow.subgraph
// (the addi block) and the carry must remain at graph level so the
// feedback edge closes inside the graph body.
// ILP-LABEL: @graph_feedback
// ILP: dataflow.graph
// ILP: dataflow.carry
// ILP: dataflow.subgraph
// ILP-NEXT: arith.addi
// ILP-NEXT: dataflow.yield
// ILP: dataflow.yield
// ILP-NOT: dataflow.subgraph

// ILPDIAG: warning: loom-ilp-partitioner: HiGHS solution induced a multi-block SSA cycle
// ILPDIAG-SAME: demoting block(s) to graph level to break the cycle
