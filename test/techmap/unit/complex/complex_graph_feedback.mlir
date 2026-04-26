// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Stress: a graph with a self-feedback loop. dataflow.carry's third
// operand is the result of an arith.addi which itself consumes carry's
// output. Graph-region semantics permit this intra-block self reference.
// The partitioner library here covers dataflow.carry but does NOT cover
// arith.addi when keeping it inside a subgraph would create an
// inter-block cycle (the addi feeds back into carry in the next block).
//
// Expected current behavior (greedy default): wrap carry in a singleton
// subgraph, leave the feedback addi at graph level so the loop closes
// inside the graph block, not across two subgraph blocks (AC-CORR-3).
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
