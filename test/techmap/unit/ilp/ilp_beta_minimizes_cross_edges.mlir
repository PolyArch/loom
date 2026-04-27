// The cost-model beta term penalizes inter-block SSA edges. The MIP
// encodes one e[j, k] variable per cross edge with coefficient beta in
// the objective and the family of constraints that forces e[j,k] = 1
// whenever j and k end up in different blocks (or either is at graph
// level). With alpha = 0 the |blocks_with_template| term is neutralized,
// leaving beta as the dominant tie-breaker between two structurally
// admissible partitions:
//   * (muli, addi) fused into a single 2-op subgraph -> 0 cross edges,
//   * muli singleton + addi singleton -> 1 cross edge (muli -> addi).
// With beta = 0 (and alpha = 0) the two partitions tie on cost so the
// MIP returns the one preferred by HiGHS's deterministic tie-break,
// which for this input is the two-singleton partition. With beta = 10
// the fused 2-op partition strictly dominates and the MIP returns it.

// RUN: echo "techmap:" > %t.b0.yaml
// RUN: echo "  algorithm: ilp" >> %t.b0.yaml
// RUN: echo "  alpha: 0" >> %t.b0.yaml
// RUN: echo "  beta: 0" >> %t.b0.yaml
// RUN: echo "  gamma: 0" >> %t.b0.yaml
// RUN: echo "techmap:" > %t.b10.yaml
// RUN: echo "  algorithm: ilp" >> %t.b10.yaml
// RUN: echo "  alpha: 0" >> %t.b10.yaml
// RUN: echo "  beta: 10" >> %t.b10.yaml
// RUN: echo "  gamma: 0" >> %t.b10.yaml
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.b0.yaml" \
// RUN:   | FileCheck --check-prefix=B0 %s
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.b10.yaml" \
// RUN:   | FileCheck --check-prefix=B10 %s

func.func @fu_muli(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  return
}

func.func @fu_addi(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  return
}

func.func @fu_muli_addi(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %m = fabric.op [@arith.addi] (%k, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %m : !fabric.bits<32>
  }
  return
}

// With beta = 0 the cost is flat across all admissible partitions; HiGHS
// breaks the tie by returning the two-singleton solution.
// B0-LABEL: @graph_chain
// B0: dataflow.subgraph
// B0-NEXT: arith.muli
// B0-NEXT: dataflow.yield
// B0: dataflow.subgraph
// B0-NEXT: arith.addi
// B0-NEXT: dataflow.yield

// With beta = 10 the fused 2-op subgraph (0 cross edges) strictly wins.
// B10-LABEL: @graph_chain
// B10: dataflow.subgraph
// B10-NEXT: arith.muli
// B10-NEXT: arith.addi
// B10-NEXT: dataflow.yield
// B10-NOT: dataflow.subgraph
func.func @graph_chain(%a: i32, %b: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32) -> i32 {
    %p = arith.muli %x, %y : i32
    %q = arith.addi %p, %y : i32
    dataflow.yield %q : i32
  }
  return %r : i32
}
