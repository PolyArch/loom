// When the graph body exceeds the ILP partitioner's size threshold
// (kILPMaxOps == 200), the ILP partitioner falls back to greedy and
// emits a diagnostic explaining why. Synthesize a 220-op chain via a
// helper python script.

// RUN: echo "techmap:" > %t.ilp.yaml
// RUN: echo "  algorithm: ilp" >> %t.ilp.yaml
// RUN: python3 %S/gen_big_chain.py 220 > %t.big.mlir
// RUN: loom %t.big.mlir -loom-partition-graph-into-subgraphs="config=%t.ilp.yaml" > %t.out.mlir 2> %t.diag
// RUN: FileCheck --check-prefix=DIAG %s < %t.diag
// RUN: FileCheck %s < %t.out.mlir

// DIAG: warning: loom-ilp-partitioner: graph has more than the supported ILP size
// DIAG-SAME: > 200 ops

// Greedy fallback produces 220 singleton subgraphs.
// CHECK-LABEL: @graph_big
// CHECK: dataflow.subgraph
// CHECK: arith.addi
// CHECK: dataflow.subgraph
// CHECK: arith.addi
// CHECK: dataflow.yield
