// RUN: loom-alignment-test %s | FileCheck %s

// dataflow.subgraph with one graph-region back-edge: dataflow.carry
// consumes %nxt, which is produced by an arith.addi *after* the carry
// in body program order. The Alignment facade must classify that single
// operand consumption as a back-edge; backedges=1.

func.func @scc_carry(%c: i1, %i: i32, %d: i32) -> i32 {
  %r = dataflow.subgraph(%cn = %c : i1, %in = %i : i32, %dn = %d : i32) -> i32 {
    %cur = dataflow.carry %cn, %in, %nxt : i32
    %nxt = arith.addi %cur, %dn : i32
    dataflow.yield %cur : i32
  }
  return %r : i32
}

// The yield anchor is the carry itself (no back-edge promotion: the
// terminator is positioned after every body op). The single back-edge
// is the carry's `%nxt` operand.

// CHECK: func @scc_carry:
// CHECK-NEXT:   yield-anchors=1
// CHECK-NEXT:   anchor[0]=BodyOp:dataflow.carry#0
// CHECK-NEXT:   signature[0]=dataflow.carry;-;bw=32;arity=3;ohash=0x{{[0-9a-f]+}}
// CHECK-NEXT:   backedges=1
