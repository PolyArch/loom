// RUN: loom-alignment-test %s | FileCheck %s

// One arith.addi feeding the yield. The single yield anchor must point
// at the addi's only result; the signature must record the arith.addi/
// arith.subi share group (index 0 in the canonical table) and bit-width
// 32.

func.func @addi_one(%a: i32, %b: i32) -> i32 {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

// CHECK: func @addi_one:
// CHECK-NEXT:   yield-anchors=1
// CHECK-NEXT:   anchor[0]=BodyOp:arith.addi#0
// CHECK-NEXT:   signature[0]=arith.addi;0;bw=32;arity=2;ohash=0x{{[0-9a-f]+}}
// CHECK-NEXT:   backedges=0
