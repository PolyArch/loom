// RUN: loom-alignment-test %s | FileCheck %s

// dataflow.yield with two operands fed by two different body ops.
// Both anchors must be classified BodyOp pointing at their respective
// producers; yield-anchors=2.

func.func @two_yields(%a: i32, %b: i32) -> (i32, i32) {
  %r0, %r1 = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> (i32, i32) {
    %s = arith.addi %x, %y : i32
    %m = arith.muli %x, %y : i32
    dataflow.yield %s, %m : i32, i32
  }
  return %r0, %r1 : i32, i32
}

// CHECK: func @two_yields:
// CHECK-NEXT:   yield-anchors=2
// CHECK-NEXT:   anchor[0]=BodyOp:arith.addi#0
// CHECK-NEXT:   signature[0]=arith.addi;0;bw=32;arity=2;ohash=0x{{[0-9a-f]+}}
// CHECK-NEXT:   anchor[1]=BodyOp:arith.muli#0
// CHECK-NEXT:   signature[1]=arith.muli;-;bw=32;arity=2;ohash=0x{{[0-9a-f]+}}
// CHECK-NEXT:   backedges=0
