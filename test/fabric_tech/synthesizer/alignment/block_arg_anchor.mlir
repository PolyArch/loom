// RUN: loom-alignment-test %s | FileCheck %s

// The yield directly forwards a block argument. The anchor must be
// classified BlockArg with the correct argIndex (the second
// subgraph operand, so argIndex=1).

func.func @passthrough(%a: i32, %b: i32) -> i32 {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    dataflow.yield %y : i32
  }
  return %r : i32
}

// CHECK: func @passthrough:
// CHECK-NEXT:   yield-anchors=1
// CHECK-NEXT:   anchor[0]=BlockArg:#1
// CHECK-NEXT:   signature[0]=-;-;bw=0;arity=0;ohash=0x{{[0-9a-f]+}}
// CHECK-NEXT:   backedges=0
