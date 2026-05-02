// RUN: loom-alignment-test %s | FileCheck %s

// Two subgraphs differ only in the operand order of the commutative
// arith.addi: one is `%xori, %x`, the other is `%x, %xori`. The
// alignment facade canonicalizes commutative operand kinds, so the
// resulting NodeSignature::structuralHash is identical for both -- this
// is the load-bearing guarantee that synthesis does not see two
// different "anchor signatures" for what is one shared op shape.
//
// The two ops the addi consumes are deliberately different *kinds*
// (BodyOp vs BlockArg) so the canonicalization actually has work to do:
// without sorting operand kinds the two hashes would diverge.

func.func @body_then_arg(%a: i32, %b: i32) -> i32 {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %u = arith.xori %x, %y : i32
    %s = arith.addi %u, %x : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

func.func @arg_then_body(%a: i32, %b: i32) -> i32 {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %u = arith.xori %x, %y : i32
    %s = arith.addi %x, %u : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

// Both anchors have identical signature output (lit captures the
// concrete hash on the first match and re-asserts the same string on
// the second so the structural hash is provably equal).

// CHECK: func @body_then_arg:
// CHECK-NEXT:   yield-anchors=1
// CHECK-NEXT:   anchor[0]=BodyOp:arith.addi#0
// CHECK-NEXT:   signature[0]=[[SIG:arith.addi;0;bw=32;arity=2;ohash=0x[0-9a-f]+]]
// CHECK-NEXT:   backedges=0

// CHECK: func @arg_then_body:
// CHECK-NEXT:   yield-anchors=1
// CHECK-NEXT:   anchor[0]=BodyOp:arith.addi#0
// CHECK-NEXT:   signature[0]=[[SIG]]
// CHECK-NEXT:   backedges=0
