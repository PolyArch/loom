// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// The principle for matching is graph isomorphism: a fabric.fu may accept a
// user dataflow.subgraph iff some configuration of the FU produces an
// internal compute graph isomorphic to the user subgraph. Today's matcher
// (SubgraphMatcher::subgraphsStructurallyEqual) is a weaker check: it
// numbers SSA values in textual program order and compares operand
// references position-by-position. That means commutativity-preserving
// operand permutations (and other textual reorderings of an isomorphic
// DAG) currently fail to match even though the underlying graphs are
// equivalent. This test pins the present behavior.
//
// TODO: matcher is sequence-sensitive and does not yet do real graph
// isomorphism; this test pins the current limitation, update when the
// matcher is generalized.

func.func @hw_muladd(%a: !fabric.bits<32>, %b: !fabric.bits<32>,
                     %c: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>,
                 %y = %b : !fabric.bits<32>,
                 %z = %c : !fabric.bits<32>) -> !fabric.bits<32> {
    %m = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %s = fabric.op [@arith.addi] (%m, %z)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %s : !fabric.bits<32>
  }
  return
}

// Canonical pattern: same op order and same operand order as the FU body.
// The matcher accepts this and binds it to @hw_muladd.
// CHECK-LABEL: @pat_canonical
func.func @pat_canonical(%a: i32, %b: i32, %c: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.matched_fu = "@hw_muladd#0"
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32
       attributes {loom.is_pattern} {
    %m = arith.muli %x, %y : i32
    %s = arith.addi %m, %z : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

// Permuted pattern: same DAG up to commutativity of arith.addi (operands
// of the addi are swapped). Under graph isomorphism this is equivalent to
// the canonical pattern, so a real isomorphism-based matcher would still
// bind it to @hw_muladd. The current sequence-sensitive matcher rejects
// it because the operand positions disagree. We lock that present
// behavior here with CHECK-NOT on a successful match plus a positive
// check on the loom.unmatched diagnostic.
// CHECK-LABEL: @pat_addi_operands_swapped
func.func @pat_addi_operands_swapped(%a: i32, %b: i32, %c: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.unmatched
  // CHECK-NOT: loom.matched_fu
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32
       attributes {loom.is_pattern} {
    %m = arith.muli %x, %y : i32
    %s = arith.addi %z, %m : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

// Permuted pattern: same DAG with the muli operands swapped. Same gap.
// CHECK-LABEL: @pat_muli_operands_swapped
func.func @pat_muli_operands_swapped(%a: i32, %b: i32, %c: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.unmatched
  // CHECK-NOT: loom.matched_fu
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32
       attributes {loom.is_pattern} {
    %m = arith.muli %y, %x : i32
    %s = arith.addi %m, %z : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
