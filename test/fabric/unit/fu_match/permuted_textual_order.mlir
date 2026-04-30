// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// The principle for matching is graph isomorphism: a fabric.fu may accept
// a user dataflow.subgraph iff some configuration of the FU produces an
// internal compute graph that is isomorphic to the user subgraph. The
// matcher is now VF2-based, so commutativity-preserving operand
// permutations and any other textual reordering of an isomorphic DAG
// match the canonical FU compute. This test pins that behavior.

fabric.module @hw_muladd(%a : !fabric.bits<32>, %b : !fabric.bits<32>, %c : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>,
                    %pc = %c : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>,
              %y = %pb : !fabric.bits<32>,
              %z = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %m = fabric.op [@arith.muli] (%x, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %s = fabric.op [@arith.addi] (%m, %z)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %s : !fabric.bits<32>
    }
  }
  fabric.yield
}

// Canonical pattern: same op order and same operand order as the FU body.
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
// swapped). Under graph isomorphism this is equivalent to the canonical
// pattern, so the VF2 matcher binds it to @hw_muladd.
// CHECK-LABEL: @pat_addi_operands_swapped
func.func @pat_addi_operands_swapped(%a: i32, %b: i32, %c: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.matched_fu = "@hw_muladd#0"
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32
       attributes {loom.is_pattern} {
    %m = arith.muli %x, %y : i32
    %s = arith.addi %z, %m : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

// Permuted pattern: same DAG with the muli operands swapped. Same gap as
// above; the VF2 matcher accepts.
// CHECK-LABEL: @pat_muli_operands_swapped
func.func @pat_muli_operands_swapped(%a: i32, %b: i32, %c: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.matched_fu = "@hw_muladd#0"
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32
       attributes {loom.is_pattern} {
    %m = arith.muli %y, %x : i32
    %s = arith.addi %m, %z : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

// Block-arg permutation: the user wires the third arg to the muli (instead
// of the second) and the second arg to the addi. Block-arg permutation is
// a structural isomorphism, so the matcher accepts.
// CHECK-LABEL: @pat_blockarg_permuted
func.func @pat_blockarg_permuted(%a: i32, %b: i32, %c: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.matched_fu = "@hw_muladd#0"
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32
       attributes {loom.is_pattern} {
    %m = arith.muli %x, %z : i32
    %s = arith.addi %m, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
