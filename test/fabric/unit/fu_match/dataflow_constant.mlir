// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// FU contains a fabric.op with a fixed singleton arith op (arith.muli),
// no hw_params or sw_configs. The matcher pins the bare-op-identity
// matching path: a pattern wrapping arith.muli must match this FU, and
// a pattern wrapping any other op (e.g. arith.addi) must not.
//
// (Originally this slot held a dataflow.constant matcher test, but
// dataflow.constant requires a bits<0> control input which is
// incompatible with the spatial_pe uniform-W rule. The const_hex_value
// path is exercised at the IR level by op/valid.mlir; the matcher
// allowed-set logic is exercised by cmpi_predicate_match.mlir.)

fabric.module @hw_muli_only {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %b = builtin.unrealized_conversion_cast to !fabric.bits<32>
  fabric.spatial_pe(%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)
                  -> !fabric.bits<32> {
      %k = fabric.op [@arith.muli] (%x, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

// CHECK-LABEL: @pat_muli
func.func @pat_muli(%x: i32, %y: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.matched_fu = "@hw_muli_only#0"
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32) -> i32
       attributes {loom.is_pattern} {
    %k = arith.muli %a, %b : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}

// addi pattern -> unmatched (FU only supports muli).
// CHECK-LABEL: @pat_addi_unmatched
func.func @pat_addi_unmatched(%x: i32, %y: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.unmatched
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32) -> i32
       attributes {loom.is_pattern} {
    %k = arith.addi %a, %b : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}
