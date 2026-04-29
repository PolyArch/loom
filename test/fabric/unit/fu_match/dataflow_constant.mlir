// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// FU contains a fabric.op[@dataflow.constant] whose hardware supports two
// hex constants. The pattern asks for one of them and should match.
//
// dataflow.constant has a !fabric.bits<0> control input (none token). The
// enclosing spatial_pe runs at uniform bits<32>, so the FU declares its
// inner block-arg as bits<0> via the `to` clause. The PE -> FU boundary
// drops the high (32 - 0) = 32 bits of the carrier on each token.

fabric.module @hw_const(%ctrl : !fabric.bits<32>) {
  fabric.spatial_pe(%pctrl = %ctrl : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%c = %pctrl : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
      %k = fabric.op [@dataflow.constant] (%c)
           {hw_params = [{const_hex_value = ["0xdeadbeef", "0xcafebabe"]}]}
           : (!fabric.bits<0>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

// CHECK-LABEL: @pat_const_dead
func.func @pat_const_dead(%c: none) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.match_config = "op#0{const_hex_value=0xdeadbeef}"
  // CHECK-SAME: loom.matched_fu = "@hw_const#0"
  %r = dataflow.subgraph(%cc = %c : none) -> i32
       attributes {loom.is_pattern} {
    %k = dataflow.constant %cc {const_value = 3735928559 : i32} : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}

// CHECK-LABEL: @pat_const_cafe
func.func @pat_const_cafe(%c: none) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.match_config = "op#0{const_hex_value=0xcafebabe}"
  // CHECK-SAME: loom.matched_fu = "@hw_const#0"
  %r = dataflow.subgraph(%cc = %c : none) -> i32
       attributes {loom.is_pattern} {
    %k = dataflow.constant %cc {const_value = 3405691582 : i32} : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}

// Constant outside the hw set -> unmatched.
// CHECK-LABEL: @pat_const_unsupported
func.func @pat_const_unsupported(%c: none) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.unmatched
  %r = dataflow.subgraph(%cc = %c : none) -> i32
       attributes {loom.is_pattern} {
    %k = dataflow.constant %cc {const_value = 1 : i32} : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}
