// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// FU contains a fabric.op[@dataflow.constant] whose hardware supports two
// hex constants. The pattern asks for one of them and should match.

func.func @hw_const(%ctrl: !fabric.bits<0>) {
  %r = fabric.fu(%c = %ctrl : !fabric.bits<0>) -> !fabric.bits<32> {
    %k = fabric.op [@dataflow.constant] (%c)
         {hw_params = [{const_hex_value = ["0xdeadbeef", "0xcafebabe"]}]}
         : (!fabric.bits<0>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  return
}

// CHECK-LABEL: @pat_const_dead
func.func @pat_const_dead(%c: none) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.match_config = "op#0{const_hex_value=0xdeadbeef}"
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
