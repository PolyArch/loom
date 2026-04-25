// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// FU offers cmpi with a 3-predicate hardware support set. Patterns ask
// for predicates within and outside the support set.

func.func @hw_cmpi(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<1> {
    %k = fabric.op [@arith.cmpi] (%x, %y)
         {hw_params = [{predicate = ["eq", "slt", "sgt"]}]}
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
    fabric.yield %k : !fabric.bits<1>
  }
  return
}

// CHECK-LABEL: @pat_cmpi_eq
func.func @pat_cmpi_eq(%x: i32, %y: i32) -> i1 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.match_config = "op#0{predicate=eq}"
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32) -> i1
       attributes {loom.is_pattern} {
    %k = arith.cmpi eq, %a, %b : i32
    dataflow.yield %k : i1
  }
  return %r : i1
}

// CHECK-LABEL: @pat_cmpi_slt
func.func @pat_cmpi_slt(%x: i32, %y: i32) -> i1 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.match_config = "op#0{predicate=slt}"
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32) -> i1
       attributes {loom.is_pattern} {
    %k = arith.cmpi slt, %a, %b : i32
    dataflow.yield %k : i1
  }
  return %r : i1
}

// "ne" is NOT in the hw_params predicate set; should not match.
// CHECK-LABEL: @pat_cmpi_ne_unmatched
func.func @pat_cmpi_ne_unmatched(%x: i32, %y: i32) -> i1 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.unmatched
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32) -> i1
       attributes {loom.is_pattern} {
    %k = arith.cmpi ne, %a, %b : i32
    dataflow.yield %k : i1
  }
  return %r : i1
}
