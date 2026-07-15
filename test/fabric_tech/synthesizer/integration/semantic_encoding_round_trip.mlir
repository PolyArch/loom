// RUN: loom %s -loom-generalize-subgraphs-to-fu='dump-stats=true config=%p/anchor.yaml' 2>&1 | FileCheck %s

// CHECK: synth-stat group=equality
// CHECK-SAME: reason=success
// CHECK-SAME: covered=2/2
// CHECK-SAME: encodings=2
// CHECK-SAME: covered_encodings=2
// CHECK-SAME: extra_capability=0
// CHECK: synth-stat group=strict_superset
// CHECK-SAME: reason=success
// CHECK-SAME: covered=2/2
// CHECK-SAME: encodings=4
// CHECK-SAME: covered_encodings=2
// CHECK-SAME: extra_capability=2
// CHECK: synth-stat group=width_widening
// CHECK-SAME: reason=success
// CHECK-SAME: covered=2/2
// CHECK-SAME: encodings=2
// CHECK-SAME: covered_encodings=2
// CHECK-SAME: extra_capability=0

func.func @equality_add(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "equality"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %v = arith.addi %x, %y : i32
    dataflow.yield %v : i32
  }
  return %r : i32
}

func.func @equality_sub(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "equality"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %v = arith.subi %x, %y : i32
    dataflow.yield %v : i32
  }
  return %r : i32
}

func.func @superset_add_and(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "strict_superset"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %v0 = arith.addi %x, %y : i32
    %v1 = arith.andi %v0, %y : i32
    dataflow.yield %v1 : i32
  }
  return %r : i32
}

func.func @superset_sub_or(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "strict_superset"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %v0 = arith.subi %x, %y : i32
    %v1 = arith.ori %v0, %y : i32
    dataflow.yield %v1 : i32
  }
  return %r : i32
}

func.func @width_i16(%a: i16, %b: i16) -> i16
    attributes {loom.synth_group = "width_widening"} {
  %r = dataflow.subgraph(%x = %a : i16, %y = %b : i16) -> i16 {
    %v0 = arith.addi %x, %y : i16
    %v1 = arith.andi %v0, %y : i16
    dataflow.yield %v1 : i16
  }
  return %r : i16
}

func.func @width_i32(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "width_widening"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %v0 = arith.addi %x, %y : i32
    %v1 = arith.andi %v0, %y : i32
    dataflow.yield %v1 : i32
  }
  return %r : i32
}
