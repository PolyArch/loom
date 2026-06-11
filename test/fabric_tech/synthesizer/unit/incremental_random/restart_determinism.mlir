// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/incremental_random.yaml' 2>/dev/null > %t.run1.mlir
// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/incremental_random.yaml' 2>/dev/null > %t.run2.mlir
// RUN: diff %t.run1.mlir %t.run2.mlir
// RUN: FileCheck %s --input-file=%t.run1.mlir

// Same input, same seed, two consecutive runs of `incremental_random`
// must produce byte-identical wrapper IR. The first two RUN lines
// capture each invocation's stdout (the rewritten module); the diff
// asserts byte equality and the FileCheck on run1 confirms a wrapper
// was synthesized (so the equality check is meaningful).
//
// Incremental-random strategy contract: same `seed` and config produce
// the same chosen FU.

// CHECK-LABEL: fabric.module @fu_det_demo
// CHECK-SAME: loom.synthesized_for = "det_demo"
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.op [@arith.addi, @arith.subi]

func.func @det_pat_add(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "det_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

func.func @det_pat_sub(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "det_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.subi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

func.func @det_pat_add_mul(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "det_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %t = arith.addi %x, %y : i32
    %m = arith.muli %t, %z : i32
    dataflow.yield %m : i32
  }
  return %r : i32
}

func.func @det_pat_sub_mul(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "det_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %t = arith.subi %x, %y : i32
    %m = arith.muli %t, %z : i32
    dataflow.yield %m : i32
  }
  return %r : i32
}
