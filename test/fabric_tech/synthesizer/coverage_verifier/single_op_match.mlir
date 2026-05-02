// RUN: loom-coverage-test %s | FileCheck %s

// One fabric.fu containing fabric.op[@arith.addi] of width 32; one
// input subgraph that's literally `arith.addi i32`. The verifier
// enumerates the FU into a single candidate (the fixed addi) and
// reports a match for the lone input.

fabric.module @fu_addi(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)
              -> !fabric.bits<32> {
      %s = fabric.op [@arith.addi] (%x, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %s : !fabric.bits<32>
    }
  }
  fabric.yield
}

func.func @input_addi(%a: i32, %b: i32) -> i32
    attributes {loom.coverage_input = true} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

// CHECK: coverage[0] funcname=input_addi matched=true index=0
// CHECK-NEXT: all_covered=true
