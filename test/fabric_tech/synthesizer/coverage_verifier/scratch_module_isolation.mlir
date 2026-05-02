// RUN: loom-coverage-test %s --check-isolation | FileCheck %s

// The verifier must perform candidate enumeration in a scratch
// ModuleOp; the user's module must not gain extra func.funcs (and
// in particular must not gain any "candidate*" prefixed symbols).
// This test asserts both the count of user funcs after verify ==
// the count before (only the 2 input func.funcs; the FU host is now
// a fabric.module, not a func.func) and that no `candidate_*` func
// leaks into the user module.

fabric.module @fu_addi_subi(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)
              -> !fabric.bits<32> {
      %s = fabric.op [@arith.addi, @arith.subi] (%x, %y)
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

func.func @input_subi(%a: i32, %b: i32) -> i32
    attributes {loom.coverage_input = true} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.subi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

// CHECK: all_covered=true
// CHECK-NEXT: user_funcs_after=2
// CHECK-NEXT: candidate_in_user=false
