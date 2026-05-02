// RUN: loom-coverage-test %s | FileCheck %s

// One fabric.fu with fabric.op[@arith.addi, @arith.subi] op_list
// (multi-member share group); two inputs (one addi, one subi). The
// enumerator materializes both the addi and the subi candidate, so
// each input matches a distinct candidate.

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

// CHECK: coverage[0] funcname=input_addi matched=true index={{[0-9]+}}
// CHECK-NEXT: coverage[1] funcname=input_subi matched=true index={{[0-9]+}}
// CHECK-NEXT: all_covered=true
