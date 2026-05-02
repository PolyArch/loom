// RUN: loom-coverage-test %s --config %p/parallel_match.yaml | FileCheck %s

// Same fabric.fu and inputs as `multiple_inputs_all_match` but the
// SynthConfig forces parallel_match: true with workers=4. The output
// must be identical to the serial run -- that is, both inputs match
// and `all_covered=true`. Index order is unconstrained because the
// candidate-enumerator order does not change under parallel matching;
// the per-input slot writes are still index-stable.

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
