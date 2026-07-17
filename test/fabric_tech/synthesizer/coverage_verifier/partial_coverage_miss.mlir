// RUN: loom-coverage-test %s | FileCheck %s

// One fabric.fu with addi only; two inputs (addi + subi). The addi
// matches; the subi misses. all_covered must be false.

fabric.module @fu_addi_only(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)
              -> !fabric.bits<32>
        attributes {valid_encodings = [{outputs = [0 : i32], resources = [
          {resource = 0 : i32, mode = 0 : i32}
        ]}]} {
      %s = fabric.op [@arith.addi] (%x, %y)
           {hw_params = [{op = @arith.addi,
             function_type = (i32, i32) -> i32,
             input_ports = [0 : i32, 1 : i32], output_ports = [0 : i32],
             attributes = {overflowFlags = #arith.overflow<none>}}]}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %s : !fabric.bits<32>
    }
  }
  fabric.yield
}

func.func @input_addi(%a: i32, %b: i32) -> i32
    attributes {loom.coverage_input = true} {
  %s = arith.addi %a, %b : i32
  return %s : i32
}

func.func @input_subi(%a: i32, %b: i32) -> i32
    attributes {loom.coverage_input = true} {
  %s = arith.subi %a, %b : i32
  return %s : i32
}

// CHECK: coverage[0] funcname=input_addi matched=true index=0
// CHECK-NEXT: coverage[1] funcname=input_subi matched=false index=none
// CHECK-NEXT: all_covered=false
