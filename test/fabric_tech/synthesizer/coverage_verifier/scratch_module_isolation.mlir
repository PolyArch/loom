// RUN: loom-coverage-test %s --check-isolation | FileCheck %s

// Canonical projection must not materialize candidate functions in the user
// module. The legacy tool's isolation check still verifies that property.

fabric.module @fu_addi_subi(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)
              -> !fabric.bits<32>
        attributes {valid_encodings = [
          {outputs = [0 : i32], resources = [{resource = 0 : i32, mode = 0 : i32}]},
          {outputs = [0 : i32], resources = [{resource = 0 : i32, mode = 1 : i32}]}
        ]} {
      %s = fabric.op [@arith.addi, @arith.subi] (%x, %y)
           {hw_params = [
             {op = @arith.addi, function_type = (i32, i32) -> i32,
              input_ports = [0 : i32, 1 : i32], output_ports = [0 : i32],
              attributes = {overflowFlags = #arith.overflow<none>}},
             {op = @arith.subi, function_type = (i32, i32) -> i32,
              input_ports = [0 : i32, 1 : i32], output_ports = [0 : i32],
              attributes = {overflowFlags = #arith.overflow<none>}}
           ]}
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
