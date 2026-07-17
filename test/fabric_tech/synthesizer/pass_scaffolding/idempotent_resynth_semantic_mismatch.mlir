// RUN: loom %s -loom-synthesize-configured-functions 2>&1 | FileCheck %s

// The existing wrapper has the expected physical signature but materializes
// only subtraction. It must not satisfy an idempotent request for addition.

// CHECK: warning: {{.*}}group "y": symbol_conflict
// CHECK-SAME: semantic coverage failed
// CHECK-SAME: [semantic-coverage]
// CHECK: func.func @pat_addi
// CHECK-SAME: loom.synth_failed = "symbol_conflict"
// CHECK-NOT: skipping idempotent re-synth

fabric.module @fu_y(%a: !fabric.bits<32>, %b: !fabric.bits<32>)
    attributes {loom.synthesized_for = "y"} {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%aa = %pa : !fabric.bits<32>,
              %bb = %pb : !fabric.bits<32>) -> !fabric.bits<32>
        attributes {valid_encodings = [{outputs = [0 : i32], resources = [
          {resource = 0 : i32, mode = 0 : i32}
        ]}]} {
      %x = fabric.op [@arith.subi] (%aa, %bb)
           {hw_params = [{op = @arith.subi,
             function_type = (i32, i32) -> i32,
             input_ports = [0 : i32, 1 : i32],
             output_ports = [0 : i32],
             attributes = {overflowFlags = #arith.overflow<none>}}]}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %x : !fabric.bits<32>
    }
  }
  fabric.yield
}

func.func @pat_addi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "y"} {
  %sum = arith.addi %a, %b : i32
  return %sum : i32
}
