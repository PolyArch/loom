// RUN: loom-coverage-test %s --project-first-encoding | FileCheck %s

// CHECK: projection=failed
// CHECK: error='hw_params' must not mix normalized modes and legacy fields

fabric.module @projector_mixed_preflight(%a : !fabric.bits<32>,
                                         %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                      %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    %r = fabric.fu(%x = %pa : !fabric.bits<32>,
                   %y = %pb : !fabric.bits<32>) -> !fabric.bits<32>
        attributes {valid_encodings = [{outputs = [0 : i32], resources = [
          {resource = 0 : i32, mode = 0 : i32}
        ]}]} {
      %v = fabric.op [@arith.addi] (%x, %y)
           {hw_params = [
             {op = @arith.addi, function_type = (i32, i32) -> i32,
              input_ports = [0 : i32, 1 : i32],
              output_ports = [0 : i32], attributes = {}},
             {}
           ]}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}
