// RUN: loom %s -verify-diagnostics

fabric.module @malformed_op_list(%a : !fabric.bits<32>,
                                 %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                      %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    %r = fabric.fu(%x = %pa : !fabric.bits<32>,
                   %y = %pb : !fabric.bits<32>) -> !fabric.bits<32>
        attributes {valid_encodings = [{outputs = [0 : i32], resources = [
          {resource = 0 : i32, mode = 0 : i32}
        ]}]} {
      // expected-error @+1 {{'op_list' entry #0 must be a flat symbol reference}}
      %v = "fabric.op"(%x, %y) {
        op_list = ["arith.addi"],
        hw_params = [{op = @arith.addi,
          function_type = (i32, i32) -> i32,
          input_ports = [0 : i32, 1 : i32],
          output_ports = [0 : i32],
          attributes = {overflowFlags = #arith.overflow<none>}}]
      } : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}
