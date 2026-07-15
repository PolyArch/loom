// RUN: loom %s -verify-diagnostics

fabric.module @duplicate_configured_function(%a : !fabric.bits<32>,
                                             %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                      %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    // expected-error @+1 {{valid semantic encodings #0 and #1 project to isomorphic configured functions}}
    %r = fabric.fu(%x = %pa : !fabric.bits<32>,
                   %y = %pb : !fabric.bits<32>) -> !fabric.bits<32>
        attributes {valid_encodings = [
          {outputs = [0 : i32], resources = [
            {resource = 0 : i32, select = 0 : i32},
            {resource = 1 : i32, select = 0 : i32},
            {resource = 2 : i32, mode = 0 : i32},
            {resource = 4 : i32, select = 0 : i32}
          ]},
          {outputs = [0 : i32], resources = [
            {resource = 0 : i32, select = 1 : i32},
            {resource = 1 : i32, select = 1 : i32},
            {resource = 3 : i32, mode = 0 : i32},
            {resource = 4 : i32, select = 1 : i32}
          ]}
        ]} {
      %xa:2 = fabric.demux %x : !fabric.bits<32> -> 2
      %yb:2 = fabric.demux %y : !fabric.bits<32> -> 2
      %v0 = fabric.op [@arith.addi] (%xa#0, %yb#0)
            {hw_params = [{op = @arith.addi,
              function_type = (i32, i32) -> i32,
              input_ports = [0 : i32, 1 : i32],
              output_ports = [0 : i32], attributes = {}}]}
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %v1 = fabric.op [@arith.addi] (%xa#1, %yb#1)
            {hw_params = [{op = @arith.addi,
              function_type = (i32, i32) -> i32,
              input_ports = [0 : i32, 1 : i32],
              output_ports = [0 : i32], attributes = {}}]}
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %out = fabric.mux %v0, %v1 : !fabric.bits<32>
      fabric.yield %out : !fabric.bits<32>
    }
  }
  fabric.yield
}
