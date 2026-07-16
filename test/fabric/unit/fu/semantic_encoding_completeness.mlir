// RUN: loom %s -split-input-file -verify-diagnostics

fabric.module @normalized_requires_encodings(
    %a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                      %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    // expected-error @+1 {{normalized fabric.fu requires non-empty valid_encodings or complete programmed selections}}
    %r = fabric.fu(%x = %pa : !fabric.bits<32>,
                   %y = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %v = fabric.op [@arith.addi] (%x, %y)
           {hw_params = [{op = @arith.addi,
             function_type = (i32, i32) -> i32,
             input_ports = [0 : i32, 1 : i32],
             output_ports = [0 : i32], attributes = {}}]}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----

fabric.module @programmed_normalized_adapter(
    %a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                      %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    %r = fabric.fu(%x = %pa : !fabric.bits<32>,
                   %y = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %v = fabric.op [@arith.subi] (%x, %y)
           {hw_params = [{op = @arith.subi,
             function_type = (i32, i32) -> i32,
             input_ports = [0 : i32, 1 : i32],
             output_ports = [0 : i32], attributes = {}}],
            sw_configs = {mode = 0 : i32}}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----

fabric.module @programmed_adapter_requires_routing_selection(
    %a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                      %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    %r = fabric.fu(%x = %pa : !fabric.bits<32>,
                   %y = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      // expected-error @+1 {{programmed normalized fabric.fu requires an explicit selection for every routing resource}}
      %selected = fabric.mux %x, %y : !fabric.bits<32>
      %v = fabric.op [@arith.addi] (%selected, %y)
           {hw_params = [{op = @arith.addi,
             function_type = (i32, i32) -> i32,
             input_ports = [0 : i32, 1 : i32],
             output_ports = [0 : i32], attributes = {}}],
            sw_configs = {mode = 0 : i32}}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}
