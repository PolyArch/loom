// RUN: loom-coverage-test %s | FileCheck %s

fabric.module @sync_then_gate(
    %ctrl: !fabric.bits<64>, %value: !fabric.bits<64>) {
  fabric.pe [spatial] (
      %pctrl = %ctrl : !fabric.bits<64>,
      %pvalue = %value : !fabric.bits<64>)
      -> (!fabric.bits<64>, !fabric.bits<64>) {
    %r:2 = fabric.fu(
        %xctrl = %pctrl : !fabric.bits<64>,
        %xvalue = %pvalue : !fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>)
        attributes {valid_encodings = [{
          outputs = [0 : i32, 1 : i32],
          resources = [
            {resource = 0 : i32, mode = 0 : i32},
            {resource = 1 : i32, mode = 0 : i32}
          ]
        }]} {
      %synced = fabric.op [@dataflow.sync] (%xvalue) {
        hw_params = [{
          op = @dataflow.sync,
          function_type = (i32) -> (i32),
          input_ports = [0 : i32], output_ports = [0 : i32], attributes = {}
        }],
        paired_lanes = [
          {input_port = 0 : i32, output_port = 0 : i32, mask_bit = 0 : i32}
        ]
      } : (!fabric.bits<64>) -> !fabric.bits<64>
      %phase, %gated = fabric.op [@dataflow.gate] (%xctrl, %synced) {
        hw_params = [{
          op = @dataflow.gate,
          function_type = (i1, i32) -> (i1, i32),
          input_ports = [0 : i32, 1 : i32],
          output_ports = [0 : i32, 1 : i32], attributes = {}
        }]
      } : (!fabric.bits<64>, !fabric.bits<64>)
          -> (!fabric.bits<64>, !fabric.bits<64>)
      fabric.yield %phase, %gated : !fabric.bits<64>, !fabric.bits<64>
    }
  }
  fabric.yield
}

func.func @gate_sibling_is_not_covered(%ctrl: i1, %value: i32) -> i32
    attributes {loom.coverage_input = true} {
  %synced = dataflow.sync %value : (i32) -> (i32)
  %phase, %gated = dataflow.gate %ctrl, %synced : i32
  return %gated : i32
}

// CHECK: coverage[0] funcname=gate_sibling_is_not_covered matched=false index=none
// CHECK-NEXT: all_covered=false
