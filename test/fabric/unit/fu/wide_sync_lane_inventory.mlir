// RUN: loom %s -verify-diagnostics

module {

func.func @incomplete_inventory(
    %a: !fabric.bits<32>, %b: !fabric.bits<32>,
    %c: !fabric.bits<32>, %d: !fabric.bits<32>) {
  // expected-error @+1 {{paired_lanes must cover the complete physical signature}}
  %r:4 = fabric.op [@dataflow.sync] (%a, %b, %c, %d) {
    hw_params = [{
      op = @dataflow.sync,
      function_type = (i32, i32, i32, i32) -> (i32, i32, i32, i32),
      input_ports = [0 : i32, 1 : i32, 2 : i32, 3 : i32],
      output_ports = [0 : i32, 1 : i32, 2 : i32, 3 : i32],
      attributes = {}
    }],
    paired_lanes = [
      {input_port = 0 : i32, output_port = 0 : i32, mask_bit = 0 : i32},
      {input_port = 1 : i32, output_port = 1 : i32, mask_bit = 1 : i32},
      {input_port = 2 : i32, output_port = 2 : i32, mask_bit = 2 : i32}
    ]
  } : (!fabric.bits<32>, !fabric.bits<32>,
       !fabric.bits<32>, !fabric.bits<32>)
      -> (!fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>)
  return
}

func.func @malformed_inventory(
    %a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  // expected-error @+1 {{paired_lanes entry #0 requires input_port, output_port, and mask_bit}}
  %r:2 = fabric.op [@dataflow.sync] (%a, %b) {
    hw_params = [{
      op = @dataflow.sync,
      function_type = (i32, i32) -> (i32, i32),
      input_ports = [0 : i32, 1 : i32],
      output_ports = [0 : i32, 1 : i32],
      attributes = {}
    }],
    paired_lanes = [
      {input_port = 0 : i32, output_port = 0 : i32},
      {input_port = 1 : i32, output_port = 1 : i32, mask_bit = 1 : i32}
    ]
  } : (!fabric.bits<32>, !fabric.bits<32>)
      -> (!fabric.bits<32>, !fabric.bits<32>)
  return
}

func.func @non_dense_mask(
    %a: !fabric.bits<32>, %b: !fabric.bits<32>,
    %c: !fabric.bits<32>, %d: !fabric.bits<32>) {
  // expected-error @+1 {{paired_lanes mask_bit values must form a dense unique range}}
  %r:4 = fabric.op [@dataflow.sync] (%a, %b, %c, %d) {
    hw_params = [{
      op = @dataflow.sync,
      function_type = (i32, i32, i32, i32) -> (i32, i32, i32, i32),
      input_ports = [0 : i32, 1 : i32, 2 : i32, 3 : i32],
      output_ports = [0 : i32, 1 : i32, 2 : i32, 3 : i32],
      attributes = {}
    }],
    paired_lanes = [
      {input_port = 0 : i32, output_port = 0 : i32, mask_bit = 0 : i32},
      {input_port = 1 : i32, output_port = 1 : i32, mask_bit = 1 : i32},
      {input_port = 2 : i32, output_port = 2 : i32, mask_bit = 1 : i32},
      {input_port = 3 : i32, output_port = 3 : i32, mask_bit = 3 : i32}
    ]
  } : (!fabric.bits<32>, !fabric.bits<32>,
       !fabric.bits<32>, !fabric.bits<32>)
      -> (!fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>)
  return
}

func.func @duplicate_endpoint(
    %a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  // expected-error @+1 {{paired_lanes output_port values must cover each physical output exactly once}}
  %r:2 = fabric.op [@dataflow.sync] (%a, %b) {
    hw_params = [{
      op = @dataflow.sync,
      function_type = (i32, i32) -> (i32, i32),
      input_ports = [0 : i32, 1 : i32],
      output_ports = [0 : i32, 1 : i32],
      attributes = {}
    }],
    paired_lanes = [
      {input_port = 0 : i32, output_port = 0 : i32, mask_bit = 0 : i32},
      {input_port = 1 : i32, output_port = 0 : i32, mask_bit = 1 : i32}
    ]
  } : (!fabric.bits<32>, !fabric.bits<32>)
      -> (!fabric.bits<32>, !fabric.bits<32>)
  return
}

func.func @mode_order_mismatch(
    %a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  // expected-error @+1 {{hw_params mode port maps must follow paired_lanes inventory order}}
  %r:2 = fabric.op [@dataflow.sync] (%a, %b) {
    hw_params = [{
      op = @dataflow.sync,
      function_type = (i32, i32) -> (i32, i32),
      input_ports = [0 : i32, 1 : i32],
      output_ports = [0 : i32, 1 : i32],
      attributes = {}
    }],
    paired_lanes = [
      {input_port = 1 : i32, output_port = 0 : i32, mask_bit = 0 : i32},
      {input_port = 0 : i32, output_port = 1 : i32, mask_bit = 1 : i32}
    ]
  } : (!fabric.bits<32>, !fabric.bits<32>)
      -> (!fabric.bits<32>, !fabric.bits<32>)
  return
}

func.func @ordinary_op_cannot_declare_pairs(
    %a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  // expected-error @+1 {{paired_lanes requires every hw_params mode to select @dataflow.sync}}
  %r = fabric.op [@arith.addi] (%a, %b) {
    hw_params = [{
      op = @arith.addi,
      function_type = (i32, i32) -> i32,
      input_ports = [0 : i32, 1 : i32],
      output_ports = [0 : i32],
      attributes = {overflowFlags = #arith.overflow<none>}
    }],
    paired_lanes = [
      {input_port = 0 : i32, output_port = 0 : i32, mask_bit = 0 : i32}
    ]
  } : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  return
}

func.func @normalized_sync_cannot_select_bitmask(
    %a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  // expected-error @+1 {{normalized hw_params requires sw_configs = {mode = N}}}
  %r:2 = fabric.op [@dataflow.sync] (%a, %b) {
    hw_params = [{
      op = @dataflow.sync,
      function_type = (i32, i32) -> (i32, i32),
      input_ports = [0 : i32, 1 : i32],
      output_ports = [0 : i32, 1 : i32],
      attributes = {}
    }],
    paired_lanes = [
      {input_port = 0 : i32, output_port = 0 : i32, mask_bit = 1 : i32},
      {input_port = 1 : i32, output_port = 1 : i32, mask_bit = 0 : i32}
    ],
    sw_configs = {mode = 0 : i32, bitmask = "10"}
  } : (!fabric.bits<32>, !fabric.bits<32>)
      -> (!fabric.bits<32>, !fabric.bits<32>)
  return
}

func.func @legacy_sync_cannot_declare_pairs(
    %a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  // expected-error @+1 {{paired_lanes requires normalized hw_params modes}}
  %r:2 = fabric.op [@dataflow.sync] (%a, %b) {
    paired_lanes = [
      {input_port = 0 : i32, output_port = 0 : i32, mask_bit = 0 : i32},
      {input_port = 1 : i32, output_port = 1 : i32, mask_bit = 0 : i32}
    ],
    sw_configs = {bitmask = "11"}
  } : (!fabric.bits<32>, !fabric.bits<32>)
      -> (!fabric.bits<32>, !fabric.bits<32>)
  return
}

}
