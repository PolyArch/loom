// RUN: loom-coverage-test %s --project-first-encoding | FileCheck %s
// RUN: loom-coverage-test %s --verify-normalized-modes 2>&1 | FileCheck %s --check-prefix=DIRECT

// CHECK: projection=failed
// CHECK: error=paired_lanes requires every hw_params mode to select @dataflow.sync

// DIRECT: error: 'fabric.op' op paired_lanes requires every hw_params mode to select @dataflow.sync
// DIRECT-NEXT: normalized_modes=failed

fabric.module @projector_paired_lane_preflight(
    %a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  fabric.pe [spatial] (
      %pa = %a : !fabric.bits<32>, %pb = %b : !fabric.bits<32>)
      -> !fabric.bits<32> {
    %r = fabric.fu(
        %x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)
        -> !fabric.bits<32>
        attributes {valid_encodings = [{
          outputs = [0 : i32],
          resources = [{resource = 0 : i32, mode = 0 : i32}]
        }]} {
      %sum = fabric.op [@arith.addi] (%x, %y) {
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
      fabric.yield %sum : !fabric.bits<32>
    }
  }
  fabric.yield
}
