// RUN: timeout 10s loom-coverage-test %s | FileCheck %s

fabric.module @residual_lane_matching(
    %x: !fabric.bits<32>, %y: !fabric.bits<32>, %z: !fabric.bits<32>) {
  fabric.pe [spatial] (
      %px = %x : !fabric.bits<32>, %py = %y : !fabric.bits<32>,
      %pz = %z : !fabric.bits<32>) -> !fabric.bits<32> {
    %r = fabric.fu(
        %fx = %px : !fabric.bits<32>, %fy = %py : !fabric.bits<32>,
        %fz = %pz : !fabric.bits<32>) -> !fabric.bits<32>
        attributes {valid_encodings = [{
          outputs = [0 : i32],
          resources = [
            {resource = 0 : i32, mode = 0 : i32},
            {resource = 1 : i32, mode = 0 : i32}
          ]
        }]} {
      %s0, %s1, %s2, %s3 = fabric.op [@dataflow.sync]
          (%fx, %fy, %fy, %fz) {
            hw_params = [{
              op = @dataflow.sync,
              function_type = (i32, i32, i32, i32) ->
                              (i32, i32, i32, i32),
              input_ports = [0 : i32, 1 : i32, 2 : i32, 3 : i32],
              output_ports = [0 : i32, 1 : i32, 2 : i32, 3 : i32],
              attributes = {}
            }],
            paired_lanes = [
              {input_port = 0 : i32, output_port = 0 : i32, mask_bit = 0 : i32},
              {input_port = 1 : i32, output_port = 1 : i32, mask_bit = 1 : i32},
              {input_port = 2 : i32, output_port = 2 : i32, mask_bit = 2 : i32},
              {input_port = 3 : i32, output_port = 3 : i32, mask_bit = 3 : i32}
            ]
          } : (!fabric.bits<32>, !fabric.bits<32>,
               !fabric.bits<32>, !fabric.bits<32>) ->
              (!fabric.bits<32>, !fabric.bits<32>,
               !fabric.bits<32>, !fabric.bits<32>)
      %selected = fabric.op [@arith.cmpi] (%s3, %s3) {hw_params = [{
        op = @arith.cmpi,
        function_type = (i32, i32) -> i1,
        input_ports = [0 : i32, 1 : i32], output_ports = [0 : i32],
        attributes = {predicate = 0 : i64}
      }]} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %selected : !fabric.bits<32>
    }
  }
  fabric.yield
}

func.func @select_complete_residual_match(%a: i32, %b: i32) -> i1
    attributes {loom.coverage_input = true} {
  %s0, %s1, %s2 = dataflow.sync %a, %a, %b :
      (i32, i32, i32) -> (i32, i32, i32)
  %selected = arith.cmpi eq, %s2, %s2 : i32
  return %selected : i1
}

// CHECK: coverage[0] funcname=select_complete_residual_match matched=true index=0
// CHECK-SAME: lanes=[0:{1->1,2->2,3->3}] bitmasks=[0:0111]
// CHECK-NEXT: all_covered=true
