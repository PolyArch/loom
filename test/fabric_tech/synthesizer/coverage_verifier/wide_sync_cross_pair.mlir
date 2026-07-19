// RUN: loom-coverage-test %s | FileCheck %s

fabric.module @cross_pair_sync(
    %a: !fabric.bits<64>, %b: !fabric.bits<64>,
    %c: !fabric.bits<64>, %d: !fabric.bits<64>) {
  fabric.pe [spatial] (
      %pa = %a : !fabric.bits<64>, %pb = %b : !fabric.bits<64>,
      %pc = %c : !fabric.bits<64>, %pd = %d : !fabric.bits<64>)
      -> !fabric.bits<64> {
    %r = fabric.fu(
        %x0 = %pa : !fabric.bits<64>, %x1 = %pb : !fabric.bits<64>,
        %x2 = %pc : !fabric.bits<64>, %x3 = %pd : !fabric.bits<64>)
        -> !fabric.bits<64>
        attributes {valid_encodings = [{
          outputs = [0 : i32],
          resources = [
            {resource = 0 : i32, mode = 0 : i32},
            {resource = 1 : i32, mode = 0 : i32},
            {resource = 2 : i32, mode = 0 : i32}
          ]
        }]} {
      %pre = fabric.op [@arith.extui] (%x0) {hw_params = [{
        op = @arith.extui, function_type = (i8) -> i16,
        input_ports = [0 : i32], output_ports = [0 : i32], attributes = {}
      }]} : (!fabric.bits<64>) -> !fabric.bits<64>
      %s0, %s1, %s2, %s3 = fabric.op [@dataflow.sync]
          (%pre, %x1, %x2, %x3) {
            hw_params = [{
              op = @dataflow.sync,
              function_type = (i16, i32, i16, i64) ->
                              (i16, i32, i16, i64),
              input_ports = [0 : i32, 1 : i32, 2 : i32, 3 : i32],
              output_ports = [0 : i32, 1 : i32, 2 : i32, 3 : i32],
              attributes = {}
            }],
            paired_lanes = [
              {input_port = 0 : i32, output_port = 0 : i32,
               mask_bit = 0 : i32},
              {input_port = 1 : i32, output_port = 1 : i32,
               mask_bit = 1 : i32},
              {input_port = 2 : i32, output_port = 2 : i32,
               mask_bit = 2 : i32},
              {input_port = 3 : i32, output_port = 3 : i32,
               mask_bit = 3 : i32}
            ]
          } : (!fabric.bits<64>, !fabric.bits<64>,
               !fabric.bits<64>, !fabric.bits<64>)
              -> (!fabric.bits<64>, !fabric.bits<64>,
                  !fabric.bits<64>, !fabric.bits<64>)
      %post = fabric.op [@arith.extui] (%s2) {hw_params = [{
        op = @arith.extui, function_type = (i16) -> i32,
        input_ports = [0 : i32], output_ports = [0 : i32], attributes = {}
      }]} : (!fabric.bits<64>) -> !fabric.bits<64>
      fabric.yield %post : !fabric.bits<64>
    }
  }
  fabric.yield
}

func.func @cross_pair(%a: i8) -> i32
    attributes {loom.coverage_input = true} {
  %pre = arith.extui %a : i8 to i16
  %sync = dataflow.sync %pre : (i16) -> (i16)
  %post = arith.extui %sync : i16 to i32
  return %post : i32
}

// CHECK: coverage[0] funcname=cross_pair matched=false index=none
// CHECK-NEXT: all_covered=false
