// RUN: split-file %s %t
// RUN: timeout 2s loom-coverage-test %t/n8.mlir | FileCheck %s --check-prefix=N8
// RUN: timeout 2s loom-coverage-test %t/n9.mlir | FileCheck %s --check-prefix=N9

// N8: coverage[0] funcname=invalid_extra_n8 matched=false index=none
// N8-NEXT: all_covered=false

// N9: coverage[0] funcname=invalid_extra_n9 matched=false index=none
// N9-NEXT: all_covered=false

//--- n8.mlir
fabric.module @invalid_extra_n8_hardware(%a: !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>)
      -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
    %r:9 = fabric.fu(%x = %pa : !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
            !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
            !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        attributes {valid_encodings = [{
          outputs = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32,
                     5 : i32, 6 : i32, 7 : i32, 8 : i32],
          resources = [
            {resource = 0 : i32, mode = 0 : i32},
            {resource = 1 : i32, mode = 0 : i32}
          ]
        }]} {
      %sum = fabric.op [@arith.addi] (%x, %x) {hw_params = [{
        op = @arith.addi,
        function_type = (i32, i32) -> i32,
        input_ports = [0 : i32, 1 : i32], output_ports = [0 : i32],
        attributes = {overflowFlags = #arith.overflow<none>}
      }]} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7 =
          fabric.op [@dataflow.sync]
              (%sum, %sum, %sum, %sum, %sum, %sum, %sum, %sum) {
            hw_params = [{
              op = @dataflow.sync,
              function_type = (i32, i32, i32, i32, i32, i32, i32, i32) ->
                              (i32, i32, i32, i32, i32, i32, i32, i32),
              input_ports = [0 : i32, 1 : i32, 2 : i32, 3 : i32,
                             4 : i32, 5 : i32, 6 : i32, 7 : i32],
              output_ports = [0 : i32, 1 : i32, 2 : i32, 3 : i32,
                              4 : i32, 5 : i32, 6 : i32, 7 : i32],
              attributes = {}
            }],
            paired_lanes = [
              {input_port = 0 : i32, output_port = 0 : i32, mask_bit = 0 : i32},
              {input_port = 1 : i32, output_port = 1 : i32, mask_bit = 1 : i32},
              {input_port = 2 : i32, output_port = 2 : i32, mask_bit = 2 : i32},
              {input_port = 3 : i32, output_port = 3 : i32, mask_bit = 3 : i32},
              {input_port = 4 : i32, output_port = 4 : i32, mask_bit = 4 : i32},
              {input_port = 5 : i32, output_port = 5 : i32, mask_bit = 5 : i32},
              {input_port = 6 : i32, output_port = 6 : i32, mask_bit = 6 : i32},
              {input_port = 7 : i32, output_port = 7 : i32, mask_bit = 7 : i32}
            ]
          } : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
               !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
               !fabric.bits<32>, !fabric.bits<32>) ->
              (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
               !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
               !fabric.bits<32>, !fabric.bits<32>)
      fabric.yield %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %sum :
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>
    }
  }
  fabric.yield
}

func.func @invalid_extra_n8(%a: i32)
    -> (i32, i32, i32, i32, i32, i32, i32, i32)
    attributes {loom.coverage_input = true} {
  %sum = arith.addi %a, %a : i32
  %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7 =
      dataflow.sync %sum, %sum, %sum, %sum, %sum, %sum, %sum, %sum :
      (i32, i32, i32, i32, i32, i32, i32, i32) ->
      (i32, i32, i32, i32, i32, i32, i32, i32)
  return %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7 :
      i32, i32, i32, i32, i32, i32, i32, i32
}

//--- n9.mlir
fabric.module @invalid_extra_n9_hardware(%a: !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>)
      -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>) {
    %r:10 = fabric.fu(%x = %pa : !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
            !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
            !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
            !fabric.bits<32>)
        attributes {valid_encodings = [{
          outputs = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32,
                     5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32],
          resources = [
            {resource = 0 : i32, mode = 0 : i32},
            {resource = 1 : i32, mode = 0 : i32}
          ]
        }]} {
      %sum = fabric.op [@arith.addi] (%x, %x) {hw_params = [{
        op = @arith.addi,
        function_type = (i32, i32) -> i32,
        input_ports = [0 : i32, 1 : i32], output_ports = [0 : i32],
        attributes = {overflowFlags = #arith.overflow<none>}
      }]} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8 =
          fabric.op [@dataflow.sync]
              (%sum, %sum, %sum, %sum, %sum, %sum, %sum, %sum, %sum) {
            hw_params = [{
              op = @dataflow.sync,
              function_type =
                  (i32, i32, i32, i32, i32, i32, i32, i32, i32) ->
                  (i32, i32, i32, i32, i32, i32, i32, i32, i32),
              input_ports = [0 : i32, 1 : i32, 2 : i32, 3 : i32,
                             4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32],
              output_ports = [0 : i32, 1 : i32, 2 : i32, 3 : i32,
                              4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32],
              attributes = {}
            }],
            paired_lanes = [
              {input_port = 0 : i32, output_port = 0 : i32, mask_bit = 0 : i32},
              {input_port = 1 : i32, output_port = 1 : i32, mask_bit = 1 : i32},
              {input_port = 2 : i32, output_port = 2 : i32, mask_bit = 2 : i32},
              {input_port = 3 : i32, output_port = 3 : i32, mask_bit = 3 : i32},
              {input_port = 4 : i32, output_port = 4 : i32, mask_bit = 4 : i32},
              {input_port = 5 : i32, output_port = 5 : i32, mask_bit = 5 : i32},
              {input_port = 6 : i32, output_port = 6 : i32, mask_bit = 6 : i32},
              {input_port = 7 : i32, output_port = 7 : i32, mask_bit = 7 : i32},
              {input_port = 8 : i32, output_port = 8 : i32, mask_bit = 8 : i32}
            ]
          } : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
               !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
               !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) ->
              (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
               !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
               !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
      fabric.yield %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %sum :
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>
    }
  }
  fabric.yield
}

func.func @invalid_extra_n9(%a: i32)
    -> (i32, i32, i32, i32, i32, i32, i32, i32, i32)
    attributes {loom.coverage_input = true} {
  %sum = arith.addi %a, %a : i32
  %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8 =
      dataflow.sync %sum, %sum, %sum, %sum, %sum, %sum, %sum, %sum, %sum :
      (i32, i32, i32, i32, i32, i32, i32, i32, i32) ->
      (i32, i32, i32, i32, i32, i32, i32, i32, i32)
  return %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8 :
      i32, i32, i32, i32, i32, i32, i32, i32, i32
}
