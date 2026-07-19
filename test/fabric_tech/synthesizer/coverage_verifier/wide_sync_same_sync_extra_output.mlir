// RUN: split-file %s %t
// RUN: timeout 2s loom-coverage-test %t/n8.mlir | FileCheck %s --check-prefix=N8
// RUN: timeout 2s loom-coverage-test %t/n9.mlir | FileCheck %s --check-prefix=N9

// N8: coverage[0] funcname=same_sync_extra_n8 matched=false index=none
// N8-NEXT: all_covered=false

// N9: coverage[0] funcname=same_sync_extra_n9 matched=false index=none
// N9-NEXT: all_covered=false

//--- n8.mlir
fabric.module @same_sync_extra_n8_hardware(%a: !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>)
      -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
    %r:12 = fabric.fu(%x = %pa : !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
            !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
            !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
            !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        attributes {valid_encodings = [{
          outputs = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32,
                     5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32,
                     10 : i32, 11 : i32],
          resources = [
            {resource = 0 : i32, mode = 0 : i32},
            {resource = 1 : i32, mode = 0 : i32},
            {resource = 2 : i32, mode = 0 : i32},
            {resource = 3 : i32, mode = 0 : i32}
          ]
        }]} {
      %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7 =
          fabric.op [@dataflow.sync]
              (%x, %x, %x, %x, %x, %x, %x, %x) {
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
      %single0 = fabric.op [@dataflow.sync] (%x) {
        hw_params = [{
          op = @dataflow.sync, function_type = (i32) -> (i32),
          input_ports = [0 : i32], output_ports = [0 : i32], attributes = {}
        }],
        paired_lanes = [
          {input_port = 0 : i32, output_port = 0 : i32, mask_bit = 0 : i32}
        ]
      } : (!fabric.bits<32>) -> !fabric.bits<32>
      %single1 = fabric.op [@dataflow.sync] (%x) {
        hw_params = [{
          op = @dataflow.sync, function_type = (i32) -> (i32),
          input_ports = [0 : i32], output_ports = [0 : i32], attributes = {}
        }],
        paired_lanes = [
          {input_port = 0 : i32, output_port = 0 : i32, mask_bit = 0 : i32}
        ]
      } : (!fabric.bits<32>) -> !fabric.bits<32>
      %joined0, %joined1 = fabric.op [@dataflow.sync] (%single0, %single1) {
        hw_params = [{
          op = @dataflow.sync, function_type = (i32, i32) -> (i32, i32),
          input_ports = [0 : i32, 1 : i32],
          output_ports = [0 : i32, 1 : i32], attributes = {}
        }],
        paired_lanes = [
          {input_port = 0 : i32, output_port = 0 : i32, mask_bit = 0 : i32},
          {input_port = 1 : i32, output_port = 1 : i32, mask_bit = 1 : i32}
        ]
      } : (!fabric.bits<32>, !fabric.bits<32>) ->
          (!fabric.bits<32>, !fabric.bits<32>)
      fabric.yield %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7,
          %single0, %single1, %joined0, %joined1 :
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>
    }
  }
  fabric.yield
}

func.func @same_sync_extra_n8(%a: i32)
    -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
    attributes {loom.coverage_input = true} {
  %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7 =
      dataflow.sync %a, %a, %a, %a, %a, %a, %a, %a :
      (i32, i32, i32, i32, i32, i32, i32, i32) ->
      (i32, i32, i32, i32, i32, i32, i32, i32)
  %single0 = dataflow.sync %a : (i32) -> (i32)
  %single1 = dataflow.sync %a : (i32) -> (i32)
  %joined0, %joined1 = dataflow.sync %single0, %single1 :
      (i32, i32) -> (i32, i32)
  return %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %joined0, %joined1 :
      i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
}

//--- n9.mlir
fabric.module @same_sync_extra_n9_hardware(%a: !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>)
      -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>) {
    %r:13 = fabric.fu(%x = %pa : !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
            !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
            !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
            !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
            !fabric.bits<32>)
        attributes {valid_encodings = [{
          outputs = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32,
                     5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32,
                     10 : i32, 11 : i32, 12 : i32],
          resources = [
            {resource = 0 : i32, mode = 0 : i32},
            {resource = 1 : i32, mode = 0 : i32},
            {resource = 2 : i32, mode = 0 : i32},
            {resource = 3 : i32, mode = 0 : i32}
          ]
        }]} {
      %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8 =
          fabric.op [@dataflow.sync]
              (%x, %x, %x, %x, %x, %x, %x, %x, %x) {
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
      %single0 = fabric.op [@dataflow.sync] (%x) {
        hw_params = [{
          op = @dataflow.sync, function_type = (i32) -> (i32),
          input_ports = [0 : i32], output_ports = [0 : i32], attributes = {}
        }],
        paired_lanes = [
          {input_port = 0 : i32, output_port = 0 : i32, mask_bit = 0 : i32}
        ]
      } : (!fabric.bits<32>) -> !fabric.bits<32>
      %single1 = fabric.op [@dataflow.sync] (%x) {
        hw_params = [{
          op = @dataflow.sync, function_type = (i32) -> (i32),
          input_ports = [0 : i32], output_ports = [0 : i32], attributes = {}
        }],
        paired_lanes = [
          {input_port = 0 : i32, output_port = 0 : i32, mask_bit = 0 : i32}
        ]
      } : (!fabric.bits<32>) -> !fabric.bits<32>
      %joined0, %joined1 = fabric.op [@dataflow.sync] (%single0, %single1) {
        hw_params = [{
          op = @dataflow.sync, function_type = (i32, i32) -> (i32, i32),
          input_ports = [0 : i32, 1 : i32],
          output_ports = [0 : i32, 1 : i32], attributes = {}
        }],
        paired_lanes = [
          {input_port = 0 : i32, output_port = 0 : i32, mask_bit = 0 : i32},
          {input_port = 1 : i32, output_port = 1 : i32, mask_bit = 1 : i32}
        ]
      } : (!fabric.bits<32>, !fabric.bits<32>) ->
          (!fabric.bits<32>, !fabric.bits<32>)
      fabric.yield %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8,
          %single0, %single1, %joined0, %joined1 :
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>
    }
  }
  fabric.yield
}

func.func @same_sync_extra_n9(%a: i32)
    -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
    attributes {loom.coverage_input = true} {
  %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8 =
      dataflow.sync %a, %a, %a, %a, %a, %a, %a, %a, %a :
      (i32, i32, i32, i32, i32, i32, i32, i32, i32) ->
      (i32, i32, i32, i32, i32, i32, i32, i32, i32)
  %single0 = dataflow.sync %a : (i32) -> (i32)
  %single1 = dataflow.sync %a : (i32) -> (i32)
  %joined0, %joined1 = dataflow.sync %single0, %single1 :
      (i32, i32) -> (i32, i32)
  return %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %joined0, %joined1 :
      i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
}
