// RUN: split-file %s %t
// RUN: loom-coverage-test %t/order.mlir | FileCheck %s --check-prefix=ORDER
// RUN: loom-coverage-test %t/duplicate.mlir | FileCheck %s --check-prefix=DUPLICATE
// RUN: loom-coverage-test %t/distinct.mlir | FileCheck %s --check-prefix=DISTINCT
// RUN: timeout 2s loom-coverage-test %t/wide-duplicate.mlir | FileCheck %s --check-prefix=WIDE-DUPLICATE
// RUN: timeout 2s loom-coverage-test %t/wide-order.mlir | FileCheck %s --check-prefix=WIDE-ORDER
// RUN: timeout 5s loom-coverage-test %t/late-order.mlir | FileCheck %s --check-prefix=LATE-ORDER

// ORDER: coverage[0] funcname=ordered_mixed_outputs matched=true index=0
// ORDER-NEXT: coverage[1] funcname=reordered_mixed_outputs matched=false index=none
// ORDER-NEXT: all_covered=false

// DUPLICATE: coverage[0] funcname=duplicate_sync_result matched=false index=none
// DUPLICATE-NEXT: all_covered=false

// DISTINCT: coverage[0] funcname=distinct_sync_reorder matched=false index=none
// DISTINCT-NEXT: all_covered=false

// WIDE-DUPLICATE: coverage[0] funcname=wide_duplicate_result matched=false index=none
// WIDE-DUPLICATE-NEXT: all_covered=false

// WIDE-ORDER: coverage[0] funcname=wide_reordered_mixed_outputs matched=false index=none
// WIDE-ORDER-NEXT: all_covered=false

// LATE-ORDER: coverage[0] funcname=late_ordinary_control matched=true index=0
// LATE-ORDER-NEXT: coverage[1] funcname=late_ordinary_inversion matched=false index=none
// LATE-ORDER-NEXT: all_covered=false

//--- order.mlir
fabric.module @sync_with_ordinary_output(
    %a: !fabric.bits<64>, %b: !fabric.bits<64>) {
  fabric.pe [spatial] (
      %pa = %a : !fabric.bits<64>, %pb = %b : !fabric.bits<64>)
      -> (!fabric.bits<64>, !fabric.bits<64>) {
    %r:2 = fabric.fu(
        %x0 = %pa : !fabric.bits<64>, %x1 = %pb : !fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>)
        attributes {valid_encodings = [{
          outputs = [0 : i32, 1 : i32],
          resources = [
            {resource = 0 : i32, mode = 0 : i32},
            {resource = 1 : i32, mode = 0 : i32}
          ]
        }]} {
      %s0, %s1 = fabric.op [@dataflow.sync] (%x0, %x1) {
        hw_params = [{
          op = @dataflow.sync,
          function_type = (i16, i32) -> (i16, i32),
          input_ports = [0 : i32, 1 : i32],
          output_ports = [0 : i32, 1 : i32],
          attributes = {}
        }],
        paired_lanes = [
          {input_port = 0 : i32, output_port = 0 : i32, mask_bit = 1 : i32},
          {input_port = 1 : i32, output_port = 1 : i32, mask_bit = 0 : i32}
        ]
      } : (!fabric.bits<64>, !fabric.bits<64>)
          -> (!fabric.bits<64>, !fabric.bits<64>)
      %wide = fabric.op [@arith.extui] (%s0) {hw_params = [{
        op = @arith.extui,
        function_type = (i16) -> i64,
        input_ports = [0 : i32], output_ports = [0 : i32], attributes = {}
      }]} : (!fabric.bits<64>) -> !fabric.bits<64>
      fabric.yield %s1, %wide : !fabric.bits<64>, !fabric.bits<64>
    }
  }
  fabric.yield
}

func.func @ordered_mixed_outputs(%a: i16, %b: i32) -> (i32, i64)
    attributes {loom.coverage_input = true} {
  %s0, %s1 = dataflow.sync %a, %b : (i16, i32) -> (i16, i32)
  %wide = arith.extui %s0 : i16 to i64
  return %s1, %wide : i32, i64
}

func.func @reordered_mixed_outputs(%a: i16, %b: i32) -> (i64, i32)
    attributes {loom.coverage_input = true} {
  %s0, %s1 = dataflow.sync %a, %b : (i16, i32) -> (i16, i32)
  %wide = arith.extui %s0 : i16 to i64
  return %wide, %s1 : i64, i32
}

//--- duplicate.mlir
fabric.module @duplicate_sync_output(
    %a: !fabric.bits<64>, %b: !fabric.bits<64>) {
  fabric.pe [spatial] (
      %pa = %a : !fabric.bits<64>, %pb = %b : !fabric.bits<64>)
      -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) {
    %r:3 = fabric.fu(
        %x0 = %pa : !fabric.bits<64>, %x1 = %pb : !fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
        attributes {valid_encodings = [{
          outputs = [0 : i32, 1 : i32, 2 : i32],
          resources = [{resource = 0 : i32, mode = 0 : i32}]
        }]} {
      %s0, %s1 = fabric.op [@dataflow.sync] (%x0, %x1) {
        hw_params = [{
          op = @dataflow.sync,
          function_type = (i16, i32) -> (i16, i32),
          input_ports = [0 : i32, 1 : i32],
          output_ports = [0 : i32, 1 : i32],
          attributes = {}
        }],
        paired_lanes = [
          {input_port = 0 : i32, output_port = 0 : i32, mask_bit = 0 : i32},
          {input_port = 1 : i32, output_port = 1 : i32, mask_bit = 1 : i32}
        ]
      } : (!fabric.bits<64>, !fabric.bits<64>)
          -> (!fabric.bits<64>, !fabric.bits<64>)
      fabric.yield %s0, %s0, %s1 :
          !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>
    }
  }
  fabric.yield
}

func.func @duplicate_sync_result(%a: i16, %b: i32) -> (i16, i32)
    attributes {loom.coverage_input = true} {
  %s0, %s1 = dataflow.sync %a, %b : (i16, i32) -> (i16, i32)
  return %s0, %s1 : i16, i32
}

//--- distinct.mlir
fabric.module @distinct_sync_outputs(
    %a: !fabric.bits<64>, %b: !fabric.bits<64>) {
  fabric.pe [spatial] (
      %pa = %a : !fabric.bits<64>, %pb = %b : !fabric.bits<64>)
      -> (!fabric.bits<64>, !fabric.bits<64>) {
    %r:2 = fabric.fu(
        %x0 = %pa : !fabric.bits<64>, %x1 = %pb : !fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>)
        attributes {valid_encodings = [{
          outputs = [0 : i32, 1 : i32],
          resources = [
            {resource = 0 : i32, mode = 0 : i32},
            {resource = 1 : i32, mode = 0 : i32}
          ]
        }]} {
      %sa = fabric.op [@dataflow.sync] (%x0) {
        hw_params = [{
          op = @dataflow.sync,
          function_type = (i16) -> (i16),
          input_ports = [0 : i32], output_ports = [0 : i32], attributes = {}
        }],
        paired_lanes = [
          {input_port = 0 : i32, output_port = 0 : i32, mask_bit = 0 : i32}
        ]
      } : (!fabric.bits<64>) -> !fabric.bits<64>
      %sb = fabric.op [@dataflow.sync] (%x1) {
        hw_params = [{
          op = @dataflow.sync,
          function_type = (i32) -> (i32),
          input_ports = [0 : i32], output_ports = [0 : i32], attributes = {}
        }],
        paired_lanes = [
          {input_port = 0 : i32, output_port = 0 : i32, mask_bit = 0 : i32}
        ]
      } : (!fabric.bits<64>) -> !fabric.bits<64>
      fabric.yield %sa, %sb : !fabric.bits<64>, !fabric.bits<64>
    }
  }
  fabric.yield
}

func.func @distinct_sync_reorder(%a: i16, %b: i32) -> (i32, i16)
    attributes {loom.coverage_input = true} {
  %sa = dataflow.sync %a : (i16) -> (i16)
  %sb = dataflow.sync %b : (i32) -> (i32)
  return %sb, %sa : i32, i16
}

//--- wide-duplicate.mlir
fabric.module @wide_duplicate_sync_output(%a: !fabric.bits<32>) {
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
          resources = [{resource = 0 : i32, mode = 0 : i32}]
        }]} {
      %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7 =
          fabric.op [@dataflow.sync] (%x, %x, %x, %x, %x, %x, %x, %x) {
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
      fabric.yield %s0, %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7 :
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>
    }
  }
  fabric.yield
}

func.func @wide_duplicate_result(%a: i32)
    -> (i32, i32, i32, i32, i32, i32, i32, i32)
    attributes {loom.coverage_input = true} {
  %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7 =
      dataflow.sync %a, %a, %a, %a, %a, %a, %a, %a :
      (i32, i32, i32, i32, i32, i32, i32, i32) ->
      (i32, i32, i32, i32, i32, i32, i32, i32)
  return %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7 :
      i32, i32, i32, i32, i32, i32, i32, i32
}

//--- wide-order.mlir
fabric.module @wide_mixed_output_order(%a: !fabric.bits<32>) {
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
      %sum = fabric.op [@arith.addi] (%s0, %s0) {hw_params = [{
        op = @arith.addi,
        function_type = (i32, i32) -> i32,
        input_ports = [0 : i32, 1 : i32], output_ports = [0 : i32],
        attributes = {overflowFlags = #arith.overflow<none>}
      }]} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %sum :
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>
    }
  }
  fabric.yield
}

func.func @wide_reordered_mixed_outputs(%a: i32)
    -> (i32, i32, i32, i32, i32, i32, i32, i32, i32)
    attributes {loom.coverage_input = true} {
  %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8 =
      dataflow.sync %a, %a, %a, %a, %a, %a, %a, %a, %a :
      (i32, i32, i32, i32, i32, i32, i32, i32, i32) ->
      (i32, i32, i32, i32, i32, i32, i32, i32, i32)
  %sum = arith.addi %s0, %s0 : i32
  return %sum, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8 :
      i32, i32, i32, i32, i32, i32, i32, i32, i32
}

//--- late-order.mlir
fabric.module @late_ordinary_output(%a: !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>)
      -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>) {
    %r:11 = fabric.fu(%x = %pa : !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
            !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
            !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
            !fabric.bits<32>, !fabric.bits<32>)
        attributes {valid_encodings = [{
          outputs = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32,
                     5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32,
                     10 : i32],
          resources = [
            {resource = 0 : i32, mode = 0 : i32},
            {resource = 1 : i32, mode = 0 : i32}
          ]
        }]} {
      %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %s9 =
          fabric.op [@dataflow.sync]
              (%x, %x, %x, %x, %x, %x, %x, %x, %x, %x) {
            hw_params = [{
              op = @dataflow.sync,
              function_type =
                  (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) ->
                  (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32),
              input_ports = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32,
                             5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32],
              output_ports = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32,
                              5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32],
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
              {input_port = 8 : i32, output_port = 8 : i32, mask_bit = 8 : i32},
              {input_port = 9 : i32, output_port = 9 : i32, mask_bit = 9 : i32}
            ]
          } : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
               !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
               !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
               !fabric.bits<32>) ->
              (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
               !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
               !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
               !fabric.bits<32>)
      %sum = fabric.op [@arith.addi] (%s0, %s0) {hw_params = [{
        op = @arith.addi,
        function_type = (i32, i32) -> i32,
        input_ports = [0 : i32, 1 : i32], output_ports = [0 : i32],
        attributes = {overflowFlags = #arith.overflow<none>}
      }]} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %sum, %s9 :
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>
    }
  }
  fabric.yield
}

func.func @late_ordinary_control(%a: i32)
    -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
    attributes {loom.coverage_input = true} {
  %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %s9 =
      dataflow.sync %a, %a, %a, %a, %a, %a, %a, %a, %a, %a :
      (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) ->
      (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
  %sum = arith.addi %s0, %s0 : i32
  return %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %sum, %s9 :
      i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
}

func.func @late_ordinary_inversion(%a: i32)
    -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
    attributes {loom.coverage_input = true} {
  %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %s9 =
      dataflow.sync %a, %a, %a, %a, %a, %a, %a, %a, %a, %a :
      (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) ->
      (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
  %sum = arith.addi %s0, %s0 : i32
  return %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %s9, %sum :
      i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
}
