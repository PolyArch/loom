// RUN: timeout 10s loom-coverage-test %s | FileCheck %s

fabric.module @wide_same_typed_reverse(
    %c0: !fabric.bits<32>, %c1: !fabric.bits<32>,
    %c2: !fabric.bits<32>, %c3: !fabric.bits<32>,
    %c4: !fabric.bits<32>, %c5: !fabric.bits<32>,
    %c6: !fabric.bits<32>, %c7: !fabric.bits<32>,
    %e0: !fabric.bits<32>, %e1: !fabric.bits<32>) {
  fabric.pe [spatial] (
      %pc0 = %c0 : !fabric.bits<32>, %pc1 = %c1 : !fabric.bits<32>,
      %pc2 = %c2 : !fabric.bits<32>, %pc3 = %c3 : !fabric.bits<32>,
      %pc4 = %c4 : !fabric.bits<32>, %pc5 = %c5 : !fabric.bits<32>,
      %pc6 = %c6 : !fabric.bits<32>, %pc7 = %c7 : !fabric.bits<32>,
      %pe0 = %e0 : !fabric.bits<32>, %pe1 = %e1 : !fabric.bits<32>)
      -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>) {
    %r:8 = fabric.fu(
        %x0 = %pc0 : !fabric.bits<32>, %x1 = %pc1 : !fabric.bits<32>,
        %x2 = %pc2 : !fabric.bits<32>, %x3 = %pc3 : !fabric.bits<32>,
        %x4 = %pc4 : !fabric.bits<32>, %x5 = %pc5 : !fabric.bits<32>,
        %x6 = %pc6 : !fabric.bits<32>, %x7 = %pc7 : !fabric.bits<32>,
        %extra0 = %pe0 : !fabric.bits<32>,
        %extra1 = %pe1 : !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
            !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
            !fabric.bits<32>, !fabric.bits<32>)
        attributes {valid_encodings = [{
          outputs = [0 : i32, 1 : i32, 2 : i32, 3 : i32,
                     4 : i32, 5 : i32, 6 : i32, 7 : i32],
          resources = [
            {resource = 0 : i32, mode = 0 : i32},
            {resource = 1 : i32, mode = 0 : i32},
            {resource = 2 : i32, mode = 0 : i32},
            {resource = 3 : i32, mode = 0 : i32},
            {resource = 4 : i32, mode = 0 : i32},
            {resource = 5 : i32, mode = 0 : i32},
            {resource = 6 : i32, mode = 0 : i32},
            {resource = 7 : i32, mode = 0 : i32},
            {resource = 8 : i32, mode = 0 : i32}
          ]
        }]} {
      %h0, %h1, %h2, %h3, %h4, %h5, %h6, %h7, %h8,
      %h9, %h10, %h11, %h12, %h13, %h14, %h15, %h16, %h17 =
          fabric.op [@dataflow.sync] (
              %x0, %x7, %x1, %x6, %x2, %x5, %x3, %x4, %x4,
              %x3, %x5, %x2, %x6, %x1, %extra0, %x0, %extra1, %x7) {
            hw_params = [{
              op = @dataflow.sync,
              function_type = (i32, i32, i32, i32, i32, i32, i32, i32, i32,
                               i32, i32, i32, i32, i32, i32, i32, i32, i32) ->
                              (i32, i32, i32, i32, i32, i32, i32, i32, i32,
                               i32, i32, i32, i32, i32, i32, i32, i32, i32),
              input_ports = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32,
                             5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32,
                             10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32,
                             15 : i32, 16 : i32, 17 : i32],
              output_ports = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32,
                              5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32,
                              10 : i32, 11 : i32, 12 : i32, 13 : i32,
                              14 : i32, 15 : i32, 16 : i32, 17 : i32],
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
              {input_port = 9 : i32, output_port = 9 : i32, mask_bit = 9 : i32},
              {input_port = 10 : i32, output_port = 10 : i32,
               mask_bit = 10 : i32},
              {input_port = 11 : i32, output_port = 11 : i32,
               mask_bit = 11 : i32},
              {input_port = 12 : i32, output_port = 12 : i32,
               mask_bit = 12 : i32},
              {input_port = 13 : i32, output_port = 13 : i32,
               mask_bit = 13 : i32},
              {input_port = 14 : i32, output_port = 14 : i32,
               mask_bit = 14 : i32},
              {input_port = 15 : i32, output_port = 15 : i32,
               mask_bit = 15 : i32},
              {input_port = 16 : i32, output_port = 16 : i32,
               mask_bit = 16 : i32},
              {input_port = 17 : i32, output_port = 17 : i32,
               mask_bit = 17 : i32}
            ]
          } : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
               !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
               !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
               !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
               !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
               !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
              -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                  !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                  !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                  !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                  !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                  !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
      %p0 = fabric.op [@arith.cmpi] (%h1, %h1) {hw_params = [{
        op = @arith.cmpi, function_type = (i32, i32) -> i1,
        input_ports = [0 : i32, 1 : i32], output_ports = [0 : i32],
        attributes = {predicate = 0 : i64}
      }]} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %p1 = fabric.op [@arith.cmpi] (%h3, %h3) {hw_params = [{
        op = @arith.cmpi, function_type = (i32, i32) -> i1,
        input_ports = [0 : i32, 1 : i32], output_ports = [0 : i32],
        attributes = {predicate = 1 : i64}
      }]} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %p2 = fabric.op [@arith.cmpi] (%h5, %h5) {hw_params = [{
        op = @arith.cmpi, function_type = (i32, i32) -> i1,
        input_ports = [0 : i32, 1 : i32], output_ports = [0 : i32],
        attributes = {predicate = 2 : i64}
      }]} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %p3 = fabric.op [@arith.cmpi] (%h7, %h7) {hw_params = [{
        op = @arith.cmpi, function_type = (i32, i32) -> i1,
        input_ports = [0 : i32, 1 : i32], output_ports = [0 : i32],
        attributes = {predicate = 3 : i64}
      }]} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %p4 = fabric.op [@arith.cmpi] (%h9, %h9) {hw_params = [{
        op = @arith.cmpi, function_type = (i32, i32) -> i1,
        input_ports = [0 : i32, 1 : i32], output_ports = [0 : i32],
        attributes = {predicate = 4 : i64}
      }]} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %p5 = fabric.op [@arith.cmpi] (%h11, %h11) {hw_params = [{
        op = @arith.cmpi, function_type = (i32, i32) -> i1,
        input_ports = [0 : i32, 1 : i32], output_ports = [0 : i32],
        attributes = {predicate = 5 : i64}
      }]} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %p6 = fabric.op [@arith.cmpi] (%h13, %h13) {hw_params = [{
        op = @arith.cmpi, function_type = (i32, i32) -> i1,
        input_ports = [0 : i32, 1 : i32], output_ports = [0 : i32],
        attributes = {predicate = 6 : i64}
      }]} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %p7 = fabric.op [@arith.cmpi] (%h15, %h15) {hw_params = [{
        op = @arith.cmpi, function_type = (i32, i32) -> i1,
        input_ports = [0 : i32, 1 : i32], output_ports = [0 : i32],
        attributes = {predicate = 7 : i64}
      }]} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %p0, %p1, %p2, %p3, %p4, %p5, %p6, %p7 :
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>
    }
  }
  fabric.yield
}

func.func @wide_reversed_non_prefix(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %a4: i32, %a5: i32, %a6: i32, %a7: i32)
    -> (i1, i1, i1, i1, i1, i1, i1, i1)
    attributes {loom.coverage_input = true} {
  %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7,
  %s8, %s9, %s10, %s11, %s12, %s13, %s14, %s15 =
      dataflow.sync %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7,
                    %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7 :
      (i32, i32, i32, i32, i32, i32, i32, i32,
       i32, i32, i32, i32, i32, i32, i32, i32) ->
      (i32, i32, i32, i32, i32, i32, i32, i32,
       i32, i32, i32, i32, i32, i32, i32, i32)
  %p0 = arith.cmpi eq, %s8, %s8 : i32
  %p1 = arith.cmpi ne, %s9, %s9 : i32
  %p2 = arith.cmpi slt, %s10, %s10 : i32
  %p3 = arith.cmpi sle, %s11, %s11 : i32
  %p4 = arith.cmpi sgt, %s12, %s12 : i32
  %p5 = arith.cmpi sge, %s13, %s13 : i32
  %p6 = arith.cmpi ult, %s14, %s14 : i32
  %p7 = arith.cmpi ule, %s15, %s15 : i32
  return %p0, %p1, %p2, %p3, %p4, %p5, %p6, %p7 :
      i1, i1, i1, i1, i1, i1, i1, i1
}

// CHECK: coverage[0] funcname=wide_reversed_non_prefix matched=true index=0
// CHECK-SAME: lanes=[0:{17->17,12->12,10->10,8->8,6->6,4->4,2->2,0->0,
// CHECK-SAME: 1->1,3->3,5->5,7->7,9->9,11->11,13->13,15->15}]
// CHECK-SAME: bitmasks=[0:111111111111110101]
// CHECK-NEXT: all_covered=true
