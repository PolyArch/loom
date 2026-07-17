// RUN: split-file %s %t
// RUN: loom-coverage-test %t/boundary.mlir | FileCheck %s --check-prefix=BOUNDARY
// RUN: loom-coverage-test %t/width.mlir | FileCheck %s --check-prefix=WIDTH

// The FU verifier projects all encodings with boundary identity preserved and
// rejects isomorphic duplicates. These encodings differ only by the swapped
// inputs of a noncommutative operation.
// BOUNDARY: all_covered=true

// WIDTH: coverage[0] funcname=cmp_i16 matched=true index=0
// WIDTH-NEXT: all_covered=true

//--- boundary.mlir
fabric.module @boundary_identity(%a : !fabric.bits<32>,
                                 %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                      %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    %r = fabric.fu(%x = %pa : !fabric.bits<32>,
                   %y = %pb : !fabric.bits<32>) -> !fabric.bits<32>
        attributes {valid_encodings = [
          {outputs = [0 : i32], resources = [
            {resource = 0 : i32, select = 0 : i32},
            {resource = 1 : i32, select = 1 : i32},
            {resource = 2 : i32, select = 0 : i32},
            {resource = 3 : i32, select = 1 : i32},
            {resource = 4 : i32, mode = 0 : i32}
          ]},
          {outputs = [0 : i32], resources = [
            {resource = 0 : i32, select = 1 : i32},
            {resource = 1 : i32, select = 0 : i32},
            {resource = 2 : i32, select = 1 : i32},
            {resource = 3 : i32, select = 0 : i32},
            {resource = 4 : i32, mode = 0 : i32}
          ]}
        ]} {
      %xa:2 = fabric.demux %x : !fabric.bits<32> -> 2
      %yb:2 = fabric.demux %y : !fabric.bits<32> -> 2
      %lhs = fabric.mux %xa#0, %yb#0 : !fabric.bits<32>
      %rhs = fabric.mux %xa#1, %yb#1 : !fabric.bits<32>
      %out = fabric.op [@arith.subi] (%lhs, %rhs)
          {hw_params = [{op = @arith.subi,
            function_type = (i32, i32) -> i32,
            input_ports = [0 : i32, 1 : i32],
            output_ports = [0 : i32], attributes = {}}]}
          : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %out : !fabric.bits<32>
    }
  }
  fabric.yield
}

//--- width.mlir
fabric.module @heterogeneous_physical_widths(%a : !fabric.bits<64>,
                                             %b : !fabric.bits<64>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<64>,
                      %pb = %b : !fabric.bits<64>) -> !fabric.bits<64> {
    %r = fabric.fu(%x = %pa : !fabric.bits<64> to !fabric.bits<32>,
                   %y = %pb : !fabric.bits<64>) -> !fabric.bits<64>
        attributes {valid_encodings = [{outputs = [0 : i32], resources = [
          {resource = 0 : i32, mode = 0 : i32}
        ]}]} {
      %v = fabric.op [@arith.cmpi] (%x, %y)
          {hw_params = [{op = @arith.cmpi,
            function_type = (i16, i16) -> i1,
            input_ports = [0 : i32, 1 : i32],
            output_ports = [0 : i32],
            attributes = {predicate = 0 : i64}}]}
          : (!fabric.bits<32>, !fabric.bits<64>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32> to !fabric.bits<64>
    }
  }
  fabric.yield
}

func.func @cmp_i16(%a : i16, %b : i16) -> i1
    attributes {loom.coverage_input = true} {
  %result = arith.cmpi eq, %a, %b : i16
  return %result : i1
}
