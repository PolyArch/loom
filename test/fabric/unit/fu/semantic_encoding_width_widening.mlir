// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// CHECK: func.func private @fu0_subgraph_0(%{{.*}}: i16, %{{.*}}: i16) -> i16
// CHECK: func.func private @fu1_subgraph_0(%{{.*}}: i16, %{{.*}}: i16) -> i1

fabric.module @width_widening(%a : !fabric.bits<64>,
                              %b : !fabric.bits<64>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<64>,
                      %pb = %b : !fabric.bits<64>) -> !fabric.bits<64> {
    %r = fabric.fu(%x = %pa : !fabric.bits<64> to !fabric.bits<32>,
                   %y = %pb : !fabric.bits<64> to !fabric.bits<32>)
        -> !fabric.bits<64>
        attributes {valid_encodings = [{outputs = [0 : i32], resources = [
          {resource = 0 : i32, mode = 0 : i32}
        ]}]} {
      %v = fabric.op [@arith.addi] (%x, %y)
           {hw_params = [{op = @arith.addi,
             function_type = (i16, i16) -> i16,
             input_ports = [0 : i32, 1 : i32],
             output_ports = [0 : i32], attributes = {}}]}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32> to !fabric.bits<64>
    }
  }
  fabric.yield
}

// A canonical typed mode may use different physical payload capacities for
// ports tied to one software type parameter. Fixed-width software results may
// also use wider physical payloads.
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
