// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

fabric.module @swapped_sub(%a : !fabric.bits<32>,
                           %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                      %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>,
              %y = %pb : !fabric.bits<32>) -> !fabric.bits<32>
        attributes {valid_encodings = [{outputs = [0 : i32], resources = [
          {resource = 0 : i32, mode = 0 : i32}
        ]}]} {
      %r = fabric.op [@arith.subi] (%y, %x)
           {hw_params = [{op = @arith.subi,
             function_type = (i32, i32) -> i32,
             input_ports = [0 : i32, 1 : i32],
             output_ports = [0 : i32],
             attributes = {overflowFlags = #arith.overflow<none>}}]}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %r : !fabric.bits<32>
    }
  }
  fabric.yield
}

func.func @pattern(%a: i32, %b: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.actor_to_fabric_op = array<i32: 0>
  // CHECK-SAME: loom.input_port_correspondence = array<i32: 0, 1, 1, 0>
  // CHECK-SAME: loom.matched_encoding = 0 : i64
  // CHECK-SAME: loom.matched_fu = "@swapped_sub#0"
  // CHECK-SAME: loom.output_port_correspondence = array<i32: 0, 0>
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32
       attributes {loom.is_pattern} {
    %v = arith.subi %x, %y : i32
    dataflow.yield %v : i32
  }
  return %r : i32
}
