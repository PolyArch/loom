// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_PE_INSTANCE_ONLY_BODY

// A fabric.pe whose body is solely a single fabric.instance with no other ops.
fabric.pe @target(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>)
    [latency = [1 : i16, 1 : i16, 1 : i16]]
    -> (!dataflow.bits<32>) {
  ^bb0(%a: i32, %b: i32):
  %r = arith.addi %a, %b : i32
  fabric.yield %r : i32
}

fabric.module @test(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %r = fabric.pe %a, %b : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%x: i32, %y: i32):
    %t = fabric.instance @target(%x, %y) : (i32, i32) -> (i32)
    fabric.yield %t : i32
  }
  fabric.yield %r : !dataflow.bits<32>
}
