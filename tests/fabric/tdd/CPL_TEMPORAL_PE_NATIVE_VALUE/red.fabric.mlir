// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_TEMPORAL_PE_NATIVE_VALUE

// Temporal PE with tagged<i32, i4> (native value; must use tagged<bits<32>, i4>).
fabric.temporal_pe @tpe_bad(%in: !dataflow.tagged<i32, i4>)
  [num_register = 0, num_instruction = 2, reg_fifo_depth = 0]
  -> (!dataflow.tagged<i32, i4>) {
  fabric.pe @fu_add(%arg0: i32, %arg1: i32) -> (i32) {
    %r = arith.addi %arg0, %arg1 : i32
    fabric.yield %r : i32
  }
  fabric.yield
}

fabric.module @test(%a: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>) {
  %out = fabric.instance @tpe_bad(%a) : (!dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>)
  fabric.yield %out : !dataflow.tagged<i32, i4>
}
