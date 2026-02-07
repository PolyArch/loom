// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_TEMPORAL_PE_EMPTY_BODY

// Body contains only the terminator (fabric.yield), no FU definitions.
fabric.temporal_pe @tpe_empty(%in: !dataflow.tagged<i32, i4>)
  [num_register = 0, num_instruction = 2, reg_fifo_depth = 0]
  -> (!dataflow.tagged<i32, i4>) {
  fabric.yield
}

fabric.module @test(%a: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>) {
  %out = fabric.instance @tpe_empty(%a) : (!dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>)
  fabric.yield %out : !dataflow.tagged<i32, i4>
}
