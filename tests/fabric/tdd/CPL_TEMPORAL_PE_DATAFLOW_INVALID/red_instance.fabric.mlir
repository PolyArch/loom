// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_TEMPORAL_PE_DATAFLOW_INVALID

// Invalid: temporal_pe wraps an external dataflow PE via fabric.instance.
// The external PE uses dataflow.invariant, which is not allowed inside temporal_pe.
fabric.pe @df_pe(%d: i1, %a: i32) -> (i32) {
  %r = dataflow.invariant %d, %a : i1, i32 -> i32
  fabric.yield %r : i32
}

fabric.temporal_pe @tpe_bad(%in0: !dataflow.tagged<i32, i4>, %in1: !dataflow.tagged<i32, i4>)
  [num_register = 0, num_instruction = 2, reg_fifo_depth = 0]
  -> (!dataflow.tagged<i32, i4>) {
  fabric.pe @fu0(%d: i1, %a: i32) -> (i32) {
    %r = fabric.instance @df_pe(%d, %a) : (i1, i32) -> (i32)
    fabric.yield %r : i32
  }
  fabric.yield
}

fabric.module @test(%a: !dataflow.tagged<i32, i4>, %b: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>) {
  %out = fabric.instance @tpe_bad(%a, %b)
      : (!dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>)
  fabric.yield %out : !dataflow.tagged<i32, i4>
}
