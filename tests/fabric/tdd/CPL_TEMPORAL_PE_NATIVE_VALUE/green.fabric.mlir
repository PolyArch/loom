// RUN: loom --adg %s

// Temporal PE with tagged<bits<32>, i4> (correct).
fabric.temporal_pe @tpe_good(%in: !dataflow.tagged<!dataflow.bits<32>, i4>)
  [num_register = 0, num_instruction = 2, reg_fifo_depth = 0]
  -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
  fabric.pe @fu_add(%arg0: i32, %arg1: i32) -> (i32) {
    %r = arith.addi %arg0, %arg1 : i32
    fabric.yield %r : i32
  }
  fabric.yield
}

fabric.module @test(%a: !dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
  %out = fabric.instance @tpe_good(%a) : (!dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.tagged<!dataflow.bits<32>, i4>)
  fabric.yield %out : !dataflow.tagged<!dataflow.bits<32>, i4>
}
