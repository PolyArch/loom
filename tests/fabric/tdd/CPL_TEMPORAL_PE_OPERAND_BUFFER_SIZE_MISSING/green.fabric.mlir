// RUN: loom --adg %s

// Valid: shared operand buffer mode with operand_buffer_size provided.
fabric.temporal_pe @tpe_shared_buf(%in: !dataflow.tagged<!dataflow.bits<32>, i4>)
  [num_register = 0, num_instruction = 2, reg_fifo_depth = 0,
   enable_share_operand_buffer = true, operand_buffer_size = 128]
  -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
  fabric.pe @fu0(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
    ^bb0(%a: i32):
    %r = arith.addi %a, %a : i32
    fabric.yield %r : i32
  }
  fabric.yield
}

fabric.module @test(%a: !dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
  %out = fabric.instance @tpe_shared_buf(%a) : (!dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.tagged<!dataflow.bits<32>, i4>)
  fabric.yield %out : !dataflow.tagged<!dataflow.bits<32>, i4>
}
