// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_TEMPORAL_PE_REG_DISABLED

// num_register = 0 but instruction_mem uses reg(0) as a source.
fabric.temporal_pe @tpe_bad(%in0: !dataflow.tagged<!dataflow.bits<32>, i4>, %in1: !dataflow.tagged<!dataflow.bits<32>, i4>)
  [num_register = 0, num_instruction = 2, reg_fifo_depth = 0]
  {instruction_mem = ["inst[0]: when(tag=3) out(0, tag=1) = add(0) reg(0), in(1)"]}
  -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
  fabric.pe @fu_add(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
    ^bb0(%a: i32, %b: i32):
    %r = arith.addi %a, %b : i32
    fabric.yield %r : i32
  }
  fabric.yield
}

fabric.module @test(%a: !dataflow.tagged<!dataflow.bits<32>, i4>, %b: !dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
  %out = fabric.instance @tpe_bad(%a, %b) : (!dataflow.tagged<!dataflow.bits<32>, i4>, !dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.tagged<!dataflow.bits<32>, i4>)
  fabric.yield %out : !dataflow.tagged<!dataflow.bits<32>, i4>
}
