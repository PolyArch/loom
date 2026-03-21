// Test: temporal PE with one instruction slot and a single addi FU
module {
  fabric.function_unit @fu_add(%a: i32, %b: i32, %c: i32) -> (i32)
      [latency = 1, interval = 1] {
    %r = arith.addi %a, %b : i32
    %unused = arith.xori %c, %c : i32
    fabric.yield %r : i32
  }

  fabric.temporal_pe @tpe_def(
      %p0: !fabric.tagged<!fabric.bits<32>, i1>,
      %p1: !fabric.tagged<!fabric.bits<32>, i1>,
      %p2: !fabric.tagged<!fabric.bits<32>, i1>)
      -> (!fabric.tagged<!fabric.bits<32>, i1>)
      [num_register = 0 : i64, num_instruction = 1 : i64, reg_fifo_depth = 0 : i64] {
    fabric.instance @fu_add() {sym_name = "fu_add_0"} : () -> ()
    fabric.yield
  }

  fabric.module @test_temporal_pe_single_instruction(
      %a: i32, %b: i32, %c: i32) -> (i32) {
    %ta = fabric.add_tag %a {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tb = fabric.add_tag %b {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tc = fabric.add_tag %c {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tout = fabric.instance @tpe_def(%ta, %tb, %tc) {sym_name = "tpe_0"}
        : (!fabric.tagged<!fabric.bits<32>, i1>,
           !fabric.tagged<!fabric.bits<32>, i1>,
           !fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>)
    %out = fabric.del_tag %tout
        : !fabric.tagged<!fabric.bits<32>, i1> -> i32
    fabric.yield %out : i32
  }
}
