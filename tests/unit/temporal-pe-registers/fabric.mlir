module {
  fabric.temporal_sw @in0_mux
      [num_route_table = 2, connectivity_table = ["11"]]
      : (!fabric.tagged<!fabric.bits<32>, i1>,
         !fabric.tagged<!fabric.bits<32>, i1>)
        -> (!fabric.tagged<!fabric.bits<32>, i1>)

  fabric.temporal_pe @tpe_regs(
      %p0: !fabric.tagged<!fabric.bits<32>, i1>,
      %p1: !fabric.tagged<!fabric.bits<32>, i1>)
      -> (!fabric.tagged<!fabric.bits<32>, i1>)
      [
        num_register = 2 : i64,
        num_instruction = 2 : i64,
        reg_fifo_depth = 2 : i64
      ] {
    fabric.function_unit @fu_add(%a: i32, %b: i32) -> (i32)
        [latency = 1, interval = 1] {
      %r = arith.addi %a, %b : i32
      fabric.yield %r : i32
    }
    fabric.function_unit @fu_mul(%a: i32, %b: i32) -> (i32)
        [latency = 1, interval = 1] {
      %r = arith.muli %a, %b : i32
      fabric.yield %r : i32
    }
    fabric.yield
  }

  fabric.module @temporal_pe_register_test(%a: i32, %b: i32, %c: i32) -> (i32) {
    %ta = fabric.add_tag %a {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tb = fabric.add_tag %b {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tc = fabric.add_tag %c {tag = 1 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>

    %in0 = fabric.instance @in0_mux(%ta, %tc) {sym_name = "tsw_in0"}
        : (!fabric.tagged<!fabric.bits<32>, i1>,
           !fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>)

    %tout = fabric.instance @tpe_regs(%in0#0, %tb) {sym_name = "tpe_0"}
        : (!fabric.tagged<!fabric.bits<32>, i1>,
           !fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>)

    %out = fabric.del_tag %tout#0
        : !fabric.tagged<!fabric.bits<32>, i1> -> i32
    fabric.yield %out : i32
  }
}
