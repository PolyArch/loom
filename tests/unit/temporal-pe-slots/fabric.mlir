module {
  fabric.temporal_sw @in0_mux
      [num_route_table = 2, connectivity_table = ["11"]]
      : (!fabric.tagged<!fabric.bits<32>, i1>,
         !fabric.tagged<!fabric.bits<32>, i1>)
        -> (!fabric.tagged<!fabric.bits<32>, i1>)

  fabric.temporal_sw @in1_mux
      [num_route_table = 2, connectivity_table = ["11"]]
      : (!fabric.tagged<!fabric.bits<32>, i1>,
         !fabric.tagged<!fabric.bits<32>, i1>)
        -> (!fabric.tagged<!fabric.bits<32>, i1>)

  fabric.temporal_sw @out_demux
      [num_route_table = 2, connectivity_table = ["1", "1"]]
      : (!fabric.tagged<!fabric.bits<32>, i1>)
        -> (!fabric.tagged<!fabric.bits<32>, i1>,
            !fabric.tagged<!fabric.bits<32>, i1>)

  fabric.temporal_pe @tpe_slots(
      %p0: !fabric.tagged<!fabric.bits<32>, i1>,
      %p1: !fabric.tagged<!fabric.bits<32>, i1>)
      -> (!fabric.tagged<!fabric.bits<32>, i1>)
      [
        num_register = 0 : i64,
        num_instruction = 2 : i64,
        reg_fifo_depth = 0 : i64
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

  fabric.module @temporal_pe_slots_test(
      %a0: i32, %b0: i32, %a1: i32, %b1: i32) -> (i32, i32) {
    %ta0 = fabric.add_tag %a0 {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tb0 = fabric.add_tag %b0 {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %ta1 = fabric.add_tag %a1 {tag = 1 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tb1 = fabric.add_tag %b1 {tag = 1 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>

    %in0 = fabric.instance @in0_mux(%ta0, %ta1) {sym_name = "tsw_in0"}
        : (!fabric.tagged<!fabric.bits<32>, i1>,
           !fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>)
    %in1 = fabric.instance @in1_mux(%tb0, %tb1) {sym_name = "tsw_in1"}
        : (!fabric.tagged<!fabric.bits<32>, i1>,
           !fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>)

    %tout = fabric.instance @tpe_slots(%in0#0, %in1#0) {sym_name = "tpe_0"}
        : (!fabric.tagged<!fabric.bits<32>, i1>,
           !fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>)

    %out:2 = fabric.instance @out_demux(%tout#0) {sym_name = "tsw_out"}
        : (!fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>,
              !fabric.tagged<!fabric.bits<32>, i1>)
    %r0 = fabric.del_tag %out#0
        : !fabric.tagged<!fabric.bits<32>, i1> -> i32
    %r1 = fabric.del_tag %out#1
        : !fabric.tagged<!fabric.bits<32>, i1> -> i32
    fabric.yield %r0, %r1 : i32, i32
  }
}
