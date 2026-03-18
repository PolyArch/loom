module {
  fabric.spatial_sw @tagged_sw
      [connectivity_table = ["11", "11"]]
      : (!fabric.tagged<!fabric.bits<32>, i1>,
         !fabric.tagged<!fabric.bits<32>, i1>)
        -> (!fabric.tagged<!fabric.bits<32>, i1>,
            !fabric.tagged<!fabric.bits<32>, i1>)

  fabric.temporal_pe @tpe_add(
      %p0: !fabric.tagged<!fabric.bits<32>, i1>,
      %p1: !fabric.tagged<!fabric.bits<32>, i1>)
      -> (!fabric.tagged<!fabric.bits<32>, i1>)
      [
        num_register = 0 : i64,
        num_instruction = 1 : i64,
        reg_fifo_depth = 0 : i64
      ] {
    fabric.function_unit @fu_add(%a: i32, %b: i32) -> (i32)
        [latency = 1, interval = 1] {
      %r = arith.addi %a, %b : i32
      fabric.yield %r : i32
    }
    fabric.yield
  }

  fabric.module @tagged_spatial_sw_test(%a: i32, %b: i32) -> (i32) {
    %ta = fabric.add_tag %a {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tb = fabric.add_tag %b {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %sw:2 = fabric.instance @tagged_sw(%ta, %tb) {sym_name = "sw_0"}
        : (!fabric.tagged<!fabric.bits<32>, i1>,
           !fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>,
              !fabric.tagged<!fabric.bits<32>, i1>)
    %tout = fabric.instance @tpe_add(%sw#0, %sw#1) {sym_name = "tpe_0"}
        : (!fabric.tagged<!fabric.bits<32>, i1>,
           !fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>)
    %out = fabric.del_tag %tout
        : !fabric.tagged<!fabric.bits<32>, i1> -> i32
    fabric.yield %out : i32
  }
}
