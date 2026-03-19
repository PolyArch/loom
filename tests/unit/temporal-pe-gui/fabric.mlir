module {
  fabric.temporal_pe @tpe_test(
      %p0: !fabric.tagged<!fabric.bits<32>, i1>,
      %p1: !fabric.tagged<!fabric.bits<32>, i1>,
      %p2: !fabric.tagged<!fabric.bits<32>, i1>)
      -> (!fabric.tagged<!fabric.bits<32>, i1>)
      [
        num_register = 0 : i64,
        num_instruction = 2 : i64,
        reg_fifo_depth = 0 : i64
      ] {
    fabric.function_unit @fu_add(%a: i32, %b: i32, %c: i32) -> (i32)
        [latency = 1, interval = 1] {
      %r = arith.addi %a, %b : i32
      %unused = arith.xori %c, %c : i32
      fabric.yield %r : i32
    }
    fabric.function_unit @fu_mac(%a: i32, %b: i32, %c: i32) -> (i32)
        [latency = 1, interval = 1] {
      %m = arith.muli %a, %b : i32
      %s = arith.addi %m, %c : i32
      %o = fabric.mux %m, %s
          {sel = 0 : i64, discard = false, disconnect = false}
          : i32, i32 -> i32
      fabric.yield %o : i32
    }
    fabric.yield
  }

  fabric.module @temporal_pe_gui_test(%a: i32, %b: i32, %c: i32) -> (i32) {
    %ta = fabric.add_tag %a {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tb = fabric.add_tag %b {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tc = fabric.add_tag %c {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tout = fabric.instance @tpe_test(%ta, %tb, %tc) {sym_name = "tpe_0"}
        : (!fabric.tagged<!fabric.bits<32>, i1>,
           !fabric.tagged<!fabric.bits<32>, i1>,
           !fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>)
    %out = fabric.del_tag %tout
        : !fabric.tagged<!fabric.bits<32>, i1> -> i32
    fabric.yield %out : i32
  }
}
