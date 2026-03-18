module {
  %c0 = arith.constant 0 : i64
  %ta = builtin.unrealized_conversion_cast %c0 : i64
      to !fabric.tagged<!fabric.bits<32>, i1>
  %tb = builtin.unrealized_conversion_cast %c0 : i64
      to !fabric.tagged<!fabric.bits<32>, i1>
  %tc = builtin.unrealized_conversion_cast %c0 : i64
      to !fabric.tagged<!fabric.bits<32>, i1>

  %tpe:1 = fabric.temporal_pe @tpe_inline
      [num_register = 0 : i64, num_instruction = 1 : i64,
       reg_fifo_depth = 0 : i64] inputs(%ta, %tb, %tc)
      : (!fabric.tagged<!fabric.bits<32>, i1>,
         !fabric.tagged<!fabric.bits<32>, i1>,
         !fabric.tagged<!fabric.bits<32>, i1>)
        -> (!fabric.tagged<!fabric.bits<32>, i1>) {
    fabric.function_unit @fu_add(%a: i32, %b: i32, %c: i32) -> (i32)
        [latency = 1, interval = 1] {
      %r = arith.addi %a, %b : i32
      fabric.yield %r : i32
    }
    fabric.yield
  }
}
