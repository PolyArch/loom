module {
  fabric.module @inline_named_temporal_pe_test(%a: i32, %b: i32, %c: i32)
      -> (i32) {
    %ta = fabric.add_tag %a {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tb = fabric.add_tag %b {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tc = fabric.add_tag %c {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>

    %tout = fabric.temporal_pe @tpe_inline
        [num_register = 0 : i64, num_instruction = 1 : i64,
         reg_fifo_depth = 0 : i64] inputs(%ta, %tb, %tc)
        : (!fabric.tagged<!fabric.bits<32>, i1>,
           !fabric.tagged<!fabric.bits<32>, i1>,
           !fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>) {
      fabric.function_unit @fu_add(%x: i32, %y: i32, %z: i32) -> (i32)
          [latency = 1, interval = 1] {
        %sum = arith.addi %x, %y : i32
        fabric.yield %sum : i32
      }
      fabric.yield
    }

    %out = fabric.del_tag %tout
        : !fabric.tagged<!fabric.bits<32>, i1> -> i32
    fabric.yield %out : i32
  }
}
