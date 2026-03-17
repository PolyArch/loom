// ADG with one used spatial PE and one unused temporal PE to test flatten + viz.
module {
  fabric.spatial_pe @sp_add(%p0: i32, %p1: i32) -> (i32) {
    fabric.function_unit @fu_add(%a: i32, %b: i32) -> (i32)
        [latency = 1, interval = 1] {
      %r = arith.addi %a, %b : i32
      fabric.yield %r : i32
    }
    fabric.yield
  }

  fabric.temporal_pe @tpe_mul(
      %p0: !fabric.tagged<!fabric.bits<32>, i1>,
      %p1: !fabric.tagged<!fabric.bits<32>, i1>)
      -> (!fabric.tagged<!fabric.bits<32>, i1>)
      attributes {
        num_register = 0 : i64,
        num_instruction = 1 : i64,
        reg_fifo_depth = 0 : i64
      } {
    fabric.function_unit @fu_mul(%a: i32, %b: i32) -> (i32)
        [latency = 1, interval = 1] {
      %r = arith.muli %a, %b : i32
      fabric.yield %r : i32
    }
    fabric.yield
  }

  fabric.module @temporal_pe_gui_test(%a: i32, %b: i32) -> (i32) {
    %sum = fabric.instance @sp_add(%a, %b) {sym_name = "pe_0"}
        : (i32, i32) -> (i32)

    %tag0 = fabric.add_tag %a {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tag1 = fabric.add_tag %b {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tmul = fabric.instance @tpe_mul(%tag0, %tag1) {sym_name = "tpe_0"}
        : (!fabric.tagged<!fabric.bits<32>, i1>,
           !fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>)
    %tmul_mapped = fabric.map_tag %tmul {table_size = 2 : i64}
        : !fabric.tagged<!fabric.bits<32>, i1>
          -> !fabric.tagged<!fabric.bits<32>, i1>
    %mul = fabric.del_tag %tmul_mapped
        : !fabric.tagged<!fabric.bits<32>, i1> -> i32

    fabric.yield %sum : i32
  }
}
