module {
  fabric.module @temporal_components_def_in_module_test(%a: i32, %b: i32,
      %c: i32) -> (i32) {
    fabric.temporal_sw @tsw_def [num_route_table = 1]
        attributes {
          route_table = [{tag = 0 : i64, input = 0 : i64, output = 0 : i64}]
        }
        : (!fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>)

    fabric.temporal_pe @tpe_def(
        %p0: !fabric.tagged<!fabric.bits<32>, i1>,
        %p1: !fabric.tagged<!fabric.bits<32>, i1>,
        %p2: !fabric.tagged<!fabric.bits<32>, i1>)
        -> (!fabric.tagged<!fabric.bits<32>, i1>)
        [num_register = 0 : i64, num_instruction = 1 : i64,
         reg_fifo_depth = 0 : i64] {
      fabric.function_unit @fu_add(%x: i32, %y: i32, %z: i32) -> (i32)
          [latency = 1, interval = 1] {
        %sum = arith.addi %x, %y : i32
        %unused = arith.xori %z, %z : i32
        fabric.yield %sum : i32
      }
      fabric.yield
    }

    %ta = fabric.add_tag %a {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tb = fabric.add_tag %b {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tc = fabric.add_tag %c {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tsa = fabric.instance @tsw_def(%ta) {sym_name = "tsw_a"}
        : (!fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>)
    %tsb = fabric.instance @tsw_def(%tb) {sym_name = "tsw_b"}
        : (!fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>)
    %tsc = fabric.instance @tsw_def(%tc) {sym_name = "tsw_c"}
        : (!fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>)
    %tout = fabric.instance @tpe_def(%tsa, %tsb, %tsc) {sym_name = "tpe_0"}
        : (!fabric.tagged<!fabric.bits<32>, i1>,
           !fabric.tagged<!fabric.bits<32>, i1>,
           !fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>)
    %out = fabric.del_tag %tout
        : !fabric.tagged<!fabric.bits<32>, i1> -> i32
    fabric.yield %out : i32
  }
}
