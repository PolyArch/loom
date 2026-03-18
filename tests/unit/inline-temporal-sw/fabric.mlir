module {
  fabric.module @inline_temporal_sw_test(%a: i32, %ctrl: none) -> (i32, none) {
    %tag = fabric.add_tag %a {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %sw:1 = fabric.temporal_sw @tsw_inline [num_route_table = 1] (%tag)
        attributes {
          route_table = [{tag = 0 : i64, input = 0 : i64, output = 0 : i64}]
        }
        : (!fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>)
    %out = fabric.del_tag %sw#0
        : !fabric.tagged<!fabric.bits<32>, i1> -> i32
    fabric.yield %out, %ctrl : i32, none
  }
}
