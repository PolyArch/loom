module {
  %c0 = arith.constant 0 : i64
  %tag = builtin.unrealized_conversion_cast %c0 : i64
      to !fabric.tagged<!fabric.bits<32>, i1>

  %sw:1 = fabric.temporal_sw @tsw_inline [num_route_table = 1] (%tag)
      attributes {
        route_table = [{tag = 0 : i64, input = 0 : i64, output = 0 : i64}]
      }
      : (!fabric.tagged<!fabric.bits<32>, i1>)
        -> (!fabric.tagged<!fabric.bits<32>, i1>)
}
