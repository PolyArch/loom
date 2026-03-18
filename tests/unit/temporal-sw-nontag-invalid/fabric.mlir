module {
  fabric.temporal_sw @bad_tsw
      [num_route_table = 1, connectivity_table = ["1"]]
      : (!fabric.bits<32>) -> (!fabric.bits<32>)

  fabric.module @temporal_sw_nontag_invalid_test(%a: i32) -> (i32) {
    %ta = fabric.add_tag %a {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %u = fabric.del_tag %ta
        : !fabric.tagged<!fabric.bits<32>, i1> -> i32
    fabric.yield %u : i32
  }
}
