module {
  fabric.spatial_sw @tagged_merge
      [connectivity_table = ["11"]]
      : (!fabric.tagged<!fabric.bits<32>, i4>,
         !fabric.tagged<!fabric.bits<32>, i4>)
        -> (!fabric.tagged<!fabric.bits<32>, i4>)

  fabric.temporal_sw @tagged_split
      [num_route_table = 2, connectivity_table = ["1", "1"]]
      : (!fabric.tagged<!fabric.bits<32>, i3>)
        -> (!fabric.tagged<!fabric.bits<32>, i3>,
            !fabric.tagged<!fabric.bits<32>, i3>)

  fabric.module @tagged_runtime_tag_distinct_test(%a: i32, %b: i32) -> (i32, i32) {
    %tag0 = fabric.add_tag %a {tag = 1 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i4>
    %tag1 = fabric.add_tag %b {tag = 2 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i4>

    %merged:1 = fabric.instance @tagged_merge(%tag0, %tag1)
        {sym_name = "sw_merge"}
        : (!fabric.tagged<!fabric.bits<32>, i4>,
           !fabric.tagged<!fabric.bits<32>, i4>)
          -> (!fabric.tagged<!fabric.bits<32>, i4>)

    %shrunk = fabric.map_tag %merged#0
        [table_size = 16 : i64]
        attributes {table = [0 : i64, 1 : i64, 2 : i64, 3 : i64,
                             4 : i64, 5 : i64, 6 : i64, 7 : i64,
                             0 : i64, 1 : i64, 2 : i64, 3 : i64,
                             4 : i64, 5 : i64, 6 : i64, 7 : i64]}
        : !fabric.tagged<!fabric.bits<32>, i4>
          -> !fabric.tagged<!fabric.bits<32>, i3>

    %split:2 = fabric.instance @tagged_split(%shrunk)
        {sym_name = "tsw_split"}
        : (!fabric.tagged<!fabric.bits<32>, i3>)
          -> (!fabric.tagged<!fabric.bits<32>, i3>,
              !fabric.tagged<!fabric.bits<32>, i3>)

    %out0 = fabric.del_tag %split#0
        : !fabric.tagged<!fabric.bits<32>, i3> -> i32
    %out1 = fabric.del_tag %split#1
        : !fabric.tagged<!fabric.bits<32>, i3> -> i32
    fabric.yield %out0, %out1 : i32, i32
  }
}
