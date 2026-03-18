module {
  fabric.spatial_sw @addr_leaf0
      [connectivity_table = ["1"]]
      : (!fabric.tagged<!fabric.bits<57>, i1>)
        -> (!fabric.tagged<!fabric.bits<57>, i1>)

  fabric.spatial_sw @addr_leaf1
      [connectivity_table = ["1"]]
      : (!fabric.tagged<!fabric.bits<57>, i1>)
        -> (!fabric.tagged<!fabric.bits<57>, i1>)

  fabric.spatial_sw @addr_root
      [connectivity_table = ["11"]]
      : (!fabric.tagged<!fabric.bits<57>, i1>,
         !fabric.tagged<!fabric.bits<57>, i1>)
        -> (!fabric.tagged<!fabric.bits<57>, i1>)

  fabric.temporal_sw @ld_data_demux
      [num_route_table = 2, connectivity_table = ["1", "1"]]
      : (!fabric.tagged<!fabric.bits<32>, i1>)
        -> (!fabric.tagged<!fabric.bits<32>, i1>,
            !fabric.tagged<!fabric.bits<32>, i1>)

  fabric.temporal_sw @ld_done_demux
      [num_route_table = 2, connectivity_table = ["1", "1"]]
      : (!fabric.tagged<none, i1>)
        -> (!fabric.tagged<none, i1>,
            !fabric.tagged<none, i1>)

  fabric.spatial_pe @load_pe(%p0: index, %p1: i32, %p2: none) -> (i32, index) {
    fabric.function_unit @fu_load(%arg0: index, %arg1: i32, %arg2: none)
        -> (i32, index) [latency = 1, interval = 1] {
      %0, %1 = handshake.load [%arg0] %arg1, %arg2 : index, i32
      fabric.yield %0, %1 : i32, index
    }
    fabric.yield
  }

  fabric.module @extmemory_tagged_map_hierarchy_test(
      %dram: memref<?xi32>,
      %idx0: index, %idx1: index,
      %ctrl0: none, %ctrl1: none)
      -> (i32, i32, none, none) {
    %pe0:2 = fabric.instance @load_pe(%idx0, %lddata0, %ctrl0)
        {sym_name = "pe_0"} : (index, i32, none) -> (i32, index)
    %pe1:2 = fabric.instance @load_pe(%idx1, %lddata1, %ctrl1)
        {sym_name = "pe_1"} : (index, i32, none) -> (i32, index)

    %tag_addr0 = fabric.add_tag %pe0#1 {tag = 0 : i64}
        : index -> !fabric.tagged<!fabric.bits<57>, i1>
    %tag_addr1_raw = fabric.add_tag %pe1#1 {tag = 0 : i64}
        : index -> !fabric.tagged<!fabric.bits<57>, i1>
    %tag_addr1 = fabric.map_tag %tag_addr1_raw
        [table_size = 2 : i64] attributes {table = [1 : i64, 0 : i64]}
        : !fabric.tagged<!fabric.bits<57>, i1>
          -> !fabric.tagged<!fabric.bits<57>, i1>

    %addr_l0:1 = fabric.instance @addr_leaf0(%tag_addr0)
        {sym_name = "sw_leaf0"}
        : (!fabric.tagged<!fabric.bits<57>, i1>)
          -> (!fabric.tagged<!fabric.bits<57>, i1>)
    %addr_l1:1 = fabric.instance @addr_leaf1(%tag_addr1)
        {sym_name = "sw_leaf1"}
        : (!fabric.tagged<!fabric.bits<57>, i1>)
          -> (!fabric.tagged<!fabric.bits<57>, i1>)
    %sw_ld_addr:1 = fabric.instance @addr_root(%addr_l0#0, %addr_l1#0)
        {sym_name = "sw_root"}
        : (!fabric.tagged<!fabric.bits<57>, i1>,
           !fabric.tagged<!fabric.bits<57>, i1>)
          -> (!fabric.tagged<!fabric.bits<57>, i1>)

    %ext0:2 = fabric.extmemory @extmem_0
        [ldCount = 2, stCount = 0, lsqDepth = 0,
         memrefType = memref<?xi32>, numRegion = 1]
        (%dram, %sw_ld_addr#0)
        : (memref<?xi32>, !fabric.tagged<!fabric.bits<57>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>,
              !fabric.tagged<none, i1>)

    %ld_data_tag = fabric.map_tag %ext0#0
        [table_size = 2 : i64] attributes {table = [1 : i64, 0 : i64]}
        : !fabric.tagged<!fabric.bits<32>, i1>
          -> !fabric.tagged<!fabric.bits<32>, i1>
    %ld_done_tag = fabric.map_tag %ext0#1
        [table_size = 2 : i64] attributes {table = [1 : i64, 0 : i64]}
        : !fabric.tagged<none, i1>
          -> !fabric.tagged<none, i1>

    %tsw_ld_data:2 = fabric.instance @ld_data_demux(%ld_data_tag)
        {sym_name = "tsw_0"}
        : (!fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>,
              !fabric.tagged<!fabric.bits<32>, i1>)
    %tsw_ld_done:2 = fabric.instance @ld_done_demux(%ld_done_tag)
        {sym_name = "tsw_1"}
        : (!fabric.tagged<none, i1>)
          -> (!fabric.tagged<none, i1>,
              !fabric.tagged<none, i1>)

    %lddata0 = fabric.del_tag %tsw_ld_data#0
        : !fabric.tagged<!fabric.bits<32>, i1> -> i32
    %lddata1 = fabric.del_tag %tsw_ld_data#1
        : !fabric.tagged<!fabric.bits<32>, i1> -> i32
    %lddone0 = fabric.del_tag %tsw_ld_done#0
        : !fabric.tagged<none, i1> -> none
    %lddone1 = fabric.del_tag %tsw_ld_done#1
        : !fabric.tagged<none, i1> -> none

    fabric.yield %pe0#0, %pe1#0, %lddone0, %lddone1
        : i32, i32, none, none
  }
}
