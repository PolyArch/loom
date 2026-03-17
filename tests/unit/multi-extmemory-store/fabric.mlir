// One hardware extmemory shared by two software store-only extmemory regions.
module {
  fabric.temporal_sw @st_addr_mux
      [num_route_table = 2, connectivity_table = ["11"]]
      : (!fabric.tagged<!fabric.bits<57>, i1>,
         !fabric.tagged<!fabric.bits<57>, i1>)
        -> (!fabric.tagged<!fabric.bits<57>, i1>)

  fabric.temporal_sw @st_data_mux
      [num_route_table = 2, connectivity_table = ["11"]]
      : (!fabric.tagged<!fabric.bits<32>, i1>,
         !fabric.tagged<!fabric.bits<32>, i1>)
        -> (!fabric.tagged<!fabric.bits<32>, i1>)

  fabric.temporal_sw @st_done_demux
      [num_route_table = 2, connectivity_table = ["1", "1"]]
      : (!fabric.tagged<none, i1>)
        -> (!fabric.tagged<none, i1>,
            !fabric.tagged<none, i1>)

  fabric.spatial_pe @store_pe(%p0: index, %p1: i32, %p2: none) -> (i32, index) {
    fabric.function_unit @fu_store(%arg0: index, %arg1: i32, %arg2: none)
        -> (i32, index) [latency = 1, interval = 1] {
      %0, %1 = handshake.store [%arg0] %arg1, %arg2 : index, i32
      fabric.yield %0, %1 : i32, index
    }
    fabric.yield
  }

  fabric.module @multi_extmemory_store_test(
      %dram: memref<?xi32>,
      %idx0: index, %val0: i32,
      %idx1: index, %val1: i32,
      %ctrl0: none, %ctrl1: none)
      -> (none, none) {
    %ext0 = fabric.extmemory @extmem_0
        [ldCount = 0, stCount = 2, lsqDepth = 0,
         memrefType = memref<?xi32>, numRegion = 2]
        (%dram, %tsw_st_addr#0, %tsw_st_data#0)
        : (memref<?xi32>,
           !fabric.tagged<!fabric.bits<57>, i1>,
           !fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<none, i1>)

    %pe0:2 = fabric.instance @store_pe(%idx0, %val0, %ctrl0)
        {sym_name = "pe_0"} : (index, i32, none) -> (i32, index)
    %pe1:2 = fabric.instance @store_pe(%idx1, %val1, %ctrl1)
        {sym_name = "pe_1"} : (index, i32, none) -> (i32, index)

    %tag_st_data0 = fabric.add_tag %pe0#0 {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tag_st_addr0 = fabric.add_tag %pe0#1 {tag = 0 : i64}
        : index -> !fabric.tagged<!fabric.bits<57>, i1>
    %tag_st_data1 = fabric.add_tag %pe1#0 {tag = 1 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tag_st_addr1 = fabric.add_tag %pe1#1 {tag = 1 : i64}
        : index -> !fabric.tagged<!fabric.bits<57>, i1>

    %tsw_st_addr:1 = fabric.instance @st_addr_mux(%tag_st_addr0, %tag_st_addr1)
        {sym_name = "tsw_0"}
        : (!fabric.tagged<!fabric.bits<57>, i1>,
           !fabric.tagged<!fabric.bits<57>, i1>)
          -> (!fabric.tagged<!fabric.bits<57>, i1>)
    %tsw_st_data:1 = fabric.instance @st_data_mux(%tag_st_data0, %tag_st_data1)
        {sym_name = "tsw_1"}
        : (!fabric.tagged<!fabric.bits<32>, i1>,
           !fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>)
    %tsw_done:2 = fabric.instance @st_done_demux(%ext0)
        {sym_name = "tsw_2"}
        : (!fabric.tagged<none, i1>)
          -> (!fabric.tagged<none, i1>,
              !fabric.tagged<none, i1>)

    %done0 = fabric.del_tag %tsw_done#0
        : !fabric.tagged<none, i1> -> none
    %done1 = fabric.del_tag %tsw_done#1
        : !fabric.tagged<none, i1> -> none

    fabric.yield %done0, %done1 : none, none
  }
}
