// One hardware extmemory hosting one load region and one store region.
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

  fabric.spatial_pe @load_pe(%p0: index, %p1: i32, %p2: none) -> (i32, index) {
    fabric.function_unit @fu_load(%arg0: index, %arg1: i32, %arg2: none)
        -> (i32, index) [latency = 1, interval = 1] {
      %0, %1 = handshake.load [%arg0] %arg1, %arg2 : index, i32
      fabric.yield %0, %1 : i32, index
    }
    fabric.yield
  }

  fabric.spatial_pe @store_pe(%p0: index, %p1: i32, %p2: none) -> (i32, index) {
    fabric.function_unit @fu_store(%arg0: index, %arg1: i32, %arg2: none)
        -> (i32, index) [latency = 1, interval = 1] {
      %0, %1 = handshake.store [%arg0] %arg1, %arg2 : index, i32
      fabric.yield %0, %1 : i32, index
    }
    fabric.yield
  }

  fabric.module @mixed_extmemory_loadstore_test(
      %dram: memref<?xi32>,
      %idx_ld: index, %idx_st: index, %val_st: i32,
      %ctrl_ld: none, %ctrl_st: none)
      -> (i32, none, none) {
    %tag_ld_addr = fabric.add_tag %pe_ld#1 {tag = 0 : i64}
        : index -> !fabric.tagged<!fabric.bits<57>, i1>
    %tag_st_data0 = fabric.add_tag %pe_st#0 {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tag_st_addr0 = fabric.add_tag %pe_st#1 {tag = 0 : i64}
        : index -> !fabric.tagged<!fabric.bits<57>, i1>
    %tag_st_data1 = fabric.add_tag %pe_st#0 {tag = 1 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tag_st_addr1 = fabric.add_tag %pe_st#1 {tag = 1 : i64}
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

    %ext0:3 = fabric.extmemory @extmem_0
        [ldCount = 1, stCount = 2, lsqDepth = 0,
         memrefType = memref<?xi32>, numRegion = 2]
        (%dram, %tag_ld_addr, %tsw_st_addr#0, %tsw_st_data#0)
        : (memref<?xi32>,
           !fabric.tagged<!fabric.bits<57>, i1>,
           !fabric.tagged<!fabric.bits<57>, i1>,
           !fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>,
              !fabric.tagged<none, i1>,
              !fabric.tagged<none, i1>)

    %pe_ld:2 = fabric.instance @load_pe(%idx_ld, %lddata, %ctrl_ld)
        {sym_name = "pe_ld"} : (index, i32, none) -> (i32, index)
    %pe_st:2 = fabric.instance @store_pe(%idx_st, %val_st, %ctrl_st)
        {sym_name = "pe_st"} : (index, i32, none) -> (i32, index)
    %tsw_st_done:2 = fabric.instance @st_done_demux(%ext0#2)
        {sym_name = "tsw_2"}
        : (!fabric.tagged<none, i1>)
          -> (!fabric.tagged<none, i1>,
              !fabric.tagged<none, i1>)

    %lddata = fabric.del_tag %ext0#0
        : !fabric.tagged<!fabric.bits<32>, i1> -> i32
    %lddone = fabric.del_tag %ext0#1
        : !fabric.tagged<none, i1> -> none
    %stdone0 = fabric.del_tag %tsw_st_done#0
        : !fabric.tagged<none, i1> -> none
    %stdone = fabric.del_tag %tsw_st_done#1
        : !fabric.tagged<none, i1> -> none

    fabric.yield %pe_ld#0, %lddone, %stdone : i32, none, none
  }
}
