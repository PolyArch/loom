module {
  fabric.spatial_sw @ld_addr_mux
      [connectivity_table = ["11"]]
      : (!fabric.tagged<!fabric.bits<57>, i1>,
         !fabric.tagged<!fabric.bits<57>, i1>)
        -> (!fabric.tagged<!fabric.bits<57>, i1>)

  fabric.spatial_sw @st_addr_mux
      [connectivity_table = ["11"]]
      : (!fabric.tagged<!fabric.bits<57>, i1>,
         !fabric.tagged<!fabric.bits<57>, i1>)
        -> (!fabric.tagged<!fabric.bits<57>, i1>)

  fabric.spatial_sw @st_data_mux
      [connectivity_table = ["11"]]
      : (!fabric.tagged<!fabric.bits<32>, i1>,
         !fabric.tagged<!fabric.bits<32>, i1>)
        -> (!fabric.tagged<!fabric.bits<32>, i1>)

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

  fabric.module @memory_tagged_spatial_ingress_test(
      %idx_ld0: index, %idx_ld1: index,
      %idx_st0: index, %val_st0: i32,
      %idx_st1: index, %val_st1: i32,
      %ctrl_ld0: none, %ctrl_ld1: none,
      %ctrl_st0: none, %ctrl_st1: none)
      -> (i32, i32, none, none, none, none) {
    %pe_ld0:2 = fabric.instance @load_pe(%idx_ld0, %lddata0, %ctrl_ld0)
        {sym_name = "pe_ld0"} : (index, i32, none) -> (i32, index)
    %pe_ld1:2 = fabric.instance @load_pe(%idx_ld1, %lddata1, %ctrl_ld1)
        {sym_name = "pe_ld1"} : (index, i32, none) -> (i32, index)
    %pe_st0:2 = fabric.instance @store_pe(%idx_st0, %val_st0, %ctrl_st0)
        {sym_name = "pe_st0"} : (index, i32, none) -> (i32, index)
    %pe_st1:2 = fabric.instance @store_pe(%idx_st1, %val_st1, %ctrl_st1)
        {sym_name = "pe_st1"} : (index, i32, none) -> (i32, index)

    %tag_ld_addr0 = fabric.add_tag %pe_ld0#1 {tag = 0 : i64}
        : index -> !fabric.tagged<!fabric.bits<57>, i1>
    %tag_ld_addr1 = fabric.add_tag %pe_ld1#1 {tag = 1 : i64}
        : index -> !fabric.tagged<!fabric.bits<57>, i1>
    %tag_st_data0 = fabric.add_tag %pe_st0#0 {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tag_st_addr0 = fabric.add_tag %pe_st0#1 {tag = 0 : i64}
        : index -> !fabric.tagged<!fabric.bits<57>, i1>
    %tag_st_data1 = fabric.add_tag %pe_st1#0 {tag = 1 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %tag_st_addr1 = fabric.add_tag %pe_st1#1 {tag = 1 : i64}
        : index -> !fabric.tagged<!fabric.bits<57>, i1>

    %sw_ld_addr:1 = fabric.instance @ld_addr_mux(%tag_ld_addr0, %tag_ld_addr1)
        {sym_name = "sw_0"}
        : (!fabric.tagged<!fabric.bits<57>, i1>,
           !fabric.tagged<!fabric.bits<57>, i1>)
          -> (!fabric.tagged<!fabric.bits<57>, i1>)
    %sw_st_addr:1 = fabric.instance @st_addr_mux(%tag_st_addr0, %tag_st_addr1)
        {sym_name = "sw_1"}
        : (!fabric.tagged<!fabric.bits<57>, i1>,
           !fabric.tagged<!fabric.bits<57>, i1>)
          -> (!fabric.tagged<!fabric.bits<57>, i1>)
    %sw_st_data:1 = fabric.instance @st_data_mux(%tag_st_data0, %tag_st_data1)
        {sym_name = "sw_2"}
        : (!fabric.tagged<!fabric.bits<32>, i1>,
           !fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>)

    %mem0:3 = fabric.memory @mem_0
        [ldCount = 2, stCount = 2, lsqDepth = 0,
         memrefType = memref<256xi32>, numRegion = 1]
        (%sw_ld_addr#0, %sw_st_addr#0, %sw_st_data#0)
        : (!fabric.tagged<!fabric.bits<57>, i1>,
           !fabric.tagged<!fabric.bits<57>, i1>,
           !fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>,
              !fabric.tagged<none, i1>,
              !fabric.tagged<none, i1>)

    %tsw_ld_data:2 = fabric.instance @ld_data_demux(%mem0#0)
        {sym_name = "tsw_0"}
        : (!fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>,
              !fabric.tagged<!fabric.bits<32>, i1>)
    %tsw_ld_done:2 = fabric.instance @ld_done_demux(%mem0#1)
        {sym_name = "tsw_1"}
        : (!fabric.tagged<none, i1>)
          -> (!fabric.tagged<none, i1>,
              !fabric.tagged<none, i1>)
    %tsw_st_done:2 = fabric.instance @st_done_demux(%mem0#2)
        {sym_name = "tsw_2"}
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
    %stdone0 = fabric.del_tag %tsw_st_done#0
        : !fabric.tagged<none, i1> -> none
    %stdone1 = fabric.del_tag %tsw_st_done#1
        : !fabric.tagged<none, i1> -> none

    fabric.yield %pe_ld0#0, %pe_ld1#0, %stdone0, %stdone1, %lddone0, %lddone1
        : i32, i32, none, none, none, none
  }
}
