// One hardware extmemory shared by two software extmemory regions.
module {
  fabric.temporal_sw @addr_mux
      [num_route_table = 2, connectivity_table = ["11"]]
      : (!fabric.tagged<!fabric.bits<57>, i1>,
         !fabric.tagged<!fabric.bits<57>, i1>)
        -> (!fabric.tagged<!fabric.bits<57>, i1>)

  fabric.temporal_sw @data_demux
      [num_route_table = 2, connectivity_table = ["1", "1"]]
      : (!fabric.tagged<!fabric.bits<32>, i1>)
        -> (!fabric.tagged<!fabric.bits<32>, i1>,
            !fabric.tagged<!fabric.bits<32>, i1>)

  fabric.temporal_sw @done_demux
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

  fabric.module @multi_extmemory_test(
      %dram: memref<?xi32>,
      %idx0: index, %idx1: index,
      %ctrl0: none, %ctrl1: none)
      -> (i32, none, i32, none) {
    %ext0:2 = fabric.extmemory @extmem_0
        [ldCount = 2, stCount = 0, lsqDepth = 0,
         memrefType = memref<?xi32>, numRegion = 2]
        (%dram, %tsw_addr#0)
        : (memref<?xi32>, !fabric.tagged<!fabric.bits<57>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>,
              !fabric.tagged<none, i1>)

    %pe0:2 = fabric.instance @load_pe(%idx0, %lddata0, %ctrl0)
        {sym_name = "pe_0"} : (index, i32, none) -> (i32, index)
    %pe1:2 = fabric.instance @load_pe(%idx1, %lddata1, %ctrl1)
        {sym_name = "pe_1"} : (index, i32, none) -> (i32, index)

    %tag_addr0 = fabric.add_tag %pe0#1 {tag = 0 : i64}
        : index -> !fabric.tagged<!fabric.bits<57>, i1>
    %tag_addr1 = fabric.add_tag %pe1#1 {tag = 1 : i64}
        : index -> !fabric.tagged<!fabric.bits<57>, i1>

    %tsw_addr:1 = fabric.instance @addr_mux(%tag_addr0, %tag_addr1)
        {sym_name = "tsw_0"}
        : (!fabric.tagged<!fabric.bits<57>, i1>,
           !fabric.tagged<!fabric.bits<57>, i1>)
          -> (!fabric.tagged<!fabric.bits<57>, i1>)

    %tsw_data:2 = fabric.instance @data_demux(%ext0#0)
        {sym_name = "tsw_1"}
        : (!fabric.tagged<!fabric.bits<32>, i1>)
          -> (!fabric.tagged<!fabric.bits<32>, i1>,
              !fabric.tagged<!fabric.bits<32>, i1>)
    %tsw_done:2 = fabric.instance @done_demux(%ext0#1)
        {sym_name = "tsw_2"}
        : (!fabric.tagged<none, i1>)
          -> (!fabric.tagged<none, i1>,
              !fabric.tagged<none, i1>)

    %lddata0 = fabric.del_tag %tsw_data#0
        : !fabric.tagged<!fabric.bits<32>, i1> -> i32
    %lddata1 = fabric.del_tag %tsw_data#1
        : !fabric.tagged<!fabric.bits<32>, i1> -> i32
    %done0 = fabric.del_tag %tsw_done#0
        : !fabric.tagged<none, i1> -> none
    %done1 = fabric.del_tag %tsw_done#1
        : !fabric.tagged<none, i1> -> none

    fabric.yield %pe0#0, %done0, %pe1#0, %done1 : i32, none, i32, none
  }
}
