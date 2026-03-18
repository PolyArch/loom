module {
  fabric.spatial_sw @ld_addr_mux
      [connectivity_table = ["11"]]
      : (!fabric.tagged<!fabric.bits<64>, i1>,
         !fabric.tagged<!fabric.bits<64>, i1>)
        -> (!fabric.tagged<!fabric.bits<57>, i1>)

  fabric.temporal_sw @ld_data_demux
      [num_route_table = 2, connectivity_table = ["1", "1"]]
      : (!fabric.tagged<!fabric.bits<64>, i1>)
        -> (!fabric.tagged<!fabric.bits<64>, i1>,
            !fabric.tagged<!fabric.bits<64>, i1>)

  fabric.temporal_sw @ld_done_demux
      [num_route_table = 2, connectivity_table = ["1", "1"]]
      : (!fabric.tagged<none, i1>)
        -> (!fabric.tagged<none, i1>,
            !fabric.tagged<none, i1>)

  fabric.temporal_pe @load_tpe(
      %p0: !fabric.tagged<!fabric.bits<64>, i1>,
      %p1: !fabric.tagged<!fabric.bits<64>, i1>,
      %p2: !fabric.tagged<!fabric.bits<64>, i1>,
      %p3: !fabric.tagged<!fabric.bits<64>, i1>,
      %p4: !fabric.tagged<!fabric.bits<64>, i1>,
      %p5: !fabric.tagged<!fabric.bits<64>, i1>)
      -> (!fabric.tagged<!fabric.bits<64>, i1>,
          !fabric.tagged<!fabric.bits<64>, i1>,
          !fabric.tagged<!fabric.bits<64>, i1>,
          !fabric.tagged<!fabric.bits<64>, i1>)
      [
        num_register = 0 : i64,
        num_instruction = 2 : i64,
        reg_fifo_depth = 0 : i64
      ] {
    fabric.function_unit @fu_load(%addr: index, %data: i32, %ctrl: none)
        -> (i32, index) [latency = 1, interval = 1] {
      %0, %1 = handshake.load [%addr] %data, %ctrl : index, i32
      fabric.yield %0, %1 : i32, index
    }

    fabric.function_unit @fu_load_1(%addr: index, %data: i32, %ctrl: none)
        -> (i32, index) [latency = 1, interval = 1] {
      %0, %1 = handshake.load [%addr] %data, %ctrl : index, i32
      fabric.yield %0, %1 : i32, index
    }

    fabric.yield
  }

  fabric.module @extmemory_tagged_direct_boundary_test(
      %dram: memref<?xi32>,
      %idx0: index, %idx1: index,
      %ctrl0: none, %ctrl1: none)
      -> (i32, i32, none, none) {
    %tag_idx0 = fabric.add_tag %idx0 {tag = 0 : i64}
        : index -> !fabric.tagged<!fabric.bits<64>, i1>
    %tag_idx1 = fabric.add_tag %idx1 {tag = 1 : i64}
        : index -> !fabric.tagged<!fabric.bits<64>, i1>
    %tag_ctrl0 = fabric.add_tag %ctrl0 {tag = 0 : i64}
        : none -> !fabric.tagged<!fabric.bits<64>, i1>
    %tag_ctrl1 = fabric.add_tag %ctrl1 {tag = 1 : i64}
        : none -> !fabric.tagged<!fabric.bits<64>, i1>

    %tsw_ld_data:2 = fabric.instance @ld_data_demux(%ext0#0)
        {sym_name = "tsw_0"}
        : (!fabric.tagged<!fabric.bits<64>, i1>)
          -> (!fabric.tagged<!fabric.bits<64>, i1>,
              !fabric.tagged<!fabric.bits<64>, i1>)
    %tsw_ld_done:2 = fabric.instance @ld_done_demux(%ext0#1)
        {sym_name = "tsw_1"}
        : (!fabric.tagged<none, i1>)
          -> (!fabric.tagged<none, i1>,
              !fabric.tagged<none, i1>)

    %tpe0:4 = fabric.instance @load_tpe(
        %tag_idx0, %tag_ctrl0, %tsw_ld_data#0,
        %tag_idx1, %tag_ctrl1, %tsw_ld_data#1)
        {sym_name = "tpe_0"}
        : (!fabric.tagged<!fabric.bits<64>, i1>,
           !fabric.tagged<!fabric.bits<64>, i1>,
           !fabric.tagged<!fabric.bits<64>, i1>,
           !fabric.tagged<!fabric.bits<64>, i1>,
           !fabric.tagged<!fabric.bits<64>, i1>,
           !fabric.tagged<!fabric.bits<64>, i1>)
          -> (!fabric.tagged<!fabric.bits<64>, i1>,
              !fabric.tagged<!fabric.bits<64>, i1>,
              !fabric.tagged<!fabric.bits<64>, i1>,
              !fabric.tagged<!fabric.bits<64>, i1>)

    %sw_ld_addr:1 = fabric.instance @ld_addr_mux(%tpe0#1, %tpe0#3)
        {sym_name = "sw_0"}
        : (!fabric.tagged<!fabric.bits<64>, i1>,
           !fabric.tagged<!fabric.bits<64>, i1>)
          -> (!fabric.tagged<!fabric.bits<57>, i1>)

    %ext0:2 = fabric.extmemory @extmem_0
        [ldCount = 2, stCount = 0, lsqDepth = 0,
         memrefType = memref<?xi32>, numRegion = 1]
        (%dram, %sw_ld_addr#0)
        : (memref<?xi32>, !fabric.tagged<!fabric.bits<57>, i1>)
          -> (!fabric.tagged<!fabric.bits<64>, i1>,
              !fabric.tagged<none, i1>)

    %data0 = fabric.del_tag %tpe0#0
        : !fabric.tagged<!fabric.bits<64>, i1> -> i32
    %data1 = fabric.del_tag %tpe0#2
        : !fabric.tagged<!fabric.bits<64>, i1> -> i32
    %lddone0 = fabric.del_tag %tsw_ld_done#0
        : !fabric.tagged<none, i1> -> none
    %lddone1 = fabric.del_tag %tsw_ld_done#1
        : !fabric.tagged<none, i1> -> none

    fabric.yield %data0, %data1, %lddone0, %lddone1
        : i32, i32, none, none
  }
}
