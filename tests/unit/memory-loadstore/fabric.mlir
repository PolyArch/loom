// One on-chip fabric.memory with a load PE and a store PE.
module {
  fabric.spatial_pe @load_pe(%p0: !fabric.bits<64>, %p1: !fabric.bits<64>, %p2: !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>) {
    fabric.function_unit @fu_load(%arg0: index, %arg1: i32, %arg2: none)
        -> (i32, index) [latency = 1, interval = 1] {
      %0, %1 = handshake.load [%arg0] %arg1, %arg2 : index, i32
      fabric.yield %0, %1 : i32, index
    }
    fabric.yield
  }

  fabric.spatial_pe @store_pe(%p0: !fabric.bits<64>, %p1: !fabric.bits<64>, %p2: !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>) {
    fabric.function_unit @fu_store(%arg0: index, %arg1: i32, %arg2: none)
        -> (i32, index) [latency = 1, interval = 1] {
      %0, %1 = handshake.store [%arg0] %arg1, %arg2 : index, i32
      fabric.yield %0, %1 : i32, index
    }
    fabric.yield
  }

  fabric.module @memory_loadstore_test(
      %idx_ld: !fabric.bits<64>, %idx_st: !fabric.bits<64>,
      %val_st: !fabric.bits<64>, %ctrl_ld: !fabric.bits<64>,
      %ctrl_st: !fabric.bits<64>)
      -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) {
    %mem0:3 = fabric.memory @mem_0
        [ldCount = 1, stCount = 1, lsqDepth = 0,
         memrefType = memref<256xi32>, numRegion = 1]
        (%pe_ld#1, %pe_st#1, %pe_st#0)
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
          -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)

    %pe_ld:2 = fabric.instance @load_pe(%idx_ld, %mem0#0, %ctrl_ld)
        {sym_name = "pe_ld"}
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
          -> (!fabric.bits<64>, !fabric.bits<64>)
    %pe_st:2 = fabric.instance @store_pe(%idx_st, %val_st, %ctrl_st)
        {sym_name = "pe_st"}
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
          -> (!fabric.bits<64>, !fabric.bits<64>)

    fabric.yield %pe_ld#0, %mem0#2, %mem0#1
        : !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>
  }
}
