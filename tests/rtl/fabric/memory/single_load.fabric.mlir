// Test: on-chip fabric.memory with a single load PE
module {
  fabric.memory @mem_def
      [ldCount = 1, stCount = 0, lsqDepth = 0,
       memrefType = memref<256xi32>, numRegion = 1]
      : (!fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>)

  fabric.spatial_pe @load_pe(%p0: !fabric.bits<64>, %p1: !fabric.bits<64>,
                              %p2: !fabric.bits<64>)
      -> (!fabric.bits<64>, !fabric.bits<64>) {
    fabric.function_unit @fu_load(%arg0: index, %arg1: i32, %arg2: none)
        -> (i32, index) [latency = 1, interval = 1] {
      %0, %1 = handshake.load [%arg0] %arg1, %arg2 : index, i32
      fabric.yield %0, %1 : i32, index
    }
    fabric.yield
  }

  fabric.module @test_memory_single_load(
      %idx: !fabric.bits<64>, %ctrl: !fabric.bits<64>)
      -> (!fabric.bits<64>, !fabric.bits<64>) {
    %mem0:2 = fabric.instance @mem_def(%pe_ld#1) {sym_name = "mem_0"}
        : (!fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>)
    %pe_ld:2 = fabric.instance @load_pe(%idx, %mem0#0, %ctrl)
        {sym_name = "pe_ld"}
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
          -> (!fabric.bits<64>, !fabric.bits<64>)
    fabric.yield %pe_ld#0, %mem0#1 : !fabric.bits<64>, !fabric.bits<64>
  }
}
