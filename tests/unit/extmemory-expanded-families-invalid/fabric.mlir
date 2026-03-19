module {
  fabric.module @extmemory_expanded_families_invalid(
      %dram: memref<?xi32>, %a: !fabric.bits<64>, %b: !fabric.bits<64>,
      %c: !fabric.bits<64>, %d: !fabric.bits<64>)
      -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>,
          !fabric.bits<64>, !fabric.bits<64>) {
    %ext0:5 = fabric.extmemory @extmem_0
        [ldCount = 2, stCount = 1, lsqDepth = 0,
         memrefType = memref<?xi32>, numRegion = 1]
        (%dram, %a, %b, %c, %d)
        : (memref<?xi32>, !fabric.bits<64>, !fabric.bits<64>,
           !fabric.bits<64>, !fabric.bits<64>)
          -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>,
              !fabric.bits<64>, !fabric.bits<64>)
    fabric.yield %ext0#0, %ext0#1, %ext0#2, %ext0#3, %ext0#4
        : !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>,
          !fabric.bits<64>, !fabric.bits<64>
  }
}
