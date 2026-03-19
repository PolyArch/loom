module {
  fabric.module @memory_expanded_families_invalid(
      %a: !fabric.bits<64>, %b: !fabric.bits<64>,
      %c: !fabric.bits<64>, %d: !fabric.bits<64>)
      -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>,
          !fabric.bits<64>, !fabric.bits<64>) {
    %mem0:5 = fabric.memory @mem_0
        [ldCount = 2, stCount = 1, lsqDepth = 0,
         memrefType = memref<256xi32>, numRegion = 1]
        (%a, %b, %c, %d)
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>,
           !fabric.bits<64>)
          -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>,
              !fabric.bits<64>, !fabric.bits<64>)
    fabric.yield %mem0#0, %mem0#1, %mem0#2, %mem0#3, %mem0#4
        : !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>,
          !fabric.bits<64>, !fabric.bits<64>
  }
}
