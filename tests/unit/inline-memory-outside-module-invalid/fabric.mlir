module {
  %n = arith.constant 1 : index
  %dram = memref.alloc(%n) : memref<?xi32>
  %c0 = arith.constant 0 : i64
  %addr = builtin.unrealized_conversion_cast %c0 : i64 to !fabric.bits<64>

  %ext:2 = fabric.extmemory @ext_inline
      [ldCount = 1, stCount = 0, lsqDepth = 0,
       memrefType = memref<?xi32>, numRegion = 1]
      (%dram, %addr)
      : (memref<?xi32>, !fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>)
}
