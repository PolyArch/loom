// RUN: loom --adg %s

// Valid: 4-region addr_offset_table with non-overlapping half-open tag ranges.
// Regions: [0,2), [2,4), [4,6), [6,8) with different base addresses.
fabric.module @multi_region_4(%ldaddr: !dataflow.bits<57>) -> (!dataflow.bits<32>, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0, numRegion = 4,
       addr_offset_table = array<i64:
         1, 0, 2, 0,
         1, 2, 4, 64,
         1, 4, 6, 128,
         1, 6, 8, 192>]
      (%ldaddr)
      : memref<256xi32>, (!dataflow.bits<57>) -> (!dataflow.bits<32>, none)
  fabric.yield %lddata, %lddone : !dataflow.bits<32>, none
}
