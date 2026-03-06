// RUN: loom --adg %s

// Valid: addr_offset_table with full contiguous tag coverage [0,8).
// At runtime, any tag in [0,8) will match a valid region, so RT_MEMORY_NO_MATCH
// will never fire for tags in range.
fabric.module @full_coverage(%ldaddr: !dataflow.bits<57>) -> (!dataflow.bits<32>, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0, numRegion = 2,
       addr_offset_table = array<i64: 1, 0, 4, 0,  1, 4, 8, 64>]
      (%ldaddr)
      : memref<128xi32>, (!dataflow.bits<57>) -> (!dataflow.bits<32>, none)
  fabric.yield %lddata, %lddone : !dataflow.bits<32>, none
}
