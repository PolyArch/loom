// RUN: loom --adg %s

// Valid: numRegion=1 and addr_offset_table has exactly 4 values (1 entry).
fabric.module @table_length_ok(%ldaddr: !dataflow.bits<57>) -> (!dataflow.bits<32>, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0, numRegion = 1,
       addr_offset_table = array<i64: 1, 0, 4, 0>]
      (%ldaddr)
      : memref<64xi32>, (!dataflow.bits<57>) -> (!dataflow.bits<32>, none)
  fabric.yield %lddata, %lddone : !dataflow.bits<32>, none
}
