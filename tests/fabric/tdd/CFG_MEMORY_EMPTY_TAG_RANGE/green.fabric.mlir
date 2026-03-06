// RUN: loom --adg %s

// Valid addr_offset_table: start_tag < end_tag for all valid entries.
// Entry format: [valid, start_tag, end_tag, base_addr] repeated per region.
fabric.module @valid_tag_range(%ldaddr: !dataflow.bits<57>) -> (!dataflow.bits<32>, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0, numRegion = 2,
       addr_offset_table = array<i64: 1, 0, 4, 0,  1, 4, 8, 64>]
      (%ldaddr)
      : memref<128xi32>, (!dataflow.bits<57>) -> (!dataflow.bits<32>, none)
  fabric.yield %lddata, %lddone : !dataflow.bits<32>, none
}
