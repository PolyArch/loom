// RUN: loom --adg %s

// Valid: addr_offset_table with non-zero base addresses for address translation.
// Region 0: tags [0,4), base_addr=0x100 (256)
// Region 1: tags [4,8), base_addr=0x200 (512)
// Address translation: physical_addr = base_addr + logical_index * sizeof(i32)
fabric.module @addr_translation(%ldaddr: !dataflow.bits<57>, %staddr: !dataflow.bits<57>, %stdata: !dataflow.bits<32>) -> (!dataflow.bits<32>, none, none) {
  %lddata, %lddone, %stdone = fabric.memory
      [ldCount = 1, stCount = 1, lsqDepth = 1, numRegion = 2,
       addr_offset_table = array<i64: 1, 0, 4, 256,  1, 4, 8, 512>]
      (%ldaddr, %staddr, %stdata)
      : memref<256xi32>, (!dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<32>) -> (!dataflow.bits<32>, none, none)
  fabric.yield %lddata, %lddone, %stdone : !dataflow.bits<32>, none, none
}
