// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: addr_offset_table

// Invalid: numRegion=3 but addr_offset_table has only 8 values (2 entries).
// Table length must be numRegion * 4 = 12, but only 8 values provided.
fabric.module @config_width_mismatch(%ldaddr: !dataflow.bits<57>) -> (!dataflow.bits<32>, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0, numRegion = 3,
       addr_offset_table = array<i64:
         1, 0, 2, 0,
         1, 2, 4, 64>]
      (%ldaddr)
      : memref<256xi32>, (!dataflow.bits<57>) -> (!dataflow.bits<32>, none)
  fabric.yield %lddata, %lddone : !dataflow.bits<32>, none
}
