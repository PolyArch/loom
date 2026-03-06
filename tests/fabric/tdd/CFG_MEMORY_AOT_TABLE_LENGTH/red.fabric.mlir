// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: addr_offset_table

// Invalid: numRegion=2 but addr_offset_table has only 4 values (1 entry).
fabric.module @table_length(%ldaddr: !dataflow.bits<57>) -> (!dataflow.bits<32>, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0, numRegion = 2,
       addr_offset_table = array<i64: 1, 0, 4, 0>]
      (%ldaddr)
      : memref<128xi32>, (!dataflow.bits<57>) -> (!dataflow.bits<32>, none)
  fabric.yield %lddata, %lddone : !dataflow.bits<32>, none
}
