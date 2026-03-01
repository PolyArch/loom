// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CFG_MEMORY_OVERLAP_TAG_REGION

// Invalid: regions [0,6) and [4,8) overlap in the half-open interval [4,6).
fabric.module @overlap(%ldaddr: !dataflow.bits<57>) -> (!dataflow.bits<32>, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0, numRegion = 2,
       addr_offset_table = array<i64: 1, 0, 6, 0,  1, 4, 8, 64>]
      (%ldaddr)
      : memref<128xi32>, (!dataflow.bits<57>) -> (!dataflow.bits<32>, none)
  fabric.yield %lddata, %lddone : !dataflow.bits<32>, none
}
