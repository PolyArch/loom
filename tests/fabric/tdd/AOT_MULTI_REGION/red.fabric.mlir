// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CFG_MEMORY_OVERLAP_TAG_REGION

// Invalid: 4-region table where regions [2,5) and [4,6) overlap at [4,5).
fabric.module @multi_region_overlap(%ldaddr: index) -> (i32, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0, numRegion = 4,
       addr_offset_table = array<i64:
         1, 0, 2, 0,
         1, 2, 5, 64,
         1, 4, 6, 128,
         1, 6, 8, 192>]
      (%ldaddr)
      : memref<256xi32>, (index) -> (i32, none)
  fabric.yield %lddata, %lddone : i32, none
}
