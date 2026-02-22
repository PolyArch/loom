// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_MEMORY_INVALID_REGION

// numRegion = 0 is invalid on fabric.extmemory.
fabric.module @bad_ext_region(%m: memref<?xi32>, %ldaddr: index) -> (i32, none) {
  %lddata, %lddone = fabric.extmemory
      [ldCount = 1, stCount = 0, numRegion = 0]
      (%m, %ldaddr)
      : memref<?xi32>, (memref<?xi32>, index) -> (i32, none)
  fabric.yield %lddata, %lddone : i32, none
}
