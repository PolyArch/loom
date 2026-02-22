// RUN: loom --adg %s

// Valid: two adjacent non-overlapping half-open regions [0,4) and [4,8).
fabric.module @no_overlap(%ldaddr: index) -> (i32, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0, numRegion = 2,
       addr_offset_table = array<i64: 1, 0, 4, 0,  1, 4, 8, 64>]
      (%ldaddr)
      : memref<128xi32>, (index) -> (i32, none)
  fabric.yield %lddata, %lddone : i32, none
}
