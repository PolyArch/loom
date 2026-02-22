// RUN: loom --adg %s

// numRegion = 1 (default) is valid on fabric.memory.
fabric.module @valid_region(%ldaddr: index) -> (i32, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0]
      (%ldaddr)
      : memref<64xi32>, (index) -> (i32, none)
  fabric.yield %lddata, %lddone : i32, none
}
