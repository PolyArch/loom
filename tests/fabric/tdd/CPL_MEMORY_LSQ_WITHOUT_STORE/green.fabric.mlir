// RUN: loom --adg %s

// A valid fabric.memory with stCount = 0 and lsqDepth omitted (defaults to 0).
fabric.module @ld_only(%ldaddr: index) -> (i32, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0]
      (%ldaddr)
      : memref<64xi32>, (index) -> (i32, none)
  fabric.yield %lddata, %lddone : i32, none
}
