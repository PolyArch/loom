// RUN: loom --adg %s

// A valid fabric.memory with stCount > 0 and lsqDepth >= 1.
fabric.module @valid_st(%ldaddr: index, %staddr: index, %stdata: i32) -> (i32, none, none) {
  %lddata, %lddone, %stdone = fabric.memory
      [ldCount = 1, stCount = 1, lsqDepth = 4]
      (%ldaddr, %staddr, %stdata)
      : memref<64xi32>, (index, index, i32) -> (i32, none, none)
  fabric.yield %lddata, %lddone, %stdone : i32, none, none
}
