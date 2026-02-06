// RUN: loom --adg %s

// A valid fabric.memory with correct address types (index for single-port).
fabric.module @valid_addr(%ldaddr: index, %staddr: index, %stdata: i32) -> (i32, none, none) {
  %lddata, %lddone, %stdone = fabric.memory
      [ldCount = 1, stCount = 1, lsqDepth = 1]
      (%ldaddr, %staddr, %stdata)
      : memref<64xi32>, (index, index, i32) -> (i32, none, none)
  fabric.yield %lddata, %lddone, %stdone : i32, none, none
}
