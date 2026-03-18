// Two software on-chip memories mapped onto one hardware fabric.memory.
module {
  handshake.func @multi_memory(
      %idx0: index, %idx1: index,
      %ctrl0: none, %ctrl1: none, ...)
      -> (i32, none, i32, none)
      attributes {
        argNames = ["idx0", "idx1", "ctrl0", "ctrl1"],
        resNames = ["data0", "done0", "data1", "done1"]
      } {
    %lddata0, %ldaddr0 = load [%idx0] %memif0#0, %ctrl0 : index, i32
    %memif0:2 = memory[ld = 1, st = 0]
        (%ldaddr0) {id = 0 : i32, lsq = false}
        : memref<256xi32>, (index) -> (i32, none)
    %lddata1, %ldaddr1 = load [%idx1] %memif1#0, %ctrl1 : index, i32
    %memif1:2 = memory[ld = 1, st = 0]
        (%ldaddr1) {id = 1 : i32, lsq = false}
        : memref<256xi32>, (index) -> (i32, none)
    return %lddata0, %memif0#1, %lddata1, %memif1#1 : i32, none, i32, none
  }
}
