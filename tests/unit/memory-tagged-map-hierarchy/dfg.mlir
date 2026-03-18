module {
  handshake.func @memory_tagged_map_hierarchy(
      %idx0: index, %idx1: index,
      %ctrl0: none, %ctrl1: none, ...)
      -> (i32, i32, none, none)
      attributes {
        argNames = ["idx0", "idx1", "ctrl0", "ctrl1"],
        resNames = ["data0", "data1", "done0", "done1"]
      } {
    %data0, %addr0 = load [%idx0] %memif#0, %ctrl0 : index, i32
    %data1, %addr1 = load [%idx1] %memif#1, %ctrl1 : index, i32
    %memif:4 = memory[ld = 2, st = 0]
        (%addr0, %addr1) {id = 0 : i32, lsq = false}
        : memref<256xi32>, (index, index) -> (i32, i32, none, none)
    return %data0, %data1, %memif#2, %memif#3 : i32, i32, none, none
  }
}
