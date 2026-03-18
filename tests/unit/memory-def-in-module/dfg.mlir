module {
  handshake.func @memory_single_load(%idx: index, %ctrl: none, ...)
      -> (i32, none)
      attributes {
        argNames = ["idx", "ctrl"],
        resNames = ["load_data", "load_done"]
      } {
    %lddata, %ldaddr = load [%idx] %memif#0, %ctrl : index, i32
    %memif:2 = memory[ld = 1, st = 0]
        (%ldaddr) {id = 0 : i32, lsq = false}
        : memref<256xi32>, (index) -> (i32, none)
    return %lddata, %memif#1 : i32, none
  }
}
