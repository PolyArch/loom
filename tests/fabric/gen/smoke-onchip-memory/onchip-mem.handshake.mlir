module {
  handshake.func @onchip_rom(%idx: i32, %ctrl: none, ...) -> (i32, none)
      attributes {argNames = ["idx", "ctrl"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["result", "done"]} {
    %idx_ext = arith.index_cast %idx : i32 to index
    %dataResult, %addressResults = handshake.load [%idx_ext] %0#0, %ctrl : index, i32
    %0:2 = handshake.memory[ld = 1, st = 0] (%addressResults) {id = 0 : i32, lsq = false} : memref<256xi32>, (index) -> (i32, none)
    handshake.return %dataResult, %0#1 : i32, none
  }
}
