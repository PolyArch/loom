module {
  handshake.func @onchip_rw(%val: f32, %idx: i32, %ctrl: none, ...) -> (f32, none)
      attributes {argNames = ["val", "idx", "ctrl"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["result", "done"]} {
    %idx_ext = arith.index_cast %idx : i32 to index
    %dataResult, %addressResults = handshake.load [%idx_ext] %0#0, %ctrl : index, f32
    %dataResult_st, %addressResult_st = handshake.store [%idx_ext] %val, %ctrl : index, f32
    %0:3 = handshake.memory[ld = 1, st = 1] (%dataResult_st, %addressResult_st, %addressResults) {id = 0 : i32, lsq = false} : memref<64xf32>, (f32, index, index) -> (f32, none, none)
    handshake.return %dataResult, %0#1 : f32, none
  }
}
