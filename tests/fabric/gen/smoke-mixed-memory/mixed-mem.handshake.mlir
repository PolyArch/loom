module {
  handshake.func @mixed_mem(%extbuf: memref<?xf32>, %idx: i32, %ctrl: none, ...) -> (f32, none)
      attributes {argNames = ["extbuf", "idx", "ctrl"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["result", "done"]} {
    %idx_ext = arith.index_cast %idx : i32 to index
    // Load from external memory
    %extData, %extAddr = handshake.load [%idx_ext] %ext#0, %ctrl : index, f32
    %ext:2 = handshake.extmemory[ld = 1, st = 0] (%extbuf : memref<?xf32>) (%extAddr) {id = 0 : i32} : (index) -> (f32, none)
    // Store to on-chip scratchpad
    %stData, %stAddr = handshake.store [%idx_ext] %extData, %ctrl : index, f32
    %onchip:3 = handshake.memory[ld = 1, st = 1] (%stData, %stAddr, %idx_ext) {id = 1 : i32, lsq = false} : memref<64xf32>, (f32, index, index) -> (f32, none, none)
    handshake.return %onchip#0, %ext#1 : f32, none
  }
}
