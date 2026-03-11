module {
  handshake.func @mem_load_onchip(%addr: index, %ctrl: none, ...) -> (i32, none)
      attributes {argNames = ["addr", "ctrl"], loom.annotations = ["loom.accel"],
                  resNames = ["data", "done"]} {
    %ld_data, %ld_addr = handshake.load [%addr] %mem_data, %ctrl : index, i32
    %mem_data, %mem_done = handshake.memory[ld = 1, st = 0] (%ld_addr) {id = 0 : i32, lsq = false} : memref<256xi32>, (index) -> (i32, none)
    handshake.return %ld_data, %mem_done : i32, none
  }
}
