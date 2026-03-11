module {
  handshake.func @mem_store_onchip(%addr: index, %data: i32, %ctrl: none, ...) -> (none)
      attributes {argNames = ["addr", "data", "ctrl"], loom.annotations = ["loom.accel"],
                  resNames = ["done"]} {
    %st_data, %st_addr = handshake.store [%addr] %data, %ctrl : index, i32
    %st_done = handshake.memory[ld = 0, st = 1] (%st_data, %st_addr) {id = 0 : i32, lsq = false} : memref<256xi32>, (i32, index) -> none
    handshake.return %st_done : none
  }
}
