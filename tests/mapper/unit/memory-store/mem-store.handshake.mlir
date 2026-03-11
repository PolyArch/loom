module {
  handshake.func @mem_store(%mem: memref<?xi32, strided<[1], offset: ?>>, %addr: index, %data: i32, %ctrl: none, ...) -> (none)
      attributes {argNames = ["mem", "addr", "data", "ctrl"], loom.annotations = ["loom.accel"],
                  resNames = ["done"]} {
    %st_data, %st_addr = handshake.store [%addr] %data, %ctrl : index, i32
    %st_done = handshake.extmemory[ld = 0, st = 1] (%mem : memref<?xi32, strided<[1], offset: ?>>) (%st_data, %st_addr) {id = 0 : i32} : (i32, index) -> none
    handshake.return %st_done : none
  }
}
