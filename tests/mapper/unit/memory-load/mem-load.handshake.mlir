module {
  handshake.func @mem_load(%mem: memref<?xi32, strided<[1], offset: ?>>, %addr: index, %ctrl: none, ...) -> (i32, none)
      attributes {argNames = ["mem", "addr", "ctrl"], loom.annotations = ["loom.accel"],
                  resNames = ["data", "done"]} {
    %ld_data, %ld_addr = handshake.load [%addr] %mem_data, %ctrl : index, i32
    %mem_data, %mem_done = handshake.extmemory[ld = 1, st = 0] (%mem : memref<?xi32, strided<[1], offset: ?>>) (%ld_addr) {id = 0 : i32} : (index) -> (i32, none)
    handshake.return %ld_data, %mem_done : i32, none
  }
}
