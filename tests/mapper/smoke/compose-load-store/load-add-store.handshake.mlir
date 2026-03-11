module {
  handshake.func @load_add_store(%mem: memref<?xi32, strided<[1], offset: ?>>, %addr: index, %val: i32, %ctrl: none, ...) -> (none)
      attributes {argNames = ["mem", "addr", "val", "ctrl"], loom.annotations = ["loom.accel"],
                  resNames = ["done"]} {
    %ld_data, %ld_addr = handshake.load [%addr] %mem_data, %ctrl : index, i32
    %mem_data, %mem_done = handshake.extmemory[ld = 1, st = 1] (%mem : memref<?xi32, strided<[1], offset: ?>>) (%ld_addr, %st_data, %st_addr) {id = 0 : i32} : (index, i32, index) -> (i32, none)
    %sum = arith.addi %ld_data, %val : i32
    %st_data, %st_addr = handshake.store [%addr] %sum, %mem_done : index, i32
    handshake.return %mem_done : none
  }
}
