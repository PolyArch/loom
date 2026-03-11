module {
  handshake.func @mem_load_store(
      %mem: memref<?xf32, strided<[1], offset: ?>>,
      %idx: i32, %ctrl: none, ...) -> (f32, none)
      attributes {argNames = ["mem", "idx", "ctrl"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["result", "done"]} {
    %idx_ext = arith.index_cast %idx : i32 to index
    %dataResult, %addressResults = handshake.load [%idx_ext] %0#0, %ctrl : index, f32
    %0:2 = handshake.extmemory[ld = 1, st = 0]
        (%mem : memref<?xf32, strided<[1], offset: ?>>)
        (%addressResults) {id = 0 : i32} : (index) -> (f32, none)
    handshake.return %dataResult, %0#1 : f32, none
  }
}
