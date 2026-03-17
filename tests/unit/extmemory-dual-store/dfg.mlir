// One software extmemory region with two store ports sharing one hardware interface.
module {
  handshake.func @dual_store(
      %mem: memref<?xi32, strided<[1], offset: ?>>,
      %idx0: index, %val0: i32,
      %idx1: index, %val1: i32,
      %ctrl0: none, %ctrl1: none, ...)
      -> (none, none)
      attributes {
        argNames = ["mem", "idx0", "val0", "idx1", "val1", "ctrl0", "ctrl1"],
        resNames = ["done0", "done1"]
      } {
    %stdata0, %staddr0 = store [%idx0] %val0, %ctrl0 : index, i32
    %stdata1, %staddr1 = store [%idx1] %val1, %ctrl1 : index, i32
    %memif:2 = extmemory[ld = 0, st = 2]
        (%mem : memref<?xi32, strided<[1], offset: ?>>)
        (%stdata0, %staddr0, %stdata1, %staddr1)
        {id = 0 : i32} : (i32, index, i32, index) -> (none, none)
    return %memif#0, %memif#1 : none, none
  }
}
