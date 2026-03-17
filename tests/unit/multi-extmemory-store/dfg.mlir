// Two software extmemory store regions sharing one hardware extmemory interface.
module {
  handshake.func @dual_store(
      %mem0: memref<?xi32, strided<[1], offset: ?>>,
      %mem1: memref<?xi32, strided<[1], offset: ?>>,
      %idx0: index, %val0: i32,
      %idx1: index, %val1: i32,
      %ctrl0: none, %ctrl1: none, ...)
      -> (none, none)
      attributes {
        argNames = ["mem0", "mem1", "idx0", "val0", "idx1", "val1", "ctrl0", "ctrl1"],
        resNames = ["done0", "done1"]
      } {
    %stdata0, %staddr0 = store [%idx0] %val0, %ctrl0 : index, i32
    %stdata1, %staddr1 = store [%idx1] %val1, %ctrl1 : index, i32
    %done0 = extmemory[ld = 0, st = 1]
        (%mem0 : memref<?xi32, strided<[1], offset: ?>>) (%stdata0, %staddr0)
        {id = 0 : i32} : (i32, index) -> none
    %done1 = extmemory[ld = 0, st = 1]
        (%mem1 : memref<?xi32, strided<[1], offset: ?>>) (%stdata1, %staddr1)
        {id = 1 : i32} : (i32, index) -> none
    return %done0, %done1 : none, none
  }
}
