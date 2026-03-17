// Two software extmemory regions sharing one hardware extmemory interface.
module {
  handshake.func @dual_load(
      %mem0: memref<?xi32, strided<[1], offset: ?>>,
      %mem1: memref<?xi32, strided<[1], offset: ?>>,
      %idx0: index, %idx1: index, %ctrl0: none, %ctrl1: none, ...)
      -> (i32, none, i32, none)
      attributes {
        argNames = ["mem0", "mem1", "idx0", "idx1", "ctrl0", "ctrl1"],
        resNames = ["data0", "done0", "data1", "done1"]
      } {
    %data0, %addr0 = load [%idx0] %memif0#0, %ctrl0 : index, i32
    %data1, %addr1 = load [%idx1] %memif1#0, %ctrl1 : index, i32
    %memif0:2 = extmemory[ld = 1, st = 0]
        (%mem0 : memref<?xi32, strided<[1], offset: ?>>) (%addr0)
        {id = 0 : i32} : (index) -> (i32, none)
    %memif1:2 = extmemory[ld = 1, st = 0]
        (%mem1 : memref<?xi32, strided<[1], offset: ?>>) (%addr1)
        {id = 1 : i32} : (index) -> (i32, none)
    return %data0, %memif0#1, %data1, %memif1#1 : i32, none, i32, none
  }
}
