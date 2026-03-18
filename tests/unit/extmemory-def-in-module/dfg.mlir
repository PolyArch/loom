module {
  handshake.func @tiny_load(%mem: memref<?xi32, strided<[1], offset: ?>>,
      %idx: index, %ctrl: none, ...) -> (i32, none)
      attributes {argNames = ["mem", "idx", "ctrl"], resNames = ["data", "done"]} {
    %data, %addr = load [%idx] %memif#0, %ctrl : index, i32
    %memif:2 = extmemory[ld = 1, st = 0]
        (%mem : memref<?xi32, strided<[1], offset: ?>>) (%addr)
        {id = 0 : i32} : (index) -> (i32, none)
    return %data, %memif#1 : i32, none
  }
}
