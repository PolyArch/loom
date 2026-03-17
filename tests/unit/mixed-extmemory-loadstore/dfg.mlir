// One software load region and one software store region sharing one hardware extmemory.
module {
  handshake.func @load_store_mix(
      %mem_ld: memref<?xi32, strided<[1], offset: ?>>,
      %mem_st: memref<?xi32, strided<[1], offset: ?>>,
      %idx_ld: index, %idx_st: index,
      %val_st: i32,
      %ctrl_ld: none, %ctrl_st: none, ...)
      -> (i32, none, none)
      attributes {
        argNames = ["mem_ld", "mem_st", "idx_ld", "idx_st", "val_st", "ctrl_ld", "ctrl_st"],
        resNames = ["load_data", "load_done", "store_done"]
      } {
    %lddata, %ldaddr = load [%idx_ld] %memif_ld#0, %ctrl_ld : index, i32
    %stdata, %staddr = store [%idx_st] %val_st, %ctrl_st : index, i32
    %memif_ld:2 = extmemory[ld = 1, st = 0]
        (%mem_ld : memref<?xi32, strided<[1], offset: ?>>) (%ldaddr)
        {id = 0 : i32} : (index) -> (i32, none)
    %stdone = extmemory[ld = 0, st = 1]
        (%mem_st : memref<?xi32, strided<[1], offset: ?>>) (%stdata, %staddr)
        {id = 1 : i32} : (i32, index) -> none
    return %lddata, %memif_ld#1, %stdone : i32, none, none
  }
}
