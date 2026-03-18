// Public on-chip memory built through the high-level switch-bank helper.
module {
  handshake.func @builder_memory_public(
      %idx_ld: index, %idx_st: index, %val_st: i32,
      %ctrl_ld: none, %ctrl_st: none, ...)
      -> (i32, none, none)
      attributes {
        argNames = ["idx_ld", "idx_st", "val_st", "ctrl_ld", "ctrl_st"],
        resNames = ["load_data", "store_done", "load_done"]
      } {
    %lddata, %ldaddr = load [%idx_ld] %memif#0, %ctrl_ld : index, i32
    %stdata, %staddr = store [%idx_st] %val_st, %ctrl_st : index, i32
    %memif:3 = memory[ld = 1, st = 1]
        (%stdata, %staddr, %ldaddr) {id = 0 : i32, lsq = false}
        : memref<256xi32>, (i32, index, index) -> (i32, none, none)
    return %lddata, %memif#1, %memif#2 : i32, none, none
  }
}
