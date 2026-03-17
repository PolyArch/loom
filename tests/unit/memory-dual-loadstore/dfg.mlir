// One software on-chip memory with two load ports and two store ports.
module {
  handshake.func @memory_dual_loadstore(
      %idx_ld0: index, %idx_ld1: index,
      %idx_st0: index, %val_st0: i32,
      %idx_st1: index, %val_st1: i32,
      %ctrl_ld0: none, %ctrl_ld1: none,
      %ctrl_st0: none, %ctrl_st1: none, ...)
      -> (i32, i32, none, none, none, none)
      attributes {
        argNames = ["idx_ld0", "idx_ld1", "idx_st0", "val_st0", "idx_st1", "val_st1", "ctrl_ld0", "ctrl_ld1", "ctrl_st0", "ctrl_st1"],
        resNames = ["data0", "data1", "st_done0", "st_done1", "load_done0", "load_done1"]
      } {
    %lddata0, %ldaddr0 = load [%idx_ld0] %memif#0, %ctrl_ld0 : index, i32
    %lddata1, %ldaddr1 = load [%idx_ld1] %memif#1, %ctrl_ld1 : index, i32
    %stdata0, %staddr0 = store [%idx_st0] %val_st0, %ctrl_st0 : index, i32
    %stdata1, %staddr1 = store [%idx_st1] %val_st1, %ctrl_st1 : index, i32
    %memif:6 = memory[ld = 2, st = 2]
        (%stdata0, %staddr0, %stdata1, %staddr1, %ldaddr0, %ldaddr1)
        {id = 0 : i32, lsq = false}
        : memref<256xi32>, (i32, index, i32, index, index, index)
          -> (i32, i32, none, none, none, none)
    return %lddata0, %lddata1, %memif#2, %memif#3, %memif#4, %memif#5
        : i32, i32, none, none, none, none
  }
}
