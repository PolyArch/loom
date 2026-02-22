// RUN: loom --adg %s

// Valid: numRegion=3 with correctly-sized addr_offset_table (3 entries, 12 values).
// CONFIG_WIDTH contribution: numRegion * (1 + 2*TAG_WIDTH + ADDR_WIDTH)
// Each entry has 4 fields: valid(1), start_tag, end_tag, base_addr.
fabric.module @config_width_3region(%ldaddr: index) -> (i32, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0, numRegion = 3,
       addr_offset_table = array<i64:
         1, 0, 2, 0,
         1, 2, 4, 64,
         1, 4, 6, 128>]
      (%ldaddr)
      : memref<256xi32>, (index) -> (i32, none)
  fabric.yield %lddata, %lddone : i32, none
}
