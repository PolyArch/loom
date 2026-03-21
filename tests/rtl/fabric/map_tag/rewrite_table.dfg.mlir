// Companion DFG for rewrite_table.fabric.mlir.
module {
  handshake.func @map_tag_test(%in: i32) -> (i32)
      attributes {argNames = ["in"], resNames = ["out"]} {
    handshake.return %in : i32
  }
}
