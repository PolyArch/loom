// Companion DFG for passthrough.fabric.mlir.
module {
  handshake.func @del_tag_test(%in: i32) -> (i32)
      attributes {argNames = ["in"], resNames = ["out"]} {
    handshake.return %in : i32
  }
}
