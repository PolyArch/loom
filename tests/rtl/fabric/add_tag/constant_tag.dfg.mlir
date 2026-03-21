// Companion DFG for constant_tag.fabric.mlir.
// Minimal pass-through kernel for tag-attachment verification.
module {
  handshake.func @tag_test(%in: i32) -> (i32)
      attributes {argNames = ["in"], resNames = ["out"]} {
    handshake.return %in : i32
  }
}
