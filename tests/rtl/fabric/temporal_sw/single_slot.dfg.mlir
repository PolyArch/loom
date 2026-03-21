// Companion DFG for single_slot.fabric.mlir.
module {
  handshake.func @tsw_test(%in: i32) -> (i32)
      attributes {argNames = ["in"], resNames = ["out"]} {
    handshake.return %in : i32
  }
}
