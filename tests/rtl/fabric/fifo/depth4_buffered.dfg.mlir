// Companion DFG for depth4_buffered.fabric.mlir.
module {
  handshake.func @fifo_test(%in: i32) -> (i32)
      attributes {argNames = ["in"], resNames = ["out"]} {
    handshake.return %in : i32
  }
}
