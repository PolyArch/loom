// Companion DFG for crossbar_2x2.fabric.mlir.
module {
  handshake.func @sw_test(%a: i32, %b: i32) -> (i32, i32)
      attributes {argNames = ["a", "b"], resNames = ["out0", "out1"]} {
    handshake.return %a, %b : i32, i32
  }
}
