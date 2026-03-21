// Companion DFG for single_instruction.fabric.mlir.
module {
  handshake.func @tpe_test(%a: i32, %b: i32) -> (i32)
      attributes {argNames = ["a", "b"], resNames = ["result"]} {
    %0 = arith.addi %a, %b : i32
    handshake.return %0 : i32
  }
}
