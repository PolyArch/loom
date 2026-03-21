// Companion DFG for single_instruction.fabric.mlir.
// ADG boundary: 3 inputs (a:i32, b:i32, c:i32) -> 1 output (i32)
// DFG computes: result = (a + b) + c
module {
  handshake.func @tpe_test(%a: i32, %b: i32, %c: i32) -> (i32)
      attributes {argNames = ["a", "b", "c"], resNames = ["result"]} {
    %0 = arith.addi %a, %b : i32
    %1 = arith.addi %0, %c : i32
    handshake.return %1 : i32
  }
}
