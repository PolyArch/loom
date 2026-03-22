// Companion DFG for single_instruction.fabric.mlir.
// ADG observable behaviour: result = a + b + c (two addi ops matching FU capability)
// ADG boundary: 3 inputs (a:i32, b:i32, c:i32) -> 1 output (i32)
module {
  handshake.func @tpe_test(%a: i32, %b: i32, %c: i32) -> (i32)
      attributes {argNames = ["a", "b", "c"], resNames = ["result"]} {
    %sum = arith.addi %a, %b : i32
    %result = arith.addi %sum, %c : i32
    handshake.return %result : i32
  }
}
