// Companion DFG for single_instruction.fabric.mlir.
// ADG observable behaviour: result = a + b (c is consumed but does not affect output)
// ADG boundary: 3 inputs (a:i32, b:i32, c:i32) -> 1 output (i32)
module {
  handshake.func @tpe_test(%a: i32, %b: i32, %c: i32) -> (i32)
      attributes {argNames = ["a", "b", "c"], resNames = ["result"]} {
    %0 = arith.addi %a, %b : i32
    // c is consumed (matching ADG fu_add's xori %c,%c) but does not affect result
    %unused = arith.xori %c, %c : i32
    handshake.return %0 : i32
  }
}
