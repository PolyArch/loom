module {
  handshake.func @chess_3x3_test(%a: i32, %b: i32, %c: i32) -> (i32)
      attributes {argNames = ["a", "b", "c"], resNames = ["result"]} {
    %0 = arith.addi %a, %b : i32
    %1 = arith.addi %0, %c : i32
    handshake.return %1 : i32
  }
}
