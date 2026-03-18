module {
  handshake.func @cube_2x2x2(%a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["a", "b"], resNames = ["result"]} {
    %sum = arith.addi %a, %b : i32
    return %sum : i32
  }
}
