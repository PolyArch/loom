module {
  handshake.func @temporal_reg_chain(%a: i32, %b: i32, %c: i32, ...) -> (i32)
      attributes {argNames = ["a", "b", "c"], resNames = ["result"]} {
    %sum = arith.addi %a, %b : i32
    %prod = arith.muli %sum, %c : i32
    return %prod : i32
  }
}
