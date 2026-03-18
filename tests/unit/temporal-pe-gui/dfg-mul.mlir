module {
  handshake.func @temporal_mul(%a: i32, %b: i32, %c: i32, ...) -> (i32)
      attributes {argNames = ["a", "b", "c"], resNames = ["prod"]} {
    %0 = arith.muli %a, %b : i32
    return %0 : i32
  }
}
