module {
  handshake.func @chess_customized(%a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["a", "b"], resNames = ["result"]} {
    %prod = arith.muli %a, %b : i32
    return %prod : i32
  }
}
