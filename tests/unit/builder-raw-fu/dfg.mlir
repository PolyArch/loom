module {
  handshake.func @builder_raw_fu(%a: i32, %b: i32, %c: i32, ...) -> (i32)
      attributes {argNames = ["a", "b", "c"], resNames = ["result"]} {
    %m = arith.muli %a, %b : i32
    %s = arith.addi %m, %c : i32
    return %s : i32
  }
}
