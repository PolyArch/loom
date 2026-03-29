module {
  handshake.func @techmap_cmpsel(%a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["a", "b"], resNames = ["result"]} {
    %cmp = arith.cmpi slt, %a, %b : i32
    %sel = arith.select %cmp, %a, %b : i32
    return %sel : i32
  }
}
