module {
  handshake.func @temporal_mac(%a: i32, %b: i32, %c: i32, ...) -> (i32)
      attributes {argNames = ["a", "b", "c"], resNames = ["mac"]} {
    %0 = arith.muli %a, %b : i32
    %1 = arith.addi %0, %c : i32
    return %1 : i32
  }
}
