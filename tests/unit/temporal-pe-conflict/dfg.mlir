module {
  handshake.func @temporal_conflict(%a0: i32, %b0: i32, %a1: i32, %b1: i32, %c1: i32, ...) -> (i32, i32)
      attributes {argNames = ["a0", "b0", "a1", "b1", "c1"], resNames = ["mul_only", "mac_result"]} {
    %mul_only = arith.muli %a0, %b0 : i32
    %prod = arith.muli %a1, %b1 : i32
    %mac = arith.addi %prod, %c1 : i32
    return %mul_only, %mac : i32, i32
  }
}
