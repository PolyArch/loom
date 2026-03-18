module {
  handshake.func @temporal_dual_slot(%a0: i32, %b0: i32, %a1: i32, %b1: i32, ...) -> (i32, i32)
      attributes {argNames = ["a0", "b0", "a1", "b1"], resNames = ["sum", "prod"]} {
    %0 = arith.addi %a0, %b0 : i32
    %1 = arith.muli %a1, %b1 : i32
    return %0, %1 : i32, i32
  }
}
