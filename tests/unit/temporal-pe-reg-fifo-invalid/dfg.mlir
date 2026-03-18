module {
  handshake.func @temporal_reg_fifo_invalid(%a: i32, %b: i32, %c: i32, ...) -> (i32)
      attributes {argNames = ["a", "b", "c"], resNames = ["sum"]} {
    %0 = arith.addi %a, %b : i32
    return %0 : i32
  }
}
