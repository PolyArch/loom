module {
  handshake.func @temporal_buffer_size_missing(%a: i32, %b: i32, %c: i32, ...) -> (i32)
      attributes {argNames = ["a", "b", "c"], resNames = ["sum"]} {
    %0 = arith.addi %a, %b : i32
    return %0 : i32
  }
}
