module {
  handshake.func @mapper_fifo_bufferization(%a: i32, %b: i32, %c: i32, ...)
      -> (i32) attributes {argNames = ["a", "b", "c"], resNames = ["out"]} {
    %0 = arith.addi %a, %b : i32
    %1 = arith.addi %0, %c : i32
    return %1 : i32
  }
}
