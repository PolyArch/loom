module {
  handshake.func @two_stream_diverge(%a: i32, %b: i32, %c: i32, %d: i32, ...) -> (i32, i32)
      attributes {argNames = ["a", "b", "c", "d"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["out0", "out1"]} {
    %0 = arith.addi %a, %b : i32
    %1 = arith.addi %c, %d : i32
    handshake.return %0, %1 : i32, i32
  }
}
