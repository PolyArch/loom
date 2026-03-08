module {
  handshake.func @add_mul(%a: i32, %b: i32, %c: i32, ...) -> (i32)
      attributes {argNames = ["a", "b", "c"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %0 = arith.addi %a, %b : i32
    %1 = arith.muli %0, %c : i32
    handshake.return %1 : i32
  }
}
