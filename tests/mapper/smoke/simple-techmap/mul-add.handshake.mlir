module {
  handshake.func @mul_add(%a: i32, %b: i32, %c: i32, ...) -> (i32)
      attributes {argNames = ["a", "b", "c"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %0 = arith.muli %a, %b : i32
    %1 = arith.addi %0, %c : i32
    handshake.return %1 : i32
  }
}
