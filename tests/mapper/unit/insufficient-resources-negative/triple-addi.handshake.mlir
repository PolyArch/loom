module {
  handshake.func @triple_addi(%a: i32, %b: i32, %c: i32, %d: i32, ...) -> (i32)
      attributes {argNames = ["a", "b", "c", "d"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %0 = arith.addi %a, %b : i32
    %1 = arith.addi %c, %d : i32
    %2 = arith.addi %0, %1 : i32
    handshake.return %2 : i32
  }
}
