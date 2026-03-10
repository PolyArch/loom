module {
  handshake.func @quad_addi(%a: i32, %b: i32, %c: i32, ...) -> (i32)
      attributes {argNames = ["a", "b", "c"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %0 = arith.addi %a, %b : i32
    %1 = arith.addi %0, %c : i32
    %2 = arith.addi %1, %a : i32
    %3 = arith.addi %2, %b : i32
    handshake.return %3 : i32
  }
}
