module {
  handshake.func @dual_reg_chain(%a: i32, %b: i32, %c: i32, %d: i32, ...) -> (i32)
      attributes {argNames = ["a", "b", "c", "d"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %0 = arith.addi %a, %b : i32
    %1 = arith.muli %0, %c : i32
    %2 = arith.addi %1, %d : i32
    handshake.return %2 : i32
  }
}
