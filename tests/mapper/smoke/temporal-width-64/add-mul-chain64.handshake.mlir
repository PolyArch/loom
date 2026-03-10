module {
  handshake.func @add_mul_chain64(%a: i64, %b: i64, %c: i64, ...) -> (i64)
      attributes {argNames = ["a", "b", "c"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %0 = arith.addi %a, %b : i64
    %1 = arith.muli %0, %c : i64
    handshake.return %1 : i64
  }
}
