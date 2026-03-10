module {
  handshake.func @single_addi64(%a: i64, %b: i64, ...) -> (i64)
      attributes {argNames = ["a", "b"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %0 = arith.addi %a, %b : i64
    handshake.return %0 : i64
  }
}
