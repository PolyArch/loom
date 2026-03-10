module {
  handshake.func @dual_addi16(%a: i16, %b: i16, %c: i16, ...) -> (i16)
      attributes {argNames = ["a", "b", "c"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %0 = arith.addi %a, %b : i16
    %1 = arith.addi %0, %c : i16
    handshake.return %1 : i16
  }
}
