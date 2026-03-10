module {
  handshake.func @single_addi16(%a: i16, %b: i16, ...) -> (i16)
      attributes {argNames = ["a", "b"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %0 = arith.addi %a, %b : i16
    handshake.return %0 : i16
  }
}
