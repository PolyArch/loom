module {
  handshake.func @single_subi(%a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = arith.subi %a, %b : i32
    handshake.return %c : i32
  }
}
