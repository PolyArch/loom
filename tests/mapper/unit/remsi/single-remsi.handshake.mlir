module {
  handshake.func @single_remsi(%a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = arith.remsi %a, %b : i32
    handshake.return %c : i32
  }
}
