module {
  handshake.func @single_negf(%a: f32, ...) -> (f32)
      attributes {argNames = ["a"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = arith.negf %a : f32
    handshake.return %c : f32
  }
}
