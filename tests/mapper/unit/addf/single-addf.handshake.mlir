module {
  handshake.func @single_addf(%a: f32, %b: f32, ...) -> (f32)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = arith.addf %a, %b : f32
    handshake.return %c : f32
  }
}
