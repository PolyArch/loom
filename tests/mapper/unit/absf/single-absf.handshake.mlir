module {
  handshake.func @single_absf(%a: f32, ...) -> (f32)
      attributes {argNames = ["a"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = math.absf %a : f32
    handshake.return %c : f32
  }
}
