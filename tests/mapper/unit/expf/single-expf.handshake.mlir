module {
  handshake.func @single_expf(%a: f32, ...) -> (f32)
      attributes {argNames = ["a"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = math.exp %a : f32
    handshake.return %c : f32
  }
}
