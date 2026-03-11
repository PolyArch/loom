module {
  handshake.func @single_sinf(%a: f32, ...) -> (f32)
      attributes {argNames = ["a"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = math.sin %a : f32
    handshake.return %c : f32
  }
}
