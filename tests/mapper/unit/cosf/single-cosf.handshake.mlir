module {
  handshake.func @single_cosf(%a: f32, ...) -> (f32)
      attributes {argNames = ["a"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = math.cos %a : f32
    handshake.return %c : f32
  }
}
