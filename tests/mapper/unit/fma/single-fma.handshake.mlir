module {
  handshake.func @single_fma(%a: f32, %b: f32, %c: f32, ...) -> (f32)
      attributes {argNames = ["a", "b", "c"], loom.annotations = ["loom.accel"], resNames = ["d"]} {
    %d = math.fma %a, %b, %c : f32
    handshake.return %d : f32
  }
}
