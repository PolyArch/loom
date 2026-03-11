module {
  handshake.func @single_log2f(%a: f32, ...) -> (f32)
      attributes {argNames = ["a"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = math.log2 %a : f32
    handshake.return %c : f32
  }
}
