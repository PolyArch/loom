module {
  handshake.func @single_sitofp(%a: i32, ...) -> (f32)
      attributes {argNames = ["a"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = arith.sitofp %a : i32 to f32
    handshake.return %c : f32
  }
}
