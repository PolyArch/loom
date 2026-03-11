module {
  handshake.func @single_fptosi(%a: f32, ...) -> (i32)
      attributes {argNames = ["a"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = arith.fptosi %a : f32 to i32
    handshake.return %c : i32
  }
}
