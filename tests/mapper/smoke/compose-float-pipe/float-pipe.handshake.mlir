module {
  handshake.func @float_pipe(%a: i32, %b: f32, ...) -> (i32)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %f = arith.sitofp %a : i32 to f32
    %prod = arith.mulf %f, %b : f32
    %out = arith.fptosi %prod : f32 to i32
    handshake.return %out : i32
  }
}
