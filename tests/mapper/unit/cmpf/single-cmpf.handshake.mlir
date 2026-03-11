module {
  handshake.func @single_cmpf(%a: f32, %b: f32, ...) -> (i1)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = arith.cmpf olt, %a, %b : f32
    handshake.return %c : i1
  }
}
