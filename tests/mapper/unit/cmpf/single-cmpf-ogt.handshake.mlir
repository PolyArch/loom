module {
  handshake.func @single_cmpf_ogt(%a: f32, %b: f32, ...) -> (i1)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = arith.cmpf ogt, %a, %b : f32
    handshake.return %c : i1
  }
}
