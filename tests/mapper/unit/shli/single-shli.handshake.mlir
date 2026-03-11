module {
  handshake.func @single_shli(%a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = arith.shli %a, %b : i32
    handshake.return %c : i32
  }
}
