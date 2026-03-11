module {
  handshake.func @single_xori(%a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = arith.xori %a, %b : i32
    handshake.return %c : i32
  }
}
