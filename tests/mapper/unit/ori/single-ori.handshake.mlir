module {
  handshake.func @single_ori(%a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = arith.ori %a, %b : i32
    handshake.return %c : i32
  }
}
