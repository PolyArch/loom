module {
  handshake.func @single_extsi(%a: i32, ...) -> (i64)
      attributes {argNames = ["a"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = arith.extsi %a : i32 to i64
    handshake.return %c : i64
  }
}
