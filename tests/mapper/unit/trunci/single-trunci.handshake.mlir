module {
  handshake.func @single_trunci(%a: i64, ...) -> (i32)
      attributes {argNames = ["a"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = arith.trunci %a : i64 to i32
    handshake.return %c : i32
  }
}
