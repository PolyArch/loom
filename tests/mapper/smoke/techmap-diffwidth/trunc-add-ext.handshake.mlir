module {
  handshake.func @trunc_add_ext(%a: i64, %b: i64, ...) -> (i64)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %sum = arith.addi %a, %b : i64
    %t = arith.trunci %sum : i64 to i32
    %c = arith.extsi %t : i32 to i64
    handshake.return %c : i64
  }
}
