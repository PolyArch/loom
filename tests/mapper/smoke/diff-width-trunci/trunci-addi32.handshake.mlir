module {
  handshake.func @trunci_addi32(%a: i64, %b: i64, ...) -> (i32)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %ta = arith.trunci %a : i64 to i32
    %tb = arith.trunci %b : i64 to i32
    %c = arith.addi %ta, %tb : i32
    handshake.return %c : i32
  }
}
