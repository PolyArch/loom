module {
  handshake.func @roundtrip(%a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %ea = arith.extsi %a : i32 to i64
    %eb = arith.extsi %b : i32 to i64
    %sum64 = arith.addi %ea, %eb : i64
    %c = arith.trunci %sum64 : i64 to i32
    handshake.return %c : i32
  }
}
