module {
  handshake.func @ext_add(%a: i32, %b: i32, ...) -> (i64)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %sum = arith.addi %a, %b : i32
    %c = arith.extsi %sum : i32 to i64
    handshake.return %c : i64
  }
}
