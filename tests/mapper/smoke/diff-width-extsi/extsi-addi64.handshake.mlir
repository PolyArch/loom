module {
  handshake.func @extsi_addi64(%a: i32, %b: i32, ...) -> (i64)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %ea = arith.extsi %a : i32 to i64
    %eb = arith.extsi %b : i32 to i64
    %c = arith.addi %ea, %eb : i64
    handshake.return %c : i64
  }
}
