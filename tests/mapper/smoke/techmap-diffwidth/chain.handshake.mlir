module {
  handshake.func @chain(%a: i32, %b: i32, %c: i32, %d: i32, ...) -> (i64)
      attributes {argNames = ["a", "b", "c", "d"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %sum1 = arith.addi %a, %b : i32
    %ext1 = arith.extsi %sum1 : i32 to i64
    %sum2 = arith.addi %c, %d : i32
    %ext2 = arith.extsi %sum2 : i32 to i64
    %sum64 = arith.addi %ext1, %ext2 : i64
    %trunc = arith.trunci %sum64 : i64 to i32
    %out = arith.extsi %trunc : i32 to i64
    handshake.return %out : i64
  }
}
