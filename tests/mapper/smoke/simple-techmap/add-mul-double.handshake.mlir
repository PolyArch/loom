module {
  handshake.func @add_mul_double(%a: i32, %b: i32, %c: i32, ...) -> (i32)
      attributes {argNames = ["a", "b", "c"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %0 = arith.addi %a, %b : i32
    %d = arith.muli %0, %c : i32
    %out = arith.addi %d, %d : i32
    handshake.return %out : i32
  }
}
