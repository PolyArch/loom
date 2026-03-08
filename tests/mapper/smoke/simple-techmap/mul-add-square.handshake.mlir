module {
  handshake.func @mul_add_square(%a: i32, %b: i32, %c: i32, ...) -> (i32)
      attributes {argNames = ["a", "b", "c"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %0 = arith.muli %a, %b : i32
    %d = arith.addi %0, %c : i32
    %out = arith.muli %d, %d : i32
    handshake.return %out : i32
  }
}
