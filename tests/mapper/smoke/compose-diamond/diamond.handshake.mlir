module {
  handshake.func @diamond_add_mul_sub(%a: i32, %b: i32, %c: i32, ...) -> (i32)
      attributes {argNames = ["a", "b", "c"], loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %sum = arith.addi %a, %b : i32
    %prod = arith.muli %a, %c : i32
    %out = arith.subi %sum, %prod : i32
    handshake.return %out : i32
  }
}
