module {
  handshake.func @tree_add_mul(%a: i32, %b: i32, %c: i32, ...) -> (i32)
      attributes {argNames = ["a", "b", "c"], loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %sum = arith.addi %a, %b : i32
    %prod = arith.muli %b, %c : i32
    %out = arith.addi %sum, %prod : i32
    handshake.return %out : i32
  }
}
