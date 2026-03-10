module {
  handshake.func @straight_line(%a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["a", "b"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["c"]} {
    %c = arith.addi %a, %b : i32
    handshake.return %c : i32
  }
}
