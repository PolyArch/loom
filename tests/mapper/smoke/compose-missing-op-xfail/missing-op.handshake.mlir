module {
  handshake.func @missing_op(%a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %0 = arith.muli %a, %b : i32
    handshake.return %0 : i32
  }
}
