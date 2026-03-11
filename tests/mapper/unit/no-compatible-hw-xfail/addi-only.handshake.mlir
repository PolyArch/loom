module {
  handshake.func @addi_only(%a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = arith.addi %a, %b : i32
    handshake.return %c : i32
  }
}
