module {
  handshake.func @builder_sw_grid(%a: i32, %b: i32, %c: i32, ...)
      -> (i32)
      attributes {
        argNames = ["a", "b", "c"],
        resNames = ["y"]
      } {
    %x = arith.addi %a, %b : i32
    %y = arith.addi %x, %c : i32
    return %y : i32
  }
}
