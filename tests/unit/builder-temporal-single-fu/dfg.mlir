module {
  handshake.func @builder_temporal_single_fu(%a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["a", "b"], resNames = ["sum"]} {
    %sum = arith.addi %a, %b : i32
    return %sum : i32
  }
}
