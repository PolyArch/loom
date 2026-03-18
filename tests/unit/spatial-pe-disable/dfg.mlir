module {
  handshake.func @spatial_pe_disable(%a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["a", "b"], resNames = ["sum"]} {
    %0 = arith.addi %a, %b : i32
    return %0 : i32
  }
}
