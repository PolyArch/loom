module {
  handshake.func @tagged_spatial_sw_decomposable_invalid(%a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["a", "b"], resNames = ["sum"]} {
    %0 = arith.addi %a, %b : i32
    return %0 : i32
  }
}
