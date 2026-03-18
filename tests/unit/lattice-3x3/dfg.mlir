module {
  handshake.func @lattice_3x3(%a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["a", "b"], resNames = ["result"]} {
    %sum = arith.addi %a, %b : i32
    return %sum : i32
  }
}
