module {
  handshake.func @spatial_pe_exclusive(%a: i32, %b: i32, %c: i32, %d: i32,
                                       %ctrl: none, ...)
      -> (i32, i32, none)
      attributes {argNames = ["a", "b", "c", "d", "ctrl"],
                  resNames = ["sum", "prod", "done"]} {
    %sum = arith.addi %a, %b : i32
    %prod = arith.muli %c, %d : i32
    return %sum, %prod, %ctrl : i32, i32, none
  }
}
