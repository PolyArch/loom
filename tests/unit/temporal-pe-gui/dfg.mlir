// Simple DFG used to smoke-test temporal_pe visualization presence.
module {
  handshake.func @add_only(%a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["a", "b"], resNames = ["sum"]} {
    %0 = arith.addi %a, %b : i32
    return %0 : i32
  }
}
