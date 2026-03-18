module {
  handshake.func @builder_switch_bank_test(%a: i32, %b: i32, %c: i32, ...) -> (i32) attributes {argNames = ["a", "b", "c"], resNames = ["sum"]} {
    %tmp = arith.addi %a, %b : i32
    %sum = arith.addi %tmp, %c : i32
    return %sum : i32
  }
}
