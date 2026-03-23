module {
  handshake.func @system_adg_builder_test(%lhs: i32, %rhs: i32, ...) -> (i32) attributes {argNames = ["lhs", "rhs"], resNames = ["sum"]} {
    %sum = arith.addi %lhs, %rhs : i32
    return %sum : i32
  }
}
