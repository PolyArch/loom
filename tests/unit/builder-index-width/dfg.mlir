module {
  handshake.func @builder_index_width_test(%lhs: i32, %rhs: i32, %bias: i32, ...) -> (i32) attributes {argNames = ["lhs", "rhs", "bias"], resNames = ["sum"]} {
    %tmp = arith.addi %lhs, %rhs : i32
    %sum = arith.addi %tmp, %bias : i32
    return %sum : i32
  }
}
