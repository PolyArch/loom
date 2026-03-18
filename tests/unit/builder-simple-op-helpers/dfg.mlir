module {
  handshake.func @builder_simple_op_helpers_test(%lhs: i64, %rhs: i64, ...) -> (i32) attributes {argNames = ["lhs", "rhs"], resNames = ["sum"]} {
    %lhs32 = arith.trunci %lhs : i64 to i32
    %rhs32 = arith.trunci %rhs : i64 to i32
    %sum = arith.addi %lhs32, %rhs32 : i32
    return %sum : i32
  }
}
