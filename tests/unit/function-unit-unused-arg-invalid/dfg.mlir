module {
  handshake.func @function_unit_unused_arg_invalid(%arg0: i32, %arg1: i32) -> (i32) {
    %0 = arith.addi %arg0, %arg1 : i32
    handshake.return %0 : i32
  }
}
