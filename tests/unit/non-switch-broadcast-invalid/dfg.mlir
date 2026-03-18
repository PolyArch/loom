module {
  handshake.func @non_switch_broadcast_invalid(%arg0: i32, %arg1: i32)
      -> (i32, i32) {
    %0 = arith.addi %arg0, %arg1 : i32
    handshake.return %0, %0 : i32, i32
  }
}
