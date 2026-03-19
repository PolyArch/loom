module {
  handshake.func @function_unit_nonstream_timing_na_invalid(%a: i32, %b: i32) -> (i32) {
    %0 = arith.addi %a, %b : i32
    handshake.return %0 : i32
  }
}
