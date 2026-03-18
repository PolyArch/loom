module {
  handshake.func @function_unit_undriven_output_invalid(%arg0: i32) -> (i32, i32) {
    handshake.return %arg0, %arg0 : i32, i32
  }
}
