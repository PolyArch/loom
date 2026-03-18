module {
  handshake.func @function_unit_passthrough_invalid(%arg0: i32) -> (i32) {
    handshake.return %arg0 : i32
  }
}
