module {
  handshake.func @fifo_depth_invalid(%a: i32, ...) -> (i32)
      attributes {argNames = ["a"], resNames = ["out"]} {
    return %a : i32
  }
}
