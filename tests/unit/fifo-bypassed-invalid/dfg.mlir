module {
  handshake.func @fifo_bypassed_invalid(%a: i32, ...) -> (i32)
      attributes {argNames = ["a"], resNames = ["out"]} {
    return %a : i32
  }
}
