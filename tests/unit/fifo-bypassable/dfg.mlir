module {
  handshake.func @fifo_bypassable(%a: i32, ...) -> (i32)
      attributes {argNames = ["a"], resNames = ["out"]} {
    return %a : i32
  }
}
