module {
  handshake.func @temporal_sw_nontag_invalid(%a: i32, ...) -> (i32)
      attributes {argNames = ["a"], resNames = ["out"]} {
    return %a : i32
  }
}
