module {
  handshake.func @tagged_runtime_tag_collapse(%a: i32, %b: i32, ...) -> (i32, i32)
      attributes {argNames = ["a", "b"], resNames = ["out0", "out1"]} {
    return %a, %b : i32, i32
  }
}
