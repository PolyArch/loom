module {
  handshake.func @tiny(%a: i32, %ctrl: none, ...) -> (i32, none)
      attributes {argNames = ["a", "ctrl"], resNames = ["out", "done"]} {
    return %a, %ctrl : i32, none
  }
}
