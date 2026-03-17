// Tech-map target: mul-only should map to fu_mac with sel=0.
module {
  handshake.func @tiny(%a: i32, %b: i32, %c: i32, %ctrl: none, ...) -> (i32, none) attributes {argNames = ["a", "b", "c", "ctrl"], resNames = ["result", "done"]} {
    %prod = arith.muli %a, %b : i32
    return %prod, %ctrl : i32, none
  }
}
