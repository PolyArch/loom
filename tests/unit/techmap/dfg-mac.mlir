// Tech-map target: multiply then add should collapse into fu_mac.
module {
  handshake.func @tiny(%a: i32, %b: i32, %c: i32, %ctrl: none, ...) -> (i32, none) attributes {argNames = ["a", "b", "c", "ctrl"], resNames = ["result", "done"]} {
    %prod = arith.muli %a, %b : i32
    %sum = arith.addi %prod, %c : i32
    return %sum, %ctrl : i32, none
  }
}
