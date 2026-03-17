// Tech-map target: add-only should map to fu_add.
module {
  handshake.func @tiny(%a: i32, %b: i32, %c: i32, %ctrl: none, ...) -> (i32, none) attributes {argNames = ["a", "b", "c", "ctrl"], resNames = ["result", "done"]} {
    %sum = arith.addi %a, %b : i32
    return %sum, %ctrl : i32, none
  }
}
